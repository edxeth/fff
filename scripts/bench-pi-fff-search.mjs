#!/usr/bin/env node
import { spawnSync } from "node:child_process";
import { existsSync, mkdirSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const root = dirname(dirname(fileURLToPath(import.meta.url)));
const model = process.env.PI_FFF_BENCH_MODEL;
const caseTimeoutMs = Number(process.env.PI_FFF_BENCH_TIMEOUT_MS ?? 150000);
const outDir =
  process.env.PI_FFF_BENCH_OUT ??
  join(root, ".tmp", "pi-fff-bench", new Date().toISOString().replace(/[:.]/g, "-"));

const repos = [
  { name: "fff", path: process.env.PI_FFF_BENCH_FFF_REPO ?? root },
  { name: "monolith", path: process.env.PI_FFF_BENCH_MONOLITH_REPO },
  { name: "pi-subagents", path: process.env.PI_FFF_BENCH_PI_SUBAGENTS_REPO },
].filter((repo) => repo.path && existsSync(repo.path));

const prompts = [
  {
    name: "security-audit-map",
    repo: "monolith",
    text: (repo) =>
      `In ${repo}, act like a senior engineer starting a 3-day security audit. Map the auth/session/security surface without reading files yet. Find relevant files for wallet login, OAuth/social sign-in, SIWS/SIWT/SIWE, nonce/session/refresh-token handling, CSRF/CORS/input validation, rate limiting, Turnstile, audit logging, and admin 2FA. Use search tools heavily but keep result noise low. Summarize only the relevant files grouped by subsystem and mention which searches were noisy or inconclusive.`,
  },
  {
    name: "exact-symbol-refactor",
    repo: "monolith",
    text: (repo) =>
      `In ${repo}, prepare for a large refactor. Without reading files, map definitions and references for these exact symbols and related files: sanitizeRedirect, verifyTurnstile, logAdminAction, AdminTwoFactorPage, useOAuthSession, refreshToken, AuthChallenge, WalletAccount, RateLimiter, generateReferralCode. Use search tools aggressively but avoid broad regex noise. Summarize files grouped by symbol and say which symbols look absent or ambiguous.`,
  },
  {
    name: "broad-regex-stress",
    repo: "monolith",
    text: (repo) =>
      `In ${repo}, deliberately stress content search. Run one grep with this exact regex pattern: auth|session|token|user|error|state|handler|service|manager|controller in ${repo}. Then say whether the tool warned you that the regex was broad, and summarize only the first-page result shape.`,
  },
  {
    name: "mixed-rust-ts-search",
    repo: "fff",
    text: (repo) =>
      `In ${repo}, act like a senior engineer tracing the search pipeline. Without reading files, map the files for fuzzy file search, grep, multi-pattern grep, query parsing, FFI bindings, cursor pagination, and pi extension tool registration. Use search tools heavily but keep noise low. Summarize relevant files by subsystem.`,
  },
  {
    name: "ui-render-input-trace",
    repo: "pi-subagents",
    text: (repo) =>
      `In ${repo}/src, inspect code paths for scroll, render, handleInput, Widget/Dialog/Selector classes. Use grep/find tools as appropriate, then summarize only which files seem relevant. Do not read files yet.`,
  },
];

const promptFilter = new Set(
  (process.env.PI_FFF_BENCH_PROMPTS ?? "")
    .split(",")
    .map((name) => name.trim())
    .filter(Boolean),
);
const selectedPrompts =
  promptFilter.size > 0
    ? prompts.filter((prompt) => promptFilter.has(prompt.name))
    : prompts.filter((prompt) => repos.some((repo) => repo.name === prompt.repo));

const variants = [
  {
    name: "builtin",
    args: ["--no-extensions", "--tools", "grep,find"],
  },
  {
    name: "fff",
    args: [
      "--no-extensions",
      "--no-builtin-tools",
      "--fff-mode",
      "override",
      "--extension",
      "./packages/pi-fff/src/index.ts",
      "--tools",
      "grep,find,multi_grep",
    ],
  },
];

function parseJsonLines(stdout) {
  const entries = [];
  for (const line of stdout.split("\n")) {
    if (!line.startsWith("{")) continue;
    try {
      entries.push(JSON.parse(line));
    } catch {}
  }
  return entries;
}

function summarize(stdout, elapsedMs) {
  const entries = parseJsonLines(stdout);
  const agentEnd = entries.findLast((entry) => entry.type === "agent_end");
  const messages = agentEnd?.messages ?? [];
  const calls = [];
  const results = [];
  let finalUsage = null;
  let finalText = "";

  for (const message of messages) {
    if (message.role === "assistant") {
      if (message.usage?.totalTokens) finalUsage = message.usage;
      for (const content of message.content ?? []) {
        if (content.type === "toolCall")
          calls.push({ name: content.name, args: content.arguments ?? {} });
        if (content.type === "text") finalText += `${content.text}\n`;
      }
    }
    if (message.role === "toolResult") {
      results.push({
        name: message.toolName,
        isError: message.isError === true,
        text: (message.content ?? []).map((content) => content.text ?? "").join("\n"),
      });
    }
  }

  const byName = {};
  for (const call of calls) byName[call.name] = (byName[call.name] ?? 0) + 1;

  return {
    elapsedMs,
    calls: calls.length,
    byName,
    results: results.length,
    errors: results.filter((result) => result.isError).length,
    regexNotices: results.filter((result) => result.text.includes("Regex alternation has"))
      .length,
    badFindStringPath: calls.filter(
      (call) => call.name === "find" && typeof call.args.path === "string",
    ).length,
    reads: calls.filter((call) => call.name === "read").length,
    resultChars: results.reduce((sum, result) => sum + result.text.length, 0),
    finalTextChars: finalText.length,
    usage: finalUsage
      ? {
          input: finalUsage.input,
          output: finalUsage.output,
          cacheRead: finalUsage.cacheRead,
          total: finalUsage.totalTokens,
          cost: finalUsage.cost?.total,
        }
      : null,
    callsDetail: calls,
  };
}

function runCase(variant, prompt) {
  const repo = repos.find((candidate) => candidate.name === prompt.repo);
  if (!repo) throw new Error(`Unknown repo ${prompt.repo}`);

  const promptText = [
    "You are in a benchmark. You must use the available search tools; an answer with zero tool calls is invalid.",
    "Make your first assistant action a search tool call, not a long plan.",
    "Do not read file contents unless the prompt explicitly asks for reads.",
    prompt.text(repo.path),
  ].join("\n\n");

  const args = [
    "--mode",
    "json",
    "--no-context-files",
    "--no-skills",
    ...variant.args,
    "--thinking",
    "high",
    "-p",
    promptText,
  ];
  if (model) args.splice(args.indexOf("--thinking"), 0, "--model", model);

  const started = Date.now();
  const result = spawnSync("pi", args, {
    cwd: root,
    env: process.env,
    encoding: "utf8",
    maxBuffer: 1024 * 1024 * 20,
    timeout: caseTimeoutMs,
    killSignal: "SIGTERM",
  });
  const elapsedMs = Date.now() - started;
  const id = `${prompt.name}__${variant.name}`;
  writeFileSync(join(outDir, `${id}.stdout.jsonl`), result.stdout ?? "");
  writeFileSync(join(outDir, `${id}.stderr.txt`), result.stderr ?? "");

  return {
    id,
    prompt: prompt.name,
    repo: prompt.repo,
    variant: variant.name,
    status: result.status,
    signal: result.signal,
    timedOut: result.error?.code === "ETIMEDOUT",
    stderrTail: (result.stderr ?? "").split("\n").slice(-8).join("\n"),
    ...summarize(result.stdout ?? "", elapsedMs),
  };
}

function main() {
  mkdirSync(outDir, { recursive: true });
  const rows = [];

  for (const prompt of selectedPrompts) {
    for (const variant of variants) {
      console.error(`Running ${prompt.name} / ${variant.name}`);
      const row = runCase(variant, prompt);
      rows.push(row);
      writeFileSync(join(outDir, "summary.json"), JSON.stringify(rows, null, 2));
    }
  }

  const report = [
    `# pi-fff benchmark ${new Date().toISOString()}`,
    "",
    `Model: ${model ?? "pi default"}`,
    `Per-case timeout: ${caseTimeoutMs}ms`,
    `Output: ${outDir}`,
    "",
    "| Prompt | Variant | Exit | Wall s | Calls | Errors | Bad find path | Regex notices | Result chars | Tokens | Cost |",
    "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ...rows.map((row) =>
      [
        row.prompt,
        row.variant,
        row.status,
        (row.elapsedMs / 1000).toFixed(1),
        row.calls,
        row.errors,
        row.badFindStringPath,
        row.regexNotices,
        row.resultChars,
        row.usage?.total ?? "",
        row.usage?.cost?.toFixed?.(6) ?? "",
      ]
        .join(" | ")
        .replace(/^/, "|")
        .replace(/$/, "|"),
    ),
    "",
  ].join("\n");
  writeFileSync(join(outDir, "report.md"), report);
  console.log(report);
}

main();
