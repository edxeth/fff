/**
 * pi-fff: FFF-powered file search extension for pi
 *
 * Overrides built-in `find` and `grep` tools with FFF and can also replace
 * @-mention autocomplete suggestions in the interactive editor.
 */

import { execSync } from "node:child_process";
import fs from "node:fs";
import path from "node:path";
import type {
  GrepCursor,
  GrepMode,
  GrepResult,
  MixedItem,
  SearchResult,
} from "@edxeth/fff-node";
import { FileFinder } from "@edxeth/fff-node";
import type { ExtensionAPI } from "@mariozechner/pi-coding-agent";
import { CustomEditor } from "@mariozechner/pi-coding-agent";
import {
  type AutocompleteItem,
  type AutocompleteProvider,
  Text,
} from "@mariozechner/pi-tui";
import { Type } from "@sinclair/typebox";
import { buildQuery } from "./query";

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const DEFAULT_GREP_LIMIT = 20;
const DEFAULT_FIND_LIMIT = 30;
const MAX_CACHED_FINDERS = 4;
const GREP_MAX_LINE_LENGTH = 500;
const MENTION_MAX_RESULTS = 20;

type FffMode = "tools-and-ui" | "tools-only" | "override";

const VALID_MODES: FffMode[] = ["tools-and-ui", "tools-only", "override"];

interface ToolNames {
  grep: string;
  find: string;
  multiGrep: string;
}

const FFF_TOOL_NAMES: ToolNames = {
  grep: "ffgrep",
  find: "fffind",
  multiGrep: "fff-multi-grep",
};
const OVERRIDE_TOOL_NAMES: ToolNames = {
  grep: "grep",
  find: "find",
  multiGrep: "multi_grep",
};

function resolveToolNames(mode: FffMode): ToolNames {
  return mode === "override" ? OVERRIDE_TOOL_NAMES : FFF_TOOL_NAMES;
}

// ---------------------------------------------------------------------------
// Cursor store — simple bounded Map for pagination cursors
// ---------------------------------------------------------------------------

const cursorCache = new Map<string, GrepCursor>();
let cursorCounter = 0;

function storeCursor(cursor: GrepCursor): string {
  const id = `fff_c${++cursorCounter}`;
  cursorCache.set(id, cursor);
  if (cursorCache.size > 200) {
    const first = cursorCache.keys().next().value;
    if (first) cursorCache.delete(first);
  }
  return id;
}

function getCursor(id: string): GrepCursor | undefined {
  return cursorCache.get(id);
}

// Find pagination uses a page-index cursor: native `fileSearch` takes
// pageIndex/pageSize, so the cursor is just the next page index paired with
// the query+limit that produced it. Stored tokens are opaque IDs to the agent.
interface FindCursor {
  basePath: string;
  query: string;
  pattern: string;
  pageSize: number;
  nextPageIndex: number;
}

const findCursorCache = new Map<string, FindCursor>();
let findCursorCounter = 0;

function storeFindCursor(cursor: FindCursor): string {
  const id = `${++findCursorCounter}`;
  findCursorCache.set(id, cursor);
  if (findCursorCache.size > 200) {
    const first = findCursorCache.keys().next().value;
    if (first) findCursorCache.delete(first);
  }
  return id;
}

function getFindCursor(id: string): FindCursor | undefined {
  return findCursorCache.get(id);
}

// ---------------------------------------------------------------------------
// Output formatting helpers
// ---------------------------------------------------------------------------

function truncateLine(line: string, max = GREP_MAX_LINE_LENGTH): string {
  const trimmed = line.trim();
  return trimmed.length <= max ? trimmed : `${trimmed.slice(0, max)}...`;
}

const HOT_FRECENCY = 25;
const WARM_FRECENCY = 20;

// Shared annotation helper for both find-output paths and grep-output file
// headers. Returns at most ONE tag so output stays scannable. Priority:
// git-dirty (most actionable — file is changing right now) beats frecency
// (historically often-touched). Keeping one function ensures the two tools
// never drift in how they surface git/frecency signal.
export function fffFileAnnotation(item: {
  gitStatus?: string;
  totalFrecencyScore?: number;
  accessFrecencyScore?: number;
}): string {
  const git = item.gitStatus;
  if (git && git !== "clean" && git !== "unknown" && git !== "") {
    return `  [${git} in git]`;
  }

  const frecency = item.totalFrecencyScore ?? item.accessFrecencyScore ?? 0;
  if (frecency >= HOT_FRECENCY) return "  [VERY often touched file]";
  if (frecency >= WARM_FRECENCY) return "  [often touched file]";

  return "";
}

// fff-core native definition classifier (byte-level scanner in Rust) is enabled
// via GrepOptions.classifyDefinitions. Each GrepMatch carries isDefinition for
// downstream consumers; pi-fff does NOT use it to re-sort.
//
// Ordering policy: NO CUSTOM SORTING. The engine already returns items in
// frecency order (most-accessed files first). pi-fff only groups consecutive
// matches into per-file blocks and preserves whatever order the engine
// provided — inside a file we keep matches in source-line order because the
// engine emits them that way.

function formatGrepOutput(result: GrepResult): string {
  if (result.items.length === 0) return "No matches found";

  // Build file-grouped output in the order files first appear in the result.
  // This preserves native frecency ordering across files without re-sorting.
  const lines: string[] = [];
  let currentFile = "";
  let _shown = 0;

  for (const match of result.items) {
    if (match.relativePath !== currentFile) {
      if (lines.length > 0) lines.push("");
      currentFile = match.relativePath;
      lines.push(`${currentFile}${fffFileAnnotation(match)}`);
    }

    match.contextBefore?.forEach((line: string, i: number) => {
      const lineNum = match.lineNumber - match.contextBefore!.length + i;
      lines.push(` ${lineNum}- ${truncateLine(line)}`);
    });

    lines.push(` ${match.lineNumber}: ${truncateLine(match.lineContent)}`);
    _shown++;

    match.contextAfter?.forEach((line: string, i: number) => {
      const lineNum = match.lineNumber + 1 + i;
      lines.push(` ${lineNum}- ${truncateLine(line)}`);
    });
  }

  return lines.join("\n");
}

// Weak-match threshold is derived from the query length, matching the
// scoring formula in crates/fff-core/src/score.rs: a perfect match scores
// `len * 16`, so we treat anything below 50% of that as scattered fuzzy noise.
// When the top score is weak, trim output to a small sample instead of dumping
// the full limit worth of noise into the agent's context.
const FIND_WEAK_SAMPLE_SIZE = 5;

function weakScoreThreshold(pattern: string): number {
  const perfect = pattern.length * 12;
  return Math.floor((perfect * 50) / 100);
}

interface FormattedFind {
  output: string;
  weak: boolean;
  shownCount: number;
  literalTailSuppressed: boolean;
}

function normalizeLiteralPattern(pattern: string): string | null {
  const trimmed = pattern.trim().toLowerCase();
  return /^[a-z0-9._-]+$/.test(trimmed) ? trimmed : null;
}

function pathHasLiteralSegment(relativePath: string, pattern: string): boolean {
  const literal = normalizeLiteralPattern(pattern);
  if (!literal) return false;

  return relativePath
    .toLowerCase()
    .split("/")
    .some((segment) => segment === literal || segment.startsWith(`${literal}.`));
}

function patternLooksLikePath(pattern: string): boolean {
  return /[\\/]|[*?[{]/.test(pattern);
}

function pathLikePatternMessage(_pattern: string): string {
  return "Path/glob belongs in path, not pattern";
}

function pathLooksLikeMultiplePaths(pathConstraint: string): boolean {
  const parts = pathConstraint.trim().split(/\s+/).filter(Boolean);
  if (parts.length < 2) return false;
  return parts.every((part) => part.includes("/") || part.includes("\\"));
}

function multiplePathsMessage(): string {
  return "Multiple paths are not supported in path; use one file, directory, or glob";
}

function formatFindOutput(
  result: SearchResult,
  limit: number,
  pattern: string,
): FormattedFind {
  if (result.items.length === 0) {
    return {
      output: "No files found matching pattern",
      weak: false,
      shownCount: 0,
      literalTailSuppressed: false,
    };
  }

  // NO CUSTOM SORTING — trust native frecency order from the engine.
  const reordered = result.items.map((item) => ({ item }));

  // Peek at the top native score to decide whether results are scattered
  // fuzzy noise (query length-scaled threshold from score.rs).
  const topScore = result.scores[0]?.total ?? 0;
  const weak = topScore < weakScoreThreshold(pattern);
  const literalFiltered =
    !weak &&
    pathHasLiteralSegment(result.items[0]?.relativePath ?? "", pattern) &&
    result.totalMatched > FIND_WEAK_SAMPLE_SIZE;
  const effective = weak ? Math.min(FIND_WEAK_SAMPLE_SIZE, limit) : limit;
  const shown = literalFiltered
    ? reordered
        .filter((p) => pathHasLiteralSegment(p.item.relativePath, pattern))
        .slice(0, effective)
    : reordered.slice(0, effective);

  return {
    output: shown
      .map((p) => `${p.item.relativePath}${fffFileAnnotation(p.item)}`)
      .join("\n"),
    weak,
    shownCount: shown.length,
    literalTailSuppressed: literalFiltered && shown.length < result.totalMatched,
  };
}

// ---------------------------------------------------------------------------
// Mention autocomplete helpers
// ---------------------------------------------------------------------------

function extractAtPrefix(textBeforeCursor: string): string | null {
  const match = textBeforeCursor.match(/(?:^|[ \t])(@(?:"[^"]*|[^\s]*))$/);
  return match?.[1] ?? null;
}

function buildAtCompletionValue(path: string): string {
  return path.includes(" ") ? `@"${path}"` : `@${path}`;
}

function createFffMentionProvider(
  getItems: (query: string, signal: AbortSignal) => Promise<AutocompleteItem[]>,
): AutocompleteProvider {
  return {
    async getSuggestions(lines, cursorLine, cursorCol, options) {
      const currentLine = lines[cursorLine] || "";
      const prefix = extractAtPrefix(currentLine.slice(0, cursorCol));
      if (!prefix || options.signal.aborted) return null;

      const query = prefix.startsWith('@"') ? prefix.slice(2) : prefix.slice(1);
      const items = await getItems(query, options.signal);
      return options.signal.aborted || items.length === 0 ? null : { items, prefix };
    },
    applyCompletion(_lines, cursorLine, cursorCol, item, prefix) {
      const currentLine = _lines[cursorLine] || "";
      const before = currentLine.slice(0, cursorCol - prefix.length);
      const after = currentLine.slice(cursorCol);
      const newLine = before + item.value + after;
      const newCursorCol = cursorCol - prefix.length + item.value.length;
      return {
        lines: [..._lines.slice(0, cursorLine), newLine, ..._lines.slice(cursorLine + 1)],
        cursorLine,
        cursorCol: newCursorCol,
      };
    },
  };
}

// FffEditor is defined inside fffExtension() so it can capture `getMentionItems`
// via closure rather than via a 4th constructor parameter. This makes the class
// safe to subclass via `new SubClass(tui, theme, keybindings)` -- the pattern
// pi-vim and pi-image-attachments use to compose editors. See:
// https://github.com/badlogic/pi-mono/issues/3935

// ---------------------------------------------------------------------------
// Extension
// ---------------------------------------------------------------------------

export default function fffExtension(pi: ExtensionAPI) {
  const finders = new Map<string, FileFinder>();
  let activeBasePath: string | null = null;
  // Concurrent ensureFinder() callers share in-flight promises by base path so
  // FileFinder.create() (which takes native DB locks) runs at most once per
  // base path at a time — otherwise parallel tool calls would race and
  // deadlock at the native layer (issue #403).
  const finderPromises = new Map<string, Promise<FileFinder>>();
  const finderLocks = new Map<string, Promise<void>>();
  const finderActiveOps = new Map<string, number>();
  let activeCwd = process.cwd();

  // Mode resolution: flag > env > default
  let currentMode: FffMode =
    (pi.getFlag("fff-mode") as FffMode) ??
    (process.env.PI_FFF_MODE as FffMode) ??
    "tools-and-ui";

  const toolNames = resolveToolNames(currentMode);

  // DB path resolution: flag > env > undefined (use fff-node defaults)
  const frecencyDbPath =
    (pi.getFlag("fff-frecency-db") as string | undefined) ??
    process.env.FFF_FRECENCY_DB ??
    undefined;
  const historyDbPath =
    (pi.getFlag("fff-history-db") as string | undefined) ??
    process.env.FFF_HISTORY_DB ??
    undefined;

  function getMode(): FffMode {
    return currentMode;
  }

  function setMode(mode: FffMode): void {
    currentMode = mode;
  }

  function shouldEnableMentions(): boolean {
    return currentMode !== "tools-only";
  }

  function resolveGitRoot(dir: string): string | null {
    try {
      const root = execSync("git rev-parse --show-toplevel", {
        cwd: dir,
        encoding: "utf8",
        stdio: ["ignore", "pipe", "ignore"],
        timeout: 3000,
      }).trim();
      return root || null;
    } catch {
      return null;
    }
  }

  function expandHomePath(pathConstraint: string): string {
    const home = process.env.HOME ?? process.env.USERPROFILE;
    if (!home) return pathConstraint;
    return pathConstraint.replace(/^~($|\/|\\)/, (_, sep) => home + sep);
  }

  function concreteStatPath(pathConstraint: string, cwd = activeCwd): string {
    const expanded = expandHomePath(pathConstraint);
    const absolute = path.isAbsolute(expanded) ? expanded : path.resolve(cwd, expanded);
    const wildcard = absolute.search(/[*?[{]/);
    const concrete = wildcard === -1 ? absolute : absolute.slice(0, wildcard);
    if (wildcard === -1) return absolute;
    return concrete.endsWith(path.sep) ? concrete.slice(0, -1) : path.dirname(concrete);
  }

  function hasHiddenSegment(pathConstraint: string): boolean {
    return pathConstraint
      .split(/[\\/]+/)
      .some((segment) => segment.startsWith(".") && segment !== "." && segment !== "..");
  }

  function invalidPathMessage(pathConstraint: string, cwd = activeCwd): string | null {
    const statPath = concreteStatPath(pathConstraint, cwd);
    return fs.existsSync(statPath)
      ? null
      : `Path not found: ${statPath || pathConstraint}`;
  }

  function absolutePathBase(pathConstraint: string): {
    basePath: string;
    pathConstraint?: string;
  } {
    const wildcard = pathConstraint.search(/[*?[{]/);
    const hasWildcard = wildcard !== -1;
    const concrete = hasWildcard ? pathConstraint.slice(0, wildcard) : pathConstraint;
    const concreteDir = concrete.endsWith(path.sep)
      ? concrete.slice(0, -1)
      : path.dirname(concrete);
    const statPath = hasWildcard ? concreteDir : pathConstraint;
    const isDir = fs.existsSync(statPath) && fs.statSync(statPath).isDirectory();
    const fallbackBase = isDir ? statPath : path.dirname(statPath);
    const gitRoot = resolveGitRoot(fallbackBase);
    const basePath = gitRoot ?? fallbackBase;
    const relative = path.relative(basePath, pathConstraint).replaceAll(path.sep, "/");
    const pathValue =
      relative && relative !== "**" && relative !== "**/*" ? relative : undefined;
    return { basePath, pathConstraint: pathValue };
  }

  function resolveSearchBase(pathConstraint: string | undefined): {
    basePath: string;
    pathConstraint?: string;
  } {
    if (!pathConstraint) return { basePath: activeCwd, pathConstraint };
    const expanded = expandHomePath(pathConstraint);
    if (path.isAbsolute(expanded)) return absolutePathBase(expanded);
    if (expanded === ".." || expanded.startsWith(`..${path.sep}`)) {
      return absolutePathBase(path.resolve(activeCwd, expanded));
    }
    if (/\s/.test(expanded) && fs.existsSync(concreteStatPath(expanded))) {
      return absolutePathBase(path.resolve(activeCwd, expanded));
    }
    if (hasHiddenSegment(expanded) && fs.existsSync(concreteStatPath(expanded))) {
      return absolutePathBase(path.resolve(activeCwd, expanded));
    }
    return { basePath: activeCwd, pathConstraint };
  }

  function trimFinderCache() {
    while (finders.size >= MAX_CACHED_FINDERS) {
      const evictable = [...finders.entries()].find(
        ([basePath]) => (finderActiveOps.get(basePath) ?? 0) === 0,
      );
      if (!evictable) return;

      const [oldestBase, oldestFinder] = evictable;
      if (!oldestFinder.isDestroyed) oldestFinder.destroy();
      finders.delete(oldestBase);
      if (activeBasePath === oldestBase) activeBasePath = null;
    }
  }

  function ensureFinder(basePath: string): Promise<FileFinder> {
    const existing = finders.get(basePath);
    if (existing && !existing.isDestroyed) return Promise.resolve(existing);
    const pending = finderPromises.get(basePath);
    if (pending) return pending;

    const promise = (async () => {
      trimFinderCache();
      const useDatabases = basePath === activeCwd;
      const result = FileFinder.create({
        basePath,
        frecencyDbPath: useDatabases ? frecencyDbPath : undefined,
        historyDbPath: useDatabases ? historyDbPath : undefined,
        aiMode: true,
      });

      if (!result.ok)
        throw new Error(`Failed to create FFF file finder: ${result.error}`);

      const finder = result.value;
      finders.set(basePath, finder);
      activeBasePath = basePath;
      await finder.waitForScan(15000);
      return finder;
    })().finally(() => {
      finderPromises.delete(basePath);
    });

    finderPromises.set(basePath, promise);
    return promise;
  }

  function destroyFinder() {
    for (const finder of finders.values()) {
      if (!finder.isDestroyed) finder.destroy();
    }
    finders.clear();
    finderLocks.clear();
    finderActiveOps.clear();
    activeBasePath = null;
  }

  async function withFinderLease<T>(
    basePath: string,
    work: (finder: FileFinder) => T | Promise<T>,
  ): Promise<T> {
    const previous = finderLocks.get(basePath) ?? Promise.resolve();
    let release!: () => void;
    const current = new Promise<void>((resolve) => {
      release = resolve;
    });
    finderLocks.set(
      basePath,
      previous.then(
        () => current,
        () => current,
      ),
    );

    await previous.catch(() => undefined);
    finderActiveOps.set(basePath, (finderActiveOps.get(basePath) ?? 0) + 1);
    try {
      const finder = await ensureFinder(basePath);
      return await work(finder);
    } finally {
      const remaining = (finderActiveOps.get(basePath) ?? 1) - 1;
      if (remaining > 0) finderActiveOps.set(basePath, remaining);
      else finderActiveOps.delete(basePath);
      release();
      if (finderLocks.get(basePath) === current) finderLocks.delete(basePath);
    }
  }

  function getActiveFinder(): FileFinder | null {
    if (!activeBasePath) return null;
    const finder = finders.get(activeBasePath);
    return finder && !finder.isDestroyed ? finder : null;
  }

  async function getMentionItems(
    query: string,
    signal: AbortSignal,
  ): Promise<AutocompleteItem[]> {
    if (signal.aborted) return [];
    const result = await withFinderLease(activeCwd, (finder) => {
      if (signal.aborted) return null;
      return finder.mixedSearch(query, { pageSize: MENTION_MAX_RESULTS });
    });
    if (!result) return [];
    if (!result.ok) return [];

    return result.value.items.slice(0, MENTION_MAX_RESULTS).map((mixed: MixedItem) => {
      if (mixed.type === "directory") {
        return {
          value: buildAtCompletionValue(mixed.item.relativePath),
          label: mixed.item.dirName,
          description: mixed.item.relativePath,
        };
      }
      return {
        value: buildAtCompletionValue(mixed.item.relativePath),
        label: mixed.item.fileName,
        description: mixed.item.relativePath,
      };
    });
  }

  // Editor wrapper that injects FFF @-mention autocomplete alongside base provider.
  // Defined inside fffExtension() so the class methods capture `getMentionItems`
  // via closure. Subclasses constructed as `new Sub(tui, theme, keybindings)` by
  // composability wrappers (pi-vim, pi-image-attachments) still get a working
  // mention provider because the closure binding is preserved across subclassing.
  class FffEditor extends CustomEditor {
    private baseProvider: AutocompleteProvider | undefined;

    override setAutocompleteProvider(provider: AutocompleteProvider): void {
      this.baseProvider = provider;
      // Create composite provider that handles @-mentions and falls back to base
      const mentionProvider = createFffMentionProvider(getMentionItems);
      const compositeProvider: AutocompleteProvider = {
        getSuggestions: async (lines, cursorLine, cursorCol, options) => {
          // Try @-mention first
          const mentionResult = await mentionProvider.getSuggestions(
            lines,
            cursorLine,
            cursorCol,
            options,
          );
          if (mentionResult) return mentionResult;
          // Fall back to base provider
          return (
            this.baseProvider?.getSuggestions(lines, cursorLine, cursorCol, options) ??
            null
          );
        },
        applyCompletion: (lines, cursorLine, cursorCol, item, prefix) => {
          // Let mention provider handle @ completions, base provider for others
          if (prefix?.startsWith("@")) {
            return mentionProvider.applyCompletion!(
              lines,
              cursorLine,
              cursorCol,
              item,
              prefix,
            );
          }
          return (
            this.baseProvider?.applyCompletion?.(
              lines,
              cursorLine,
              cursorCol,
              item,
              prefix,
            ) ?? { lines, cursorLine, cursorCol }
          );
        },
      };
      super.setAutocompleteProvider(compositeProvider);
    }
  }

  function applyEditorMode(ctx: {
    ui: {
      setEditorComponent: (
        factory: ((tui: any, theme: any, keybindings: any) => any) | undefined,
      ) => void;
    };
  }) {
    if (!shouldEnableMentions()) {
      ctx.ui.setEditorComponent(undefined);
    } else {
      ctx.ui.setEditorComponent(
        (tui: any, theme: any, keybindings: any) =>
          new FffEditor(tui, theme, keybindings),
      );
    }
  }

  // --- Flags / lifecycle ---

  pi.registerFlag("fff-mode", {
    description: "FFF mode: tools-and-ui | tools-only | override",
    type: "string",
  });

  pi.registerFlag("fff-frecency-db", {
    description: "Path to the frecency database (overrides FFF_FRECENCY_DB env)",
    type: "string",
  });

  pi.registerFlag("fff-history-db", {
    description: "Path to the query history database (overrides FFF_HISTORY_DB env)",
    type: "string",
  });

  pi.on("session_start", async (_event, ctx) => {
    try {
      activeCwd = ctx.cwd;
      await withFinderLease(activeCwd, () => undefined);
      applyEditorMode(ctx);
    } catch (e: unknown) {
      ctx.ui.notify(
        `FFF init failed: ${e instanceof Error ? e.message : String(e)}`,
        "error",
      );
    }
  });

  pi.on("session_shutdown", async () => {
    destroyFinder();
  });

  // --- Shared render helpers ---

  const renderTextResult = (
    result: { content?: { type: string; text?: string }[] },
    options: { expanded?: boolean },
    theme: any,
    context: any,
    maxLines = 15,
  ) => {
    const text = (context.lastComponent as Text | undefined) ?? new Text("", 0, 0);
    const output = result.content?.find((c) => c.type === "text")?.text?.trim() ?? "";
    if (!output) {
      text.setText(theme.fg("muted", "No output"));
      return text;
    }

    const lines = output.split("\n");
    const displayLines = lines.slice(0, options.expanded ? lines.length : maxLines);
    let content = `\n${displayLines.map((line: string) => theme.fg("toolOutput", line)).join("\n")}`;
    if (lines.length > displayLines.length) {
      content += theme.fg(
        "muted",
        `\n... (${lines.length - displayLines.length} more lines)`,
      );
    }
    text.setText(content);
    return text;
  };

  // --- grep tool ---

  const grepSchema = Type.Object({
    pattern: Type.String({
      description: "Search pattern (literal text or regex)",
    }),
    path: Type.Optional(
      Type.String({
        description:
          "Single path constraint: one file, one directory, or one glob. Do not pass multiple paths. Applied to the full repo-relative path.",
      }),
    ),
    exclude: Type.Optional(
      Type.Union([Type.String(), Type.Array(Type.String())], {
        description:
          "Exclude paths (comma/space-separated or array). Same syntax as path: directory prefix ('test/'), filename with extension ('config.json'), or glob ('*.min.js', '**/*.{rs,go}'). A leading '!' is optional and ignored — both 'test/' and '!test/' work. Example: 'test/,*.min.js,!vendor/'.",
      }),
    ),
    caseSensitive: Type.Optional(
      Type.Boolean({
        description:
          "Force case-sensitive matching. Default uses smart-case (case-insensitive when pattern is all lowercase).",
      }),
    ),
    context: Type.Optional(
      Type.Number({ description: "Context lines before+after each match" }),
    ),
    limit: Type.Optional(
      Type.Number({
        description: `Max matches (default ${DEFAULT_GREP_LIMIT})`,
      }),
    ),
    cursor: Type.Optional(
      Type.String({ description: "Pagination cursor from previous result" }),
    ),
  });

  pi.registerTool({
    name: toolNames.grep,
    label: toolNames.grep,
    description: `Grep file contents. Smart-case, auto-detects regex vs literal, git-aware. Results are ranked by frecency (most-accessed files first); matches within a file stay in source order. Default limit ${DEFAULT_GREP_LIMIT}.`,
    promptSnippet: "Grep contents",
    promptGuidelines: [
      "Prefer bare identifiers as patterns. Literal queries are most efficient.",
      "Use path for include ('src/', '*.ts') and exclude for noise ('test/,*.min.js').",
      "caseSensitive: true when you need exact case (smart-case otherwise).",
      "Never combine paths in one call. For multiple files, make separate grep calls.",
      "After 1-2 greps, read the top match instead of more greps.",
    ],
    parameters: grepSchema,

    async execute(_toolCallId, params, signal) {
      if (signal?.aborted) throw new Error("Operation aborted");

      if (params.path && pathLooksLikeMultiplePaths(params.path)) {
        throw new Error(multiplePathsMessage());
      }
      const invalidPath = params.path ? invalidPathMessage(params.path) : null;
      if (invalidPath) throw new Error(invalidPath);

      const searchBase = resolveSearchBase(params.path);
      const effectiveLimit = Math.max(1, params.limit ?? DEFAULT_GREP_LIMIT);
      const query = buildQuery(
        searchBase.pathConstraint,
        params.pattern,
        params.exclude,
        searchBase.basePath,
      );
      // Auto-detect: regex if the pattern has regex metacharacters AND parses
      // as a valid regex, otherwise plain literal. The fuzzy fallback below
      // only kicks in for plain mode — regex queries are intentional.
      const hasRegexSyntax =
        params.pattern !== params.pattern.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
      let mode: GrepMode = hasRegexSyntax ? "regex" : "plain";
      if (mode === "regex") {
        try {
          new RegExp(params.pattern);
        } catch {
          mode = "plain";
        }
      }

      // Guard: the agent keeps calling grep with '.*' or similar wildcard-only regex
      // to try to read a whole file. That's not what grep is for — return a terse error
      // steering them to a real pattern, preventing dozens of wasted retries.
      const p = params.pattern.trim();
      const isWildcardOnly =
        hasRegexSyntax &&
        /^(?:[.^$]*(?:[.][*+?]|\*|\+)[.^$]*|[.^$\s]*|\.\*\??|\.\*[+?]?|\.\+\??|\.|\*|\?)$/.test(
          p,
        );

      if (isWildcardOnly) {
        return {
          content: [
            {
              type: "text",
              text: `Pattern '${params.pattern}' matches everything — grep needs a concrete substring or identifier. Example: \`pattern: 'MyClass'\` or \`pattern: 'export function'\`.`,
            },
          ],
          details: { totalMatched: 0, totalFiles: 0 },
        };
      }

      // caseSensitive override flips smartCase off; omitting it keeps smart-case
      // (case-insensitive when pattern is all lowercase).
      const smartCase = params.caseSensitive !== true;

      const grepResult = await withFinderLease(searchBase.basePath, (finder) =>
        finder.grep(query, {
          mode,
          smartCase,
          maxMatchesPerFile: Math.min(effectiveLimit, 50),
          cursor: (params.cursor ? getCursor(params.cursor) : null) ?? null,
          beforeContext: params.context ?? 0,
          afterContext: params.context ?? 0,
          classifyDefinitions: true,
        }),
      );

      if (!grepResult.ok) throw new Error(grepResult.error);

      let result = grepResult.value;
      let fuzzyNotice: string | null = null;

      // Fuzzy fallback helps broad plain greps, but excludes mean exact filtering.
      if (
        result.items.length === 0 &&
        !params.cursor &&
        !params.exclude &&
        mode !== "regex"
      ) {
        const fuzzy = await withFinderLease(searchBase.basePath, (finder) =>
          finder.grep(query, {
            mode: "fuzzy",
            smartCase,
            maxMatchesPerFile: Math.min(effectiveLimit, 50),
            cursor: null,
            beforeContext: 0,
            afterContext: 0,
            classifyDefinitions: true,
          }),
        );

        if (fuzzy.ok && fuzzy.value.items.length > 0) {
          fuzzyNotice = `0 exact matches. Maybe you meant this?`;
          result = fuzzy.value;
        }
      }

      if (result.items.length === 0) throw new Error("No matches found");

      let output = formatGrepOutput(result);
      const notices: string[] = [];
      if (result.regexFallbackError) {
        notices.push(`Invalid regex: ${result.regexFallbackError}, used literal match`);
      }
      if (result.nextCursor) {
        notices.push(`Continue with cursor="${storeCursor(result.nextCursor)}"`);
      }

      if (notices.length > 0) output += `\n\n[${notices.join(". ")}]`;
      if (fuzzyNotice) output = `[${fuzzyNotice}]\n${output}`;

      return {
        content: [{ type: "text", text: output }],
        details: {
          totalMatched: result.totalMatched,
          totalFiles: result.totalFiles,
        },
      };
    },

    renderCall(args, theme, context) {
      const text = (context.lastComponent as Text | undefined) ?? new Text("", 0, 0);
      const pattern = args?.pattern ?? "";
      const path = args?.path ?? ".";
      let content =
        theme.fg("toolTitle", theme.bold(toolNames.grep)) +
        " " +
        theme.fg("accent", `/${pattern}/`) +
        theme.fg("toolOutput", ` in ${path}`);
      if (args?.limit !== undefined)
        content += theme.fg("toolOutput", ` limit ${args.limit}`);
      if (args?.cursor) content += theme.fg("muted", ` (page)`);
      text.setText(content);
      return text;
    },

    renderResult(result, options, theme, context) {
      return renderTextResult(result, options, theme, context, 15);
    },
  });

  // --- find tool ---

  const findSchema = Type.Object({
    pattern: Type.String({
      description:
        "Fuzzy filename search and glob search. Frecency-ranked, git-aware. Multi-word = narrower (AND) not bound to order, use for multi word related concept search. Prefer this over ls/find/bash as the first exploration step whenever the user names a concept, feature, or symbol — it surfaces the relevant files in one call. Only use ls/read on a directory when you specifically need the alphabetical layout of an unknown repo, or when a concept search returned nothing.",
    }),
    path: Type.Optional(
      Type.String({
        description:
          "Single path constraint: one file, one directory, or one glob. Do not pass multiple paths. Applied to the full repo-relative path.",
      }),
    ),
    exclude: Type.Optional(
      Type.Union([Type.String(), Type.Array(Type.String())], {
        description:
          "Exclude paths (comma/space-separated or array). Same syntax as path: directory prefix ('test/'), filename with extension ('config.json'), or glob ('*.min.js', '**/*.{rs,go}'). A leading '!' is optional and ignored — both 'test/' and '!test/' work. Example: 'test/,*.min.js,!vendor/'.",
      }),
    ),
    limit: Type.Optional(
      Type.Number({
        description: `Max results per page (default ${DEFAULT_FIND_LIMIT})`,
      }),
    ),
    cursor: Type.Optional(
      Type.String({ description: "Pagination cursor from previous result" }),
    ),
  });

  pi.registerTool({
    name: toolNames.find,
    label: toolNames.find,
    description: `Fuzzy path search and glob search. Matches against the whole repo-relative path, not just the filename. Frecency-ranked, git-aware. Multi-word = narrower (AND). Default limit ${DEFAULT_FIND_LIMIT}.`,
    promptSnippet: "Find files by path or glob",
    promptGuidelines: [
      "Matches the WHOLE path, not just the filename — `profile` hits `chrome/browser/profiles/x.cc` too.",
      "Keep queries to 1-2 terms; extra words narrow.",
      "Use one path constraint only: one file, directory, or glob.",
      "Use for paths, not content. Use grep for content.",
      "For exact path matches use a glob in `path` — e.g. path: '**/profile.h' for exact filename, or path: 'src/**/profile.h' scoped to a subtree. Bare patterns are fuzzy.",
      "To list everything inside a directory, pass path: 'dir/**' with an empty or wildcard pattern instead of using pattern alone.",
      "Use exclude: 'test/,*.min.js' to cut noise in large repos.",
    ],
    parameters: findSchema,

    async execute(_toolCallId, params, signal) {
      if (signal?.aborted) throw new Error("Operation aborted");

      // Resume from a prior cursor if supplied — cursor owns basePath+query+pageSize
      // so the agent can't accidentally mix patterns across pages.
      const resumed = params.cursor ? getFindCursor(params.cursor) : undefined;
      if (!params.cursor && params.path && pathLooksLikeMultiplePaths(params.path)) {
        throw new Error(multiplePathsMessage());
      }
      const invalidPath =
        !params.cursor && params.path ? invalidPathMessage(params.path) : null;
      if (invalidPath) throw new Error(invalidPath);

      const resolvedBase = resolveSearchBase(params.path);
      const basePath = resumed?.basePath ?? resolvedBase.basePath;
      const effectiveLimit = resumed
        ? resumed.pageSize
        : Math.max(1, params.limit ?? DEFAULT_FIND_LIMIT);
      const query = resumed
        ? resumed.query
        : buildQuery(
            resolvedBase.pathConstraint,
            params.pattern,
            params.exclude,
            resolvedBase.basePath,
          );
      const pattern = resumed ? resumed.pattern : params.pattern;
      if (!resumed && patternLooksLikePath(pattern)) {
        throw new Error(pathLikePatternMessage(pattern));
      }
      const pageIndex = resumed?.nextPageIndex ?? 0;

      const searchResult = await withFinderLease(basePath, (finder) =>
        finder.fileSearch(query, {
          pageIndex,
          pageSize: effectiveLimit,
        }),
      );
      if (!searchResult.ok) throw new Error(searchResult.error);

      let result = searchResult.value;
      if (result.items.length === 0 && /\s/.test(pattern.trim())) {
        const scopedQuery = buildQuery(
          resolvedBase.pathConstraint,
          "",
          params.exclude,
          basePath,
        );
        const fallback = await withFinderLease(basePath, (finder) =>
          finder.fileSearch(scopedQuery, {
            pageIndex: 0,
            pageSize: Math.max(effectiveLimit, 500),
          }),
        );
        if (fallback.ok) {
          const needle = pattern.trim().toLowerCase();
          const pairs = fallback.value.items
            .map((item, index) => ({ item, score: fallback.value.scores[index] }))
            .filter(({ item }) => item.relativePath.toLowerCase().includes(needle))
            .slice(0, effectiveLimit);
          if (pairs.length > 0) {
            result = {
              ...fallback.value,
              items: pairs.map((pair) => pair.item),
              scores: pairs.map((pair) => pair.score),
              totalMatched: pairs.length,
            };
          }
        }
      }
      if (result.items.length === 0) throw new Error("No files found matching pattern");

      const formatted = formatFindOutput(result, effectiveLimit, pattern);
      let output = formatted.output;

      // Infer hasMore: native fileSearch fills pageSize when more results
      // exist, so if we got a full page AND totalMatched exceeds what we've
      // shown so far there's another page to fetch.
      const shownSoFar = pageIndex * effectiveLimit + result.items.length;
      const hasMore =
        result.items.length >= effectiveLimit && result.totalMatched > shownSoFar;

      const notices: string[] = [];
      if (formatted.weak && formatted.shownCount > 0)
        notices.push(
          `Query "${pattern}" produced only weak scattered fuzzy matches. Output capped at ${formatted.shownCount}/${result.totalMatched}.`,
        );
      const hiddenFuzzyMatches = result.totalMatched - formatted.shownCount;
      if (formatted.literalTailSuppressed && hiddenFuzzyMatches >= 1000)
        notices.push(`${formatted.shownCount} exact matches shown. Fuzzy tail hidden`);

      if (!formatted.weak && !formatted.literalTailSuppressed && hasMore) {
        const remaining = result.totalMatched - shownSoFar;
        const cursorId = storeFindCursor({
          basePath,
          query,
          pattern,
          pageSize: effectiveLimit,
          nextPageIndex: pageIndex + 1,
        });
        notices.push(`${remaining} more. Next page: find cursor="${cursorId}"`);
      }

      if (notices.length > 0) output += `\n\n[${notices.join(". ")}]`;
      return {
        content: [{ type: "text", text: output }],
        details: {
          totalMatched: result.totalMatched,
          totalFiles: result.totalFiles,
          pageIndex,
          hasMore,
        },
      };
    },

    renderCall(args, theme, context) {
      const text = (context.lastComponent as Text | undefined) ?? new Text("", 0, 0);
      const pattern = args?.pattern ?? "";
      const path = args?.path ?? ".";
      let content =
        theme.fg("toolTitle", theme.bold(toolNames.find)) +
        " " +
        theme.fg("accent", pattern) +
        theme.fg("toolOutput", ` in ${path}`);
      if (args?.limit !== undefined)
        content += theme.fg("toolOutput", ` (limit ${args.limit})`);
      if (args?.cursor) content += theme.fg("muted", ` (page)`);
      text.setText(content);
      return text;
    },

    renderResult(result, options, theme, context) {
      return renderTextResult(result, options, theme, context, 20);
    },
  });

  // --- multi_grep tool ---
  // My latest tests are showing that the multi grep tool is only harmful, trying to get rid of it
  const enableMultiGrep = process.env.PI_FFF_MULTIGREP === "1";

  if (enableMultiGrep) {
    const multiGrepSchema = Type.Object({
      patterns: Type.Array(Type.String(), {
        description:
          "Literal patterns (OR). Include snake_case/camelCase/PascalCase variants.",
      }),
      constraints: Type.Optional(
        Type.String({ description: "File filter, e.g. '*.{ts,tsx} !test/'" }),
      ),
      context: Type.Optional(Type.Number({ description: "Context lines before+after" })),
      limit: Type.Optional(
        Type.Number({
          description: `Max matches (default ${DEFAULT_GREP_LIMIT})`,
        }),
      ),
      cursor: Type.Optional(Type.String({ description: "Pagination cursor" })),
    });

    pi.registerTool({
      name: toolNames.multiGrep,
      label: toolNames.multiGrep,
      description:
        "Search file contents for ANY of multiple literal patterns (OR, SIMD Aho-Corasick). Faster than regex alternation.",
      promptSnippet: "Multi-pattern OR content search",
      promptGuidelines: [
        "Use when searching for several identifiers at once.",
        "Include all naming-convention variants (snake/camel/Pascal).",
        "Patterns are literal. Use constraints for file filters.",
      ],
      parameters: multiGrepSchema,

      async execute(_toolCallId, params, signal) {
        if (signal?.aborted) throw new Error("Operation aborted");
        if (!params.patterns?.length)
          throw new Error("patterns array must have at least 1 element");

        const effectiveLimit = Math.max(1, params.limit ?? DEFAULT_GREP_LIMIT);

        const grepResult = await withFinderLease(activeCwd, (finder) =>
          finder.multiGrep({
            patterns: params.patterns,
            constraints: params.constraints,
            maxMatchesPerFile: Math.min(effectiveLimit, 50),
            smartCase: true,
            cursor: (params.cursor ? getCursor(params.cursor) : null) ?? null,
            beforeContext: params.context ?? 0,
            afterContext: params.context ?? 0,
          }),
        );

        if (!grepResult.ok) throw new Error(grepResult.error);

        const result = grepResult.value;
        if (result.items.length === 0) throw new Error("No matches found");

        let output = formatGrepOutput(result);

        const notices: string[] = [];
        if (result.items.length >= effectiveLimit)
          notices.push(`${effectiveLimit}+ matches (refine patterns)`);
        if (result.nextCursor)
          notices.push(
            `More available. cursor="${storeCursor(result.nextCursor)}" to continue`,
          );

        if (notices.length > 0) output += `\n\n[${notices.join(". ")}]`;

        return {
          content: [{ type: "text", text: output }],
          details: {
            totalMatched: result.totalMatched,
            totalFiles: result.totalFiles,
            patterns: params.patterns,
          },
        };
      },

      renderCall(args, theme, context) {
        const text = (context.lastComponent as Text | undefined) ?? new Text("", 0, 0);
        const patterns = args?.patterns ?? [];
        const constraints = args?.constraints;
        let content =
          theme.fg("toolTitle", theme.bold(toolNames.multiGrep)) +
          " " +
          theme.fg("accent", patterns.map((p: string) => `"${p}"`).join(", "));
        if (constraints) content += theme.fg("toolOutput", ` (${constraints})`);
        if (args?.cursor) content += theme.fg("muted", ` (page)`);
        text.setText(content);
        return text;
      },

      renderResult(result, options, theme, context) {
        return renderTextResult(result, options, theme, context, 15);
      },
    });
  } // end if (enableMultiGrep)

  // --- commands ---

  pi.registerCommand("fff-mode", {
    description: "Show or set FFF mode: /fff-mode [tools-and-ui | tools-only | override]",
    handler: async (args, ctx) => {
      const arg = (args || "").trim();

      // No args - show current mode
      if (!arg) {
        const mode = getMode();
        const flag = pi.getFlag("fff-mode") ?? "unset";
        const env = process.env.PI_FFF_MODE ?? "unset";
        ctx.ui.notify(`Current mode: '${mode}'\nFlag: ${flag}, Env: ${env}`, "info");
        return;
      }

      // Validate and set mode
      if (!VALID_MODES.includes(arg as FffMode)) {
        ctx.ui.notify(`Usage: /fff-mode [${VALID_MODES.join(" | ")}]`, "warning");
        return;
      }

      const newMode = arg as FffMode;
      const oldMode = getMode();
      setMode(newMode);

      // Apply immediately using the shared function
      applyEditorMode(ctx);

      const note =
        (oldMode === "override") !== (newMode === "override")
          ? " (tool name change requires restart)"
          : "";
      ctx.ui.notify(`Mode changed: '${oldMode}' → '${newMode}'${note}`, "info");
    },
  });

  pi.registerCommand("fff-health", {
    description: "Show FFF file finder health and status",
    handler: async (_args, ctx) => {
      const finder = getActiveFinder();
      if (!finder) {
        ctx.ui.notify("FFF not initialized", "warning");
        return;
      }

      const health = finder.healthCheck();
      if (!health.ok) {
        ctx.ui.notify(`Health check failed: ${health.error}`, "error");
        return;
      }

      const h = health.value;
      const lines = [
        `FFF v${h.version}`,
        `Mode: ${getMode()}`,
        `Git: ${h.git.repositoryFound ? `yes (${h.git.workdir ?? "unknown"})` : "no"}`,
        `Picker: ${h.filePicker.initialized ? `${h.filePicker.indexedFiles ?? 0} files` : "not initialized"}`,
        `Frecency: ${h.frecency.initialized ? "active" : "disabled"}`,
        `Query tracker: ${h.queryTracker.initialized ? "active" : "disabled"}`,
      ];

      const progress = finder.getScanProgress();
      if (progress.ok) {
        lines.push(
          `Scanning: ${progress.value.isScanning ? "yes" : "no"} (${progress.value.scannedFilesCount} files)`,
        );
      }

      ctx.ui.notify(lines.join("\n"), "info");
    },
  });

  pi.registerCommand("fff-rescan", {
    description: "Trigger FFF to rescan files",
    handler: async (_args, ctx) => {
      const finder = getActiveFinder();
      if (!finder) {
        ctx.ui.notify("FFF not initialized", "warning");
        return;
      }

      const result = finder.scanFiles();
      if (!result.ok) {
        ctx.ui.notify(`Rescan failed: ${result.error}`, "error");
        return;
      }

      ctx.ui.notify("FFF rescan triggered", "info");
    },
  });
}
