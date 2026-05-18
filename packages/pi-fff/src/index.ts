/**
 * pi-fff: FFF-powered file search extension for pi
 *
 * Overrides built-in `find` and `grep` tools with FFF.
 */

import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { Text } from "@earendil-works/pi-tui";
import { Type } from "@sinclair/typebox";
import type {
  GrepCursor,
  GrepMode,
  FileItem,
  GrepResult,
  InitOptions,
  Score,
  SearchResult,
} from "@edxeth/fff-node";
import { FileFinder } from "@edxeth/fff-node";
import {
  buildQuery,
  buildSearchScopes,
  invalidPathMessage,
  normalizeExcludes,
  normalizePathConstraint,
  mergeGrepResults,
  mergeFindResults,
  resolveSearchBase,
} from "./query";
import {
  getFindSourceSearchNotice,
  getMultiGrepPhraseMissNotice,
  getRegexAlternationNotice,
  shouldShowRegexAlternationNotice,
} from "./regex-diagnostics";

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const DEFAULT_GREP_LIMIT = 20;
const DEFAULT_FIND_LIMIT = 30;
const MAX_CACHED_FINDERS = 4;
const GREP_MAX_LINE_LENGTH = 500;

type FffMode = "tools-and-ui" | "tools-only" | "override";
type FffInitOptions = InitOptions & { includeIgnored?: boolean };

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

interface StoredGrepCursor {
  cursor: GrepCursor;
  includeIgnored: boolean;
}

const cursorCache = new Map<string, StoredGrepCursor>();
let cursorCounter = 0;

function storeCursor(cursor: GrepCursor, includeIgnored = false): string {
  const id = `fff_c${++cursorCounter}`;
  cursorCache.set(id, { cursor, includeIgnored });
  if (cursorCache.size > 200) {
    const first = cursorCache.keys().next().value;
    if (first) cursorCache.delete(first);
  }
  return id;
}

function getCursor(id: string): StoredGrepCursor | undefined {
  return cursorCache.get(id);
}

// Find pagination uses a page-index cursor: native `fileSearch` takes
// pageIndex/pageSize, so the cursor stores per-scope page indices.
// Multi-scope cursors handle multiple independent searches.
interface FindCursor {
  // Each scope tracks its own search state within one base directory
  scopes: Array<{
    basePath: string;
    query: string;
    nextPageIndex: number;
    scopePrefix?: string;
  }>;
  pattern: string;
  pageSize: number;
  includeIgnored: boolean;
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
  patternIndices?: number[];
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
  if (result.items.length === 0) {
    if (result.regexFallbackError) return result.regexFallbackError;
    return "No matches found";
  }

  // Build file-grouped output in the order files first appear in the result.
  // This preserves native frecency ordering across files without re-sorting.
  const lines: string[] = [];
  let currentFile = "";
  let currentFilePatterns: Set<number> | null = null;
  let _shown = 0;

  // Detect if this is a multi-pattern result by checking first match
  const isMultiPattern = result.items[0]?.patternIndices !== undefined;

  for (const match of result.items) {
    if (match.relativePath !== currentFile) {
      if (lines.length > 0) lines.push("");
      currentFile = match.relativePath;
      currentFilePatterns = isMultiPattern ? new Set() : null;

      let header = currentFile;
      const annotation = fffFileAnnotation(match);
      if (annotation) header += annotation;
      lines.push(header);
    }

    match.contextBefore?.forEach((line: string, i: number) => {
      const lineNum = match.lineNumber - match.contextBefore!.length + i;
      lines.push(` ${lineNum}- ${truncateLine(line)}`);
    });

    // Build pattern prefix for this match line
    let patternPrefix = "";
    if (isMultiPattern && match.patternIndices) {
      const uniqueIndices = new Set(match.patternIndices);
      if (uniqueIndices.size === 1) {
        patternPrefix = `[${[...uniqueIndices][0]}] `;
        currentFilePatterns?.add([...uniqueIndices][0]);
      } else {
        // Multiple patterns on same line — show all
        const sorted = [...uniqueIndices].sort((a, b) => a - b);
        patternPrefix = `[${sorted.join("+")}] `;
        sorted.forEach((i) => currentFilePatterns?.add(i));
      }
    }

    lines.push(
      ` ${patternPrefix}${match.lineNumber}: ${truncateLine(match.lineContent)}`,
    );
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

function pathLikePatternMessage(pattern: string): string {
  return `Path/glob belongs in path, not pattern. Use pattern: "" with path: ["${pattern}"] to list or scope files.`;
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
// Render helpers — defensive normalization for renderCall/renderResult
// ---------------------------------------------------------------------------

// Normalize `args.path` for display: string, string[], or undefined → safe string.
// Prevents renderCall from throwing when pi passes a scalar path.
export function formatRenderPath(path: unknown): string {
  if (Array.isArray(path)) return path.length > 0 ? path.join(", ") : ".";
  if (typeof path === "string" && path.length > 0) return path;
  return ".";
}

// Normalize `args.patterns` for display: must be string[]. Never throws.
export function formatRenderPatterns(patterns: unknown): string[] {
  if (Array.isArray(patterns))
    return patterns.filter((p): p is string => typeof p === "string");
  return [];
}

// ---------------------------------------------------------------------------
// Extension
// ---------------------------------------------------------------------------

export default function fffExtension(pi: ExtensionAPI) {
  const finders = new Map<string, FileFinder>();
  let activeFinderKey: string | null = null;
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

  async function noResultsMessage(
    base: string,
    basePath: string,
    pathConstraint: string | undefined,
    includeIgnored: boolean,
  ): Promise<string> {
    if (includeIgnored || !pathConstraint) return base;

    const ignored = await withFinderLease(basePath, (finder) => {
      const checker = finder as FileFinder & {
        isPathIgnored?: (path: string) => { ok: boolean; value?: boolean };
      };
      return checker.isPathIgnored?.(pathConstraint);
    });

    if (ignored?.ok && ignored.value === true) {
      return `${base}. Path is ignored. Retry with \`includeIgnored: true\`.`;
    }
    return base;
  }

  function finderKey(basePath: string, includeIgnored: boolean): string {
    return `${includeIgnored ? "ignored" : "normal"}:${basePath}`;
  }

  function trimFinderCache() {
    while (finders.size >= MAX_CACHED_FINDERS) {
      const evictable = [...finders.entries()].find(
        ([key]) => (finderActiveOps.get(key) ?? 0) === 0,
      );
      if (!evictable) return;

      const [oldestKey, oldestFinder] = evictable;
      if (!oldestFinder.isDestroyed) oldestFinder.destroy();
      finders.delete(oldestKey);
      if (activeFinderKey === oldestKey) activeFinderKey = null;
    }
  }

  function ensureFinder(basePath: string, includeIgnored = false): Promise<FileFinder> {
    const key = finderKey(basePath, includeIgnored);
    const existing = finders.get(key);
    if (existing && !existing.isDestroyed) return Promise.resolve(existing);
    const pending = finderPromises.get(key);
    if (pending) return pending;

    const promise = (async () => {
      trimFinderCache();
      const useDatabases = basePath === activeCwd && !includeIgnored;
      const isWorkspace = basePath === activeCwd;
      const result = FileFinder.create({
        basePath,
        frecencyDbPath: useDatabases ? frecencyDbPath : undefined,
        historyDbPath: useDatabases ? historyDbPath : undefined,
        aiMode: true,
        disableContentIndexing: !isWorkspace,
        disableMmapCache: !isWorkspace,
        includeIgnored,
      } as FffInitOptions);

      if (!result.ok)
        throw new Error(`Failed to create FFF file finder: ${result.error}`);

      const finder = result.value;
      finders.set(key, finder);
      if (!includeIgnored) activeFinderKey = key;
      await finder.waitForScan(15000);
      return finder;
    })().finally(() => {
      finderPromises.delete(key);
    });

    finderPromises.set(key, promise);
    return promise;
  }

  function destroyFinder() {
    for (const finder of finders.values()) {
      if (!finder.isDestroyed) finder.destroy();
    }
    finders.clear();
    finderLocks.clear();
    finderActiveOps.clear();
    activeFinderKey = null;
  }

  async function withFinderLease<T>(
    basePath: string,
    work: (finder: FileFinder) => T | Promise<T>,
    includeIgnored = false,
  ): Promise<T> {
    const key = finderKey(basePath, includeIgnored);
    const previous = finderLocks.get(key) ?? Promise.resolve();
    let release!: () => void;
    const current = new Promise<void>((resolve) => {
      release = resolve;
    });
    finderLocks.set(
      key,
      previous.then(
        () => current,
        () => current,
      ),
    );

    await previous.catch(() => undefined);
    finderActiveOps.set(key, (finderActiveOps.get(key) ?? 0) + 1);
    try {
      const finder = await ensureFinder(basePath, includeIgnored);
      return await work(finder);
    } finally {
      const remaining = (finderActiveOps.get(key) ?? 1) - 1;
      if (remaining > 0) finderActiveOps.set(key, remaining);
      else finderActiveOps.delete(key);
      release();
      if (finderLocks.get(key) === current) finderLocks.delete(key);
    }
  }

  function getActiveFinder(): FileFinder | null {
    if (!activeFinderKey) return null;
    const finder = finders.get(activeFinderKey);
    return finder && !finder.isDestroyed ? finder : null;
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

  // ── Multi-path search helpers ──

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
      description:
        "One precise literal or narrowly scoped regex. If the request names several exact terms, use multi_grep instead. Regex alternatives are unanchored and can flood results.",
    }),
    path: Type.Optional(
      Type.Array(Type.String(), {
        description:
          "Path constraints: files, directories, or globs. Each entry is one scope \u2014 searches OR across all entries. Paths with spaces work correctly as single entries.",
      }),
    ),
    exclude: Type.Optional(
      Type.Union([Type.String(), Type.Array(Type.String())], {
        description:
          "Exclude paths (comma/space-separated or array). Same syntax as path: directory prefix ('test/'), filename with extension ('config.json'), or glob ('*.min.js', '**/*.{rs,go}'). A leading '!' is optional and ignored — both 'test/' and '!test/' work. Example: 'test/,*.min.js,!vendor/'.",
      }),
    ),
    includeIgnored: Type.Optional(
      Type.Boolean({
        description:
          "Include files matched by .gitignore, .ignore, git excludes, and global gitignore. Default false. Use when the target file or directory exists but normal search cannot see it because it is ignored, such as node_modules or build output.",
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
    description: `Search file contents for one precise pattern. Smart-case, auto-detects regex vs literal, git-aware. Use multi_grep when the request lists exact terms. Default limit ${DEFAULT_GREP_LIMIT}.`,
    promptSnippet: "Search content for one precise pattern",
    promptGuidelines: [
      "Use for content, not paths.",
      "Use one precise literal identifier or phrase first; regex only when structure matters.",
      "If the user gives a list of exact symbols, methods, classes, or terms, use multi_grep instead of repeated grep calls.",
      "Do not combine broad concepts into one OR regex. Split broad searches or scope them with path.",
      "Regex alternatives are substring matches unless anchored. Prefer \\bfoo\\b or class\\s+\\w*Widget over foo|Widget.",
      "Use path to scope searches by directory, file, or glob. Multiple paths search as OR: path: ['src/', 'tests/'].",
      "Set includeIgnored only when intentionally searching ignored files such as node_modules or build output.",
      "Set caseSensitive: true only when exact case matters; otherwise smart-case applies.",
      "Use multi_grep for 2-6 exact identifiers or naming variants instead of regex alternation.",
      "Avoid broad unanchored terms unless scoped by path, paired with context, or bounded with word boundaries.",
      "After 1-2 greps, read the best match instead of widening search.",
    ],
    parameters: grepSchema,

    async execute(_toolCallId, params, signal) {
      if (signal?.aborted) throw new Error("Operation aborted");

      const rawPaths = params.path ?? [];
      const paths = Array.isArray(rawPaths)
        ? rawPaths
        : typeof rawPaths === "string"
          ? [rawPaths]
          : [];
      const invalidPath = paths.length > 0 ? invalidPathMessage(paths) : null;
      if (invalidPath) throw new Error(invalidPath);

      const scopes = buildSearchScopes(paths, params.pattern, params.exclude, activeCwd);
      const effectiveLimit = Math.max(1, params.limit ?? DEFAULT_GREP_LIMIT);
      const storedCursor = params.cursor ? getCursor(params.cursor) : undefined;
      const includeIgnored =
        storedCursor?.includeIgnored ?? params.includeIgnored === true;

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

      // Run grep across all search scopes, merge results
      const grepOptions = {
        mode,
        smartCase,
        maxMatchesPerFile: Math.min(effectiveLimit, 50),
        cursor: storedCursor?.cursor ?? null,
        beforeContext: params.context ?? 0,
        afterContext: params.context ?? 0,
        classifyDefinitions: true,
      };

      const grepResults: GrepResult[] = [];
      for (const scope of scopes) {
        const r = await withFinderLease(
          scope.basePath,
          (finder) => finder.grep(scope.query, grepOptions),
          includeIgnored,
        );
        if (!r.ok) throw new Error(r.error);
        // Apply scopePrefix: keep only files under the space-containing directory
        if (scope.scopePrefix) {
          const filtered = r.value.items.filter((m) =>
            m.relativePath.startsWith(scope.scopePrefix!),
          );
          grepResults.push({
            ...r.value,
            items: filtered,
            totalMatched: filtered.length,
          });
        } else {
          grepResults.push(r.value);
        }
        // If using a cursor, only run the first scope (cursor is single-scope)
        if (storedCursor) break;
      }

      let result = mergeGrepResults(grepResults);
      let fuzzyNotice: string | null = null;

      // Fuzzy fallback: if exact produces no results, try fuzzy on each scope
      if (
        result.items.length === 0 &&
        !params.cursor &&
        !params.exclude &&
        mode !== "regex"
      ) {
        const fuzzyResults: GrepResult[] = [];
        for (const scope of scopes) {
          const r = await withFinderLease(
            scope.basePath,
            (finder) =>
              finder.grep(scope.query, { ...grepOptions, mode: "fuzzy", cursor: null }),
            includeIgnored,
          );
          if (r.ok && r.value.items.length > 0) fuzzyResults.push(r.value);
        }
        if (fuzzyResults.length > 0) {
          fuzzyNotice = `0 exact matches. Maybe you meant this?`;
          result = mergeGrepResults(fuzzyResults);
        }
      }

      let output = formatGrepOutput(result);
      const notices: string[] = [];
      if (result.items.length === 0 && scopes.length > 0) {
        const noResults = await noResultsMessage(
          "No matches found",
          scopes[0].basePath,
          paths[0],
          includeIgnored,
        );
        if (noResults !== "No matches found") notices.push(noResults);
      }
      const regexAlternationNotice =
        mode === "regex" &&
        shouldShowRegexAlternationNotice(
          result.items,
          effectiveLimit,
          result.nextCursor !== null,
        )
          ? getRegexAlternationNotice(params.pattern)
          : null;
      if (regexAlternationNotice) notices.push(regexAlternationNotice);
      if (result.regexFallbackError) {
        notices.push(`Invalid regex: ${result.regexFallbackError}, used literal match`);
      }
      if (result.nextCursor) {
        notices.push(
          `Continue with cursor="${storeCursor(result.nextCursor, includeIgnored)}"`,
        );
      }
      if (includeIgnored) notices.unshift("ignored files included");

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
      const path = formatRenderPath(args?.path);
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
        "Fuzzy filename/path query only, not code symbol search. Use an empty string when path already contains an exact directory, file, or glob. Frecency-ranked, git-aware. Multi-word = narrower (AND) not bound to order.",
    }),
    path: Type.Optional(
      Type.Array(Type.String(), {
        description:
          "Path constraints: files, directories, or globs. Each entry is one scope \u2014 searches OR across all entries. Paths with spaces work correctly as single entries.",
      }),
    ),
    exclude: Type.Optional(
      Type.Union([Type.String(), Type.Array(Type.String())], {
        description:
          "Exclude paths (comma/space-separated or array). Same syntax as path: directory prefix ('test/'), filename with extension ('config.json'), or glob ('*.min.js', '**/*.{rs,go}'). A leading '!' is optional and ignored — both 'test/' and '!test/' work. Example: 'test/,*.min.js,!vendor/'.",
      }),
    ),
    includeIgnored: Type.Optional(
      Type.Boolean({
        description:
          "Include files matched by .gitignore, .ignore, git excludes, and global gitignore. Default false. Use when the target file or directory exists but normal search cannot see it because it is ignored, such as node_modules or build output.",
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
    description: `Fuzzy file/path search, not source symbol search. Exact directories, files, and globs belong in path, not pattern. Matches against the whole repo-relative path. Frecency-ranked, git-aware. Default limit ${DEFAULT_FIND_LIMIT}.`,
    promptSnippet: "Find files by fuzzy path query",
    promptGuidelines: [
      "Use for paths, not content; use grep for content.",
      "Do not use find for methods, classes, constants, or UI verbs unless they are likely file names; use grep or multi_grep for source symbols.",
      "Pattern is fuzzy over the whole repo-relative path, not just the basename.",
      "Keep pattern to 1-2 terms; extra words narrow results.",
      "Put exact paths, directories, and globs in path, not pattern, e.g. path: ['**/profile.h'].",
      "Multiple paths search as OR: path: ['src/', 'tests/'].",
      "For directory contents, use pattern: '' with path: ['dir/**']; do not use pattern: '*'.",
      "Use exclude to cut noise, e.g. 'test/,*.min.js'.",
      "Set includeIgnored only when intentionally searching ignored files such as node_modules or build output.",
    ],
    parameters: findSchema,

    async execute(_toolCallId, params, signal) {
      if (signal?.aborted) throw new Error("Operation aborted");

      // Resume from a prior cursor if supplied — cursor owns basePath+query+pageSize
      // so the agent can't accidentally mix patterns across pages.
      const resumed = params.cursor ? getFindCursor(params.cursor) : undefined;
      const rawPaths = !params.cursor ? (params.path ?? []) : [];
      const paths = Array.isArray(rawPaths)
        ? rawPaths
        : typeof rawPaths === "string"
          ? [rawPaths]
          : [];
      const invalidPath =
        !params.cursor && paths.length > 0 ? invalidPathMessage(paths) : null;
      if (invalidPath) throw new Error(invalidPath);

      const scopes = resumed
        ? resumed.scopes.map((s) => ({
            basePath: s.basePath,
            query: s.query,
            scopePrefix: s.scopePrefix,
          }))
        : buildSearchScopes(paths, params.pattern, params.exclude, activeCwd);
      const effectiveLimit = resumed
        ? resumed.pageSize
        : Math.max(1, params.limit ?? DEFAULT_FIND_LIMIT);
      const basePath = scopes.length > 0 ? scopes[0].basePath : activeCwd;
      const pattern = resumed ? resumed.pattern : params.pattern;
      if (!resumed && patternLooksLikePath(pattern)) {
        throw new Error(pathLikePatternMessage(pattern));
      }
      const pageIndex = resumed?.scopes[0]?.nextPageIndex ?? 0;
      const includeIgnored = resumed?.includeIgnored ?? params.includeIgnored === true;

      let result: SearchResult;
      let hasMore = false;
      if (resumed) {
        // Resume: iterate all cursor scopes, merge results
        const results: SearchResult[] = [];
        for (const scopeInfo of resumed.scopes) {
          const r = await withFinderLease(
            scopeInfo.basePath,
            (finder) =>
              finder.fileSearch(scopeInfo.query, {
                pageIndex: scopeInfo.nextPageIndex,
                pageSize: effectiveLimit,
              }),
            includeIgnored,
          );
          if (!r.ok) throw new Error(r.error);
          // Apply scopePrefix for space-paths resolved to git root
          if (scopeInfo.scopePrefix && r.value.items.length > 0) {
            const keep: Array<{ item: FileItem; score: Score }> = [];
            for (let i = 0; i < r.value.items.length; i++) {
              if (r.value.items[i].relativePath.startsWith(scopeInfo.scopePrefix!)) {
                keep.push({ item: r.value.items[i], score: r.value.scores[i] });
              }
            }
            results.push({
              items: keep.map((k) => k.item),
              scores: keep.map((k) => k.score),
              totalMatched: keep.length,
              totalFiles: r.value.totalFiles,
            });
          } else {
            results.push(r.value);
          }
        }
        result = mergeFindResults(results);
      } else if (paths.length <= 1) {
        // Fast path: single scope
        const scope = scopes[0] ?? {
          basePath,
          query: buildQuery([], pattern, params.exclude, basePath),
        };
        const searchResult = await withFinderLease(
          scope.basePath,
          (finder) =>
            finder.fileSearch(scope.query, {
              pageIndex: 0,
              pageSize: effectiveLimit,
            }),
          includeIgnored,
        );
        if (!searchResult.ok) throw new Error(searchResult.error);
        result = searchResult.value;
      } else {
        // Multi-path: search each scope, merge
        const results: SearchResult[] = [];
        let anyScopeFull = false;
        for (const scope of scopes) {
          const r = await withFinderLease(
            scope.basePath,
            (finder) =>
              finder.fileSearch(scope.query, { pageIndex: 0, pageSize: effectiveLimit }),
            includeIgnored,
          );
          if (r.ok) {
            anyScopeFull = anyScopeFull || r.value.items.length >= effectiveLimit;
            // Apply scopePrefix: keep only files under the space-containing directory
            if (scope.scopePrefix && r.value.items.length > 0) {
              const keep: Array<{ item: FileItem; score: Score }> = [];
              for (let i = 0; i < r.value.items.length; i++) {
                if (r.value.items[i].relativePath.startsWith(scope.scopePrefix!)) {
                  keep.push({ item: r.value.items[i], score: r.value.scores[i] });
                }
              }
              results.push({
                items: keep.map((k) => k.item),
                scores: keep.map((k) => k.score),
                totalMatched: keep.length,
                totalFiles: r.value.totalFiles,
              });
            } else {
              results.push(r.value);
            }
          }
        }
        result = mergeFindResults(results);
        // For multi-scope, hasMore is based on whether ANY scope filled its page
        if (anyScopeFull) {
          hasMore = true;
        }
      }
      // Space-in-pattern fallback: only for single-scope searches
      if (
        result.items.length === 0 &&
        /\s/.test(pattern.trim()) &&
        !resumed &&
        paths.length <= 1
      ) {
        const scope = scopes[0] ?? { basePath, query: "" };
        const scopedQuery = buildQuery(paths, "", params.exclude, scope.basePath);
        const fallback = await withFinderLease(
          scope.basePath,
          (finder) =>
            finder.fileSearch(scopedQuery, {
              pageIndex: 0,
              pageSize: Math.max(effectiveLimit, 500),
            }),
          includeIgnored,
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

      let regexFallbackUsed = false;
      let regexFallbackAlts: string[] = [];

      // Regex alternation fallback: only for single-scope searches
      if (!params.cursor && !resumed && paths.length <= 1 && pattern.includes("|")) {
        const scope = scopes[0] ?? { basePath: activeCwd, query: "" };
        const alternatives = pattern
          .split("|")
          .map((s) => s.trim().replace(/[()]/g, ""))
          .filter(Boolean);
        regexFallbackAlts = alternatives;
        if (alternatives.length > 1) {
          const seen = new Set<string>();
          const merged: FileItem[] = [];
          let totalFiles = 0;
          for (const alt of alternatives) {
            const altQuery = buildQuery(paths, alt, params.exclude, scope.basePath);
            const altResult = await withFinderLease(
              scope.basePath,
              (finder) =>
                finder.fileSearch(altQuery, {
                  pageIndex: 0,
                  pageSize: effectiveLimit,
                }),
              includeIgnored,
            );
            if (altResult.ok) {
              totalFiles = Math.max(totalFiles, altResult.value.totalFiles);
              for (const item of altResult.value.items) {
                if (!seen.has(item.relativePath)) {
                  seen.add(item.relativePath);
                  merged.push(item);
                }
              }
            }
          }
          if (merged.length > 0) {
            const total = weakScoreThreshold(pattern) + 1;
            const scores: Score[] = merged.map(() => ({
              total,
              baseScore: total,
              filenameBonus: 0,
              specialFilenameBonus: 0,
              frecencyBoost: 0,
              distancePenalty: 0,
              currentFilePenalty: 0,
              comboMatchBoost: 0,
              exactMatch: false,
              matchType: "alternation",
            }));
            result = {
              items: merged,
              scores,
              totalMatched: merged.length,
              totalFiles,
            };
            regexFallbackUsed = true;
          }
        }
      }

      const formatted = formatFindOutput(result, effectiveLimit, pattern);
      let output = formatted.output;

      // Infer hasMore for single-scope (multi-scope tracked in search block above).
      if (!resumed && !hasMore) {
        const shownSoFar = pageIndex * effectiveLimit + result.items.length;
        hasMore =
          result.items.length >= effectiveLimit &&
          (result.totalMatched ?? result.items.length) > shownSoFar;
      }

      const notices: string[] = [];
      if (result.items.length === 0) {
        const noResults = await noResultsMessage(
          "No files found matching pattern",
          basePath,
          paths[0],
          includeIgnored,
        );
        if (noResults !== "No files found matching pattern") notices.push(noResults);
      }
      const sourceSearchNotice = getFindSourceSearchNotice(pattern);
      if (sourceSearchNotice) notices.push(sourceSearchNotice);
      if (formatted.weak && formatted.shownCount > 0)
        notices.push(
          `Query "${pattern}" produced only weak scattered fuzzy matches. Output capped at ${formatted.shownCount}/${result.totalMatched}.`,
        );
      const hiddenFuzzyMatches = result.totalMatched - formatted.shownCount;
      if (formatted.literalTailSuppressed && hiddenFuzzyMatches >= 1000)
        notices.push(`${formatted.shownCount} exact matches shown. Fuzzy tail hidden`);

      if (
        !formatted.weak &&
        !formatted.literalTailSuppressed &&
        hasMore &&
        !regexFallbackUsed
      ) {
        const remaining =
          result.totalMatched - (pageIndex * effectiveLimit + result.items.length);
        const cursorId = storeFindCursor({
          scopes: scopes.map((s) => ({
            basePath: s.basePath,
            query: s.query,
            nextPageIndex: pageIndex + 1,
            scopePrefix: s.scopePrefix,
          })),
          pattern,
          pageSize: effectiveLimit,
          includeIgnored,
        });
        notices.push(`${remaining} more. Next page: find cursor="${cursorId}"`);
      }
      if (regexFallbackUsed) {
        notices.push(
          `Regex alternation (|) in pattern treated as ${regexFallbackAlts.length} searches: ${regexFallbackAlts.map((s) => `"${s}"`).join(", ")}`,
        );
      }
      if (includeIgnored) notices.unshift("ignored files included");

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
      const path = formatRenderPath(args?.path);
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
  // Enabled by default. Disable with `PI_FFF_MULTIGREP=0`
  const enableMultiGrep = process.env.PI_FFF_MULTIGREP !== "0";

  if (enableMultiGrep) {
    const multiGrepSchema = Type.Object({
      patterns: Type.Array(Type.String(), {
        description:
          "Exact literal patterns (OR), not regex. Include snake_case/camelCase/PascalCase variants.",
        minItems: 1,
        maxItems: 20,
      }),
      path: Type.Optional(
        Type.Array(Type.String(), {
          description:
            "Path constraints: files, directories, or globs. Each entry is one scope \u2014 searches OR across all entries. Paths with spaces work correctly as single entries.",
        }),
      ),
      exclude: Type.Optional(
        Type.Union([Type.String(), Type.Array(Type.String())], {
          description:
            "Exclude paths (comma/space-separated or array). Same syntax as path: directory prefix ('test/'), filename with extension ('config.json'), or glob ('*.min.js', '**/*.{rs,go}'). A leading '!' is optional and ignored. Example: 'test/,*.min.js,!vendor/'.",
        }),
      ),
      includeIgnored: Type.Optional(
        Type.Boolean({
          description:
            "Include files matched by .gitignore, .ignore, git excludes, and global gitignore. Default false. Use when the target file or directory exists but normal search cannot see it because it is ignored, such as node_modules or build output.",
        }),
      ),
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
        "Search file contents for ANY of multiple exact literal patterns (OR logic). Use this when the request lists concrete identifiers/terms; not for regex structure.",
      promptSnippet: "Search exact literal alternatives",
      promptGuidelines: [
        "Use for content searches with 2-10 exact identifiers, method/class names, UI verbs, or naming variants.",
        "When a prompt lists several exact terms to inspect, this should usually be the first content search.",
        "For classes/types/functions, search bare names first (Widget), not phrases (class Widget); use grep regex when structure matters.",
        "Prefer this over grep regex alternation when each pattern is a concrete string.",
        "Patterns are ORed literals, not regexes or globs; use grep for class\\s+\\w*Widget-style structure.",
        "Do not use for broad concepts or unrelated keywords; run separate scoped searches instead.",
        "Use constraints for file filters, e.g. '*.{ts,tsx} !test/'.",
        "Output tags each match with the pattern index.",
        "Set includeIgnored only when intentionally searching ignored files such as node_modules or build output.",
      ],
      parameters: multiGrepSchema,

      async execute(_toolCallId, params, signal) {
        if (signal?.aborted) throw new Error("Operation aborted");
        if (!params.patterns?.length)
          throw new Error("patterns array must have at least 1 element");
        const rawMgPaths = params.path ?? [];
        const mgPaths = Array.isArray(rawMgPaths)
          ? rawMgPaths
          : typeof rawMgPaths === "string"
            ? [rawMgPaths]
            : [];
        const invalidPath = mgPaths.length > 0 ? invalidPathMessage(mgPaths) : null;
        if (invalidPath) throw new Error(invalidPath);

        const effectiveLimit = Math.max(1, params.limit ?? DEFAULT_GREP_LIMIT);
        const includeIgnored = params.includeIgnored === true;

        // Group paths by resolved basePath; each group runs one multiGrep call
        const baseGroups = new Map<string, string[]>();
        const mgScopePrefixes = new Map<string, string>();
        if (mgPaths.length === 0) {
          baseGroups.set(activeCwd, []);
        } else {
          for (const p of mgPaths) {
            const { basePath, pathConstraint, scopePrefix } = resolveSearchBase(
              p,
              activeCwd,
            );
            const key = basePath;
            if (!baseGroups.has(key)) baseGroups.set(key, []);
            if (pathConstraint) baseGroups.get(key)!.push(pathConstraint);
            // Track scopePrefix for space-path post-filtering
            if (scopePrefix) {
              if (!mgScopePrefixes.has(key)) mgScopePrefixes.set(key, scopePrefix);
            }
          }
        }

        // Resolve: when a group has path constraints, combine them with exclude + explicit constraints
        const mgResults: GrepResult[] = [];
        for (const [basePath, pathConstraints] of baseGroups) {
          const parts: string[] = [];
          for (const pc of pathConstraints) {
            const normalized = normalizePathConstraint(pc, basePath);
            if (normalized) parts.push(normalized);
          }
          parts.push(...normalizeExcludes(params.exclude, basePath));
          if (params.constraints) parts.push(params.constraints);
          const effectiveConstraints = parts.join(" ");

          const r = await withFinderLease(
            basePath,
            (finder) =>
              finder.multiGrep({
                patterns: params.patterns,
                constraints: effectiveConstraints,
                maxMatchesPerFile: Math.min(effectiveLimit, 50),
                smartCase: true,
                cursor: (params.cursor ? getCursor(params.cursor)?.cursor : null) ?? null,
                beforeContext: params.context ?? 0,
                afterContext: params.context ?? 0,
                classifyDefinitions: true,
              }),
            includeIgnored,
          );
          if (!r.ok) throw new Error(r.error);
          // Apply scopePrefix filtering for space-paths resolved to git root
          if (mgScopePrefixes.has(basePath)) {
            const prefix = mgScopePrefixes.get(basePath)!;
            const filtered = r.value.items.filter((m) =>
              m.relativePath.startsWith(prefix),
            );
            mgResults.push({
              ...r.value,
              items: filtered,
              totalMatched: filtered.length,
            });
          } else {
            mgResults.push(r.value);
          }
          // Cursor: only run first scope
          if (params.cursor) break;
        }

        const result = mergeGrepResults(mgResults);
        let output = formatGrepOutput(result);

        const notices: string[] = [];
        if (result.items.length === 0) {
          const phraseMissNotice = getMultiGrepPhraseMissNotice(params.patterns);
          if (phraseMissNotice) notices.push(phraseMissNotice);
        }
        if (result.items.length >= effectiveLimit)
          notices.push(`${effectiveLimit}+ matches (refine patterns)`);
        if (result.nextCursor)
          notices.push(
            `More available. cursor="${storeCursor(result.nextCursor, includeIgnored)}" to continue`,
          );
        if (includeIgnored) notices.unshift("ignored files included");

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
        const patterns = formatRenderPatterns(args?.patterns);
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
