import path from "node:path";
import type { GrepResult, SearchResult } from "@edxeth/fff-node";

export function mergeGrepResults(results: GrepResult[], scopePrefix?: string): GrepResult {
  if (results.length <= 1 && !scopePrefix)
    return results[0] ?? { items: [], totalMatched: 0, totalFilesSearched: 0, totalFiles: 0, filteredFileCount: 0, nextCursor: null };
  const seen = new Set<string>();
  const items: GrepResult["items"] = [];
  let totalMatched = 0;
  let totalFiles = 0;
  let totalFilesSearched = 0;
  for (const r of results) {
    for (const match of r.items) {
      if (scopePrefix && !match.relativePath.startsWith(scopePrefix)) continue;
      const key = `${match.relativePath}:${match.lineNumber}`;
      if (!seen.has(key)) {
        seen.add(key);
        items.push(match);
      }
    }
    totalMatched += r.totalMatched;
    totalFiles = Math.max(totalFiles, r.totalFiles);
    totalFilesSearched += r.totalFilesSearched;
  }
  return { items, totalMatched, totalFiles, totalFilesSearched, filteredFileCount: 0, nextCursor: null };
}

export function mergeFindResults(results: SearchResult[], scopePrefix?: string): SearchResult {
  if (results.length <= 1 && !scopePrefix)
    return results[0] ?? { items: [], scores: [], totalMatched: 0, totalFiles: 0 };
  const seen = new Set<string>();
  const items: SearchResult["items"] = [];
  const scores: SearchResult["scores"] = [];
  let totalFiles = 0;
  for (const r of results) {
    totalFiles = Math.max(totalFiles, r.totalFiles);
    for (let i = 0; i < r.items.length; i++) {
      const key = r.items[i].relativePath;
      if (scopePrefix && !key.startsWith(scopePrefix)) continue;
      if (!seen.has(key)) {
        seen.add(key);
        items.push(r.items[i]);
        scores.push(r.scores[i]);
      }
    }
  }
  if (scores.length > 1) {
    const idx = items.map((_, i) => i).sort((a, b) => (scores[b]?.total ?? 0) - (scores[a]?.total ?? 0));
    return {
      items: idx.map((i) => items[i]),
      scores: idx.map((i) => scores[i]),
      totalMatched: items.length,
      totalFiles,
    };
  }
  return { items, scores, totalMatched: items.length, totalFiles };
}




// ── Path resolution helpers (extracted from index.ts for testability) ──

import { execSync } from "node:child_process";
import fs from "node:fs";

export function resolveGitRoot(dir: string): string | null {
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


export function expandHomePath(pathConstraint: string): string {
    const home = process.env.HOME ?? process.env.USERPROFILE;
    if (!home) return pathConstraint;
    return pathConstraint.replace(/^~($|\/|\\)/, (_, sep) => home + sep);
  }

export function concreteStatPath(pathConstraint: string, cwd = process.cwd()): string {
    const expanded = expandHomePath(pathConstraint);
    const absolute = path.isAbsolute(expanded) ? expanded : path.resolve(cwd, expanded);
    const wildcard = absolute.search(/[*?[{]/);
    const concrete = wildcard === -1 ? absolute : absolute.slice(0, wildcard);
    if (wildcard === -1) return absolute;
    return concrete.endsWith(path.sep) ? concrete.slice(0, -1) : path.dirname(concrete);
  }

export function hasHiddenSegment(pathConstraint: string): boolean {
    return pathConstraint
      .split(/[\\/]+/)
      .some((segment) => segment.startsWith(".") && segment !== "." && segment !== "..");
  }

export function invalidPathMessage(paths: string[], cwd = process.cwd()): string | null {
    for (const p of paths) {
      const statPath = concreteStatPath(p, cwd);
      if (!fs.existsSync(statPath)) {
        return `Path not found: ${statPath || p}`;
      }
    }
    return null;
  }

export function absolutePathBase(pathConstraint: string): {
    basePath: string;
    pathConstraint?: string;
    scopePrefix?: string;
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
    
    // Path with spaces: resolve to git root when possible for git annotations.
    if (/\s/.test(fallbackBase)) {
      const gitRoot = resolveGitRoot(fallbackBase);
      if (gitRoot) {
        const relative = path.relative(gitRoot, fallbackBase).replaceAll(path.sep, "/");
        return { basePath: gitRoot, pathConstraint: undefined, scopePrefix: relative + "/" };
      }
      return { basePath: fallbackBase, pathConstraint: undefined };
    }
    
    const gitRoot = resolveGitRoot(fallbackBase);
    const basePath = gitRoot ?? fallbackBase;
    const relative = path.relative(basePath, pathConstraint).replaceAll(path.sep, "/");
    const pathValue =
      relative && relative !== "**" && relative !== "**/*" ? relative : undefined;
    return { basePath, pathConstraint: pathValue };
  }

export function resolveSearchBase(pathConstraint: string | undefined, activeCwd = process.cwd()): {
    basePath: string;
    pathConstraint?: string;
    /** When set, search results must be filtered to only files under this prefix.
     * Used for space-containing paths resolved to a git root — the space prevents
     * use as a query constraint, so we search the git root fully and post-filter. */
    scopePrefix?: string;
  } {
    if (!pathConstraint) return { basePath: activeCwd, pathConstraint };
    // Treat "." and "./" as workspace root — search everything in base.
    if (pathConstraint === "." || pathConstraint === "./") return { basePath: activeCwd };
    const expanded = expandHomePath(pathConstraint);
    if (path.isAbsolute(expanded)) return absolutePathBase(expanded);
    if (expanded === ".." || expanded.startsWith(`..${path.sep}`)) {
      return absolutePathBase(path.resolve(activeCwd, expanded));
    }
    // Path with spaces: resolve to git root when possible so git annotations work.
    // The space prevents use as a query-string constraint, so we search the git root
    // fully and post-filter results to the space-containing path via scopePrefix.
    if (/\s/.test(expanded) && fs.existsSync(concreteStatPath(expanded, activeCwd))) {
      const concrete = concreteStatPath(expanded, activeCwd);
      const gitRoot = resolveGitRoot(concrete);
      if (gitRoot) {
        const relative = path.relative(gitRoot, concrete).replaceAll(path.sep, "/");
        return { basePath: gitRoot, pathConstraint: undefined, scopePrefix: relative + "/" };
      }
      return { basePath: concrete, pathConstraint: undefined };
    }
    // Hidden segment path: resolve directly as the base directory.
    if (hasHiddenSegment(expanded) && fs.existsSync(concreteStatPath(expanded, activeCwd))) {
      return { basePath: concreteStatPath(expanded, activeCwd), pathConstraint: undefined };
    }
    return { basePath: activeCwd, pathConstraint };
  }

export function normalizePathConstraint(
  pathConstraint: string,
  cwd = process.cwd(),
): string | null {
  let trimmed = pathConstraint.trim();
  if (!trimmed) return trimmed;

  if (path.isAbsolute(trimmed)) {
    const relative = path.relative(cwd, trimmed).replaceAll(path.sep, "/");
    if (relative === "") return null;
    if (relative.startsWith("../") || relative === ".." || path.isAbsolute(relative)) {
      throw new Error(
        `Path constraint must be relative to the workspace: ${pathConstraint}`,
      );
    }
    trimmed = relative;
  }

  if (trimmed === "." || trimmed === "./") return null;
  // Strip a leading `./` so `./**/*.rs` and `**/*.rs` behave identically.
  if (trimmed.startsWith("./")) trimmed = trimmed.slice(2);

  // Collapse simple trailing recursive directory globs to directory-prefix.
  const recursiveDir = trimmed.match(/^(.*)\/\*\*(?:\/\*)?$/);
  if (recursiveDir) {
    const dir = recursiveDir[1];
    if (dir && !/[*?[{]/.test(dir)) return `${dir}/`;
  }

  // Already signals path-constraint syntax to the parser.
  if (trimmed.startsWith("/") || trimmed.endsWith("/")) return trimmed;
  // Globs (`*.ts`, `src/**/*.cc`, `{src,lib}`) are handled by the parser.
  if (/[*?[{]/.test(trimmed)) return trimmed;
  // Filename with extension (`main.rs`, `config.json`) → FilePath constraint.
  const lastSegment = trimmed.split("/").pop() ?? "";
  if (/\.[a-zA-Z][a-zA-Z0-9]{0,9}$/.test(lastSegment)) return trimmed;
  // Bare directory prefix → append `/` so the parser sees a PathSegment.
  return `${trimmed}/`;
}

// Exclusions are emitted as `!<constraint>` tokens, which the Rust parser
// understands. Normalize each the same way as the include path.
// Tolerate callers passing already-negated forms like `!src/` by stripping
// the leading `!` before normalizing so we never double-negate (`!!src/`).
export function normalizeExcludes(
  exclude: string | string[] | undefined,
  cwd = process.cwd(),
): string[] {
  if (!exclude) return [];
  const list = Array.isArray(exclude) ? exclude : [exclude];
  const out: string[] = [];
  for (const raw of list) {
    const parts = raw
      .split(/[,\s]+/)
      .map((s) => s.trim())
      .filter(Boolean);
    for (const p of parts) {
      const stripped = p.startsWith("!") ? p.slice(1) : p;
      const normalized = normalizePathConstraint(stripped, cwd);
      if (normalized) out.push(`!${normalized}`);
    }
  }
  return out;
}

export function buildQuery(
  paths: string[],
  pattern: string,
  exclude?: string | string[],
  cwd = process.cwd(),
): string {
  const parts: string[] = [];
  for (const p of paths) {
    const constraint = normalizePathConstraint(p, cwd);
    if (constraint) parts.push(constraint);
  }
  parts.push(...normalizeExcludes(exclude, cwd));
  parts.push(pattern);
  return parts.join(" ");
}


export type SearchScope = {
    basePath: string;
    query: string;
    scopePrefix?: string;
  };

  export function buildSearchScopes(
    paths: string[],
    pattern: string,
    exclude: string | string[] | undefined,
    activeCwd = process.cwd(),
  ): SearchScope[] {
    if (paths.length === 0) {
      return [{ basePath: activeCwd, query: buildQuery([], pattern, exclude, activeCwd) }];
    }
    const scopes: SearchScope[] = [];
    for (const p of paths) {
      const { basePath, pathConstraint, scopePrefix } = resolveSearchBase(p, activeCwd);
      const constraints = pathConstraint ? [pathConstraint] : [];
      scopes.push({
        basePath,
        query: buildQuery(constraints, pattern, exclude, basePath),
        scopePrefix,
      });
    }
    return scopes;
  }

