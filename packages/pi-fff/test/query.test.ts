import { describe, expect, test } from "bun:test";
import {
  buildQuery,
  mergeFindResults,
  mergeGrepResults,
  normalizePathConstraint,
} from "../src/query";

describe("path constraint normalization", () => {
  test("converts absolute in-workspace paths to repo-relative constraints", () => {
    expect(normalizePathConstraint("/tmp/workspace/.agents/**", cwd)).toBe(".agents/");
    expect(normalizePathConstraint("/tmp/workspace/.agents/plans/**", cwd)).toBe(
      ".agents/plans/",
    );
  });

  test("rejects absolute paths outside the workspace", () => {
    expect(() => normalizePathConstraint("/tmp/other/.agents/**", cwd)).toThrow(
      "Path constraint must be relative to the workspace",
    );
  });

  test("collapses only simple trailing recursive directory globs", () => {
    expect(normalizePathConstraint(".agents/**", cwd)).toBe(".agents/");
    expect(normalizePathConstraint("src/**/*", cwd)).toBe("src/");
    expect(normalizePathConstraint("src/**/*.ts", cwd)).toBe("src/**/*.ts");
    expect(normalizePathConstraint("{src,lib}/**", cwd)).toBe("{src,lib}/**");
  });

  test("treats path='.' as workspace root (no constraint)", () => {
    expect(normalizePathConstraint(".", cwd)).toBeNull();
    expect(normalizePathConstraint("./", cwd)).toBeNull();
  });

  test("treats absolute workspace root as no constraint", () => {
    expect(normalizePathConstraint(cwd, cwd)).toBeNull();
  });

  test("bare directory path without trailing slash becomes PathSegment", () => {
    expect(normalizePathConstraint("app", cwd)).toBe("app/");
    expect(normalizePathConstraint("src/nested", cwd)).toBe("src/nested/");
  });

  test("converts absolute in-workspace file path to repo-relative", () => {
    expect(normalizePathConstraint("/tmp/workspace/src/main.rs", cwd)).toBe("src/main.rs");
  });

  test("converts absolute in-workspace directory (without trailing slash) to repo-relative", () => {
    expect(normalizePathConstraint("/tmp/workspace/src", cwd)).toBe("src/");
  });

  test("converts absolute in-workspace glob path to repo-relative glob", () => {
    expect(normalizePathConstraint("/tmp/workspace/src/**/*.ts", cwd)).toBe("src/**/*.ts");
  });
});

describe("buildQuery with path arrays", () => {
  test("single path produces same result as before", () => {
    expect(buildQuery(["app"], "needle", undefined, cwd)).toBe("app/ needle");
  });

  test("empty array uses just pattern", () => {
    expect(buildQuery([], "needle", undefined, cwd)).toBe("needle");
  });

  test("multiple paths are joined in the query", () => {
    expect(buildQuery(["src/", "tests/"], "needle", undefined, cwd)).toBe(
      "src/ tests/ needle",
    );
  });

  test("workspace root path (.) is omitted", () => {
    expect(buildQuery(["."], "needle", undefined, cwd)).toBe("needle");
    expect(buildQuery(["./"], "needle", undefined, cwd)).toBe("needle");
  });

  test("absolute in-workspace paths are normalized to relative", () => {
    expect(
      buildQuery(["/tmp/workspace/src/main.rs", "/tmp/workspace/tests/"], "needle", undefined, cwd),
    ).toBe("src/main.rs tests/ needle");
  });

  test("globs are preserved", () => {
    expect(buildQuery(["src/**/*.ts"], "needle", undefined, cwd)).toBe(
      "src/**/*.ts needle",
    );
  });

  test("exclude works with path array", () => {
    expect(
      buildQuery(["src/"], "needle", "test/", cwd),
    ).toBe("src/ !test/ needle");
  });

  test("multiple paths with various forms", () => {
    expect(
      buildQuery(
        ["src/", "lib/**/*.rs", "/tmp/workspace/tests/"],
        "search",
        undefined,
        cwd,
      ),
    ).toBe("src/ lib/**/*.rs tests/ search");
  });
});

describe("mergeGrepResults", () => {
  const makeItem = (relPath: string, line: number, content = "line") => ({
    relativePath: relPath,
    lineNumber: line,
    lineContent: content,
    matchRanges: [],
    gitStatus: "clean",
    isDefinition: false,
  });

  test("returns empty result for empty input", () => {
    const r = mergeGrepResults([]);
    expect(r.items).toEqual([]);
    expect(r.totalMatched).toBe(0);
  });

  test("single result passes through unchanged", () => {
    const r = mergeGrepResults([
      { items: [makeItem("a.rs", 1)], totalMatched: 1, totalFiles: 10, totalFilesSearched: 5, filteredFileCount: 0, nextCursor: null },
    ]);
    expect(r.items).toHaveLength(1);
    expect(r.totalMatched).toBe(1);
  });

  test("deduplicates same file+line across scopes", () => {
    const r = mergeGrepResults([
      { items: [makeItem("a.rs", 1)], totalMatched: 1, totalFiles: 10, totalFilesSearched: 5, filteredFileCount: 0, nextCursor: null },
      { items: [makeItem("a.rs", 1)], totalMatched: 1, totalFiles: 10, totalFilesSearched: 5, filteredFileCount: 0, nextCursor: null },
    ]);
    expect(r.items).toHaveLength(1);
  });

  test("keeps different file+line combinations", () => {
    const r = mergeGrepResults([
      { items: [makeItem("a.rs", 1), makeItem("a.rs", 2)], totalMatched: 2, totalFiles: 10, totalFilesSearched: 5, filteredFileCount: 0, nextCursor: null },
      { items: [makeItem("b.rs", 1)], totalMatched: 1, totalFiles: 10, totalFilesSearched: 5, filteredFileCount: 0, nextCursor: null },
    ]);
    expect(r.items).toHaveLength(3);
  });

  test("scopePrefix filters out files outside the prefix", () => {
    const r = mergeGrepResults([
      { items: [makeItem("src/a.rs", 1), makeItem("lib/b.rs", 1)], totalMatched: 2, totalFiles: 10, totalFilesSearched: 5, filteredFileCount: 0, nextCursor: null },
    ], "src/");
    expect(r.items).toHaveLength(1);
    expect(r.items[0].relativePath).toBe("src/a.rs");
  });

  test("accumulates totals across results", () => {
    const r = mergeGrepResults([
      { items: [makeItem("a.rs", 1)], totalMatched: 1, totalFiles: 20, totalFilesSearched: 5, filteredFileCount: 0, nextCursor: null },
      { items: [makeItem("b.rs", 1)], totalMatched: 1, totalFiles: 30, totalFilesSearched: 7, filteredFileCount: 0, nextCursor: null },
    ]);
    expect(r.totalMatched).toBe(2);
    expect(r.totalFiles).toBe(30); // max
    expect(r.totalFilesSearched).toBe(5 + 7); // sum
  });
});

describe("mergeFindResults", () => {
  const makeScore = (total: number) => ({
    total, baseScore: total, filenameBonus: 0, specialFilenameBonus: 0,
    frecencyBoost: 0, distancePenalty: 0, currentFilePenalty: 0,
    comboMatchBoost: 0, exactMatch: false, matchType: "fuzzy" as const,
  });

  const makeResult = (paths: string[], scores: number[], totalFiles = 10) => ({
    items: paths.map((p) => ({
      relativePath: p, fileName: p.split("/").pop()!, gitStatus: "clean",
    })),
    scores: scores.map(makeScore),
    totalMatched: paths.length,
    totalFiles,
  });

  test("returns empty result for empty input", () => {
    const r = mergeFindResults([]);
    expect(r.items).toEqual([]);
    expect(r.totalMatched).toBe(0);
  });

  test("single result passes through unchanged", () => {
    const r = mergeFindResults([makeResult(["a.rs"], [10])]);
    expect(r.items).toHaveLength(1);
    expect(r.items[0].relativePath).toBe("a.rs");
  });

  test("deduplicates same file across scopes", () => {
    const r = mergeFindResults([
      makeResult(["a.rs", "b.rs"], [10, 5]),
      makeResult(["a.rs", "c.rs"], [10, 8]),
    ]);
    expect(r.items).toHaveLength(3);
    // a.rs should appear only once with first result's score
    const aCount = r.items.filter((i: any) => i.relativePath === "a.rs").length;
    expect(aCount).toBe(1);
  });

  test("sorts merged results by score descending", () => {
    const r = mergeFindResults([
      makeResult(["low.rs"], [3]),
      makeResult(["high.rs"], [100]),
      makeResult(["mid.rs"], [50]),
    ]);
    expect(r.items[0].relativePath).toBe("high.rs");
    expect(r.items[1].relativePath).toBe("mid.rs");
    expect(r.items[2].relativePath).toBe("low.rs");
  });

  test("scopePrefix filters out files outside the prefix", () => {
    const r = mergeFindResults([
      makeResult(["src/a.rs", "lib/b.rs"], [10, 5]),
    ], "src/");
    expect(r.items).toHaveLength(1);
    expect(r.items[0].relativePath).toBe("src/a.rs");
  });

  test("scopePrefix with multi-scope deduplication", () => {
    // Two scopes, one filtered by prefix, both have overlapping files
    const r = mergeFindResults([
      makeResult(["src/a.rs", "other/b.rs"], [10, 5]),
      makeResult(["src/a.rs", "src/c.rs"], [10, 8]),
    ], "src/");
    expect(r.items).toHaveLength(2);
    expect(r.items.map((i: any) => i.relativePath).sort()).toEqual(["src/a.rs", "src/c.rs"]);
  });
});

const cwd = "/tmp/workspace";

describe("path constraint normalization", () => {
  test("converts absolute in-workspace paths to repo-relative constraints", () => {
    expect(normalizePathConstraint("/tmp/workspace/.agents/**", cwd)).toBe(".agents/");
    expect(normalizePathConstraint("/tmp/workspace/.agents/plans/**", cwd)).toBe(
      ".agents/plans/",
    );
  });

  test("rejects absolute paths outside the workspace", () => {
    expect(() => normalizePathConstraint("/tmp/other/.agents/**", cwd)).toThrow(
      "Path constraint must be relative to the workspace",
    );
  });

  test("collapses only simple trailing recursive directory globs", () => {
    expect(normalizePathConstraint(".agents/**", cwd)).toBe(".agents/");
    expect(normalizePathConstraint("src/**/*", cwd)).toBe("src/");
    expect(normalizePathConstraint("src/**/*.ts", cwd)).toBe("src/**/*.ts");
    expect(normalizePathConstraint("{src,lib}/**", cwd)).toBe("{src,lib}/**");
  });

  test("treats path='.' as workspace root (no constraint)", () => {
    expect(normalizePathConstraint(".", cwd)).toBeNull();
    expect(normalizePathConstraint("./", cwd)).toBeNull();
  });

  test("treats absolute workspace root as no constraint", () => {
    expect(normalizePathConstraint(cwd, cwd)).toBeNull();
  });

  test("bare directory path without trailing slash becomes PathSegment", () => {
    expect(normalizePathConstraint("app", cwd)).toBe("app/");
    expect(normalizePathConstraint("src/nested", cwd)).toBe("src/nested/");
  });

  test("converts absolute in-workspace file path to repo-relative", () => {
    expect(normalizePathConstraint("/tmp/workspace/src/main.rs", cwd)).toBe("src/main.rs");
  });

  test("converts absolute in-workspace directory (without trailing slash) to repo-relative", () => {
    expect(normalizePathConstraint("/tmp/workspace/src", cwd)).toBe("src/");
  });

  test("converts absolute in-workspace glob path to repo-relative glob", () => {
    expect(normalizePathConstraint("/tmp/workspace/src/**/*.ts", cwd)).toBe("src/**/*.ts");
  });
});

describe("buildQuery with path arrays", () => {
  test("single path produces same result as before", () => {
    expect(buildQuery(["app"], "needle", undefined, cwd)).toBe("app/ needle");
  });

  test("empty array uses just pattern", () => {
    expect(buildQuery([], "needle", undefined, cwd)).toBe("needle");
  });

  test("multiple paths are joined in the query", () => {
    expect(buildQuery(["src/", "tests/"], "needle", undefined, cwd)).toBe(
      "src/ tests/ needle",
    );
  });

  test("workspace root path (.) is omitted", () => {
    expect(buildQuery(["."], "needle", undefined, cwd)).toBe("needle");
    expect(buildQuery(["./"], "needle", undefined, cwd)).toBe("needle");
  });

  test("absolute in-workspace paths are normalized to relative", () => {
    expect(
      buildQuery(["/tmp/workspace/src/main.rs", "/tmp/workspace/tests/"], "needle", undefined, cwd),
    ).toBe("src/main.rs tests/ needle");
  });

  test("globs are preserved", () => {
    expect(buildQuery(["src/**/*.ts"], "needle", undefined, cwd)).toBe(
      "src/**/*.ts needle",
    );
  });

  test("exclude works with path array", () => {
    expect(
      buildQuery(["src/"], "needle", "test/", cwd),
    ).toBe("src/ !test/ needle");
  });

  test("multiple paths with various forms", () => {
    expect(
      buildQuery(
        ["src/", "lib/**/*.rs", "/tmp/workspace/tests/"],
        "search",
        undefined,
        cwd,
      ),
    ).toBe("src/ lib/**/*.rs tests/ search");
  });
});
