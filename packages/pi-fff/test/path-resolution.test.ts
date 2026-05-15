import { execSync } from "node:child_process";
import { mkdirSync, rmSync, writeFileSync } from "node:fs";
import { describe, expect, test, beforeAll } from "bun:test";
import path from "node:path";
import {
  buildSearchScopes,
  resolveSearchBase,
  expandHomePath,
  concreteStatPath,
  hasHiddenSegment,
  absolutePathBase,
  resolveGitRoot,
} from "../src/query";

const testDir = "/tmp/fff-regression-test";
const spacedDir = `${testDir}/Code Forge`;

function createGitRepo(): string {
  const d = "/tmp/fff-git-" + Date.now() + "-" + Math.random().toString(36).slice(2, 6);
  rmSync(d, { recursive: true, force: true });
  mkdirSync(d + "/pkg", { recursive: true });
  mkdirSync(d + "/My Project", { recursive: true });
  writeFileSync(d + "/pkg/main.go", "package main");
  writeFileSync(d + "/My Project/lib.rs", "pub fn hello()");
  execSync("git init", { cwd: d, stdio: "ignore" });
  execSync("git config user.email test@test.com", { cwd: d, stdio: "ignore" });
  execSync("git config user.name test", { cwd: d, stdio: "ignore" });
  execSync("git add -A", { cwd: d, stdio: "ignore" });
  execSync("git commit -m init", { cwd: d, stdio: "ignore" });
  return d;
}

beforeAll(() => {
  // Simple workspace (no git)
  rmSync(testDir, { recursive: true, force: true });
  mkdirSync(`${testDir}/src`, { recursive: true });
  mkdirSync(spacedDir, { recursive: true });
  writeFileSync(`${testDir}/src/main.rs`, "fn main() {}");
  writeFileSync(`${spacedDir}/main.rs`, "fn main() {}");
});

describe("expandHomePath", () => {
  test("returns unchanged if no tilde", () => {
    const r = expandHomePath("src/main.rs");
    expect(r).toBe("src/main.rs");
  });

  test("expands tilde to home directory", () => {
    const home = process.env.HOME!;
    const r = expandHomePath("~/src");
    expect(r).toBe(`${home}/src`);
  });

  test("expands tilde with path", () => {
    const home = process.env.HOME!;
    const r = expandHomePath("~/Code Forge/**");
    expect(r).toBe(`${home}/Code Forge/**`);
  });
});

describe("concreteStatPath", () => {
  test("resolves relative path to absolute without wildcards", () => {
    const r = concreteStatPath("src/main.rs", testDir);
    expect(r).toBe(`${testDir}/src/main.rs`);
  });

  test("strips wildcard suffix from glob", () => {
    const r = concreteStatPath("Code Forge/**", testDir);
    expect(r).toBe(spacedDir);
  });

  test("resolves absolute path with wildcard", () => {
    const r = concreteStatPath(`${spacedDir}/**`, testDir);
    expect(r).toBe(spacedDir);
  });
});

describe("hasHiddenSegment", () => {
  test("detects hidden directory", () => {
    expect(hasHiddenSegment(".hidden/")).toBe(true);
  });

  test("does not flag regular paths", () => {
    expect(hasHiddenSegment("src/")).toBe(false);
  });

  test("detects nested hidden directory", () => {
    expect(hasHiddenSegment("src/.hidden/dir")).toBe(true);
  });
});

describe("resolveGitRoot", () => {
  test("returns null for non-git directories", () => {
    expect(resolveGitRoot(testDir)).toBeNull();
  });

  test("returns git root for git directories", () => {
    const gd = createGitRepo();
    const r = resolveGitRoot(gd);
    expect(r).toBe(gd);
  });

  test("returns git root for subdirectories inside git repo", () => {
    const gd = createGitRepo();
    const r = resolveGitRoot(`${gd}/pkg`);
    expect(r).toBe(gd);
  });
});

describe("absolutePathBase", () => {
  test("resolves file path to workspace base", () => {
    const r = absolutePathBase(`${testDir}/src/main.rs`);
    // Resolves to the parent directory of the file
    expect(r.basePath).toBe(`${testDir}/src`);
    expect(r.pathConstraint).toBe("main.rs");
  });

  test("resolves glob to directory base", () => {
    const r = absolutePathBase(`${spacedDir}/**`);
    expect(r.basePath).toBe(spacedDir);
    expect(r.pathConstraint).toBeUndefined();
  });
});

describe("resolveSearchBase", () => {
  test("returns activeCwd for undefined input", () => {
    const r = resolveSearchBase(undefined, testDir);
    expect(r.basePath).toBe(testDir);
    expect(r.pathConstraint).toBeUndefined();
  });

  test("resolves relative path to workspace base", () => {
    const r = resolveSearchBase("src/", testDir);
    expect(r.basePath).toBe(testDir);
    expect(r.pathConstraint).toBe("src/");
  });

  test("resolves '.' to workspace root", () => {
    const r = resolveSearchBase(".", testDir);
    expect(r.basePath).toBe(testDir);
    expect(r.pathConstraint).toBeUndefined();
  });

  test("resolves space path to base dir (no git root)", () => {
    const r = resolveSearchBase("Code Forge/**", testDir);
    expect(r.basePath).toBe(spacedDir);
    expect(r.pathConstraint).toBeUndefined();
  });

  test("resolves space path to git root with scopePrefix", () => {
    const gd = createGitRepo();
    const r = resolveSearchBase("My Project/**", gd);
    // Should resolve to git root with scopePrefix
    expect(r.basePath).toBe(gd);
    expect(r.pathConstraint).toBeUndefined();
    expect(r.scopePrefix).toBe("My Project/");
  });

  test("resolves absolute path to git root", () => {
    const gd = createGitRepo();
    const r = resolveSearchBase(`${gd}/pkg/`, gd);
    expect(r.basePath).toBe(gd);
    expect(r.pathConstraint).toBe("pkg");
  });
});

describe("buildSearchScopes", () => {
  test("empty paths returns one scope at activeCwd", () => {
    const scopes = buildSearchScopes([], "needle", undefined, testDir);
    expect(scopes).toHaveLength(1);
    expect(scopes[0].basePath).toBe(testDir);
    expect(scopes[0].query).toContain("needle");
  });

  test("single path creates one scope", () => {
    const scopes = buildSearchScopes(["src/"], "needle", undefined, testDir);
    expect(scopes).toHaveLength(1);
    expect(scopes[0].basePath).toBe(testDir);
    expect(scopes[0].query).toContain("src/");
  });

  test("multiple paths create multiple scopes", () => {
    const scopes = buildSearchScopes(["src/", "pkg/"], "needle", undefined, testDir);
    expect(scopes).toHaveLength(2);
  });

  test("space path gets scopePrefix", () => {
    const gd = createGitRepo();
    const scopes = buildSearchScopes(["My Project/**"], "needle", undefined, gd);
    expect(scopes).toHaveLength(1);
    expect(scopes[0].scopePrefix).toBe("My Project/");
    // Without git root: no scopePrefix
    const scopes2 = buildSearchScopes(["Code Forge/**"], "needle", undefined, testDir);
    expect(scopes2).toHaveLength(1);
    expect(scopes2[0].scopePrefix).toBeUndefined();
  });
});
