import { describe, expect, test } from "bun:test";
import {
  getFindSourceSearchNotice,
  getMultiGrepPhraseMissNotice,
  getRegexAlternationNotice,
  shouldShowRegexAlternationNotice,
} from "../src/regex-diagnostics";

describe("getRegexAlternationNotice", () => {
  test("warns on several top-level bare alternatives", () => {
    const notice = getRegexAlternationNotice("alpha|beta|handleInput|class.*Widget");

    expect(notice).toContain("Regex alternation");
    expect(notice).toContain("`alpha`");
    expect(notice).toContain("`beta`");
    expect(notice).toContain("`handleInput`");
    expect(notice).toContain("multi_grep");
  });

  test("allows simple two-way alternatives", () => {
    expect(getRegexAlternationNotice("alpha|beta")).toBeNull();
  });

  test("ignores alternatives inside groups", () => {
    expect(
      getRegexAlternationNotice("class\\s+\\w*(?:Widget|Dialog|Selector)\\b"),
    ).toBeNull();
  });

  test("ignores escaped pipe characters", () => {
    expect(getRegexAlternationNotice("alpha\\|beta|gamma|delta")).toBeNull();
  });

  test("does not warn when alternatives are anchored", () => {
    expect(getRegexAlternationNotice("^alpha$|^beta$|^gamma$")).toBeNull();
  });
});

describe("shouldShowRegexAlternationNotice", () => {
  test("warns when more result pages exist", () => {
    expect(shouldShowRegexAlternationNotice([], 20, true)).toBe(true);
  });

  test("warns when a full page spans multiple files", () => {
    const matches = [
      { relativePath: "a.ts" },
      { relativePath: "b.ts" },
      { relativePath: "b.ts" },
    ];

    expect(shouldShowRegexAlternationNotice(matches, 3, false)).toBe(true);
  });

  test("does not warn for a small result set", () => {
    expect(shouldShowRegexAlternationNotice([{ relativePath: "a.ts" }], 3, false)).toBe(
      false,
    );
  });

  test("does not warn when a full page is confined to one file", () => {
    const matches = [
      { relativePath: "a.ts" },
      { relativePath: "a.ts" },
      { relativePath: "a.ts" },
    ];

    expect(shouldShowRegexAlternationNotice(matches, 3, false)).toBe(false);
  });
});

describe("getFindSourceSearchNotice", () => {
  test("warns when a find query looks like a list of source symbols", () => {
    const notice = getFindSourceSearchNotice(
      "scroll render handleInput Widget Dialog Selector",
    );

    expect(notice).toContain("source-symbol search");
    expect(notice).toContain("multi_grep");
  });

  test("does not warn for ordinary fuzzy path queries", () => {
    expect(getFindSourceSearchNotice("picker ui")).toBeNull();
  });

  test("does not warn for path-like queries", () => {
    expect(getFindSourceSearchNotice("src/**/*.ts")).toBeNull();
  });

  test("warns on a single code-shaped token when path search misses", () => {
    const notice = getFindSourceSearchNotice("handleInput");

    expect(notice).toContain("source-symbol search");
    expect(notice).toContain("handleInput");
  });

  test("clarifies file-path-only results for a single code-shaped token", () => {
    const notice = getFindSourceSearchNotice("Widget");

    expect(notice).toContain("find searches file paths");
    expect(notice).toContain("Widget");
  });

  test("does not warn for all-caps filename-like tokens", () => {
    expect(getFindSourceSearchNotice("README")).toBeNull();
  });
});

describe("getMultiGrepPhraseMissNotice", () => {
  test("warns when exact phrase patterns look like over-specific identifiers", () => {
    const notice = getMultiGrepPhraseMissNotice([
      "class Widget",
      "class Dialog",
      "class Selector",
    ]);

    expect(notice).toContain("exact substrings");
    expect(notice).toContain("Widget");
    expect(notice).toContain("Dialog");
    expect(notice).toContain("Selector");
  });

  test("does not warn for plain prose phrases", () => {
    expect(getMultiGrepPhraseMissNotice(["error message", "log entry"])).toBeNull();
  });
});
