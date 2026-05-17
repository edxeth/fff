import { describe, expect, test } from "bun:test";
import { formatRenderPath, formatRenderPatterns } from "../src/index";

describe("formatRenderPath", () => {
  test("undefined path defaults to '.'", () => {
    expect(formatRenderPath(undefined)).toBe(".");
  });

  test("null path defaults to '.'", () => {
    expect(formatRenderPath(null)).toBe(".");
  });

  test("empty array path defaults to '.'", () => {
    expect(formatRenderPath([])).toBe(".");
  });

  test("single-element array joins to one path", () => {
    expect(formatRenderPath(["src/"])).toBe("src/");
  });

  test("multi-element array joins with comma", () => {
    expect(formatRenderPath(["src/", "tests/"])).toBe("src/, tests/");
  });

  test("string path returned as-is", () => {
    expect(formatRenderPath("src/")).toBe("src/");
  });

  test("empty string defaults to '.'", () => {
    expect(formatRenderPath("")).toBe(".");
  });

  test("number path defaults to '.'", () => {
    expect(formatRenderPath(42)).toBe(".");
  });

  test("object path defaults to '.'", () => {
    expect(formatRenderPath({})).toBe(".");
  });

  test("boolean path defaults to '.'", () => {
    expect(formatRenderPath(false)).toBe(".");
  });

  test("never throws on any input", () => {
    const cases = [undefined, null, "", "src/", [], ["a", "b"], 0, true, {}, Symbol(), /x/];
    for (const c of cases) {
      expect(() => formatRenderPath(c)).not.toThrow();
    }
  });
});

describe("formatRenderPatterns", () => {
  test("undefined patterns returns empty array", () => {
    expect(formatRenderPatterns(undefined)).toEqual([]);
  });

  test("null patterns returns empty array", () => {
    expect(formatRenderPatterns(null)).toEqual([]);
  });

  test("empty array returns empty array", () => {
    expect(formatRenderPatterns([])).toEqual([]);
  });

  test("array of strings returns them", () => {
    expect(formatRenderPatterns(["foo", "bar"])).toEqual(["foo", "bar"]);
  });

  test("filters out non-string elements", () => {
    expect(formatRenderPatterns(["foo", 42, "bar", null, undefined])).toEqual(["foo", "bar"]);
  });

  test("string returns empty array (not iterable as patterns)", () => {
    expect(formatRenderPatterns("hello")).toEqual([]);
  });

  test("number returns empty array", () => {
    expect(formatRenderPatterns(42)).toEqual([]);
  });

  test("object returns empty array", () => {
    expect(formatRenderPatterns({})).toEqual([]);
  });

  test("never throws on any input", () => {
    const cases = [undefined, null, [], ["a"], "hello", 0, true, {}, Symbol(), /x/];
    for (const c of cases) {
      expect(() => formatRenderPatterns(c)).not.toThrow();
    }
  });
});
