function splitTopLevelAlternatives(pattern: string): string[] {
  const parts: string[] = [];
  let current = "";
  let escaped = false;
  let classDepth = 0;
  let parenDepth = 0;

  for (const ch of pattern) {
    if (escaped) {
      current += ch;
      escaped = false;
      continue;
    }

    if (ch === "\\") {
      current += ch;
      escaped = true;
      continue;
    }

    if (ch === "[") classDepth++;
    if (ch === "]" && classDepth > 0) classDepth--;

    if (classDepth === 0) {
      if (ch === "(") parenDepth++;
      if (ch === ")" && parenDepth > 0) parenDepth--;
      if (ch === "|" && parenDepth === 0) {
        parts.push(current);
        current = "";
        continue;
      }
    }

    current += ch;
  }

  parts.push(current);
  return parts;
}

function bareUnanchoredAlternative(part: string): string | null {
  const trimmed = part.trim();
  if (trimmed.startsWith("^") || trimmed.endsWith("$")) return null;
  return /^[A-Za-z_$][A-Za-z0-9_$]*$/.test(trimmed) ? trimmed : null;
}

function codeLikeToken(token: string): boolean {
  return /^[A-Za-z_$][A-Za-z0-9_$]*$/.test(token);
}

function hasIdentifierShape(token: string): boolean {
  return (
    token.includes("_") || token.includes("$") || /[a-z][A-Z]|[A-Z][a-z]/.test(token)
  );
}

function pathLikeQuery(query: string): boolean {
  return /[\/]|[*?[{]/.test(query);
}

export function getRegexAlternationNotice(pattern: string): string | null {
  const alternatives = splitTopLevelAlternatives(pattern);
  if (alternatives.length < 3) return null;

  const bareAlternatives = alternatives
    .map(bareUnanchoredAlternative)
    .filter((part): part is string => part !== null);

  if (bareAlternatives.length < 3) return null;

  return [
    `Regex alternation has ${alternatives.length} top-level branches, including bare unanchored alternatives: ${bareAlternatives.map((part) => `\`${part}\``).join(", ")}.`,
    "Bare alternatives match substrings and can flood results.",
    "If these are exact identifiers, use multi_grep; otherwise split scoped searches or anchor with word boundaries.",
  ].join(" ");
}

export function shouldShowRegexAlternationNotice(
  matches: Array<{ relativePath: string }>,
  limit: number,
  hasMore: boolean,
): boolean {
  if (hasMore) return true;
  if (matches.length < limit) return false;

  const files = new Set(matches.map((match) => match.relativePath));
  return files.size > 1;
}

export function getFindSourceSearchNotice(pattern: string): string | null {
  const query = pattern.trim();
  if (query.length === 0 || pathLikeQuery(query)) return null;

  const tokens = query.split(/\s+/).filter(Boolean);
  const codeTokens = tokens.filter(codeLikeToken);
  if (codeTokens.length === 0) return null;
  if (!codeTokens.some(hasIdentifierShape)) return null;
  if (tokens.length < 3 && codeTokens.length !== 1) return null;

  return [
    "This looks like source-symbol search, but find searches file paths.",
    `Use multi_grep with exact patterns such as: ${codeTokens.map((token) => `"${token}"`).join(", ")}.`,
  ].join(" ");
}

export function getMultiGrepPhraseMissNotice(patterns: string[]): string | null {
  const suggestions = new Set<string>();

  for (const pattern of patterns) {
    const tokens = pattern.trim().split(/\s+/).filter(Boolean);
    if (tokens.length < 2 || !tokens.every(codeLikeToken)) continue;

    const identifierTokens = tokens.filter(hasIdentifierShape);
    for (const token of identifierTokens) suggestions.add(token);
  }

  if (suggestions.size === 0) return null;

  return [
    "multi_grep patterns are exact substrings; phrase patterns can miss split syntax.",
    `If you meant identifiers, retry with bare patterns: ${[...suggestions].map((token) => `"${token}"`).join(", ")}.`,
  ].join(" ");
}
