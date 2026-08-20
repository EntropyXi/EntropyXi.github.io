'use strict';

const crypto = require('node:crypto');
const fs = require('node:fs');
const path = require('node:path');
const matter = require('gray-matter');
const { walkFiles, toPosixRelative } = require('./file-walker');

function normalizeBodyForHash(body) {
  return String(body)
    .replace(/\r\n/g, '\n')
    .replace(/\r/g, '\n')
    .replace(/<!--\s*more\s*-->/gi, '');
}

function sha256Hex(text) {
  return crypto.createHash('sha256').update(text, 'utf8').digest('hex');
}

function extractRawDate(rawFrontMatter) {
  const match = /^date\s*:\s*(.*?)\s*$/m.exec(rawFrontMatter || '');
  return match ? match[1].trim() : '';
}

function stableDate(data, rawFrontMatter) {
  const raw = extractRawDate(rawFrontMatter);
  if (raw) return raw;
  if (data.date instanceof Date && !Number.isNaN(data.date.valueOf())) {
    return data.date.toISOString();
  }
  if (data.date !== undefined && data.date !== null) return String(data.date);
  return '';
}

function maskCode(body) {
  const lines = String(body).split('\n');
  let fence = null;

  return lines
    .map((line) => {
      const marker = /^\s*(`{3,}|~{3,})/.exec(line)?.[1] || null;
      if (fence) {
        if (marker && marker[0] === fence[0] && marker.length >= fence.length) fence = null;
        return '';
      }
      if (marker) {
        fence = marker;
        return '';
      }
      return line.replace(/(`+)(.*?)\1/g, '');
    })
    .join('\n');
}

function isEscaped(text, index) {
  let slashCount = 0;
  for (let cursor = index - 1; cursor >= 0 && text[cursor] === '\\'; cursor -= 1) slashCount += 1;
  return slashCount % 2 === 1;
}

function findMarker(text, marker, start) {
  let index = text.indexOf(marker, start);
  while (index !== -1) {
    if (!isEscaped(text, index)) return index;
    index = text.indexOf(marker, index + marker.length);
  }
  return -1;
}

function collectDelimited(text, open, close, options = {}) {
  const { singleLine = false, maskRanges = [] } = options;
  const segments = [];
  let unclosedCount = 0;
  let cursor = 0;

  function isMasked(index) {
    return maskRanges.some(([from, to]) => index >= from && index < to);
  }

  while (cursor < text.length) {
    let openIndex = findMarker(text, open, cursor);
    while (openIndex !== -1 && isMasked(openIndex)) {
      openIndex = findMarker(text, open, openIndex + open.length);
    }
    if (openIndex === -1) break;

    const contentStart = openIndex + open.length;
    let closeIndex = findMarker(text, close, contentStart);
    while (closeIndex !== -1 && isMasked(closeIndex)) {
      closeIndex = findMarker(text, close, closeIndex + close.length);
    }
    const lineEnd = singleLine ? text.indexOf('\n', contentStart) : -1;
    if (closeIndex === -1 || (lineEnd !== -1 && closeIndex > lineEnd)) {
      unclosedCount += 1;
      cursor = lineEnd === -1 ? text.length : lineEnd + 1;
      continue;
    }

    segments.push({ content: text.slice(contentStart, closeIndex), from: openIndex, to: closeIndex + close.length });
    cursor = closeIndex + close.length;
  }

  return { segments, unclosedCount };
}

function scanMath(body) {
  const normalized = maskCode(body);
  const displayDollar = collectDelimited(normalized, '$$', '$$');
  const inlineSource = normalized.replace(/(?<!\\)\$\$/g, '  ');
  const inlineDollar = collectDelimited(inlineSource, '$', '$', { singleLine: true });
  const parentheses = collectDelimited(normalized, '\\(', '\\)', { singleLine: true });
  const brackets = collectDelimited(normalized, '\\[', '\\]');
  const allSegments = [
    ...displayDollar.segments,
    ...inlineDollar.segments,
    ...parentheses.segments,
    ...brackets.segments,
  ];
  const mathContent = allSegments.map(({ content }) => content).join('\n');

  const environments = {};
  for (const match of mathContent.matchAll(/\\begin\{([^}]+)\}/g)) {
    const name = match[1];
    environments[name] = (environments[name] || 0) + 1;
  }

  const commands = {};
  for (const match of mathContent.matchAll(/\\([A-Za-z]+)/g)) {
    const name = match[1];
    if (name === 'begin' || name === 'end') continue;
    commands[name] = (commands[name] || 0) + 1;
  }

  return {
    inlineDollarFormulaCount: inlineDollar.segments.length,
    displayDollarFormulaCount: displayDollar.segments.length,
    parenthesesFormulaCount: parentheses.segments.length,
    bracketFormulaCount: brackets.segments.length,
    unclosed: {
      inlineDollar: inlineDollar.unclosedCount,
      displayDollar: displayDollar.unclosedCount,
      parentheses: parentheses.unclosedCount,
      bracket: brackets.unclosedCount,
    },
    environments,
    commands,
  };
}

function markdownInventory(postsRoot) {
  const files = walkFiles(postsRoot, { extensions: ['.md'] });
  const sourceRoot = path.dirname(postsRoot);

  return files.map((file) => {
    const raw = fs.readFileSync(file, 'utf8');
    const parsed = matter(raw);
    const body = parsed.content || '';
    const normalizedBody = normalizeBodyForHash(body);

    return {
      sourceRelative: toPosixRelative(sourceRoot, file),
      title: parsed.data.title !== undefined ? String(parsed.data.title) : '',
      date: stableDate(parsed.data, parsed.matter),
      description: parsed.data.description !== undefined ? String(parsed.data.description) : '',
      tags: Array.isArray(parsed.data.tags) ? parsed.data.tags.map(String) : [],
      categories: Array.isArray(parsed.data.categories) ? parsed.data.categories.map(String) : [],
      mathjaxRaw: parsed.data.mathjax === undefined ? null : parsed.data.mathjax,
      mathEnabled: parsed.data.mathjax === true || parsed.data.mathjax === 'true',
      normalizedBodySha256: sha256Hex(normalizedBody),
      math: scanMath(body),
    };
  });
}

module.exports = {
  collectDelimited,
  extractRawDate,
  isEscaped,
  markdownInventory,
  maskCode,
  normalizeBodyForHash,
  scanMath,
  sha256Hex,
  stableDate,
};
