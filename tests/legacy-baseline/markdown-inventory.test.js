'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const test = require('node:test');
const {
  markdownInventory,
  normalizeBodyForHash,
  scanMath,
  sha256Hex,
} = require('../../scripts/baseline/markdown-inventory');

test('normalizeBodyForHash normalizes line endings and removes more marker', () => {
  const raw = 'line1\r\n<!-- more -->\r\nline2\r\n';
  const clean = 'line1\n\nline2\n';
  assert.equal(
    sha256Hex(normalizeBodyForHash(raw)),
    sha256Hex(normalizeBodyForHash(clean)),
  );
  assert.equal(normalizeBodyForHash(raw), clean);
});

test('scanMath counts delimiters, environments, and commands', () => {
  const body = [
    'Inline $x$ and display',
    '$$',
    '\\sum_{i=1}^n i \\\\',
    '\\begin{aligned}',
    'a &= b \\\\',
    '\\end{aligned}',
    '\\frac{1}{2} \\mathbf{x}',
    '$$',
    '\\(z + 1\\)',
    '\\[w + 1\\]',
    '`$ignored$`',
    '```tex',
    '$$ignored$$',
    '```',
  ].join('\n');

  const math = scanMath(body);
  assert.equal(math.inlineDollarFormulaCount, 1);
  assert.equal(math.displayDollarFormulaCount, 1);
  assert.equal(math.parenthesesFormulaCount, 1);
  assert.equal(math.bracketFormulaCount, 1);
  assert.deepEqual(math.unclosed, {
    inlineDollar: 0,
    displayDollar: 0,
    parentheses: 0,
    bracket: 0,
  });
  assert.deepEqual(math.environments, { aligned: 1 });
  assert.deepEqual(math.commands, {
    sum: 1,
    frac: 1,
    mathbf: 1,
  });
});

test('scanMath does not count Windows paths as LaTeX commands', () => {
  const math = scanMath('Path C:\\Windows\\System32 and $x$');
  assert.deepEqual(math.commands, {});
});

test('scanMath reports unclosed delimiters', () => {
  const math = scanMath('broken $x\nmissing \\[y');
  assert.equal(math.unclosed.inlineDollar, 1);
  assert.equal(math.unclosed.bracket, 1);
});

test('markdownInventory reads posts with gray-matter and stable raw date', () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'baseline-md-'));
  const postsRoot = path.join(root, 'source', '_posts');
  const postDir = path.join(postsRoot, '深度学习');
  fs.mkdirSync(postDir, { recursive: true });
  fs.writeFileSync(
    path.join(postDir, 'demo.md'),
    [
      '---',
      'title: Demo',
      'date: 2026-05-17 14:00:00',
      'description: Desc',
      'tags:',
      '  - 深度学习',
      'mathjax: true',
      'categories:',
      '  - 深度学习',
      '---',
      '<!-- more -->',
      '',
      '$x$ and $$y$$',
    ].join('\n'),
    'utf8',
  );

  try {
    const posts = markdownInventory(postsRoot);
    assert.equal(posts.length, 1);
    const post = posts[0];
    assert.equal(post.sourceRelative, '_posts/深度学习/demo.md');
    assert.equal(post.title, 'Demo');
    assert.equal(post.date, '2026-05-17 14:00:00');
    assert.equal(post.description, 'Desc');
    assert.deepEqual(post.tags, ['深度学习']);
    assert.deepEqual(post.categories, ['深度学习']);
    assert.equal(post.mathjaxRaw, true);
    assert.equal(post.mathEnabled, true);
    assert.equal(post.math.inlineDollarFormulaCount, 1);
    assert.equal(post.math.displayDollarFormulaCount, 1);
    assert.equal(typeof post.normalizedBodySha256, 'string');
    assert.match(post.normalizedBodySha256, /^[a-f0-9]{64}$/);
  } finally {
    fs.rmSync(root, { recursive: true, force: true });
  }
});
