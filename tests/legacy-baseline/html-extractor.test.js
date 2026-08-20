'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const test = require('node:test');
const {
  classify,
  extractHtmlRecord,
  getPathname,
  relativeFileToPathname,
} = require('../../scripts/baseline/html-extractor');

function withHtmlFile(html, relativeFile, callback) {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'baseline-html-'));
  const file = path.join(root, ...relativeFile.split('/'));
  fs.mkdirSync(path.dirname(file), { recursive: true });
  fs.writeFileSync(file, html, 'utf8');
  try {
    callback(file, root);
  } finally {
    fs.rmSync(root, { recursive: true, force: true });
  }
}

test('html-extractor preserves URL encoding in pathname', () => {
  const html = `<!DOCTYPE html>
<html><head>
<meta name="description" content="测试 &amp; 描述">
<meta property="og:title" content="示例文章">
<meta property="article:published_time" content="2026-01-23T04:46:14.000Z">
<meta property="article:tag" content="数值分析">
<meta property="article:tag" content="矩阵">
<link rel="canonical" href="https://example.com/2026/01/23/%E6%95%B0%E5%80%BC%E5%88%86%E6%9E%90/%E6%96%87%E7%AB%A0/">
<title>示例文章 | Example</title>
</head></html>`;
  withHtmlFile(html, '2026/01/23/数值分析/文章/index.html', (file, root) => {
    const record = extractHtmlRecord(file, root);
    assert.equal(record.pathname, '/2026/01/23/%E6%95%B0%E5%80%BC%E5%88%86%E6%9E%90/%E6%96%87%E7%AB%A0/');
    assert.equal(record.relativeFile, '2026/01/23/数值分析/文章/index.html');
    assert.equal(record.kind, 'article');
    assert.equal(record.title, '示例文章');
    assert.equal(record.description, '测试 & 描述');
    assert.equal(record.canonical, 'https://example.com/2026/01/23/%E6%95%B0%E5%80%BC%E5%88%86%E6%9E%90/%E6%96%87%E7%AB%A0/');
    assert.equal(record.publishedTime, '2026-01-23T04:46:14.000Z');
    assert.deepEqual(record.tags, ['数值分析', '矩阵']);
  });
});

test('getPathname falls back to encoded filesystem path when canonical is absent', () => {
  assert.equal(
    getPathname('<html></html>', '2026/01/23/数值分析/文章/index.html'),
    '/2026/01/23/%E6%95%B0%E5%80%BC%E5%88%86%E6%9E%90/%E6%96%87%E7%AB%A0/',
  );
  assert.equal(relativeFileToPathname('index.html'), '/');
  assert.equal(relativeFileToPathname('archives/index.html'), '/archives/');
});

test('html-extractor classifies known Hexo page kinds', () => {
  assert.equal(classify('index.html'), 'home');
  assert.equal(classify('2026/01/23/a/index.html'), 'article');
  assert.equal(classify('archives/index.html'), 'archive');
  assert.equal(classify('categories/deep-learning/index.html'), 'category');
  assert.equal(classify('tags/math/index.html'), 'tag');
  assert.equal(classify('about/index.html'), 'about');
  assert.equal(classify('page/2/index.html'), 'page');
  assert.equal(classify('feed.xml'), 'other');
});
