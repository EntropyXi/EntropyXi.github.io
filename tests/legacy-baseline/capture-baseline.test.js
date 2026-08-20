'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const test = require('node:test');
const { buildAssetList, buildSummary } = require('../../scripts/capture-baseline');

test('buildAssetList records encoded pathname, size, and content hash', () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'baseline-assets-'));
  const asset = path.join(root, '图 像', '示例.png');
  fs.mkdirSync(path.dirname(asset), { recursive: true });
  fs.writeFileSync(asset, 'image-bytes', 'utf8');

  try {
    const records = buildAssetList(root);
    assert.equal(records.length, 1);
    assert.deepEqual(records[0], {
      pathname: '/%E5%9B%BE%20%E5%83%8F/%E7%A4%BA%E4%BE%8B.png',
      relativeFile: '图 像/示例.png',
      size: 11,
      sha256: '2c8648d103e3dd7ad87660da0f126a1443b6d21ac1bd3ec000c5e24e2373a90c',
    });
  } finally {
    fs.rmSync(root, { recursive: true, force: true });
  }
});

test('buildSummary aggregates formula and raw mathjax types deterministically', () => {
  const emptyMath = {
    inlineDollarFormulaCount: 0,
    displayDollarFormulaCount: 0,
    parenthesesFormulaCount: 0,
    bracketFormulaCount: 0,
    unclosed: { inlineDollar: 0, displayDollar: 0, parentheses: 0, bracket: 0 },
  };
  const summary = buildSummary(
    [{ kind: 'article' }],
    [{ relativeFile: 'a.png' }],
    [
      { mathjaxRaw: 'true', mathEnabled: true, math: { ...emptyMath, displayDollarFormulaCount: 2 } },
      { mathjaxRaw: null, mathEnabled: false, math: emptyMath },
    ],
  );

  assert.equal(summary.mathPostCount, 1);
  assert.equal(summary.math.displayDollarFormulaCount, 2);
  assert.deepEqual(summary.mathjaxRawTypes, { null: 1, boolean: 0, string: 1, number: 0, other: 0 });
});
