'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const test = require('node:test');
const { walkFiles } = require('../../scripts/baseline/file-walker');

function makeTempFixture() {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'baseline-walker-'));
  fs.mkdirSync(path.join(root, '.obsidian'), { recursive: true });
  fs.mkdirSync(path.join(root, 'visible', 'nested'), { recursive: true });
  fs.writeFileSync(path.join(root, '.obsidian', 'secret.md'), '# secret\n');
  fs.writeFileSync(path.join(root, '.hidden.md'), '# hidden file\n');
  fs.writeFileSync(path.join(root, 'visible', 'a.md'), '# a\n');
  fs.writeFileSync(path.join(root, 'visible', 'nested', 'b.md'), '# b\n');
  fs.writeFileSync(path.join(root, 'visible', 'c.txt'), 'c\n');
  return root;
}

test('walkFiles excludes hidden directories and files by default', () => {
  const root = makeTempFixture();
  try {
    const files = walkFiles(root, { extensions: ['.md'] });
    const relative = files.map((file) => path.relative(root, file).split(path.sep).join('/'));
    assert.deepEqual(relative, ['visible/a.md', 'visible/nested/b.md']);
  } finally {
    fs.rmSync(root, { recursive: true, force: true });
  }
});

test('walkFiles returns deterministic sorted results across repeated calls', () => {
  const root = makeTempFixture();
  try {
    const first = walkFiles(root).map((file) => path.relative(root, file).split(path.sep).join('/'));
    const second = walkFiles(root).map((file) => path.relative(root, file).split(path.sep).join('/'));
    assert.deepEqual(first, second);
    assert.deepEqual(first, [
      'visible/a.md',
      'visible/c.txt',
      'visible/nested/b.md',
    ]);
  } finally {
    fs.rmSync(root, { recursive: true, force: true });
  }
});

test('walkFiles can include hidden entries when explicitly requested', () => {
  const root = makeTempFixture();
  try {
    const files = walkFiles(root, { extensions: ['.md'], includeHidden: true });
    const relative = files.map((file) => path.relative(root, file).split(path.sep).join('/'));
    assert.ok(relative.includes('.obsidian/secret.md'));
  } finally {
    fs.rmSync(root, { recursive: true, force: true });
  }
});

test('walkFiles fails loudly when the root cannot be enumerated', () => {
  const missing = path.join(os.tmpdir(), `missing-baseline-${Date.now()}`);
  assert.throws(() => walkFiles(missing), /Unable to enumerate baseline directory/);
});
