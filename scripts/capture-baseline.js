'use strict';

const { execFileSync } = require('node:child_process');
const crypto = require('node:crypto');
const fs = require('node:fs');
const path = require('node:path');
const { walkFiles, toPosixRelative } = require('./baseline/file-walker');
const { extractHtmlInventory, relativeFileToPathname } = require('./baseline/html-extractor');
const { markdownInventory } = require('./baseline/markdown-inventory');

const SCHEMA_VERSION = 1;
const ROOT = path.resolve(__dirname, '..');
const PUBLIC_DIR = path.join(ROOT, 'public');
const POSTS_DIR = path.join(ROOT, 'source', '_posts');
const OUTPUT_FILE = path.join(ROOT, 'tests', 'fixtures', 'legacy-baseline.json');

function getGeneratedFromCommit() {
  try {
    const commit = execFileSync('git', ['rev-parse', 'HEAD'], {
      cwd: ROOT,
      encoding: 'utf8',
      stdio: ['ignore', 'pipe', 'ignore'],
    });
    return commit.trim() || null;
  } catch (error) {
    return null;
  }
}

function buildAssetList(publicDir) {
  return walkFiles(publicDir)
    .filter((file) => path.extname(file).toLowerCase() !== '.html')
    .map((file) => {
      const stat = fs.statSync(file);
      const relativeFile = toPosixRelative(publicDir, file);
      return {
        pathname: relativeFileToPathname(relativeFile),
        relativeFile,
        size: stat.size,
        sha256: crypto.createHash('sha256').update(fs.readFileSync(file)).digest('hex'),
      };
    });
}

function buildSummary(html, assets, posts) {
  const kinds = {};
  for (const record of html) {
    kinds[record.kind] = (kinds[record.kind] || 0) + 1;
  }

  const math = {
    inlineDollarFormulaCount: 0,
    displayDollarFormulaCount: 0,
    parenthesesFormulaCount: 0,
    bracketFormulaCount: 0,
    unclosedFormulaCount: 0,
  };
  const mathjaxRawTypes = { null: 0, boolean: 0, string: 0, number: 0, other: 0 };
  let mathPostCount = 0;

  for (const post of posts) {
    const record = post.math;
    const formulaCount =
      record.inlineDollarFormulaCount +
      record.displayDollarFormulaCount +
      record.parenthesesFormulaCount +
      record.bracketFormulaCount;
    if (formulaCount > 0 || post.mathEnabled) mathPostCount += 1;
    math.inlineDollarFormulaCount += record.inlineDollarFormulaCount;
    math.displayDollarFormulaCount += record.displayDollarFormulaCount;
    math.parenthesesFormulaCount += record.parenthesesFormulaCount;
    math.bracketFormulaCount += record.bracketFormulaCount;
    math.unclosedFormulaCount += Object.values(record.unclosed).reduce((sum, count) => sum + count, 0);

    const rawType = post.mathjaxRaw === null ? 'null' : typeof post.mathjaxRaw;
    const bucket = Object.hasOwn(mathjaxRawTypes, rawType) ? rawType : 'other';
    mathjaxRawTypes[bucket] += 1;
  }

  return {
    htmlCount: html.length,
    assetCount: assets.length,
    postCount: posts.length,
    mathPostCount,
    kinds,
    math,
    mathjaxRawTypes,
  };
}

function captureBaseline() {
  if (!fs.existsSync(path.join(PUBLIC_DIR, 'index.html'))) {
    console.error('CAPTURE: public/index.html is missing; run `hexo generate` first');
    process.exit(1);
  }

  const html = extractHtmlInventory(PUBLIC_DIR);
  const assets = buildAssetList(PUBLIC_DIR);
  const posts = markdownInventory(POSTS_DIR);
  const summary = buildSummary(html, assets, posts);

  const baseline = {
    schemaVersion: SCHEMA_VERSION,
    generatedFromCommit: getGeneratedFromCommit(),
    html,
    assets,
    posts,
    summary,
  };

  fs.mkdirSync(path.dirname(OUTPUT_FILE), { recursive: true });
  fs.writeFileSync(OUTPUT_FILE, `${JSON.stringify(baseline, null, 2)}\n`, 'utf8');

  console.log(`CAPTURE: wrote ${path.relative(ROOT, OUTPUT_FILE)}`);
  console.log(`CAPTURE: ${summary.htmlCount} html, ${summary.assetCount} assets, ${summary.postCount} posts`);
}

if (require.main === module) {
  captureBaseline();
}

module.exports = {
  buildAssetList,
  buildSummary,
  captureBaseline,
  getGeneratedFromCommit,
};
