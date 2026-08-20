'use strict';

const crypto = require('node:crypto');
const fs = require('node:fs');
const path = require('node:path');
const { getCanonical } = require('./baseline/html-extractor');

const ROOT = path.resolve(__dirname, '..');
const DEFAULT_BASELINE = path.join(ROOT, 'tests', 'fixtures', 'legacy-baseline.json');

function parseArguments(args) {
  const options = { baseline: DEFAULT_BASELINE, baseUrl: null, output: null, includeAssets: true };
  for (let index = 0; index < args.length; index += 1) {
    const argument = args[index];
    if (argument === '--baseline') options.baseline = path.resolve(args[++index]);
    else if (argument === '--base') options.baseUrl = args[++index];
    else if (argument === '--output') options.output = path.resolve(args[++index]);
    else if (argument === '--skip-assets') options.includeAssets = false;
    else throw new Error(`Unknown argument: ${argument}`);
  }
  return options;
}

function deriveBaseUrl(baseline) {
  const canonical = baseline.html.find((record) => record.canonical)?.canonical;
  if (!canonical) throw new Error('Baseline has no canonical URL from which to derive the production origin');
  return new URL('/', canonical).href;
}

function sha256Hex(buffer) {
  return crypto.createHash('sha256').update(buffer).digest('hex');
}

class ProductionBaselineChecker {
  constructor({ fetchImpl = globalThis.fetch, concurrency = 8, timeoutMs = 15_000 } = {}) {
    if (typeof fetchImpl !== 'function') throw new TypeError('fetchImpl must be a function');
    this.fetchImpl = fetchImpl;
    this.concurrency = concurrency;
    this.timeoutMs = timeoutMs;
  }

  async fetch(pathname, baseUrl) {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), this.timeoutMs);
    try {
      return await this.fetchImpl(new URL(pathname, baseUrl), {
        redirect: 'follow',
        signal: controller.signal,
        headers: { 'user-agent': 'EntropyXi-Astro-Migration-Baseline/1.0' },
      });
    } finally {
      clearTimeout(timeout);
    }
  }

  async checkHtml(record, baseUrl) {
    try {
      const response = await this.fetch(record.pathname, baseUrl);
      const html = await response.text();
      const finalPathname = new URL(response.url || new URL(record.pathname, baseUrl)).pathname;
      const canonical = getCanonical(html);
      const canonicalPathname = canonical ? new URL(canonical, baseUrl).pathname : '';
      const expectedCanonicalPathname = record.canonical
        ? new URL(record.canonical, baseUrl).pathname
        : record.pathname;
      const errors = [];
      if (response.status !== 200) errors.push(`expected HTTP 200, received ${response.status}`);
      if (finalPathname !== record.pathname) errors.push(`final pathname ${finalPathname} != ${record.pathname}`);
      if (canonicalPathname !== expectedCanonicalPathname) {
        errors.push(`canonical pathname ${canonicalPathname || '<missing>'} != ${expectedCanonicalPathname}`);
      }
      return { pathname: record.pathname, status: response.status, finalPathname, canonicalPathname, errors };
    } catch (error) {
      return { pathname: record.pathname, status: null, finalPathname: '', canonicalPathname: '', errors: [error.message] };
    }
  }

  async checkAsset(record, baseUrl) {
    try {
      const response = await this.fetch(record.pathname, baseUrl);
      const content = Buffer.from(await response.arrayBuffer());
      const actualSha256 = sha256Hex(content);
      const errors = [];
      if (response.status !== 200) errors.push(`expected HTTP 200, received ${response.status}`);
      if (actualSha256 !== record.sha256) errors.push(`sha256 ${actualSha256} != ${record.sha256}`);
      return { pathname: record.pathname, status: response.status, sha256: actualSha256, errors };
    } catch (error) {
      return { pathname: record.pathname, status: null, sha256: '', errors: [error.message] };
    }
  }

  async mapConcurrent(records, check) {
    const results = new Array(records.length);
    let nextIndex = 0;
    const workers = Array.from({ length: Math.min(this.concurrency, records.length) }, async () => {
      while (nextIndex < records.length) {
        const currentIndex = nextIndex;
        nextIndex += 1;
        results[currentIndex] = await check(records[currentIndex]);
      }
    });
    await Promise.all(workers);
    return results;
  }

  async run(baseline, baseUrl, { includeAssets = true } = {}) {
    const normalizedBaseUrl = new URL('/', baseUrl).href;
    const html = await this.mapConcurrent(baseline.html, (record) => this.checkHtml(record, normalizedBaseUrl));
    const assets = includeAssets
      ? await this.mapConcurrent(baseline.assets, (record) => this.checkAsset(record, normalizedBaseUrl))
      : [];
    const failureCount = [...html, ...assets].filter((result) => result.errors.length > 0).length;
    return {
      schemaVersion: 1,
      baseUrl: normalizedBaseUrl,
      summary: {
        htmlChecked: html.length,
        assetsChecked: assets.length,
        failureCount,
      },
      html,
      assets,
    };
  }
}

async function main() {
  const options = parseArguments(process.argv.slice(2));
  const baseline = JSON.parse(fs.readFileSync(options.baseline, 'utf8'));
  const baseUrl = options.baseUrl || deriveBaseUrl(baseline);
  const checker = new ProductionBaselineChecker();
  const report = await checker.run(baseline, baseUrl, { includeAssets: options.includeAssets });
  const output = `${JSON.stringify(report, null, 2)}\n`;

  if (options.output) {
    fs.mkdirSync(path.dirname(options.output), { recursive: true });
    fs.writeFileSync(options.output, output, 'utf8');
  } else {
    process.stdout.write(output);
  }

  console.error(
    `PRODUCTION: ${report.summary.htmlChecked} html, ${report.summary.assetsChecked} assets, ${report.summary.failureCount} failures`,
  );
  if (report.summary.failureCount > 0) process.exitCode = 1;
}

if (require.main === module) {
  main().catch((error) => {
    console.error(`PRODUCTION: ${error.stack || error.message}`);
    process.exitCode = 1;
  });
}

module.exports = {
  ProductionBaselineChecker,
  deriveBaseUrl,
  parseArguments,
  sha256Hex,
};
