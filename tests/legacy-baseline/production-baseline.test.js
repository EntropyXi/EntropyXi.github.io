'use strict';

const assert = require('node:assert/strict');
const test = require('node:test');
const { ProductionBaselineChecker, deriveBaseUrl } = require('../../scripts/check-production-baseline');

function fakeResponse({ body, status = 200, url }) {
  const buffer = Buffer.from(body);
  return {
    status,
    url,
    async text() {
      return buffer.toString('utf8');
    },
    async arrayBuffer() {
      return buffer;
    },
  };
}

test('deriveBaseUrl uses the first canonical origin', () => {
  assert.equal(
    deriveBaseUrl({ html: [{ canonical: '' }, { canonical: 'https://example.com/a/' }] }),
    'https://example.com/',
  );
});

test('ProductionBaselineChecker validates HTML pathname and canonical', async () => {
  const fetchImpl = async (url) =>
    fakeResponse({
      body: '<link rel="canonical" href="https://example.com/%E6%96%87%E7%AB%A0/">',
      url: url.href,
    });
  const checker = new ProductionBaselineChecker({ fetchImpl, concurrency: 1 });
  const baseline = {
    html: [
      {
        pathname: '/%E6%96%87%E7%AB%A0/',
        canonical: 'https://example.com/%E6%96%87%E7%AB%A0/',
      },
    ],
    assets: [],
  };
  const report = await checker.run(baseline, 'https://example.com/');
  assert.equal(report.summary.failureCount, 0);
  assert.equal(report.html[0].status, 200);
});

test('ProductionBaselineChecker reports status and canonical mismatches', async () => {
  const fetchImpl = async () =>
    fakeResponse({ body: '<html></html>', status: 404, url: 'https://example.com/missing/' });
  const checker = new ProductionBaselineChecker({ fetchImpl, concurrency: 1 });
  const report = await checker.run(
    {
      html: [{ pathname: '/expected/', canonical: 'https://example.com/expected/' }],
      assets: [],
    },
    'https://example.com/',
  );
  assert.equal(report.summary.failureCount, 1);
  assert.match(report.html[0].errors.join(' '), /HTTP 200/);
  assert.match(report.html[0].errors.join(' '), /canonical pathname/);
});
