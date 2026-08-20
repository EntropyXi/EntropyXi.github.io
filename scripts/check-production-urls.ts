import { readFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const baselineFile = path.join(root, "tests", "fixtures", "legacy-baseline.json");
const baseUrl = "https://entropyxi.github.io";

interface BaselineHtmlRecord {
  pathname: string;
  kind: string;
}

interface BaselineAssetRecord {
  pathname: string;
}

interface Baseline {
  html: BaselineHtmlRecord[];
  assets: BaselineAssetRecord[];
}

const errors: string[] = [];
let htmlChecked = 0;
let assetsChecked = 0;

async function checkHtml(record: BaselineHtmlRecord): Promise<void> {
  const response = await fetch(`${baseUrl}${record.pathname}`, { redirect: "follow" });
  htmlChecked += 1;
  if (response.status !== 200) {
    errors.push(`${record.pathname}: HTTP ${response.status}`);
    return;
  }
  const finalPathname = new URL(response.url).pathname;
  if (finalPathname !== record.pathname) {
    errors.push(`${record.pathname}: final pathname ${finalPathname}`);
    return;
  }
  const html = await response.text();
  const canonical = html.match(/<link rel="canonical" href="([^"]+)"/)?.[1];
  if (canonical !== undefined) {
    const canonicalPathname = new URL(canonical).pathname;
    if (canonicalPathname !== record.pathname) {
      errors.push(`${record.pathname}: canonical ${canonicalPathname}`);
    }
  }
}

async function checkAsset(record: BaselineAssetRecord): Promise<void> {
  const response = await fetch(`${baseUrl}${record.pathname}`, { redirect: "follow" });
  assetsChecked += 1;
  if (response.status !== 200) {
    errors.push(`${record.pathname}: HTTP ${response.status}`);
  }
}

async function main(): Promise<void> {
  const baseline = JSON.parse(readFileSync(baselineFile, "utf8")) as Baseline;
  const preservedAssets = new Set([
    "/atom.xml",
    "/search.xml",
    "/sitemap.xml",
    "/images/eigenvalue-error.png",
  ]);

  for (const record of baseline.html) await checkHtml(record);
  for (const record of baseline.assets) {
    if (!preservedAssets.has(record.pathname)) continue;
    await checkAsset(record);
  }

  if (errors.length > 0) {
    for (const error of errors) console.error(`SMOKE: ${error}`);
    process.exit(1);
  }

  console.log(`SMOKE: ${htmlChecked} HTML and ${assetsChecked} assets passed`);
}

void main();
