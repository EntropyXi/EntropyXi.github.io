import { gzipSync } from "node:zlib";
import { readdir, readFile } from "node:fs/promises";
import path from "node:path";

async function main(): Promise<void> {
  const assetDirectory = path.resolve("dist", "_astro");
  const totalGzipBudget = 24 * 1024;
  const singleAssetGzipBudget = 8 * 1024;

  const assetNames = (await readdir(assetDirectory))
    .filter((name) => name.endsWith(".js"))
    .sort();

  const assets = await Promise.all(
    assetNames.map(async (name) => {
      const contents = await readFile(path.join(assetDirectory, name));
      return {
        name,
        rawBytes: contents.byteLength,
        gzipBytes: gzipSync(contents, { level: 9 }).byteLength,
      };
    }),
  );

  const totalRawBytes = assets.reduce(
    (total, asset) => total + asset.rawBytes,
    0,
  );
  const totalGzipBytes = assets.reduce(
    (total, asset) => total + asset.gzipBytes,
    0,
  );
  const oversizedAssets = assets.filter(
    ({ gzipBytes }) => gzipBytes > singleAssetGzipBudget,
  );

  if (totalGzipBytes > totalGzipBudget || oversizedAssets.length > 0) {
    const details = oversizedAssets
      .map(({ name, gzipBytes }) => `${name}: ${gzipBytes} B gzip`)
      .join(", ");
    throw new Error(
      `BUNDLE budget exceeded: total ${totalGzipBytes}/${totalGzipBudget} B gzip` +
        (details ? `; oversized assets: ${details}` : ""),
    );
  }

  console.log(
    `BUNDLE: ${assets.length} first-party JS assets, ${totalRawBytes} B raw, ` +
      `${totalGzipBytes}/${totalGzipBudget} B gzip passed`,
  );
}

void main();
