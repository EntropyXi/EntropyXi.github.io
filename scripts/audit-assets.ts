import fs from "node:fs";
import path from "node:path";

interface AssetBudget {
  file: string;
  maxSizeBytes: number;
}

const HERO_BUDGETS: AssetBudget[] = [
  { file: "astro-public/images/hero/suisen-hero-750w.webp", maxSizeBytes: 400 * 1024 },
  { file: "astro-public/images/hero/suisen-hero-1440w.webp", maxSizeBytes: 800 * 1024 },
  { file: "astro-public/images/hero/suisen-hero-3840w.webp", maxSizeBytes: 2 * 1024 * 1024 },
  { file: "astro-public/images/hero/suisen-hero-750w.png", maxSizeBytes: 600 * 1024 },
  { file: "astro-public/images/hero/suisen-hero-1440w.png", maxSizeBytes: 2 * 1024 * 1024 },
];

function runAssetAudit(): void {
  console.log("Auditing responsive hero image assets...");
  let failed = false;

  for (const item of HERO_BUDGETS) {
    const fullPath = path.resolve(process.cwd(), item.file);
    if (!fs.existsSync(fullPath)) {
      console.error(`❌ Missing required hero asset: ${item.file}`);
      failed = true;
      continue;
    }

    const stat = fs.statSync(fullPath);
    const sizeKb = (stat.size / 1024).toFixed(1);
    const maxKb = (item.maxSizeBytes / 1024).toFixed(1);

    if (stat.size > item.maxSizeBytes) {
      console.error(
        `❌ Asset ${item.file} exceeded budget: ${sizeKb} KB (Budget: ${maxKb} KB)`,
      );
      failed = true;
    } else {
      console.log(`✅ ${item.file}: ${sizeKb} KB (Budget: ${maxKb} KB)`);
    }
  }

  if (failed) {
    process.exit(1);
  }
  console.log("All hero assets meet performance budgets.");
}

runAssetAudit();
