import { existsSync, rmSync, readdirSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");

// 目标清理的本地临时目录与缓存
const cleanTargets = [
  "dist",
  ".astro",
  "test-results",
  "playwright-report",
  "blob-report",
  ".codegraph",
  ".agents",
];

for (const target of cleanTargets) {
  const full = path.join(root, target);
  if (existsSync(full)) {
    rmSync(full, { recursive: true, force: true });
    console.log(`[Clean] Removed: ${target}`);
  }
}

// 清理 audit-screenshots 中未跟踪的冗余本地测试截图（保留 index.json / evidence.json 与必要基准图）
const auditScreenshotsDir = path.join(root, "audit-screenshots");
if (existsSync(auditScreenshotsDir)) {
  const cleanUntrackedScreenshots = (dir: string): void => {
    for (const entry of readdirSync(dir, { withFileTypes: true })) {
      const fullPath = path.join(dir, entry.name);
      if (entry.isDirectory()) {
        cleanUntrackedScreenshots(fullPath);
        // 如果目录为空，则清理空目录
        try {
          if (readdirSync(fullPath).length === 0) {
            rmSync(fullPath, { recursive: true, force: true });
          }
        } catch {}
      } else if (
        (entry.name.endsWith(".png") || entry.name.endsWith(".webp")) &&
        !entry.name.startsWith("article-ddim") &&
        !dir.endsWith("phase-6") &&
        !dir.endsWith("phase-7")
      ) {
        rmSync(fullPath, { force: true });
      }
    }
  };

  cleanUntrackedScreenshots(auditScreenshotsDir);
  console.log("[Clean] Pruned untracked screenshots in audit-screenshots.");
}

console.log("[Clean] Workspace temporary build artifacts cleaned successfully.");
