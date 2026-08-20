import { cpSync, existsSync, rmSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const source = path.join(root, 'node_modules', 'mathjax', 'es5');
const targetDir = path.join(root, 'astro-public', 'vendor', 'mathjax');

if (!existsSync(path.join(source, 'tex-mml-chtml.js'))) {
  throw new Error(`MathJax bundle not found: ${source}`);
}

rmSync(targetDir, { recursive: true, force: true });
cpSync(source, targetDir, { recursive: true });
