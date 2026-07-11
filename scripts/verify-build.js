const fs = require('node:fs');
const path = require('node:path');

const root = path.resolve(__dirname, '..', 'public');
const htmlFiles = [];
const errors = [];

function walk(dir) {
  for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
    const full = path.join(dir, entry.name);
    if (entry.isDirectory()) walk(full);
    else if (entry.name.endsWith('.html')) htmlFiles.push(full);
  }
}

if (!fs.existsSync(path.join(root, 'index.html'))) {
  console.error('VERIFY: public/index.html is missing');
  process.exit(1);
}

walk(root);
for (const file of htmlFiles) {
  const html = fs.readFileSync(file, 'utf8');
  const relative = path.relative(root, file).replaceAll('\\', '/');
  const description = html.match(/<meta name="description" content="([^"]*)"/i)?.[1] ?? '';

  if (/\.mjx-container|MathJax_Display/.test(description)) {
    errors.push(`${relative}: meta description contains CSS`);
  }
  if (relative.endsWith('/README/index.html')) {
    errors.push(`${relative}: README was published as a post`);
  }
  if (/<article[\s\S]*?\$\$[\s\S]*?<\/article>/i.test(html)) {
    errors.push(`${relative}: display-math delimiters leaked into article HTML`);
  }
}

if (errors.length) {
  for (const error of errors) console.error(`VERIFY: ${error}`);
  process.exit(1);
}

console.log(`VERIFY: ${htmlFiles.length} HTML files passed`);
