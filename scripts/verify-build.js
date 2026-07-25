const fs = require('node:fs');
const path = require('node:path');

if (require.main === module) {
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

  const postPages = htmlFiles.filter(file =>
    path.relative(root, file).replaceAll('\\', '/').match(/^\d{4}\/\d{2}\/\d{2}\//)
  );

  const postsRoot = path.resolve(__dirname, '..', 'source', '_posts');
  const expectedPostCount = (function count(dir) {
    let n = 0;
    for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
      const full = path.join(dir, entry.name);
      if (entry.isDirectory()) n += count(full);
      else if (entry.name.endsWith('.md')) n += 1;
    }
    return n;
  })(postsRoot);

  if (postPages.length !== expectedPostCount) {
    errors.push(`expected ${expectedPostCount} generated posts, found ${postPages.length}`);
  }

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
  const articleBody = html.match(/<article[\s\S]*?<\/article>/i)?.[0] ?? '';
  if (articleBody) {
    const bodyWithoutPandocMath = articleBody.replace(/<span\s+class="math display">[\s\S]*?<\/span>/g, '').replace(/<span\s+class="math inline">[\s\S]*?<\/span>/g, '');
    if (/\$\$/.test(bodyWithoutPandocMath)) {
      errors.push(`${relative}: display-math delimiters leaked into article HTML`);
    }
  }
    if (relative.match(/^\d{4}\/\d{2}\/\d{2}\//) && !description.trim()) {
      errors.push(`${relative}: post has no meta description`);
    }
  }

  if (errors.length) {
    for (const error of errors) console.error(`VERIFY: ${error}`);
    process.exit(1);
  }

  console.log(`VERIFY: ${htmlFiles.length} HTML files passed`);
}
