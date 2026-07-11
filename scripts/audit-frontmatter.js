const fs = require('node:fs');
const path = require('node:path');
const matter = require('gray-matter');

if (require.main === module) {
  const root = path.resolve(__dirname, '..', 'source', '_posts');
  const errors = [];

  function walk(dir) {
    for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
      const full = path.join(dir, entry.name);
      if (entry.isDirectory()) walk(full);
      else if (entry.name.endsWith('.md')) validate(full);
    }
  }

  function validate(file) {
    const { data } = matter(fs.readFileSync(file, 'utf8'));
    const relative = path.relative(root, file);
    for (const key of ['title', 'date', 'description', 'tags', 'categories']) {
      if (data[key] === undefined || data[key] === '') errors.push(`${relative}: missing ${key}`);
    }
    if (!Array.isArray(data.tags)) errors.push(`${relative}: tags must be a YAML list`);
    if (!Array.isArray(data.categories)) errors.push(`${relative}: categories must be a YAML list`);
    for (const tag of Array.isArray(data.tags) ? data.tags : []) {
      if (/[,，]/.test(String(tag))) errors.push(`${relative}: combined tag "${tag}"`);
    }
    const date = new Date(data.date);
    if (!Number.isNaN(date.valueOf()) && date > new Date()) errors.push(`${relative}: future date ${data.date}`);
  }

  walk(root);
  if (errors.length) {
    errors.forEach(error => console.error(`FRONTMATTER: ${error}`));
    process.exit(1);
  }
  console.log('FRONTMATTER: all posts passed');
}
