'use strict';

const fs = require('node:fs');
const path = require('node:path');

function isHidden(name) {
  return name.startsWith('.');
}

function toPosixRelative(root, file) {
  return path.relative(root, file).split(path.sep).join('/');
}

/**
 * Deterministically enumerate files under root.
 *
 * Hidden entries (dot-prefixed files and directories) are excluded by default.
 * Only regular files are returned; symlinks are intentionally ignored.
 *
 * @param {string} root absolute directory to walk
 * @param {{extensions?: string[], includeHidden?: boolean}} [options]
 * @returns {string[]} absolute file paths, sorted by POSIX relative path
 */
function walkFiles(root, options = {}) {
  const { extensions = null, includeHidden = false } = options;
  const extensionSet = extensions
    ? new Set(extensions.map((ext) => (ext.startsWith('.') ? ext.toLowerCase() : `.${ext.toLowerCase()}`)))
    : null;

  const results = [];

  function visit(dir) {
    try {
      const entries = fs.readdirSync(dir, { withFileTypes: true });
      entries.sort((a, b) => (a.name < b.name ? -1 : a.name > b.name ? 1 : 0));

      for (const entry of entries) {
        if (!includeHidden && isHidden(entry.name)) continue;
        const full = path.join(dir, entry.name);
        if (entry.isDirectory()) {
          visit(full);
        } else if (entry.isFile()) {
          if (!extensionSet || extensionSet.has(path.extname(entry.name).toLowerCase())) {
            results.push(full);
          }
        }
      }
    } catch (error) {
      throw new Error(`Unable to enumerate baseline directory: ${dir}`, { cause: error });
    }
  }

  visit(root);

  results.sort((a, b) => {
    const ra = toPosixRelative(root, a);
    const rb = toPosixRelative(root, b);
    return ra < rb ? -1 : ra > rb ? 1 : 0;
  });

  return results;
}

module.exports = {
  isHidden,
  toPosixRelative,
  walkFiles,
};
