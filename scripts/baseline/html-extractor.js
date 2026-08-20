'use strict';

const fs = require('node:fs');
const path = require('node:path');
const { walkFiles, toPosixRelative } = require('./file-walker');

const HTML_EXTENSIONS = ['.html'];

function parseTagAttributes(tag) {
  const attributes = {};
  const attributePattern = /([:\w-]+)\s*=\s*(?:"([^"]*)"|'([^']*)'|([^\s"'=<>`]+))/g;
  let match;
  while ((match = attributePattern.exec(tag)) !== null) {
    const key = match[1].toLowerCase();
    const value = match[2] !== undefined ? match[2] : match[3] !== undefined ? match[3] : match[4] || '';
    attributes[key] = value;
  }
  return attributes;
}

function decodeHtmlEntities(value) {
  return String(value)
    .replace(/&#x([0-9a-f]+);/gi, (_, hex) => String.fromCodePoint(parseInt(hex, 16)))
    .replace(/&#([0-9]+);/g, (_, dec) => String.fromCodePoint(parseInt(dec, 10)))
    .replace(/&quot;/g, '"')
    .replace(/&#39;/g, "'")
    .replace(/&lt;/g, '<')
    .replace(/&gt;/g, '>')
    .replace(/&amp;/g, '&')
    .replace(/&nbsp;/g, ' ');
}

function findMetaByAttribute(html, attributeName, attributeValue) {
  return findMetasByAttribute(html, attributeName, attributeValue)[0] || '';
}

function findMetasByAttribute(html, attributeName, attributeValue) {
  const values = [];
  const metaPattern = /<meta\b[^>]*>/gi;
  let match;
  while ((match = metaPattern.exec(html)) !== null) {
    const attributes = parseTagAttributes(match[0]);
    const current = attributes[attributeName.toLowerCase()];
    if (current !== undefined && current.toLowerCase() === attributeValue.toLowerCase()) {
      values.push(attributes.content !== undefined ? decodeHtmlEntities(attributes.content) : '');
    }
  }
  return values;
}

function getMetaName(html, name) {
  return findMetaByAttribute(html, 'name', name);
}

function getMetaProperty(html, property) {
  return findMetaByAttribute(html, 'property', property);
}

function getMetaProperties(html, property) {
  return findMetasByAttribute(html, 'property', property);
}

function getCanonical(html) {
  const linkPattern = /<link\b[^>]*>/gi;
  let match;
  while ((match = linkPattern.exec(html)) !== null) {
    const attributes = parseTagAttributes(match[0]);
    if (attributes.rel && attributes.rel.toLowerCase().split(/\s+/).includes('canonical')) {
      return attributes.href || '';
    }
  }
  return '';
}

function getTitle(html) {
  const ogTitle = getMetaProperty(html, 'og:title');
  if (ogTitle) return ogTitle.trim();

  const titlePattern = /<title\b[^>]*>([\s\S]*?)<\/title>/i;
  const match = titlePattern.exec(html);
  if (!match) return '';

  return match[1].replace(/\s+/g, ' ').trim();
}

function classify(relativeFile) {
  if (relativeFile === 'index.html') return 'home';
  if (/^archives\//.test(relativeFile)) return 'archive';
  if (/^categories\//.test(relativeFile)) return 'category';
  if (/^tags\//.test(relativeFile)) return 'tag';
  if (/^about\//.test(relativeFile)) return 'about';
  if (/^\d{4}\/\d{2}\/\d{2}\//.test(relativeFile)) return 'article';
  if (/^page\//.test(relativeFile)) return 'page';
  return 'other';
}

function relativeFileToPathname(relativeFile) {
  if (relativeFile === 'index.html') return '/';
  const parts = relativeFile.split('/');
  const last = parts[parts.length - 1];
  if (last === 'index.html') {
    parts.pop();
    return `/${parts.map((part) => encodeURIComponent(part)).join('/')}${parts.length ? '/' : ''}`;
  }
  return `/${parts.map((part) => encodeURIComponent(part)).join('/')}`;
}

function getPathname(html, relativeFile) {
  const canonical = getCanonical(html);
  if (canonical) {
    try {
      return new URL(canonical).pathname;
    } catch (error) {
      // Fall through to filesystem-derived pathname for malformed canonical URLs.
    }
  }
  return relativeFileToPathname(relativeFile);
}

function extractHtmlRecord(filePath, publicRoot) {
  const html = fs.readFileSync(filePath, 'utf8');
  const relativeFile = toPosixRelative(publicRoot, filePath);
  return {
    pathname: getPathname(html, relativeFile),
    relativeFile,
    kind: classify(relativeFile),
    title: getTitle(html),
    description: getMetaName(html, 'description'),
    canonical: getCanonical(html),
    publishedTime: getMetaProperty(html, 'article:published_time'),
    tags: getMetaProperties(html, 'article:tag'),
  };
}

function extractHtmlInventory(publicRoot) {
  return walkFiles(publicRoot, { extensions: HTML_EXTENSIONS }).map((file) =>
    extractHtmlRecord(file, publicRoot),
  );
}

module.exports = {
  classify,
  decodeHtmlEntities,
  extractHtmlInventory,
  extractHtmlRecord,
  getCanonical,
  getMetaName,
  getMetaProperties,
  getMetaProperty,
  getPathname,
  getTitle,
  parseTagAttributes,
  relativeFileToPathname,
};
