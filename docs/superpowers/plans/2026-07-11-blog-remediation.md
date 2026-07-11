# Blog Remediation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Restore correct mathematical rendering and metadata, improve article discovery, and make the Hexo/NexT build reproducible and maintainable.

**Architecture:** Keep NexT 8 as the only active theme and make it the sole owner of client-side MathJax loading. Add a small post-build verifier that treats broken math, polluted metadata, and accidental posts as build failures. Apply content metadata and repository cleanup only after the rendering baseline is green.

**Tech Stack:** Hexo 8, NexT 8.27, Pandoc renderer, MathJax 3, Node.js 20, GitHub Actions, PowerShell/local shell.

## Global Constraints

- Preserve all current user edits in `source/_posts`; inspect `git diff` before modifying overlapping files.
- Keep `theme: next`; do not reactivate or customize AnZhiYu.
- Do not change published permalinks during this remediation.
- Do not deploy until local generation and `npm run verify` both pass.
- Use small commits at the end of every task; never combine theme deletion with rendering changes.

## File Map

- `_config.yml`: Hexo site, renderer, publication, URL, and date policy.
- `_config.next.yml`: NexT presentation, MathJax loader, excerpts, search, and statistics.
- `source/_data/post-body-end.njk`: Giscus only after MathJax override removal.
- `source/_data/styles.styl`: the single location for site-wide formula and responsive styles.
- `scripts/verify-build.js`: deterministic checks over generated HTML.
- `scripts/audit-frontmatter.js`: validates post metadata and reports malformed tags/dates.
- `package.json`: reproducible commands and dependency ownership.
- `.github/workflows/deploy.yml`: clean install, build, verification, and Pages deployment.
- `source/_posts/**/*.md`: descriptions, excerpts, dates, and tag normalization.
- `source/_posts/深度学习/流匹配与扩散模型/体系/README.md`: remove from publishable posts.
- `README.md`: contributor workflow and validation commands.

---

## Phase 1 — Restore Correctness

### Task 1: Add a failing generated-site verifier

**Files:**
- Create: `scripts/verify-build.js`
- Modify: `package.json`

**Interfaces:**
- Consumes: generated files under `public/`.
- Produces: exit code `0` when required pages and metadata are valid; non-zero with one line per violation.

- [x] **Step 1: Add the verifier script in failing-first form**

Create `scripts/verify-build.js` with these exact checks:

```js
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
  const relative = path.relative(root, file).replaceAll('\\\\', '/');
  const description = html.match(/<meta name="description" content="([^"]*)"/i)?.[1] ?? '';

  if (/\.mjx-container|MathJax_Display/.test(description)) {
    errors.push(`${relative}: meta description contains CSS`);
  }
  if (relative.endsWith('/README/index.html')) {
    errors.push(`${relative}: README was published as a post`);
  }
  if (/<article[\\s\\S]*?\\$\\$[\\s\\S]*?<\\/article>/i.test(html)) {
    errors.push(`${relative}: display-math delimiters leaked into article HTML`);
  }
}

if (errors.length) {
  for (const error of errors) console.error(`VERIFY: ${error}`);
  process.exit(1);
}

console.log(`VERIFY: ${htmlFiles.length} HTML files passed`);
```

- [x] **Step 2: Register the command**

Add to `package.json` scripts:

```json
"verify": "node scripts/verify-build.js",
"check": "npm run clean && npm run build && npm run verify"
```

- [x] **Step 3: Run it against the current broken output**

Run: `npm run build && npm run verify`

Expected: non-zero exit with at least `meta description contains CSS` and `README was published as a post`.

- [x] **Step 4: Commit the regression guard**

```bash
git add package.json scripts/verify-build.js
git commit -m "test: detect broken generated blog output"
```

### Task 2: Remove the conflicting MathJax initialization

**Files:**
- Modify: `source/_data/post-body-end.njk:1-11`
- Modify: `_config.yml:120-130`
- Modify: `_config.next.yml:150-158`
- Modify: `source/_data/styles.styl`

**Interfaces:**
- Consumes: posts marked `mathjax: true`.
- Produces: NexT-owned MathJax loading with no early replacement of `window.MathJax`.

- [x] **Step 1: Preserve a reproducible failing browser case**

Run:

```bash
npm run clean && npm run build && npm run server
```

Open `/2026/05/17/深度学习/流匹配与扩散模型/体系/1.%20从SDE开始/` and confirm the browser console contains `Cannot read properties of undefined (reading 'document')` and the page has zero `mjx-container` elements.

- [x] **Step 2: Remove the pre-configuration block from `post-body-end.njk`**

Delete the leading `<script>...</script>` block entirely. The file must begin with:

```njk
{% if page.comments !== false %}
<div class="post-comments">
```

- [x] **Step 3: Make NexT the only MathJax loader**

Keep this in `_config.next.yml`:

```yaml
math:
  every_page: false
  mathjax:
    enable: true
    tags: none
```

Remove the top-level `mathjax:` block from `_config.yml`. Keep Pandoc's pass-through configuration temporarily:

```yaml
pandoc:
  math: --mathjax
```

This separates Markdown parsing from browser-side rendering and avoids two runtime configurations.

- [x] **Step 4: Keep formula CSS centralized and remove unsafe inline forcing**

In `source/_data/styles.styl`, replace the inline-math override:

```stylus
.mjx-container[display="false"] {
  display: inline !important;
}
```

with MathJax 3 selectors that do not convert display equations into inline boxes:

```stylus
mjx-container[jax="CHTML"][display="true"] {
  display: block;
  max-width: 100%;
  overflow-x: auto;
  overflow-y: hidden;
  -webkit-overflow-scrolling: touch;
}

mjx-container[jax="CHTML"]:not([display="true"]) {
  display: inline;
}
```

Remove the mobile rule that forces every `mjx-container[jax="CHTML"]` to `display: inline`.

- [x] **Step 5: Generate and verify in a browser**

Run: `npm run clean && npm run build`

Expected: build exits `0`; Pandoc may still warn that it leaves TeX for MathJax, but the browser console must contain no MathJax error, `document.querySelectorAll('mjx-container').length` must be greater than `0`, and displayed equations must render rather than expose `$$`/LaTeX source.

Verify at desktop `1280×800` and mobile `390×844`; expected document width equals viewport width and long equations scroll inside their container.

- [x] **Step 6: Commit**

```bash
git add _config.yml _config.next.yml source/_data/post-body-end.njk source/_data/styles.styl
git commit -m "fix: restore NexT MathJax rendering"
```

### Task 3: Remove inline CSS from posts and repair descriptions

**Files:**
- Modify: all `source/_posts/**/*.md` files containing `<style>`.
- Modify: `scaffolds/post.md`.
- Modify: `scripts/verify-build.js`.

**Interfaces:**
- Consumes: global formula styles from `source/_data/styles.styl`.
- Produces: clean page descriptions and a reusable post template.

- [x] **Step 1: Inventory inline style blocks**

Run: `rg -l '<style>' source/_posts`

Expected: a finite list of posts to edit. Save the list in the commit description; do not run an unreviewed regex rewrite across mathematical content.

- [x] **Step 2: Delete only duplicated MathJax `<style>...</style>` blocks**

For each matched post, remove the style block while preserving frontmatter, prose, equations, and `<!-- more -->`.

- [x] **Step 3: Add explicit descriptions to key posts**

Use plain text, 60–160 Chinese characters, no Markdown or TeX. Example:

```yaml
description: 从随机微分方程出发，推导扩散过程的正向转移、反向时间动力学以及分数函数在逆过程中的作用。
```

Add descriptions first to all posts visible on homepage page 1, then apply the same rule to remaining posts in Phase 2.

- [x] **Step 4: Update the post scaffold**

Make `scaffolds/post.md` begin with:

```yaml
---
title: {{ title }}
date: {{ date }}
updated: {{ date }}
description:
tags: []
categories: []
mathjax: true
comments: true
---

<!-- more -->
```

- [x] **Step 5: Strengthen the verifier**

Add this generated-post check inside the verifier loop:

```js
if (relative.match(/^\\d{4}\\/\\d{2}\\/\\d{2}\\//) && !description.trim()) {
  errors.push(`${relative}: post has no meta description`);
}
```

- [x] **Step 6: Run the check**

Run: `npm run check`

Expected: no description contains CSS, and every generated post has a non-empty description.

- [x] **Step 7: Commit**

```bash
git add scaffolds/post.md scripts/verify-build.js source/_posts
git commit -m "fix: generate clean post descriptions"
```

### Task 4: Stop publishing the series README as an article

**Files:**
- Move: `source/_posts/深度学习/流匹配与扩散模型/体系/README.md` to `docs/content/flow-matching-series.md`
- Modify: `scripts/verify-build.js`

**Interfaces:**
- Consumes: internal series notes.
- Produces: 22 intentional posts and no `/README/` permalink.

- [x] **Step 1: Confirm README is not linked as a public article**

Run: `rg -n '体系/README|README/' source _config*.yml README.md`

Expected: no intentional public link. If a public link exists, replace it with the first article or a deliberate series-index page before moving the file.

- [x] **Step 2: Move the document without rewriting its content**

Create `docs/content/` and move the file to `docs/content/flow-matching-series.md` using a normal filesystem move so Git records a rename.

- [x] **Step 3: Assert the expected post count**

Add to `scripts/verify-build.js` after walking files:

```js
const postPages = htmlFiles.filter(file =>
  path.relative(root, file).replaceAll('\\\\', '/').match(/^\\d{4}\\/\\d{2}\\/\\d{2}\\//)
);
if (postPages.length !== 22) {
  errors.push(`expected 22 generated posts, found ${postPages.length}`);
}
```

- [x] **Step 4: Verify**

Run: `npm run check`

Expected: verifier reports 22 post pages and no README page.

- [x] **Step 5: Commit**

```bash
git add docs/content scripts/verify-build.js source/_posts
git commit -m "fix: exclude internal series README from posts"
```

---

## Phase 2 — Improve Content Discovery and Metadata

### Task 5: Validate and normalize frontmatter

**Files:**
- Create: `scripts/audit-frontmatter.js`
- Modify: `package.json`
- Modify: affected `source/_posts/**/*.md`.

**Interfaces:**
- Consumes: YAML frontmatter from every Markdown post.
- Produces: errors for missing title/date/description/categories/tags, combined comma-like tag strings, and future dates.

- [x] **Step 1: Add a dependency for safe YAML parsing**

Run: `npm install --save-dev gray-matter`

Expected: `package.json` and `package-lock.json` record `gray-matter` under development dependencies.

- [x] **Step 2: Create `scripts/audit-frontmatter.js`**

```js
const fs = require('node:fs');
const path = require('node:path');
const matter = require('gray-matter');

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
```

- [x] **Step 3: Register it**

Add to `package.json`:

```json
"audit:frontmatter": "node scripts/audit-frontmatter.js",
"check": "npm run audit:frontmatter && npm run clean && npm run build && npm run verify"
```

- [x] **Step 4: Run it and correct every reported post manually**

Run: `npm run audit:frontmatter`

Expected initially: failures including the combined `深度学习，线性回归` tag and posts without descriptions. Normalize tags as arrays:

```yaml
tags:
  - 深度学习
  - 线性回归
```

Do not infer a new publication date. Preserve existing dates unless the post currently has none; for the one missing date, recover the intended date from Git history before adding it.

- [x] **Step 5: Verify and commit**

Run: `npm run check`

Expected: frontmatter, generation, and generated output all pass.

```bash
git add package.json package-lock.json scripts/audit-frontmatter.js source/_posts
git commit -m "chore: validate and normalize post metadata"
```

### Task 6: Make the homepage useful for scanning

**Files:**
- Modify: `_config.yml:73-76`
- Modify: `source/_posts/**/*.md` around `<!-- more -->`.

**Interfaces:**
- Consumes: explicit descriptions and curated pre-`<!-- more -->` introductions.
- Produces: six to eight compact, informative article cards per homepage page.

- [x] **Step 1: Establish the visual baseline**

Capture the homepage at desktop `1280×800` and mobile `390×844`. Record the visible title count and whether each card contains explanatory prose.

- [x] **Step 2: Reduce page length**

Change `_config.yml`:

```yaml
index_generator:
  path: ''
  per_page: 8
  order_by: -date
```

Also set the global `per_page: 8` to keep pagination behavior consistent.

- [x] **Step 3: Add a meaningful lead paragraph before every `<!-- more -->`**

For each post, the lead must state its question, method, or result in 1–3 sentences. Do not copy the title, formula source, or CSS. Keep the excerpt under roughly 220 Chinese characters.

- [x] **Step 4: Verify visual behavior**

Run: `npm run check && npm run server`

Expected desktop and mobile homepage cards show a short summary; page 1 contains 8 articles; no card is reduced to only title/date/read-more; mobile has no horizontal overflow.

- [x] **Step 5: Commit**

```bash
git add _config.yml source/_posts
git commit -m "content: add useful homepage excerpts"
```

### Task 7: Make update and publication policy explicit

**Files:**
- Modify: `_config.yml:63,101`
- Modify: `scaffolds/post.md`
- Modify: affected `source/_posts/**/*.md`.

**Interfaces:**
- Consumes: explicit `date` and `updated` frontmatter.
- Produces: deterministic dates across machines and no accidental future publication.

- [x] **Step 1: Change Hexo policy**

Set:

```yaml
future: false
updated_option: empty
```

- [x] **Step 2: Add `updated` only where history supports it**

Use `git log --follow --format=%aI -- <post>` to determine the last substantive content change. Do not set every article's update time to the remediation date merely because metadata was normalized.

- [x] **Step 3: Verify dates remain stable**

Run `npm run check` twice without file edits and compare generated post metadata. Expected: published/updated dates are identical between runs.

- [x] **Step 4: Commit**

```bash
git add _config.yml scaffolds/post.md source/_posts
git commit -m "fix: make post dates deterministic"
```

---

## Phase 3 — Reproducibility and Repository Hygiene

### Task 8: Make CI deterministic and enforce verification

**Files:**
- Modify: `.github/workflows/deploy.yml:27-48`
- Modify: `package.json` and `package-lock.json` only if dependency ownership changes.

**Interfaces:**
- Consumes: committed lockfile and `npm run check`.
- Produces: Pages artifact only after all validation passes.

- [x] **Step 1: Replace mutable dependency installation**

Replace the install and generate steps with:

```yaml
- name: Install Pandoc
  run: sudo apt-get update && sudo apt-get install -y pandoc

- name: Install dependencies
  run: npm ci

- name: Validate and generate
  run: npm run check

- name: Prepare Pages artifact
  run: touch public/.nojekyll
```

Delete the CI-time `npm install ... --save` command.

- [x] **Step 2: Run the same clean install locally in a disposable environment**

Do not delete the user's working `node_modules`. Use CI or a temporary clone/worktree, then run:

```bash
npm ci
npm run check
```

Expected: clean installation succeeds using only `package-lock.json`, then all checks pass.

- [x] **Step 3: Commit**

```bash
git add .github/workflows/deploy.yml package.json package-lock.json
git commit -m "ci: make Hexo deployment reproducible"
```

### Task 9: Resolve misleading traffic counters

**Files:**
- Modify: `_config.next.yml:225-230`
- Optionally modify: `README.md` if the counter is intentionally disabled.

**Interfaces:**
- Consumes: observed production counter values on `https://entropyxi.github.io`.
- Produces: credible statistics or no statistics.

- [x] **Step 1: Check production rather than localhost**

Open the production homepage and one post. Record the displayed site UV, site PV, and page PV. Do not treat localhost values as production evidence.

- [x] **Step 2: Choose based on evidence**

If production values are credible, keep the current block. If they are obviously inflated or unstable, set:

```yaml
busuanzi_count:
  enable: false
  site_uv: false
  site_pv: false
  page_pv: false
```

Do not replace it with another analytics vendor in this remediation; that is a separate privacy and product decision.

- [x] **Step 3: Verify and commit**

Run: `npm run check`

Expected: generated footer and post metadata match the chosen policy.

```bash
git add _config.next.yml README.md
git commit -m "fix: show only credible traffic statistics"
```

### Task 10: Remove unused theme copies safely

**Files:**
- Delete after explicit review: tracked `themes/anzhiyu/`
- Delete or ignore after explicit review: untracked `themes/themes/`, `themes/landscape/`
- Modify: `.gitignore`
- Modify: `README.md`

**Interfaces:**
- Consumes: confirmed active theme `next` from npm.
- Produces: a repository with no misleading inactive theme source.

- [x] **Step 1: Prove no active configuration references AnZhiYu**

Run:

```bash
rg -n "anzhiyu|themes/themes|themes/landscape" _config*.yml package.json source scripts .github README.md
npm run check
```

Expected: no runtime reference to AnZhiYu; `theme: next`; build passes before deletion.

- [x] **Step 2: Review user-owned differences before deletion**

Run:

```bash
git status --short themes
git log --oneline -- themes/anzhiyu
```

If the theme contains intentional custom work that must be archived, preserve it in a separate branch or repository before deletion. Do not silently discard it.

- [x] **Step 3: Remove only confirmed inactive directories**

Delete `themes/anzhiyu`, `themes/themes`, and `themes/landscape` only after Step 2 passes and the user authorizes removal of untracked copies. Preserve `themes/.gitkeep` if the directory is otherwise empty.

- [x] **Step 4: Prevent recurrence**

Add to `.gitignore`:

```gitignore
themes/*
!themes/.gitkeep
```

Document in `README.md` that NexT is installed through npm and customized only through `_config.next.yml` plus `source/_data/`.

- [x] **Step 5: Run full regression after deletion**

Run: `npm ci` in a disposable clean environment, then `npm run check`.

Expected: output is identical in structure, 22 posts are generated, formulas work in the browser, and no file is sourced from deleted theme directories.

- [x] **Step 6: Commit theme cleanup separately**

```bash
git add .gitignore README.md themes
git commit -m "chore: remove inactive theme copies"
```

### Task 11: Final end-to-end acceptance

**Files:**
- Modify only files needed to correct failures found here.

**Interfaces:**
- Consumes: all prior tasks.
- Produces: release-ready source branch without deploying it.

- [x] **Step 1: Run automated acceptance**

Run:

```bash
npm ci
npm run check
git status --short
```

Expected: installation and checks pass; status contains only intentional plan implementation changes.

- [x] **Step 2: Test the primary reader flow on desktop**

At `1280×800`, verify homepage → article → table of contents → search → categories. Expected: links work, summaries are visible, MathJax renders, search returns the article, and no console errors occur.

- [x] **Step 3: Test the primary reader flow on mobile**

At `390×844`, verify homepage, menu, search, article headings, long equations, code blocks, and Giscus container. Expected: no document-level horizontal overflow, controls are visible, equations scroll locally when needed, and content appears after the NexT entrance animation.

- [x] **Step 4: Inspect metadata**

For homepage and two representative posts, inspect `title`, `meta description`, canonical URL, `lang`, RSS link, and sitemap entry. Expected: descriptions are human prose and contain no CSS or raw TeX.

- [x] **Step 5: Record the result**

Update `README.md` with:

```markdown
## Local validation

```bash
npm ci
npm run check
npm run server
```

`npm run check` validates frontmatter, generates the site, and inspects generated HTML for known regressions.
```

- [x] **Step 6: Commit**

```bash
git add README.md
git commit -m "docs: document blog validation workflow"
```

## Release Gate

Do not deploy unless all conditions are true:

- `npm ci` succeeds from the committed lockfile.
- `npm run check` exits `0`.
- Exactly 22 intentional post pages are generated.
- Representative math-heavy posts contain rendered MathJax and no raw display delimiters.
- Browser console has no MathJax or PJAX errors on direct load and PJAX navigation.
- Homepage cards contain useful summaries on desktop and mobile.
- Article descriptions contain human prose, not CSS/TeX.
- Production traffic counters are either credible or disabled.
- Theme cleanup is a separate reviewed commit.

## Recommended Rollout

1. Merge Phase 1 and verify the deployed preview first; it fixes reader-visible correctness.
2. Merge Phase 2 after editorial review of descriptions, dates, and excerpts.
3. Merge Phase 3 last; theme deletion and CI changes should be easy to revert independently.
4. After production deployment, smoke-test one math-heavy post through both a direct URL and an internal PJAX navigation.
