import { execSync, spawn } from 'child_process';
import fs from 'fs';

async function isServerReady(url) {
  try {
    const res = await fetch(url);
    return res.ok;
  } catch {
    return false;
  }
}

async function run() {
  const urls = [
    { name: 'home', path: '/' },
    { name: 'search', path: '/search/' },
    { name: 'article-normal', path: '/2026/08/05/技术随笔/WSL2安装失败排查：从DISM_COM损坏到NetCfg残留/' },
    { name: 'article-ddim', path: '/2026/05/10/深度学习/流匹配与扩散模型/DDIM/' }
  ];

  const baseUrl = 'http://127.0.0.1:4321';
  if (!fs.existsSync('audit-screenshots/phase-8/lighthouse')) {
    fs.mkdirSync('audit-screenshots/phase-8/lighthouse', { recursive: true });
  }

  let server;
  if (!(await isServerReady(baseUrl))) {
    console.log('Starting preview server...');
    server = spawn('npm', ['run', 'preview:test'], { stdio: 'ignore', shell: true });
    for (let i = 0; i < 30; i++) {
      await new Promise(r => setTimeout(r, 1000));
      if (await isServerReady(baseUrl)) break;
    }
  }

  for (const u of urls) {
    console.log('Running Lighthouse for ' + u.name + '...');
    try {
      execSync('npx lighthouse "' + baseUrl + encodeURI(u.path) + '" --output=json --output-path=audit-screenshots/phase-8/lighthouse/' + u.name + '.json --chrome-flags="--headless" --throttling-method=simulate --form-factor=mobile --screenEmulation.disabled=false --screenEmulation.width=390 --screenEmulation.height=844 --screenEmulation.deviceScaleFactor=1', { stdio: 'inherit', shell: true });
    } catch (err) {
      console.error('Failed to run Lighthouse for ' + u.name, err);
    }
  }

  if (server) server.kill();
}
run();
