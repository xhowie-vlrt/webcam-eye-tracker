#!/usr/bin/env node
// Download the MediaPipe runtime and the face-landmarker model into ./vendor,
// then point the pages at it by adding <meta name="eyetracker-assets">.
//
// After this the app makes zero third-party requests: it works offline, behind
// a strict CSP, and cannot be broken by a CDN outage. That matters more for a
// distributed build than the ~17 MB it costs.
//
//   node tools/fetch-assets.mjs           download + wire up
//   node tools/fetch-assets.mjs --revert  remove ./vendor and go back to the CDN

import { execFileSync } from 'node:child_process';
import { mkdtempSync, rmSync, mkdirSync, cpSync, existsSync, statSync, readFileSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join, dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const VENDOR = join(ROOT, 'vendor');
const PACKAGE = '@mediapipe/tasks-vision@0.10.18';
const MODEL_URL =
  'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task';
const PAGES = ['index.html', join('examples', 'embed.html')];
const META = '<meta name="eyetracker-assets" content="vendor">';

if (process.argv.includes('--revert')) {
  rmSync(VENDOR, { recursive: true, force: true });
  for (const page of PAGES) setMeta(join(ROOT, page), false);
  console.log('vendor/ removed; the app will use the CDN again.');
  process.exit(0);
}

const tmp = mkdtempSync(join(tmpdir(), 'eyetracker-assets-'));
try {
  console.log(`fetching ${PACKAGE} ...`);
  const packed = execFileSync('npm', ['pack', PACKAGE, '--silent'], {
    cwd: tmp,
    encoding: 'utf8',
  })
    .trim()
    .split('\n')
    .pop();
  execFileSync('tar', ['xzf', packed], { cwd: tmp });

  const from = join(tmp, 'package');
  const to = join(VENDOR, 'tasks-vision');
  mkdirSync(to, { recursive: true });
  cpSync(join(from, 'vision_bundle.mjs'), join(to, 'vision_bundle.mjs'));
  cpSync(join(from, 'wasm'), join(to, 'wasm'), { recursive: true });

  console.log('fetching the face landmarker model ...');
  const models = join(VENDOR, 'models');
  mkdirSync(models, { recursive: true });
  const res = await fetch(MODEL_URL);
  if (!res.ok) throw new Error(`model download failed: HTTP ${res.status}`);
  writeFileSync(join(models, 'face_landmarker.task'), Buffer.from(await res.arrayBuffer()));

  for (const page of PAGES) setMeta(join(ROOT, page), true);

  report(to, models);
} finally {
  rmSync(tmp, { recursive: true, force: true });
}

/** Add or remove the meta tag that switches js/config.js over to ./vendor. */
function setMeta(file, enable) {
  if (!existsSync(file)) return;
  let html = readFileSync(file, 'utf8');
  const has = html.includes('name="eyetracker-assets"');
  if (enable && !has) {
    // Embedded pages live one directory down, so the base is relative to them.
    const depth = file.includes(`${join('examples', '')}`) ? '../vendor' : 'vendor';
    html = html.replace('</head>', `${META.replace('"vendor"', `"${depth}"`)}\n</head>`);
  } else if (!enable && has) {
    html = html.replace(/\n?[ \t]*<meta name="eyetracker-assets"[^>]*>/g, '');
  } else {
    return;
  }
  writeFileSync(file, html);
  console.log(`${enable ? 'wired up' : 'unwired'} ${file.replace(`${ROOT}/`, '')}`);
}

function report(...dirs) {
  const mb = (bytes) => `${(bytes / 1024 / 1024).toFixed(1)} MB`;
  let total = 0;
  const walk = (p) => {
    const st = statSync(p);
    if (st.isDirectory()) {
      for (const child of readdir(p)) walk(join(p, child));
    } else {
      total += st.size;
    }
  };
  for (const d of dirs) walk(d);
  console.log(`\nvendor/ ready (${mb(total)}). The app now runs without any CDN.`);
  console.log('Revert with: npm run setup:revert');
}

function readdir(p) {
  return execFileSync('ls', ['-A', p], { encoding: 'utf8' }).split('\n').filter(Boolean);
}
