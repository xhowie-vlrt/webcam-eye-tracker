// Offline support. The app shell is cached on install; the MediaPipe wasm and
// model (several MB, fetched from a CDN) are cached on first use so a second
// visit works without a network.

const SHELL_CACHE = 'eyetracker-shell-v1';
const CDN_CACHE = 'eyetracker-cdn-v1';

const SHELL = [
  './',
  'index.html',
  'icon.svg',
  'manifest.webmanifest',
  'css/style.css',
  'js/app.js',
  'js/eyetracker.js',
  'js/tracker.js',
  'js/features.js',
  'js/math.js',
  'js/gaze.js',
  'js/filter.js',
  'js/fixation.js',
  'js/points.js',
  'js/calibration-ui.js',
  'js/heatmap.js',
  'js/scanpath.js',
];

const CDN_HOSTS = ['cdn.jsdelivr.net', 'storage.googleapis.com'];

self.addEventListener('install', (event) => {
  event.waitUntil(
    caches
      .open(SHELL_CACHE)
      // addAll is all-or-nothing; one 404 would leave the app uncached.
      .then((cache) => Promise.allSettled(SHELL.map((url) => cache.add(url))))
      .then(() => self.skipWaiting())
  );
});

self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches
      .keys()
      .then((keys) =>
        Promise.all(
          keys
            .filter((k) => k !== SHELL_CACHE && k !== CDN_CACHE)
            .map((k) => caches.delete(k))
        )
      )
      .then(() => self.clients.claim())
  );
});

self.addEventListener('fetch', (event) => {
  const { request } = event;
  if (request.method !== 'GET') return;
  const url = new URL(request.url);

  if (CDN_HOSTS.includes(url.hostname)) {
    // Immutable, versioned assets: cache first, and keep what we already have
    // if the network is gone.
    event.respondWith(
      caches.open(CDN_CACHE).then(async (cache) => {
        const hit = await cache.match(request);
        if (hit) return hit;
        const response = await fetch(request);
        if (response.ok || response.type === 'opaque') {
          cache.put(request, response.clone());
        }
        return response;
      })
    );
    return;
  }

  if (url.origin !== self.location.origin) return;

  // Same-origin app files: network first so a deploy is picked up immediately,
  // falling back to the cached shell when offline.
  event.respondWith(
    fetch(request)
      .then((response) => {
        const copy = response.clone();
        caches.open(SHELL_CACHE).then((cache) => cache.put(request, copy));
        return response;
      })
      .catch(async () => {
        const hit = await caches.match(request);
        if (hit) return hit;
        if (request.mode === 'navigate') return caches.match('index.html');
        throw new Error('offline and not cached');
      })
  );
});
