/**
 * MineGuard — shared runtime configuration.
 * Included by every page (home, rul, about) so the API base URL is resolved
 * in exactly one place instead of being hardcoded per-file.
 *
 * Resolution order:
 *   1. <meta name="api-base-url" content="..."> in the page <head>, if present
 *      and non-empty — the easiest override for a one-off deploy, no JS edits needed.
 *   2. If running on localhost/127.0.0.1, use the local dev backend.
 *   3. Otherwise fall back to PRODUCTION_API_URL below — set this once you've
 *      deployed the backend (e.g. to Render, Railway, Fly.io).
 */
(function () {
  // --- EDIT THIS after deploying the backend ---
  const PRODUCTION_API_URL = 'https://mineguard-api.onrender.com';

  const metaTag = document.querySelector('meta[name="api-base-url"]');
  const metaValue = metaTag && metaTag.content && metaTag.content.trim();

  const isLocalhost = ['localhost', '127.0.0.1', ''].includes(window.location.hostname);
  const LOCAL_API_URL = 'http://localhost:5000';

  window.MINEGUARD_API_URL = metaValue || (isLocalhost ? LOCAL_API_URL : PRODUCTION_API_URL);
})();
