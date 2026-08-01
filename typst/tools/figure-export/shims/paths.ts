// Shim for web/lib/paths.ts.
//
// On the site getAssetPath() prepends the GitHub-Pages base path. The exporter
// reads the same files straight off disk, so the identity function is correct
// here and keeps the returned string usable as a key into the JSON loader.
export function getAssetPath(path: string): string {
  return path;
}
