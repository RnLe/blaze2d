/**
 * Resolves a static asset path, handling the base path for GitHub Pages deployment.
 * 
 * @param path The absolute path to the asset (e.g., '/data/file.json')
 * @returns The full path with base path prepended (e.g., '/blaze2d/data/file.json')
 */
export function getAssetPath(path: string): string {
  const basePath = process.env.NEXT_PUBLIC_BASE_PATH || '';
  if (!basePath) return path;

  // If path already starts with basePath (e.g. /blaze2d/...), return it
  if (path.startsWith(basePath)) return path;

  // Ensure path starts with /
  const cleanPath = path.startsWith('/') ? path : `/${path}`;
  return `${basePath}${cleanPath}`;
}
