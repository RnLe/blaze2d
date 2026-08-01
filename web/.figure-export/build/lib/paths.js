"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.getAssetPath = getAssetPath;
// Shim for web/lib/paths.ts.
//
// On the site getAssetPath() prepends the GitHub-Pages base path. The exporter
// reads the same files straight off disk, so the identity function is correct
// here and keeps the returned string usable as a key into the JSON loader.
function getAssetPath(path) {
    return path;
}
