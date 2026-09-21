// Aliased to "canvas" in browser builds (see next.config.ts). Node-only
// dependencies of the PDF viewer reach for it; an empty object satisfies them.
const emptyModule = {};
export default emptyModule;
