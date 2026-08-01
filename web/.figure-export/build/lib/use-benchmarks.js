"use strict";
// Synchronous replacement for web/lib/use-benchmarks.ts.
//
// The real hooks start from the embedded FALLBACK_* constants and swap in the
// hoisted JSON from web/public/data/ once a useEffect-driven fetch resolves.
// Under react-dom/server that effect never fires, so rendering the untouched
// hooks would silently export the fallback numbers instead of the published
// ones. Each hook here reads the same file the browser would fetch and reports
// the settled state directly, which is why the exported figures carry the same
// values as the live site.
//
// Signatures mirror the originals exactly ({ data, loading, error, source }) so
// the chart components are used unmodified.
Object.defineProperty(exports, "__esModule", { value: true });
exports.useSingleCoreBenchmarks = useSingleCoreBenchmarks;
exports.useMultiCoreBenchmarks = useMultiCoreBenchmarks;
exports.useSeries7Benchmarks = useSeries7Benchmarks;
exports.useSeries5Benchmarks = useSeries5Benchmarks;
exports.useSeries3Benchmarks = useSeries3Benchmarks;
exports.useSeries4Benchmarks = useSeries4Benchmarks;
exports.useSeries1Benchmarks = useSeries1Benchmarks;
exports.useSeries2Benchmarks = useSeries2Benchmarks;
const data_source_1 = require("./data-source");
function settled(assetPath) {
    return {
        data: (0, data_source_1.loadJson)(assetPath),
        loading: false,
        error: null,
        source: 'static',
    };
}
function useSingleCoreBenchmarks() {
    return settled('/data/benchmarks/single-core.json');
}
function useMultiCoreBenchmarks() {
    return settled('/data/benchmarks/multi-core.json');
}
function useSeries7Benchmarks() {
    return settled('/data/benchmarks/series7-scaling.json');
}
function useSeries5Benchmarks() {
    return settled('/data/benchmarks/series5-memory.json');
}
function useSeries3Benchmarks() {
    return settled('/data/benchmarks/series3-resolution.json');
}
function useSeries4Benchmarks() {
    return settled('/data/benchmarks/series4-iterations.json');
}
function useSeries1Benchmarks() {
    return settled('/data/benchmarks/series1-epsilon.json');
}
function useSeries2Benchmarks() {
    return settled('/data/benchmarks/series2-bands.json');
}
