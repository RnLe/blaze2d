'use client';
"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.default = SpeedupScalingChart;
const jsx_runtime_1 = require("react/jsx-runtime");
const react_1 = require("react");
const LineChart_1 = __importDefault(require("./LineChart"));
const use_benchmarks_1 = require("../../lib/use-benchmarks");
// Colors matching the bar chart style
const COLORS = {
    Blaze: '#eaf1fe', // Light blue-white (same as bar charts)
    MPB: '#5477c4', // Blue-gray (MPB OMP - same as bar charts)
    MPB_Multiproc: '#a3befa', // Light blue for MPB Multiprocessor
    Ideal: '#666666', // Gray for ideal line
};
function SpeedupScalingChart({ width = 650, height = 380, }) {
    const { data: benchmarkData, loading } = (0, use_benchmarks_1.useSeries7Benchmarks)();
    // Transform data for line chart - compute speedup relative to 1 thread
    const { lowResSeries, highResSeries } = (0, react_1.useMemo)(() => {
        const computeSpeedup = (data) => {
            const baseThroughput = data[0]?.mean_throughput || 1;
            return data.map(d => ({
                x: d.threads,
                y: d.mean_throughput / baseThroughput,
                // Propagate std error for speedup (simplified approximation)
                std: (d.std_throughput / baseThroughput),
            }));
        };
        const maxThreads = Math.max(...benchmarkData.low.blaze.map(d => d.threads), ...benchmarkData.high.blaze.map(d => d.threads));
        // Ideal linear speedup line
        const idealData = [
            { x: 1, y: 1 },
            { x: maxThreads, y: maxThreads },
        ];
        const lowRes = [
            {
                id: 'blaze-low',
                label: 'Blaze',
                color: COLORS.Blaze,
                marker: 'circle',
                data: computeSpeedup(benchmarkData.low.blaze),
            },
            {
                id: 'mpb-omp-low',
                label: 'MPB (OpenMP)',
                color: COLORS.MPB,
                marker: 'square',
                data: computeSpeedup(benchmarkData.low.mpb_omp),
            },
            {
                id: 'mpb-multiproc-low',
                label: 'MPB (Multiprocessor)',
                color: COLORS.MPB_Multiproc,
                marker: 'triangle',
                data: computeSpeedup(benchmarkData.low.mpb_multiproc),
            },
            {
                id: 'ideal-low',
                label: 'Ideal (linear)',
                color: COLORS.Ideal,
                isReference: true,
                data: idealData,
            },
        ];
        const highRes = [
            {
                id: 'blaze-high',
                label: 'Blaze',
                color: COLORS.Blaze,
                marker: 'circle',
                data: computeSpeedup(benchmarkData.high.blaze),
            },
            {
                id: 'mpb-omp-high',
                label: 'MPB (OpenMP)',
                color: COLORS.MPB,
                marker: 'square',
                data: computeSpeedup(benchmarkData.high.mpb_omp),
            },
            {
                id: 'mpb-multiproc-high',
                label: 'MPB (Multiprocessor)',
                color: COLORS.MPB_Multiproc,
                marker: 'triangle',
                data: computeSpeedup(benchmarkData.high.mpb_multiproc),
            },
            {
                id: 'ideal-high',
                label: 'Ideal (linear)',
                color: COLORS.Ideal,
                isReference: true,
                data: idealData,
            },
        ];
        return { lowResSeries: lowRes, highResSeries: highRes };
    }, [benchmarkData]);
    if (loading) {
        return ((0, jsx_runtime_1.jsx)("div", { style: { width: width * 2 + 40, height, display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#888' }, children: "Loading benchmark data..." }));
    }
    const chartWidth = (width - 20) / 2;
    const lowResolution = benchmarkData.low.resolution;
    const highResolution = benchmarkData.high.resolution;
    return ((0, jsx_runtime_1.jsxs)("div", { style: { display: 'flex', flexDirection: 'column', gap: '0.5rem' }, children: [(0, jsx_runtime_1.jsxs)("div", { style: { display: 'flex', gap: '20px', justifyContent: 'center' }, children: [(0, jsx_runtime_1.jsx)(LineChart_1.default, { series: lowResSeries, width: chartWidth, height: height, title: `Speedup (${lowResolution}×${lowResolution})`, xLabel: "Thread Count", yLabel: "Speedup (\u00D7)", xTickFormat: (v) => `${v}`, yTickFormat: (v) => `${v.toFixed(0)}`, showErrorBars: false, xDomain: [0, 17], yDomain: [0, 17], margin: { top: 50, right: 20, bottom: 60, left: 70 } }), (0, jsx_runtime_1.jsx)(LineChart_1.default, { series: highResSeries, width: chartWidth, height: height, title: `Speedup (${highResolution}×${highResolution})`, xLabel: "Thread Count", yLabel: "Speedup (\u00D7)", xTickFormat: (v) => `${v}`, yTickFormat: (v) => `${v.toFixed(0)}`, showErrorBars: false, xDomain: [0, 17], yDomain: [0, 17], margin: { top: 50, right: 20, bottom: 60, left: 70 } })] }), (0, jsx_runtime_1.jsx)("p", { style: {
                    marginTop: '1rem',
                    fontSize: '0.875rem',
                    color: '#888',
                    lineHeight: 1.5,
                    fontFamily: 'var(--font-sans), system-ui, sans-serif',
                    fontStyle: 'italic',
                    textAlign: 'left',
                }, children: "Speedup relative to single-threaded performance. The dashed line shows ideal linear scaling. Blaze achieves near-linear speedup at low resolution, while MPB's OpenMP parallelization shows essentially no scaling benefit." })] }));
}
