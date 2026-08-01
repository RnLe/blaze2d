'use client';
"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.default = IterationDistributionChart;
const jsx_runtime_1 = require("react/jsx-runtime");
const react_1 = require("react");
const group_1 = require("@visx/group");
const shape_1 = require("@visx/shape");
const scale_1 = require("@visx/scale");
const axis_1 = require("@visx/axis");
const grid_1 = require("@visx/grid");
const text_1 = require("@visx/text");
const use_benchmarks_1 = require("../../lib/use-benchmarks");
const BarChart_1 = require("./BarChart");
// Colors matching other charts
const COLORS = {
    mpb: '#5477c4', // Blue-gray
    blaze: '#4caf50', // Green for better contrast when overlapping
};
function IterationDistributionChart({ width = 520, // Reduced from 650 (~20% narrower)
height = 380, }) {
    const { data: benchmarkData, loading } = (0, use_benchmarks_1.useSeries4Benchmarks)();
    // Compute histogram data with bin width of 2 iterations
    const { tmHistogram, teHistogram, binCenters } = (0, react_1.useMemo)(() => {
        const binWidth = 2; // Fixed bin width of 2 iterations
        // Get all iteration values to determine range
        const allIters = [
            ...(benchmarkData.TM.mpb?.k_points.map(kp => kp.iterations) || []),
            ...(benchmarkData.TM.blaze?.k_points.map(kp => kp.iterations) || []),
            ...(benchmarkData.TE.mpb?.k_points.map(kp => kp.iterations) || []),
            ...(benchmarkData.TE.blaze?.k_points.map(kp => kp.iterations) || []),
        ];
        const minIter = Math.min(...allIters);
        const maxIter = Math.max(...allIters);
        // Align to even numbers for cleaner bins
        const alignedMin = Math.floor(minIter / binWidth) * binWidth;
        const alignedMax = Math.ceil((maxIter + 1) / binWidth) * binWidth;
        const numBins = Math.ceil((alignedMax - alignedMin) / binWidth);
        // Create bin centers for labeling
        const centers = [];
        for (let i = 0; i < numBins; i++) {
            centers.push(alignedMin + i * binWidth + binWidth / 2);
        }
        // Helper to compute histogram
        const computeHist = (iterations) => {
            const counts = new Array(numBins).fill(0);
            for (const iter of iterations) {
                const binIdx = Math.min(Math.floor((iter - alignedMin) / binWidth), numBins - 1);
                if (binIdx >= 0)
                    counts[binIdx]++;
            }
            return counts;
        };
        const tmHist = {
            mpb: computeHist(benchmarkData.TM.mpb?.k_points.map(kp => kp.iterations) || []),
            blaze: computeHist(benchmarkData.TM.blaze?.k_points.map(kp => kp.iterations) || []),
        };
        const teHist = {
            mpb: computeHist(benchmarkData.TE.mpb?.k_points.map(kp => kp.iterations) || []),
            blaze: computeHist(benchmarkData.TE.blaze?.k_points.map(kp => kp.iterations) || []),
        };
        return { tmHistogram: tmHist, teHistogram: teHist, binCenters: centers };
    }, [benchmarkData]);
    if (loading) {
        return ((0, jsx_runtime_1.jsx)("div", { style: { width: width * 2 + 40, height, display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#888' }, children: "Loading benchmark data..." }));
    }
    const chartWidth = (width - 20) / 2;
    const margin = { top: 65, right: 20, bottom: 60, left: 70 };
    const innerWidth = chartWidth - margin.left - margin.right;
    const innerHeight = height - margin.top - margin.bottom;
    const renderChart = (histogram, polarization) => {
        const maxCount = Math.max(...histogram.mpb, ...histogram.blaze);
        const binIndices = histogram.mpb.map((_, i) => i);
        const xScale = (0, scale_1.scaleBand)({
            domain: binIndices,
            range: [0, innerWidth],
            padding: 0.1,
        });
        const yScale = (0, scale_1.scaleLinear)({
            domain: [0, maxCount * 1.1],
            range: [innerHeight, 0],
        });
        // Full bandwidth for overlapping bars
        const barWidth = xScale.bandwidth();
        // Create bin labels (show center value)
        const getBinLabel = (idx) => {
            const center = binCenters[idx];
            return `${Math.floor(center)}`;
        };
        // Show every Nth tick for readability
        const tickStep = binIndices.length > 20 ? 4 : binIndices.length > 10 ? 2 : 1;
        const xTickValues = binIndices.filter((_, i) => i % tickStep === 0);
        return ((0, jsx_runtime_1.jsxs)("svg", { width: chartWidth, height: height, style: { overflow: 'visible' }, children: [(0, jsx_runtime_1.jsx)(text_1.Text, { x: 0, y: 16, fontSize: 14, fontFamily: BarChart_1.CHART_STYLES.fontFamily, fill: BarChart_1.CHART_STYLES.labelColor, fontWeight: 700, children: `Iteration Distribution (${polarization})` }), (0, jsx_runtime_1.jsxs)(group_1.Group, { left: margin.left, top: margin.top, children: [(0, jsx_runtime_1.jsx)(grid_1.GridRows, { scale: yScale, width: innerWidth, stroke: BarChart_1.CHART_STYLES.gridColor, strokeOpacity: 0.4, strokeDasharray: "3,3" }), binIndices.map((binIdx) => {
                            const xPos = xScale(binIdx) || 0;
                            const mpbCount = histogram.mpb[binIdx];
                            const blazeCount = histogram.blaze[binIdx];
                            return ((0, jsx_runtime_1.jsxs)("g", { children: [mpbCount > 0 && ((0, jsx_runtime_1.jsx)(shape_1.Bar, { x: xPos, y: yScale(mpbCount), width: barWidth, height: innerHeight - yScale(mpbCount), fill: COLORS.mpb, opacity: 0.6 })), blazeCount > 0 && ((0, jsx_runtime_1.jsx)(shape_1.Bar, { x: xPos, y: yScale(blazeCount), width: barWidth, height: innerHeight - yScale(blazeCount), fill: COLORS.blaze, opacity: 0.5 }))] }, binIdx));
                        }), (0, jsx_runtime_1.jsx)(axis_1.AxisLeft, { scale: yScale, stroke: BarChart_1.CHART_STYLES.axisColor, tickStroke: BarChart_1.CHART_STYLES.axisColor, tickFormat: (v) => `${v}`, tickLabelProps: () => ({
                                fill: BarChart_1.CHART_STYLES.labelColor,
                                fontSize: 11,
                                fontFamily: BarChart_1.CHART_STYLES.fontFamily,
                                textAnchor: 'end',
                                dx: -4,
                                dy: 4,
                            }), numTicks: 5 }), (0, jsx_runtime_1.jsx)(axis_1.AxisBottom, { scale: xScale, top: innerHeight, stroke: BarChart_1.CHART_STYLES.axisColor, tickStroke: BarChart_1.CHART_STYLES.axisColor, tickValues: xTickValues, tickFormat: (v) => getBinLabel(v), tickLabelProps: () => ({
                                fill: BarChart_1.CHART_STYLES.labelColor,
                                fontSize: 9,
                                fontFamily: BarChart_1.CHART_STYLES.fontFamily,
                                textAnchor: 'middle',
                                dy: 4,
                            }) }), (0, jsx_runtime_1.jsx)(text_1.Text, { x: -innerHeight / 2, y: -55, transform: "rotate(-90)", fontSize: 11, fontFamily: BarChart_1.CHART_STYLES.fontFamily, fill: BarChart_1.CHART_STYLES.labelColor, textAnchor: "middle", children: "Count" }), (0, jsx_runtime_1.jsx)(text_1.Text, { x: innerWidth / 2, y: innerHeight + 45, fontSize: 11, fontFamily: BarChart_1.CHART_STYLES.fontFamily, fill: BarChart_1.CHART_STYLES.labelColor, textAnchor: "middle", children: "Iterations" })] }), (0, jsx_runtime_1.jsxs)(group_1.Group, { top: margin.top - 25, left: margin.left, children: [(0, jsx_runtime_1.jsx)("rect", { x: 0, y: -6, width: 12, height: 12, fill: COLORS.mpb, opacity: 0.6 }), (0, jsx_runtime_1.jsx)(text_1.Text, { x: 18, y: 4, fontSize: 11, fill: BarChart_1.CHART_STYLES.labelColor, fontFamily: BarChart_1.CHART_STYLES.fontFamily, children: "MPB" }), (0, jsx_runtime_1.jsx)("rect", { x: 60, y: -6, width: 12, height: 12, fill: COLORS.blaze, opacity: 0.5 }), (0, jsx_runtime_1.jsx)(text_1.Text, { x: 78, y: 4, fontSize: 11, fill: BarChart_1.CHART_STYLES.labelColor, fontFamily: BarChart_1.CHART_STYLES.fontFamily, children: "Blaze2D" })] })] }));
    };
    return ((0, jsx_runtime_1.jsxs)("div", { style: { display: 'flex', gap: '20px', flexWrap: 'wrap', justifyContent: 'center' }, children: [renderChart(tmHistogram, 'TM'), renderChart(teHistogram, 'TE')] }));
}
