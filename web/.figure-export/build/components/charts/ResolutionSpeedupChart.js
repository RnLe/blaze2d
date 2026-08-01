'use client';
"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.default = ResolutionSpeedupChart;
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
// Colors for speedup
const COLORS = {
    positive: '#eaf1fe', // Frost for speedup > 1
    negative: '#ef4444', // Red for speedup < 1
};
function ResolutionSpeedupChart({ width = 650, height = 380, }) {
    const { data: benchmarkData, loading } = (0, use_benchmarks_1.useSeries3Benchmarks)();
    // Calculate speedups
    const { tmSpeedups, teSpeedups, resolutions } = (0, react_1.useMemo)(() => {
        const resolutions = benchmarkData.TM.resolution;
        const tmSpeedups = benchmarkData.TM.mpb.map((mpb, i) => {
            const blaze = benchmarkData.TM.blaze[i];
            if (!mpb || !blaze || blaze.mean === 0)
                return 0;
            return mpb.mean / blaze.mean;
        });
        const teSpeedups = benchmarkData.TE.mpb.map((mpb, i) => {
            const blaze = benchmarkData.TE.blaze[i];
            if (!mpb || !blaze || blaze.mean === 0)
                return 0;
            return mpb.mean / blaze.mean;
        });
        return { tmSpeedups, teSpeedups, resolutions };
    }, [benchmarkData]);
    if (loading) {
        return ((0, jsx_runtime_1.jsx)("div", { style: { width: width * 2 + 40, height, display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#888' }, children: "Loading benchmark data..." }));
    }
    const chartWidth = (width - 20) / 2;
    const margin = { top: 65, right: 20, bottom: 60, left: 70 };
    const innerWidth = chartWidth - margin.left - margin.right;
    const innerHeight = height - margin.top - margin.bottom;
    const renderChart = (speedups, polarization) => {
        const maxValue = Math.max(...speedups, 1) * 1.15;
        const xScale = (0, scale_1.scaleBand)({
            domain: resolutions,
            range: [0, innerWidth],
            padding: 0.3,
        });
        const yScale = (0, scale_1.scaleLinear)({
            domain: [0, maxValue],
            range: [innerHeight, 0],
        });
        const barWidth = xScale.bandwidth();
        return ((0, jsx_runtime_1.jsxs)("svg", { width: chartWidth, height: height, style: { overflow: 'visible' }, children: [(0, jsx_runtime_1.jsx)(text_1.Text, { x: 0, y: 16, fontSize: 14, fontFamily: BarChart_1.CHART_STYLES.fontFamily, fill: BarChart_1.CHART_STYLES.labelColor, fontWeight: 700, children: `Speedup (${polarization})` }), (0, jsx_runtime_1.jsxs)(group_1.Group, { left: margin.left, top: margin.top, children: [(0, jsx_runtime_1.jsx)(grid_1.GridRows, { scale: yScale, width: innerWidth, stroke: BarChart_1.CHART_STYLES.gridColor, strokeOpacity: 0.4, strokeDasharray: "3,3" }), (0, jsx_runtime_1.jsx)("line", { x1: 0, y1: yScale(1), x2: innerWidth, y2: yScale(1), stroke: "#fff", strokeWidth: 1, strokeDasharray: "5,5", strokeOpacity: 0.7 }), resolutions.map((res, i) => {
                            const speedup = speedups[i];
                            const xPos = xScale(res) || 0;
                            const color = speedup >= 1 ? COLORS.positive : COLORS.negative;
                            return ((0, jsx_runtime_1.jsxs)("g", { children: [(0, jsx_runtime_1.jsx)(shape_1.Bar, { x: xPos, y: yScale(speedup), width: barWidth, height: innerHeight - yScale(speedup), fill: color }), (0, jsx_runtime_1.jsx)(text_1.Text, { x: xPos + barWidth / 2, y: yScale(speedup) - 5, fontSize: 9, fontFamily: BarChart_1.CHART_STYLES.fontFamily, fill: BarChart_1.CHART_STYLES.labelColor, textAnchor: "middle", children: `${speedup.toFixed(1)}×` })] }, res));
                        }), (0, jsx_runtime_1.jsx)(axis_1.AxisLeft, { scale: yScale, stroke: BarChart_1.CHART_STYLES.axisColor, tickStroke: BarChart_1.CHART_STYLES.axisColor, tickFormat: (v) => `${v}×`, tickLabelProps: () => ({
                                fill: BarChart_1.CHART_STYLES.labelColor,
                                fontSize: 11,
                                fontFamily: BarChart_1.CHART_STYLES.fontFamily,
                                textAnchor: 'end',
                                dx: -4,
                                dy: 4,
                            }), numTicks: 5 }), (0, jsx_runtime_1.jsx)(axis_1.AxisBottom, { scale: xScale, top: innerHeight, stroke: BarChart_1.CHART_STYLES.axisColor, tickStroke: BarChart_1.CHART_STYLES.axisColor, tickFormat: (v) => `${v}`, tickLabelProps: () => ({
                                fill: BarChart_1.CHART_STYLES.labelColor,
                                fontSize: 11,
                                fontFamily: BarChart_1.CHART_STYLES.fontFamily,
                                textAnchor: 'middle',
                                dy: 4,
                            }) }), (0, jsx_runtime_1.jsx)(text_1.Text, { x: -innerHeight / 2, y: -55, transform: "rotate(-90)", fontSize: 11, fontFamily: BarChart_1.CHART_STYLES.fontFamily, fill: BarChart_1.CHART_STYLES.labelColor, textAnchor: "middle", children: "Speedup (MPB / Blaze2D)" }), (0, jsx_runtime_1.jsx)(text_1.Text, { x: innerWidth / 2, y: innerHeight + 45, fontSize: 11, fontFamily: BarChart_1.CHART_STYLES.fontFamily, fill: BarChart_1.CHART_STYLES.labelColor, textAnchor: "middle", children: "Resolution (N\u00D7N)" })] })] }));
    };
    return ((0, jsx_runtime_1.jsxs)("div", { style: { display: 'flex', gap: '20px', flexWrap: 'wrap', justifyContent: 'center' }, children: [renderChart(tmSpeedups, 'TM'), renderChart(teSpeedups, 'TE')] }));
}
