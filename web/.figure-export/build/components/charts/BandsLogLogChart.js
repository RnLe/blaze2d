'use client';
"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.default = BandsLogLogChart;
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
    blaze: '#f97316', // Orange for better contrast
};
function BandsLogLogChart({ width = 650, height = 380, }) {
    const { data: benchmarkData, loading } = (0, use_benchmarks_1.useSeries2Benchmarks)();
    // Transform data for the charts
    const { tmData, teData } = (0, react_1.useMemo)(() => {
        const bands = benchmarkData.band_values;
        const tm = {
            bands,
            mpb: benchmarkData.TM.mpb
                .map((d, i) => ({
                band: bands[i],
                mean: d?.mean || 0,
                std: d?.std || 0,
            }))
                .filter(d => d.mean > 0),
            blaze: benchmarkData.TM.blaze
                .map((d, i) => ({
                band: bands[i],
                mean: d?.mean || 0,
                std: d?.std || 0,
            }))
                .filter(d => d.mean > 0),
        };
        const te = {
            bands,
            mpb: benchmarkData.TE.mpb
                .map((d, i) => ({
                band: bands[i],
                mean: d?.mean || 0,
                std: d?.std || 0,
            }))
                .filter(d => d.mean > 0),
            blaze: benchmarkData.TE.blaze
                .map((d, i) => ({
                band: bands[i],
                mean: d?.mean || 0,
                std: d?.std || 0,
            }))
                .filter(d => d.mean > 0),
        };
        return { tmData: tm, teData: te };
    }, [benchmarkData]);
    if (loading || tmData.mpb.length === 0) {
        return ((0, jsx_runtime_1.jsx)("div", { style: { width: width * 2 + 40, height, display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#888' }, children: "Loading benchmark data..." }));
    }
    const chartWidth = (width - 20) / 2;
    const margin = { top: 65, right: 20, bottom: 60, left: 70 };
    const innerWidth = chartWidth - margin.left - margin.right;
    const innerHeight = height - margin.top - margin.bottom;
    const renderChart = (data, polarization) => {
        const allMeans = [...data.mpb.map(d => d.mean), ...data.blaze.map(d => d.mean)];
        const minBand = Math.min(...data.bands);
        const maxBand = Math.max(...data.bands);
        const minValue = Math.min(...allMeans);
        const maxValue = Math.max(...allMeans);
        const xScale = (0, scale_1.scaleLog)({
            domain: [minBand * 0.9, maxBand * 1.1],
            range: [0, innerWidth],
            base: 10,
        });
        const yScale = (0, scale_1.scaleLog)({
            domain: [minValue * 0.8, maxValue * 1.2],
            range: [innerHeight, 0],
            base: 10,
        });
        // Format tick labels for log scale
        const formatLogTick = (value) => {
            if (value >= 1000)
                return `${(value / 1000).toFixed(0)}k`;
            if (value >= 100)
                return `${value.toFixed(0)}`;
            if (value >= 10)
                return `${value.toFixed(0)}`;
            return `${value.toFixed(1)}`;
        };
        return ((0, jsx_runtime_1.jsxs)("svg", { width: chartWidth, height: height, style: { overflow: 'visible' }, children: [(0, jsx_runtime_1.jsx)(text_1.Text, { x: 0, y: 16, fontSize: 14, fontFamily: BarChart_1.CHART_STYLES.fontFamily, fill: BarChart_1.CHART_STYLES.labelColor, fontWeight: 700, children: `Log-Log: Time vs Bands (${polarization})` }), (0, jsx_runtime_1.jsxs)(group_1.Group, { left: margin.left, top: margin.top, children: [(0, jsx_runtime_1.jsx)(grid_1.GridRows, { scale: yScale, width: innerWidth, stroke: BarChart_1.CHART_STYLES.gridColor, strokeOpacity: 0.3, strokeDasharray: "3,3", numTicks: 5 }), (0, jsx_runtime_1.jsx)(grid_1.GridColumns, { scale: xScale, height: innerHeight, stroke: BarChart_1.CHART_STYLES.gridColor, strokeOpacity: 0.3, strokeDasharray: "3,3", numTicks: 5 }), (0, jsx_runtime_1.jsx)(shape_1.LinePath, { data: data.mpb, x: d => xScale(d.band), y: d => yScale(d.mean), stroke: COLORS.mpb, strokeWidth: 2, strokeOpacity: 0.9 }), data.mpb.map((d, i) => ((0, jsx_runtime_1.jsxs)("g", { children: [(0, jsx_runtime_1.jsx)("circle", { cx: xScale(d.band), cy: yScale(d.mean), r: 4, fill: COLORS.mpb, opacity: 0.9 }), d.std > 0 && d.mean - d.std > 0 && ((0, jsx_runtime_1.jsx)(jsx_runtime_1.Fragment, { children: (0, jsx_runtime_1.jsx)("line", { x1: xScale(d.band), y1: yScale(d.mean - d.std), x2: xScale(d.band), y2: yScale(d.mean + d.std), stroke: COLORS.mpb, strokeWidth: 1, opacity: 0.7 }) }))] }, `mpb-${i}`))), (0, jsx_runtime_1.jsx)(shape_1.LinePath, { data: data.blaze, x: d => xScale(d.band), y: d => yScale(d.mean), stroke: COLORS.blaze, strokeWidth: 2, strokeOpacity: 0.9 }), data.blaze.map((d, i) => ((0, jsx_runtime_1.jsxs)("g", { children: [(0, jsx_runtime_1.jsx)("circle", { cx: xScale(d.band), cy: yScale(d.mean), r: 4, fill: COLORS.blaze, opacity: 0.9 }), d.std > 0 && d.mean - d.std > 0 && ((0, jsx_runtime_1.jsx)(jsx_runtime_1.Fragment, { children: (0, jsx_runtime_1.jsx)("line", { x1: xScale(d.band), y1: yScale(d.mean - d.std), x2: xScale(d.band), y2: yScale(d.mean + d.std), stroke: COLORS.blaze, strokeWidth: 1, opacity: 0.7 }) }))] }, `blaze-${i}`))), (0, jsx_runtime_1.jsx)(axis_1.AxisLeft, { scale: yScale, stroke: BarChart_1.CHART_STYLES.axisColor, tickStroke: BarChart_1.CHART_STYLES.axisColor, tickFormat: (v) => formatLogTick(v), tickLabelProps: () => ({
                                fill: BarChart_1.CHART_STYLES.labelColor,
                                fontSize: 10,
                                fontFamily: BarChart_1.CHART_STYLES.fontFamily,
                                textAnchor: 'end',
                                dx: -4,
                                dy: 4,
                            }), numTicks: 5 }), (0, jsx_runtime_1.jsx)(axis_1.AxisBottom, { scale: xScale, top: innerHeight, stroke: BarChart_1.CHART_STYLES.axisColor, tickStroke: BarChart_1.CHART_STYLES.axisColor, tickFormat: (v) => `${v}`, tickLabelProps: () => ({
                                fill: BarChart_1.CHART_STYLES.labelColor,
                                fontSize: 10,
                                fontFamily: BarChart_1.CHART_STYLES.fontFamily,
                                textAnchor: 'middle',
                                dy: 4,
                            }), numTicks: 5 }), (0, jsx_runtime_1.jsx)(text_1.Text, { x: -innerHeight / 2, y: -55, transform: "rotate(-90)", fontSize: 11, fontFamily: BarChart_1.CHART_STYLES.fontFamily, fill: BarChart_1.CHART_STYLES.labelColor, textAnchor: "middle", children: "Runtime (ms)" }), (0, jsx_runtime_1.jsx)(text_1.Text, { x: innerWidth / 2, y: innerHeight + 45, fontSize: 11, fontFamily: BarChart_1.CHART_STYLES.fontFamily, fill: BarChart_1.CHART_STYLES.labelColor, textAnchor: "middle", children: "Number of Bands" })] }), (0, jsx_runtime_1.jsxs)(group_1.Group, { top: margin.top - 25, left: margin.left, children: [(0, jsx_runtime_1.jsx)("line", { x1: 0, y1: 0, x2: 16, y2: 0, stroke: COLORS.mpb, strokeWidth: 2 }), (0, jsx_runtime_1.jsx)("circle", { cx: 8, cy: 0, r: 3, fill: COLORS.mpb }), (0, jsx_runtime_1.jsx)(text_1.Text, { x: 22, y: 4, fontSize: 11, fill: BarChart_1.CHART_STYLES.labelColor, fontFamily: BarChart_1.CHART_STYLES.fontFamily, children: "MPB" }), (0, jsx_runtime_1.jsx)("line", { x1: 70, y1: 0, x2: 86, y2: 0, stroke: COLORS.blaze, strokeWidth: 2 }), (0, jsx_runtime_1.jsx)("circle", { cx: 78, cy: 0, r: 3, fill: COLORS.blaze }), (0, jsx_runtime_1.jsx)(text_1.Text, { x: 92, y: 4, fontSize: 11, fill: BarChart_1.CHART_STYLES.labelColor, fontFamily: BarChart_1.CHART_STYLES.fontFamily, children: "Blaze2D" })] })] }));
    };
    return ((0, jsx_runtime_1.jsxs)("div", { style: { display: 'flex', gap: '20px', flexWrap: 'wrap', justifyContent: 'center' }, children: [renderChart(tmData, 'TM'), renderChart(teData, 'TE')] }));
}
