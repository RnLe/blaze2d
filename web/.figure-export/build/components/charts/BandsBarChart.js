'use client';
"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.default = BandsBarChart;
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
    blaze: '#eaf1fe', // Light blue-white
};
function BandsBarChart({ width = 650, height = 380, }) {
    const { data: benchmarkData, loading } = (0, use_benchmarks_1.useSeries2Benchmarks)();
    // Transform data for the bar charts
    const { tmData, teData } = (0, react_1.useMemo)(() => {
        const bands = benchmarkData.band_values;
        const tm = {
            bands,
            mpb: benchmarkData.TM.mpb.map((d, i) => ({
                band: bands[i],
                mean: d?.mean || 0,
                std: d?.std || 0,
            })),
            blaze: benchmarkData.TM.blaze.map((d, i) => ({
                band: bands[i],
                mean: d?.mean || 0,
                std: d?.std || 0,
            })),
        };
        const te = {
            bands,
            mpb: benchmarkData.TE.mpb.map((d, i) => ({
                band: bands[i],
                mean: d?.mean || 0,
                std: d?.std || 0,
            })),
            blaze: benchmarkData.TE.blaze.map((d, i) => ({
                band: bands[i],
                mean: d?.mean || 0,
                std: d?.std || 0,
            })),
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
        const maxValue = Math.max(...data.mpb.map(d => d.mean + d.std), ...data.blaze.map(d => d.mean + d.std));
        const xScale = (0, scale_1.scaleBand)({
            domain: data.bands,
            range: [0, innerWidth],
            padding: 0.2,
        });
        const yScale = (0, scale_1.scaleLinear)({
            domain: [0, maxValue * 1.1],
            range: [innerHeight, 0],
        });
        const barWidth = xScale.bandwidth() / 2 - 1;
        // Show every other tick for readability if many bands
        const tickStep = data.bands.length > 12 ? 2 : 1;
        const xTickValues = data.bands.filter((_, i) => i % tickStep === 0);
        return ((0, jsx_runtime_1.jsxs)("svg", { width: chartWidth, height: height, style: { overflow: 'visible' }, children: [(0, jsx_runtime_1.jsx)(text_1.Text, { x: 0, y: 16, fontSize: 14, fontFamily: BarChart_1.CHART_STYLES.fontFamily, fill: BarChart_1.CHART_STYLES.labelColor, fontWeight: 700, children: `Time vs Number of Bands (${polarization})` }), (0, jsx_runtime_1.jsxs)(group_1.Group, { left: margin.left, top: margin.top, children: [(0, jsx_runtime_1.jsx)(grid_1.GridRows, { scale: yScale, width: innerWidth, stroke: BarChart_1.CHART_STYLES.gridColor, strokeOpacity: 0.4, strokeDasharray: "3,3" }), data.bands.map((band, i) => {
                            const xPos = xScale(band) || 0;
                            const mpbData = data.mpb[i];
                            const blazeData = data.blaze[i];
                            return ((0, jsx_runtime_1.jsxs)("g", { children: [mpbData.mean > 0 && ((0, jsx_runtime_1.jsxs)(jsx_runtime_1.Fragment, { children: [(0, jsx_runtime_1.jsx)(shape_1.Bar, { x: xPos, y: yScale(mpbData.mean), width: barWidth, height: innerHeight - yScale(mpbData.mean), fill: COLORS.mpb, opacity: 0.8 }), mpbData.std > 0 && ((0, jsx_runtime_1.jsxs)("g", { children: [(0, jsx_runtime_1.jsx)("line", { x1: xPos + barWidth / 2, y1: yScale(mpbData.mean - mpbData.std), x2: xPos + barWidth / 2, y2: yScale(mpbData.mean + mpbData.std), stroke: COLORS.mpb, strokeWidth: 1.5 }), (0, jsx_runtime_1.jsx)("line", { x1: xPos + barWidth / 2 - 2, y1: yScale(mpbData.mean + mpbData.std), x2: xPos + barWidth / 2 + 2, y2: yScale(mpbData.mean + mpbData.std), stroke: COLORS.mpb, strokeWidth: 1.5 })] }))] })), blazeData.mean > 0 && ((0, jsx_runtime_1.jsxs)(jsx_runtime_1.Fragment, { children: [(0, jsx_runtime_1.jsx)(shape_1.Bar, { x: xPos + barWidth, y: yScale(blazeData.mean), width: barWidth, height: innerHeight - yScale(blazeData.mean), fill: COLORS.blaze, opacity: 0.8 }), blazeData.std > 0 && ((0, jsx_runtime_1.jsxs)("g", { children: [(0, jsx_runtime_1.jsx)("line", { x1: xPos + barWidth + barWidth / 2, y1: yScale(blazeData.mean - blazeData.std), x2: xPos + barWidth + barWidth / 2, y2: yScale(blazeData.mean + blazeData.std), stroke: COLORS.blaze, strokeWidth: 1.5 }), (0, jsx_runtime_1.jsx)("line", { x1: xPos + barWidth + barWidth / 2 - 2, y1: yScale(blazeData.mean + blazeData.std), x2: xPos + barWidth + barWidth / 2 + 2, y2: yScale(blazeData.mean + blazeData.std), stroke: COLORS.blaze, strokeWidth: 1.5 })] }))] }))] }, band));
                        }), (0, jsx_runtime_1.jsx)(axis_1.AxisLeft, { scale: yScale, stroke: BarChart_1.CHART_STYLES.axisColor, tickStroke: BarChart_1.CHART_STYLES.axisColor, tickFormat: (v) => `${v}`, tickLabelProps: () => ({
                                fill: BarChart_1.CHART_STYLES.labelColor,
                                fontSize: 11,
                                fontFamily: BarChart_1.CHART_STYLES.fontFamily,
                                textAnchor: 'end',
                                dx: -4,
                                dy: 4,
                            }), numTicks: 5 }), (0, jsx_runtime_1.jsx)(axis_1.AxisBottom, { scale: xScale, top: innerHeight, stroke: BarChart_1.CHART_STYLES.axisColor, tickStroke: BarChart_1.CHART_STYLES.axisColor, tickValues: xTickValues, tickFormat: (v) => `${v}`, tickLabelProps: () => ({
                                fill: BarChart_1.CHART_STYLES.labelColor,
                                fontSize: 10,
                                fontFamily: BarChart_1.CHART_STYLES.fontFamily,
                                textAnchor: 'middle',
                                dy: 4,
                            }) }), (0, jsx_runtime_1.jsx)(text_1.Text, { x: -innerHeight / 2, y: -55, transform: "rotate(-90)", fontSize: 11, fontFamily: BarChart_1.CHART_STYLES.fontFamily, fill: BarChart_1.CHART_STYLES.labelColor, textAnchor: "middle", children: "Runtime (ms)" }), (0, jsx_runtime_1.jsx)(text_1.Text, { x: innerWidth / 2, y: innerHeight + 45, fontSize: 11, fontFamily: BarChart_1.CHART_STYLES.fontFamily, fill: BarChart_1.CHART_STYLES.labelColor, textAnchor: "middle", children: "Number of Bands" })] }), (0, jsx_runtime_1.jsxs)(group_1.Group, { top: margin.top - 25, left: margin.left, children: [(0, jsx_runtime_1.jsx)("rect", { x: 0, y: -6, width: 12, height: 12, fill: COLORS.mpb, opacity: 0.8 }), (0, jsx_runtime_1.jsx)(text_1.Text, { x: 18, y: 4, fontSize: 11, fill: BarChart_1.CHART_STYLES.labelColor, fontFamily: BarChart_1.CHART_STYLES.fontFamily, children: "MPB" }), (0, jsx_runtime_1.jsx)("rect", { x: 60, y: -6, width: 12, height: 12, fill: COLORS.blaze, opacity: 0.8 }), (0, jsx_runtime_1.jsx)(text_1.Text, { x: 78, y: 4, fontSize: 11, fill: BarChart_1.CHART_STYLES.labelColor, fontFamily: BarChart_1.CHART_STYLES.fontFamily, children: "Blaze2D" })] })] }));
    };
    return ((0, jsx_runtime_1.jsxs)("div", { style: { display: 'flex', gap: '20px', flexWrap: 'wrap', justifyContent: 'center' }, children: [renderChart(tmData, 'TM'), renderChart(teData, 'TE')] }));
}
