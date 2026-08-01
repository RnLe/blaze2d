'use client';
"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.default = ResolutionBarChart;
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
function ResolutionBarChart({ width = 650, height = 380, }) {
    const { data: benchmarkData, loading } = (0, use_benchmarks_1.useSeries3Benchmarks)();
    // Transform data for the bar charts
    const { tmData, teData } = (0, react_1.useMemo)(() => {
        const tm = {
            resolutions: benchmarkData.TM.resolution,
            mpb: benchmarkData.TM.mpb.map((d, i) => ({
                resolution: benchmarkData.TM.resolution[i],
                mean: d?.mean || 0,
                std: d?.std || 0,
            })),
            blaze: benchmarkData.TM.blaze.map((d, i) => ({
                resolution: benchmarkData.TM.resolution[i],
                mean: d?.mean || 0,
                std: d?.std || 0,
            })),
        };
        const te = {
            resolutions: benchmarkData.TE.resolution,
            mpb: benchmarkData.TE.mpb.map((d, i) => ({
                resolution: benchmarkData.TE.resolution[i],
                mean: d?.mean || 0,
                std: d?.std || 0,
            })),
            blaze: benchmarkData.TE.blaze.map((d, i) => ({
                resolution: benchmarkData.TE.resolution[i],
                mean: d?.mean || 0,
                std: d?.std || 0,
            })),
        };
        return { tmData: tm, teData: te };
    }, [benchmarkData]);
    if (loading) {
        return ((0, jsx_runtime_1.jsx)("div", { style: { width: width * 2 + 40, height, display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#888' }, children: "Loading benchmark data..." }));
    }
    const chartWidth = (width - 20) / 2;
    const margin = { top: 65, right: 20, bottom: 60, left: 70 };
    const innerWidth = chartWidth - margin.left - margin.right;
    const innerHeight = height - margin.top - margin.bottom;
    const renderChart = (data, polarization) => {
        const maxValue = Math.max(...data.mpb.map(d => d.mean + d.std), ...data.blaze.map(d => d.mean + d.std));
        const xScale = (0, scale_1.scaleBand)({
            domain: data.resolutions,
            range: [0, innerWidth],
            padding: 0.3,
        });
        const yScale = (0, scale_1.scaleLinear)({
            domain: [0, maxValue * 1.1],
            range: [innerHeight, 0],
        });
        const barWidth = xScale.bandwidth() / 2 - 2;
        return ((0, jsx_runtime_1.jsxs)("svg", { width: chartWidth, height: height, style: { overflow: 'visible' }, children: [(0, jsx_runtime_1.jsx)(text_1.Text, { x: 0, y: 16, fontSize: 14, fontFamily: BarChart_1.CHART_STYLES.fontFamily, fill: BarChart_1.CHART_STYLES.labelColor, fontWeight: 700, children: `Time Comparison (${polarization})` }), (0, jsx_runtime_1.jsxs)(group_1.Group, { left: margin.left, top: margin.top, children: [(0, jsx_runtime_1.jsx)(grid_1.GridRows, { scale: yScale, width: innerWidth, stroke: BarChart_1.CHART_STYLES.gridColor, strokeOpacity: 0.4, strokeDasharray: "3,3" }), data.resolutions.map((res, i) => {
                            const xPos = xScale(res) || 0;
                            const mpbData = data.mpb[i];
                            const blazeData = data.blaze[i];
                            return ((0, jsx_runtime_1.jsxs)("g", { children: [(0, jsx_runtime_1.jsx)(shape_1.Bar, { x: xPos, y: yScale(mpbData.mean), width: barWidth, height: innerHeight - yScale(mpbData.mean), fill: COLORS.mpb }), mpbData.std > 0 && ((0, jsx_runtime_1.jsxs)("g", { children: [(0, jsx_runtime_1.jsx)("line", { x1: xPos + barWidth / 2, y1: yScale(mpbData.mean - mpbData.std), x2: xPos + barWidth / 2, y2: yScale(mpbData.mean + mpbData.std), stroke: COLORS.mpb, strokeWidth: 1.5 }), (0, jsx_runtime_1.jsx)("line", { x1: xPos + barWidth / 2 - 3, y1: yScale(mpbData.mean + mpbData.std), x2: xPos + barWidth / 2 + 3, y2: yScale(mpbData.mean + mpbData.std), stroke: COLORS.mpb, strokeWidth: 1.5 })] })), (0, jsx_runtime_1.jsx)(shape_1.Bar, { x: xPos + barWidth + 4, y: yScale(blazeData.mean), width: barWidth, height: innerHeight - yScale(blazeData.mean), fill: COLORS.blaze }), blazeData.std > 0 && ((0, jsx_runtime_1.jsxs)("g", { children: [(0, jsx_runtime_1.jsx)("line", { x1: xPos + barWidth + 4 + barWidth / 2, y1: yScale(blazeData.mean - blazeData.std), x2: xPos + barWidth + 4 + barWidth / 2, y2: yScale(blazeData.mean + blazeData.std), stroke: COLORS.blaze, strokeWidth: 1.5 }), (0, jsx_runtime_1.jsx)("line", { x1: xPos + barWidth + 4 + barWidth / 2 - 3, y1: yScale(blazeData.mean + blazeData.std), x2: xPos + barWidth + 4 + barWidth / 2 + 3, y2: yScale(blazeData.mean + blazeData.std), stroke: COLORS.blaze, strokeWidth: 1.5 })] }))] }, res));
                        }), (0, jsx_runtime_1.jsx)(axis_1.AxisLeft, { scale: yScale, stroke: BarChart_1.CHART_STYLES.axisColor, tickStroke: BarChart_1.CHART_STYLES.axisColor, tickFormat: (v) => {
                                const val = v;
                                if (val >= 1000)
                                    return `${(val / 1000).toFixed(0)}s`;
                                return `${val.toFixed(0)}ms`;
                            }, tickLabelProps: () => ({
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
                            }) }), (0, jsx_runtime_1.jsx)(text_1.Text, { x: -innerHeight / 2, y: -55, transform: "rotate(-90)", fontSize: 11, fontFamily: BarChart_1.CHART_STYLES.fontFamily, fill: BarChart_1.CHART_STYLES.labelColor, textAnchor: "middle", children: "Time (ms)" }), (0, jsx_runtime_1.jsx)(text_1.Text, { x: innerWidth / 2, y: innerHeight + 45, fontSize: 11, fontFamily: BarChart_1.CHART_STYLES.fontFamily, fill: BarChart_1.CHART_STYLES.labelColor, textAnchor: "middle", children: "Resolution (N\u00D7N)" })] }), (0, jsx_runtime_1.jsxs)(group_1.Group, { top: margin.top - 25, left: margin.left, children: [(0, jsx_runtime_1.jsx)("rect", { x: 0, y: -6, width: 12, height: 12, fill: COLORS.mpb }), (0, jsx_runtime_1.jsx)(text_1.Text, { x: 18, y: 4, fontSize: 11, fill: BarChart_1.CHART_STYLES.labelColor, fontFamily: BarChart_1.CHART_STYLES.fontFamily, children: "MPB" }), (0, jsx_runtime_1.jsx)("rect", { x: 60, y: -6, width: 12, height: 12, fill: COLORS.blaze }), (0, jsx_runtime_1.jsx)(text_1.Text, { x: 78, y: 4, fontSize: 11, fill: BarChart_1.CHART_STYLES.labelColor, fontFamily: BarChart_1.CHART_STYLES.fontFamily, children: "Blaze2D" })] })] }));
    };
    return ((0, jsx_runtime_1.jsxs)("div", { style: { display: 'flex', gap: '20px', flexWrap: 'wrap', justifyContent: 'center' }, children: [renderChart(tmData, 'TM'), renderChart(teData, 'TE')] }));
}
