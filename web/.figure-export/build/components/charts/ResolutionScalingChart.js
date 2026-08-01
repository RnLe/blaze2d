'use client';
"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.default = ResolutionScalingChart;
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
    refLine: '#999', // Gray for reference lines
};
// Marker size
const MARKER_SIZE = 5;
function ResolutionScalingChart({ width = 650, height = 380, }) {
    const { data: benchmarkData, loading } = (0, use_benchmarks_1.useSeries3Benchmarks)();
    // Transform data for the log-log plot
    const { tmData, teData, refLines } = (0, react_1.useMemo)(() => {
        const tm = {
            resolutions: benchmarkData.TM.resolution,
            mpb: benchmarkData.TM.mpb.map((d, i) => ({
                x: benchmarkData.TM.resolution[i],
                y: d?.mean || 0,
                std: d?.std || 0,
            })).filter(d => d.y > 0),
            blaze: benchmarkData.TM.blaze.map((d, i) => ({
                x: benchmarkData.TM.resolution[i],
                y: d?.mean || 0,
                std: d?.std || 0,
            })).filter(d => d.y > 0),
        };
        const te = {
            resolutions: benchmarkData.TE.resolution,
            mpb: benchmarkData.TE.mpb.map((d, i) => ({
                x: benchmarkData.TE.resolution[i],
                y: d?.mean || 0,
                std: d?.std || 0,
            })).filter(d => d.y > 0),
            blaze: benchmarkData.TE.blaze.map((d, i) => ({
                x: benchmarkData.TE.resolution[i],
                y: d?.mean || 0,
                std: d?.std || 0,
            })).filter(d => d.y > 0),
        };
        // Compute reference lines O(N²) and O(N³)
        const midIdx = Math.floor(tm.blaze.length / 2);
        const refX = tm.blaze[midIdx]?.x || 64;
        const refY = tm.blaze[midIdx]?.y || 600;
        const refLines = {
            n2: { refX, refY },
            n3: { refX, refY: refY / 5 },
        };
        return { tmData: tm, teData: te, refLines };
    }, [benchmarkData]);
    if (loading) {
        return ((0, jsx_runtime_1.jsx)("div", { style: { width: width * 2 + 40, height, display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#888' }, children: "Loading benchmark data..." }));
    }
    const chartWidth = (width - 20) / 2;
    const margin = { top: 65, right: 20, bottom: 60, left: 70 };
    const innerWidth = chartWidth - margin.left - margin.right;
    const innerHeight = height - margin.top - margin.bottom;
    // Common domain for both plots
    const xDomain = [10, 250];
    const yDomain = [30, 100000];
    // Generate power-of-10 tick values
    const getPowerOf10Ticks = (domain) => {
        const ticks = [];
        const minPow = Math.ceil(Math.log10(domain[0]));
        const maxPow = Math.floor(Math.log10(domain[1]));
        for (let pow = minPow; pow <= maxPow; pow++) {
            ticks.push(Math.pow(10, pow));
        }
        return ticks;
    };
    const xTicks = getPowerOf10Ticks(xDomain);
    const yTicks = getPowerOf10Ticks(yDomain);
    // Create scales
    const xScale = (0, scale_1.scaleLog)({
        domain: xDomain,
        range: [0, innerWidth],
        base: 10,
    });
    const yScale = (0, scale_1.scaleLog)({
        domain: yDomain,
        range: [innerHeight, 0],
        base: 10,
    });
    // Generate reference line points
    const generateRefLine = (power, refX, refY) => {
        const points = [];
        for (let x = xDomain[0]; x <= xDomain[1]; x *= 1.1) {
            const y = refY * Math.pow(x / refX, power);
            if (y >= yDomain[0] && y <= yDomain[1]) {
                points.push({ x, y });
            }
        }
        return points;
    };
    const renderChart = (data, polarization) => {
        // Reference lines for this chart
        const midIdx = Math.floor(data.blaze.length / 2);
        const refX = data.blaze[midIdx]?.x || 64;
        const refY = (data.mpb[midIdx]?.y + data.blaze[midIdx]?.y) / 2 || 1000;
        const n2Line = generateRefLine(2, refX, refY);
        const n3Line = generateRefLine(3, refX, refY / 10);
        return ((0, jsx_runtime_1.jsxs)("svg", { width: chartWidth, height: height, style: { overflow: 'visible' }, children: [(0, jsx_runtime_1.jsx)(text_1.Text, { x: 0, y: 16, fontSize: 14, fontFamily: BarChart_1.CHART_STYLES.fontFamily, fill: BarChart_1.CHART_STYLES.labelColor, fontWeight: 700, children: `Scaling Behavior (${polarization})` }), (0, jsx_runtime_1.jsxs)(group_1.Group, { left: margin.left, top: margin.top, children: [(0, jsx_runtime_1.jsx)(grid_1.GridRows, { scale: yScale, width: innerWidth, stroke: BarChart_1.CHART_STYLES.gridColor, strokeOpacity: 0.2, strokeDasharray: "3,3" }), (0, jsx_runtime_1.jsx)(grid_1.GridColumns, { scale: xScale, height: innerHeight, stroke: BarChart_1.CHART_STYLES.gridColor, strokeOpacity: 0.2, strokeDasharray: "3,3" }), (0, jsx_runtime_1.jsx)(shape_1.LinePath, { data: n2Line, x: (d) => xScale(d.x), y: (d) => yScale(d.y), stroke: COLORS.refLine, strokeWidth: 1, strokeDasharray: "4,4", strokeOpacity: 0.5 }), n2Line.length > 0 && ((0, jsx_runtime_1.jsx)(text_1.Text, { x: xScale(n2Line[n2Line.length - 1].x) - 25, y: yScale(n2Line[n2Line.length - 1].y) - 5, fontSize: 11, fontFamily: BarChart_1.CHART_STYLES.fontFamily, fill: COLORS.refLine, children: "O(N\u00B2)" })), (0, jsx_runtime_1.jsx)(shape_1.LinePath, { data: n3Line, x: (d) => xScale(d.x), y: (d) => yScale(d.y), stroke: COLORS.refLine, strokeWidth: 1, strokeDasharray: "2,2", strokeOpacity: 0.5 }), n3Line.length > 0 && ((0, jsx_runtime_1.jsx)(text_1.Text, { x: xScale(n3Line[Math.floor(n3Line.length / 2)].x) + 5, y: yScale(n3Line[Math.floor(n3Line.length / 2)].y) + 10, fontSize: 11, fontFamily: BarChart_1.CHART_STYLES.fontFamily, fill: COLORS.refLine, children: "O(N\u00B3)" })), (0, jsx_runtime_1.jsx)(shape_1.LinePath, { data: data.mpb, x: (d) => xScale(d.x), y: (d) => yScale(d.y), stroke: COLORS.mpb, strokeWidth: 2 }), data.mpb.map((d, i) => ((0, jsx_runtime_1.jsx)("circle", { cx: xScale(d.x), cy: yScale(d.y), r: MARKER_SIZE, fill: COLORS.mpb }, `mpb-${i}`))), (0, jsx_runtime_1.jsx)(shape_1.LinePath, { data: data.blaze, x: (d) => xScale(d.x), y: (d) => yScale(d.y), stroke: COLORS.blaze, strokeWidth: 2 }), data.blaze.map((d, i) => ((0, jsx_runtime_1.jsx)("rect", { x: xScale(d.x) - MARKER_SIZE, y: yScale(d.y) - MARKER_SIZE, width: MARKER_SIZE * 2, height: MARKER_SIZE * 2, fill: COLORS.blaze }, `blaze-${i}`))), (0, jsx_runtime_1.jsx)(axis_1.AxisLeft, { scale: yScale, stroke: BarChart_1.CHART_STYLES.axisColor, tickStroke: BarChart_1.CHART_STYLES.axisColor, tickFormat: (v) => {
                                const val = v;
                                if (val >= 1000)
                                    return `${(val / 1000).toFixed(0)}s`;
                                return `${val}ms`;
                            }, tickLabelProps: () => ({
                                fill: BarChart_1.CHART_STYLES.labelColor,
                                fontSize: 11,
                                fontFamily: BarChart_1.CHART_STYLES.fontFamily,
                                textAnchor: 'end',
                                dx: -4,
                                dy: 4,
                            }), tickValues: yTicks }), (0, jsx_runtime_1.jsx)(axis_1.AxisBottom, { scale: xScale, top: innerHeight, stroke: BarChart_1.CHART_STYLES.axisColor, tickStroke: BarChart_1.CHART_STYLES.axisColor, tickFormat: (v) => `${v}`, tickLabelProps: () => ({
                                fill: BarChart_1.CHART_STYLES.labelColor,
                                fontSize: 11,
                                fontFamily: BarChart_1.CHART_STYLES.fontFamily,
                                textAnchor: 'middle',
                                dy: 4,
                            }), tickValues: xTicks }), (0, jsx_runtime_1.jsx)(text_1.Text, { x: -innerHeight / 2, y: -55, transform: "rotate(-90)", fontSize: 11, fontFamily: BarChart_1.CHART_STYLES.fontFamily, fill: BarChart_1.CHART_STYLES.labelColor, textAnchor: "middle", children: "Time (ms)" }), (0, jsx_runtime_1.jsx)(text_1.Text, { x: innerWidth / 2, y: innerHeight + 45, fontSize: 11, fontFamily: BarChart_1.CHART_STYLES.fontFamily, fill: BarChart_1.CHART_STYLES.labelColor, textAnchor: "middle", children: "Resolution (N)" })] }), (0, jsx_runtime_1.jsxs)(group_1.Group, { top: margin.top - 25, left: margin.left, children: [(0, jsx_runtime_1.jsx)("circle", { cx: 6, cy: 0, r: 5, fill: COLORS.mpb }), (0, jsx_runtime_1.jsx)(text_1.Text, { x: 16, y: 4, fontSize: 11, fill: BarChart_1.CHART_STYLES.labelColor, fontFamily: BarChart_1.CHART_STYLES.fontFamily, children: "MPB" }), (0, jsx_runtime_1.jsx)("rect", { x: 60, y: -5, width: 10, height: 10, fill: COLORS.blaze }), (0, jsx_runtime_1.jsx)(text_1.Text, { x: 76, y: 4, fontSize: 11, fill: BarChart_1.CHART_STYLES.labelColor, fontFamily: BarChart_1.CHART_STYLES.fontFamily, children: "Blaze2D" })] })] }));
    };
    return ((0, jsx_runtime_1.jsxs)("div", { style: { display: 'flex', gap: '20px', flexWrap: 'wrap', justifyContent: 'center' }, children: [renderChart(tmData, 'TM'), renderChart(teData, 'TE')] }));
}
