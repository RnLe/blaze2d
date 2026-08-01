'use client';
"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.default = DeviationBoxPlotChart;
const jsx_runtime_1 = require("react/jsx-runtime");
const react_1 = require("react");
const group_1 = require("@visx/group");
const scale_1 = require("@visx/scale");
const axis_1 = require("@visx/axis");
const grid_1 = require("@visx/grid");
const text_1 = require("@visx/text");
const swr_shim_1 = __importDefault(require("../../lib/swr-shim"));
const BarChart_1 = require("./BarChart");
const paths_1 = require("../../lib/paths");
const defaultMargin = { top: 60, right: 30, bottom: 60, left: 80 };
// Colors matching band diagram
const F64_COLOR = '#a3befa'; // Full precision - light blue
const F32_COLOR = '#bbc1cb'; // Mixed precision - gray
const COMPARE_COLOR = '#435f9d'; // f32 vs f64 - reference blue
const fetcher = (url) => fetch(url).then(res => res.json());
function DeviationBoxPlotChart({ width = 400, height = 400, polarization = 'TM', }) {
    const { data, error, isLoading } = (0, swr_shim_1.default)((0, paths_1.getAssetPath)('/data/benchmarks/series6-accuracy.json'), fetcher);
    const margin = defaultMargin;
    const innerWidth = width - margin.left - margin.right;
    const innerHeight = height - margin.top - margin.bottom;
    // Extract deviation data for the selected polarization
    const deviations = data?.[polarization]?.deviations;
    // Prepare boxplot data
    const boxData = (0, react_1.useMemo)(() => {
        if (!deviations)
            return [];
        const items = [];
        if (deviations.f32_vs_mpb) {
            items.push({ label: 'Mixed (f32)', stats: deviations.f32_vs_mpb, color: F32_COLOR });
        }
        if (deviations.f64_vs_mpb) {
            items.push({ label: 'Full (f64)', stats: deviations.f64_vs_mpb, color: F64_COLOR });
        }
        if (deviations.f32_vs_f64) {
            items.push({ label: 'f32 vs f64', stats: deviations.f32_vs_f64, color: COMPARE_COLOR });
        }
        return items;
    }, [deviations]);
    // X scale (categorical)
    const xScale = (0, react_1.useMemo)(() => {
        return (0, scale_1.scaleBand)({
            domain: boxData.map(d => d.label),
            range: [0, innerWidth],
            padding: 0.4,
        });
    }, [boxData, innerWidth]);
    // Y scale (log scale for deviations)
    const { yScale, yMin, yMax } = (0, react_1.useMemo)(() => {
        // Gather all stats from both polarizations if available to ensure shared axis
        const allStats = [];
        if (data?.TM?.deviations) {
            const dev = data.TM.deviations;
            if (dev.f32_vs_mpb)
                allStats.push(dev.f32_vs_mpb);
            if (dev.f64_vs_mpb)
                allStats.push(dev.f64_vs_mpb);
            if (dev.f32_vs_f64)
                allStats.push(dev.f32_vs_f64);
        }
        if (data?.TE?.deviations) {
            const dev = data.TE.deviations;
            if (dev.f32_vs_mpb)
                allStats.push(dev.f32_vs_mpb);
            if (dev.f64_vs_mpb)
                allStats.push(dev.f64_vs_mpb);
            if (dev.f32_vs_f64)
                allStats.push(dev.f32_vs_f64);
        }
        if (!allStats.length) {
            return {
                yScale: (0, scale_1.scaleLinear)({ domain: [-6, -2], range: [innerHeight, 0] }),
                yMin: -6,
                yMax: -2
            };
        }
        // Get min/max from Q1 and Q3 (more stable than whiskers which can be 0)
        const allQ1 = allStats.map(d => d.q1).filter(v => v > 0);
        const allWhiskerHigh = allStats.map(d => d.whiskerHigh).filter(v => v > 0);
        // Use Q1 for min (since whiskerLow can be 0)
        const minVal = allQ1.length > 0 ? Math.min(...allQ1) * 0.1 : 1e-6;
        const maxVal = allWhiskerHigh.length > 0 ? Math.max(...allWhiskerHigh) * 5 : 1e-2;
        const logMin = Math.floor(Math.log10(minVal));
        const logMax = Math.ceil(Math.log10(maxVal));
        return {
            yScale: (0, scale_1.scaleLinear)({
                domain: [logMin, logMax],
                range: [innerHeight, 0],
            }),
            yMin: logMin,
            yMax: logMax
        };
    }, [data, innerHeight]);
    // Helper to convert value to y position (log scale)
    const toY = (value) => {
        // Clamp very small values to the bottom of the scale
        if (value <= 0)
            return innerHeight;
        return yScale(Math.log10(value));
    };
    // Box visual width
    const boxWidth = xScale.bandwidth() * 0.7;
    if (isLoading) {
        return ((0, jsx_runtime_1.jsx)("div", { style: { width, height, display: 'flex', alignItems: 'center', justifyContent: 'center', color: BarChart_1.CHART_STYLES.labelColor }, children: "Loading..." }));
    }
    if (error || !data) {
        return ((0, jsx_runtime_1.jsx)("div", { style: { width, height, display: 'flex', alignItems: 'center', justifyContent: 'center', color: BarChart_1.CHART_STYLES.labelColor }, children: "Error loading data" }));
    }
    return ((0, jsx_runtime_1.jsxs)("div", { className: "deviation-boxplot-container", style: { width: '100%', maxWidth: width }, children: [(0, jsx_runtime_1.jsxs)("svg", { width: width, height: height, style: { overflow: 'visible' }, children: [(0, jsx_runtime_1.jsx)(text_1.Text, { x: 0, y: 16, fontSize: 14, fontFamily: BarChart_1.CHART_STYLES.fontFamily, fill: BarChart_1.CHART_STYLES.labelColor, fontWeight: 700, children: `${polarization} Deviation Distribution` }), (0, jsx_runtime_1.jsxs)(group_1.Group, { left: margin.left, top: margin.top, children: [(0, jsx_runtime_1.jsx)(grid_1.GridRows, { scale: yScale, width: innerWidth, stroke: BarChart_1.CHART_STYLES.gridColor, strokeOpacity: 0.3, strokeDasharray: "3,3" }), boxData.map((d, i) => {
                                const x = (xScale(d.label) || 0) + (xScale.bandwidth() - boxWidth) / 2;
                                const color = d.color;
                                const stats = d.stats;
                                // For log scale, if whiskerLow is 0, use q1 as the lower bound
                                const effectiveWhiskerLow = stats.whiskerLow > 0 ? stats.whiskerLow : stats.q1;
                                const whiskerLowY = toY(effectiveWhiskerLow);
                                const q1Y = toY(stats.q1);
                                const medianY = toY(stats.median);
                                const q3Y = toY(stats.q3);
                                const whiskerHighY = toY(stats.whiskerHigh);
                                const boxHeight = q1Y - q3Y;
                                const centerX = x + boxWidth / 2;
                                return ((0, jsx_runtime_1.jsxs)(group_1.Group, { children: [stats.whiskerLow > 0 ? ((0, jsx_runtime_1.jsxs)(jsx_runtime_1.Fragment, { children: [(0, jsx_runtime_1.jsx)("line", { x1: centerX, y1: whiskerLowY, x2: centerX, y2: q1Y, stroke: color, strokeWidth: 1.5 }), (0, jsx_runtime_1.jsx)("line", { x1: x + boxWidth * 0.2, y1: whiskerLowY, x2: x + boxWidth * 0.8, y2: whiskerLowY, stroke: color, strokeWidth: 2 })] })) : (
                                        /* When whiskerLow is 0, draw line from q1 to x-axis */
                                        (0, jsx_runtime_1.jsx)("line", { x1: centerX, y1: q1Y, x2: centerX, y2: innerHeight, stroke: color, strokeWidth: 1.5, strokeDasharray: "4,2" })), (0, jsx_runtime_1.jsx)("line", { x1: centerX, y1: q3Y, x2: centerX, y2: whiskerHighY, stroke: color, strokeWidth: 1.5 }), (0, jsx_runtime_1.jsx)("line", { x1: x + boxWidth * 0.2, y1: whiskerHighY, x2: x + boxWidth * 0.8, y2: whiskerHighY, stroke: color, strokeWidth: 2 }), (0, jsx_runtime_1.jsx)("rect", { x: x, y: q3Y, width: boxWidth, height: Math.max(0, boxHeight), fill: color, fillOpacity: 0.5, stroke: color, strokeWidth: 2, rx: 2 }), (0, jsx_runtime_1.jsx)("line", { x1: x, y1: medianY, x2: x + boxWidth, y2: medianY, stroke: "#ffffff", strokeWidth: 1.5 }), stats.mean !== undefined && ((0, jsx_runtime_1.jsx)("polygon", { points: `
                      ${centerX},${toY(stats.mean) - 5}
                      ${centerX + 5},${toY(stats.mean)}
                      ${centerX},${toY(stats.mean) + 5}
                      ${centerX - 5},${toY(stats.mean)}
                    `, fill: "#ffffff", stroke: color, strokeWidth: 1 }))] }, `box-${i}`));
                            }), (0, jsx_runtime_1.jsx)(axis_1.AxisLeft, { scale: yScale, stroke: BarChart_1.CHART_STYLES.axisColor, tickStroke: BarChart_1.CHART_STYLES.axisColor, tickLabelProps: {
                                    fill: BarChart_1.CHART_STYLES.labelColor,
                                    fontSize: 10,
                                    fontFamily: BarChart_1.CHART_STYLES.fontFamily,
                                    textAnchor: 'end',
                                    dy: '0.33em',
                                    dx: -4,
                                }, tickValues: Array.from({ length: yMax - yMin + 1 }, (_, i) => yMin + i), tickFormat: (v) => {
                                    const val = Math.pow(10, v);
                                    return val.toExponential(0);
                                } }), (0, jsx_runtime_1.jsx)(text_1.Text, { x: -innerHeight / 2, y: -60, transform: "rotate(-90)", fontSize: 12, fontFamily: BarChart_1.CHART_STYLES.fontFamily, fill: BarChart_1.CHART_STYLES.labelColor, textAnchor: "middle", children: "Relative Deviation" }), (0, jsx_runtime_1.jsx)(axis_1.AxisBottom, { scale: xScale, top: innerHeight, stroke: BarChart_1.CHART_STYLES.axisColor, tickStroke: "transparent", hideTicks: true, tickLabelProps: {
                                    fill: 'transparent',
                                    fontSize: 0,
                                } })] }), (0, jsx_runtime_1.jsxs)(group_1.Group, { top: margin.top - 22, left: margin.left, children: [(0, jsx_runtime_1.jsxs)(group_1.Group, { left: 0, children: [(0, jsx_runtime_1.jsx)("circle", { cx: 6, cy: 6, r: 5, fill: F32_COLOR }), (0, jsx_runtime_1.jsx)(text_1.Text, { x: 16, y: 10, fontSize: 10, fontFamily: BarChart_1.CHART_STYLES.fontFamily, fill: BarChart_1.CHART_STYLES.labelColor, children: "Blaze f32" })] }), (0, jsx_runtime_1.jsxs)(group_1.Group, { left: 85, children: [(0, jsx_runtime_1.jsx)("circle", { cx: 6, cy: 6, r: 5, fill: F64_COLOR }), (0, jsx_runtime_1.jsx)(text_1.Text, { x: 16, y: 10, fontSize: 10, fontFamily: BarChart_1.CHART_STYLES.fontFamily, fill: BarChart_1.CHART_STYLES.labelColor, children: "Blaze f64" })] }), (0, jsx_runtime_1.jsxs)(group_1.Group, { left: 175, children: [(0, jsx_runtime_1.jsx)("circle", { cx: 6, cy: 6, r: 5, fill: COMPARE_COLOR }), (0, jsx_runtime_1.jsx)(text_1.Text, { x: 16, y: 10, fontSize: 10, fontFamily: BarChart_1.CHART_STYLES.fontFamily, fill: BarChart_1.CHART_STYLES.labelColor, children: "f32 vs f64" })] })] })] }), (0, jsx_runtime_1.jsx)("p", { style: {
                    marginTop: '1rem',
                    fontSize: '0.875rem',
                    color: BarChart_1.CHART_STYLES.captionColor,
                    lineHeight: 1.5,
                    maxWidth: width,
                    fontStyle: 'italic',
                } })] }));
}
