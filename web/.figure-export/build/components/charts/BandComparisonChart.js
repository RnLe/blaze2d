'use client';
"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.default = BandComparisonChart;
const jsx_runtime_1 = require("react/jsx-runtime");
const react_1 = require("react");
const group_1 = require("@visx/group");
const shape_1 = require("@visx/shape");
const scale_1 = require("@visx/scale");
const axis_1 = require("@visx/axis");
const grid_1 = require("@visx/grid");
const text_1 = require("@visx/text");
const swr_shim_1 = __importDefault(require("../../lib/swr-shim"));
const BarChart_1 = require("./BarChart");
const paths_1 = require("../../lib/paths");
const defaultMargin = { top: 60, right: 30, bottom: 60, left: 70 };
// Colors
const MPB_COLOR = '#435f9d'; // Reference blue
const F64_COLOR = '#4caf50'; // Full precision - green
const F32_COLOR = '#bbc1cb'; // Mixed precision - gray
// Marker sizes
const F64_MARKER_SIZE = 3;
const F32_MARKER_SIZE = 2.5;
const fetcher = (url) => fetch(url).then(res => res.json());
/**
 * Render a circle marker
 */
function CircleMarker({ x, y, size, color, opacity = 1 }) {
    return ((0, jsx_runtime_1.jsx)("circle", { cx: x, cy: y, r: size, fill: color, fillOpacity: opacity }));
}
/**
 * Render a triangle marker (pointing up)
 */
function TriangleMarker({ x, y, size, color, opacity = 1 }) {
    const h = size * 1.7;
    const points = [
        [x, y - h * 0.6],
        [x - size, y + h * 0.4],
        [x + size, y + h * 0.4],
    ].map(p => p.join(',')).join(' ');
    return ((0, jsx_runtime_1.jsx)("polygon", { points: points, fill: color, fillOpacity: opacity }));
}
function BandComparisonChart({ width = 600, height = 400, polarization = 'TM', }) {
    const { data, error, isLoading } = (0, swr_shim_1.default)((0, paths_1.getAssetPath)('/data/benchmarks/series6-accuracy.json'), fetcher);
    const margin = defaultMargin;
    const innerWidth = width - margin.left - margin.right;
    const innerHeight = height - margin.top - margin.bottom;
    // Extract data for the selected polarization
    const polData = data?.[polarization];
    const mpbData = polData?.mpb || [];
    const f32Data = polData?.blaze_f32 || [];
    const f64Data = polData?.blaze_f64 || [];
    const numBands = data?.parameters?.num_bands || 8;
    // Compute scales
    const { xScale, yScale } = (0, react_1.useMemo)(() => {
        if (!mpbData.length) {
            return {
                xScale: (0, scale_1.scaleLinear)({ domain: [0, 1], range: [0, innerWidth] }),
                yScale: (0, scale_1.scaleLinear)({ domain: [0, 1], range: [innerHeight, 0] }),
            };
        }
        const xMax = Math.max(...mpbData.map(d => d.k_distance));
        const allFreqs = mpbData.flatMap(d => d.frequencies);
        const yMax = Math.max(...allFreqs) * 1.05;
        return {
            xScale: (0, scale_1.scaleLinear)({
                domain: [0, xMax],
                range: [0, innerWidth],
            }),
            yScale: (0, scale_1.scaleLinear)({
                domain: [0, yMax],
                range: [innerHeight, 0],
            }),
        };
    }, [mpbData, innerWidth, innerHeight]);
    // Find high-symmetry points (Γ, X, M, Γ for square lattice)
    const highSymmetryPoints = (0, react_1.useMemo)(() => {
        if (!mpbData.length)
            return [];
        const points = [];
        // First point is Γ
        points.push({ distance: 0, label: 'Γ' });
        // Find X (k = 0.5, 0)
        const xPoint = mpbData.find((_, i) => {
            if (i === 0)
                return false;
            const prev = mpbData[i - 1];
            const curr = mpbData[i];
            // X point is around k_distance = 0.5
            return prev.k_distance < 0.5 && curr.k_distance >= 0.5;
        });
        if (xPoint) {
            const xIdx = mpbData.findIndex(d => d.k_distance >= 0.5);
            if (xIdx > 0) {
                points.push({ distance: mpbData[xIdx].k_distance, label: 'X' });
            }
        }
        // Find M (k = 0.5, 0.5) - around k_distance = 1.0
        const mPoint = mpbData.find((_, i) => {
            if (i === 0)
                return false;
            const prev = mpbData[i - 1];
            const curr = mpbData[i];
            return prev.k_distance < 1.0 && curr.k_distance >= 1.0;
        });
        if (mPoint) {
            const mIdx = mpbData.findIndex(d => d.k_distance >= 1.0);
            if (mIdx > 0) {
                points.push({ distance: mpbData[mIdx].k_distance, label: 'M' });
            }
        }
        // Last point is Γ again
        const lastDist = mpbData[mpbData.length - 1]?.k_distance;
        if (lastDist) {
            points.push({ distance: lastDist, label: 'Γ' });
        }
        return points;
    }, [mpbData]);
    if (isLoading) {
        return ((0, jsx_runtime_1.jsx)("div", { style: { width, height, display: 'flex', alignItems: 'center', justifyContent: 'center', color: BarChart_1.CHART_STYLES.labelColor }, children: "Loading..." }));
    }
    if (error || !data) {
        return ((0, jsx_runtime_1.jsx)("div", { style: { width, height, display: 'flex', alignItems: 'center', justifyContent: 'center', color: BarChart_1.CHART_STYLES.labelColor }, children: "Error loading data" }));
    }
    return ((0, jsx_runtime_1.jsxs)("div", { className: "band-comparison-container", style: { width: '100%', maxWidth: width }, children: [(0, jsx_runtime_1.jsxs)("svg", { width: width, height: height, style: { overflow: 'visible' }, children: [(0, jsx_runtime_1.jsx)(text_1.Text, { x: 0, y: 16, fontSize: 14, fontFamily: BarChart_1.CHART_STYLES.fontFamily, fill: BarChart_1.CHART_STYLES.labelColor, fontWeight: 700, children: `${polarization} Band Structure: MPB vs Blaze` }), (0, jsx_runtime_1.jsxs)(group_1.Group, { left: margin.left, top: margin.top, children: [(0, jsx_runtime_1.jsx)(grid_1.GridRows, { scale: yScale, width: innerWidth, stroke: BarChart_1.CHART_STYLES.gridColor, strokeOpacity: 0.3, strokeDasharray: "3,3" }), highSymmetryPoints.map((pt, i) => ((0, jsx_runtime_1.jsx)("line", { x1: xScale(pt.distance), y1: 0, x2: xScale(pt.distance), y2: innerHeight, stroke: BarChart_1.CHART_STYLES.gridColor, strokeOpacity: 0.5, strokeDasharray: "2,2" }, `vline-${i}`))), Array.from({ length: numBands }).map((_, bandIdx) => {
                                const bandData = mpbData.map(kp => ({
                                    x: kp.k_distance,
                                    y: kp.frequencies[bandIdx] || 0,
                                }));
                                return ((0, jsx_runtime_1.jsx)(shape_1.LinePath, { data: bandData, x: d => xScale(d.x), y: d => yScale(d.y), stroke: MPB_COLOR, strokeWidth: 1.5 }, `mpb-band-${bandIdx}`));
                            }), f64Data.map((kp, kIdx) => ((0, jsx_runtime_1.jsx)(group_1.Group, { children: kp.frequencies.map((freq, bandIdx) => ((0, jsx_runtime_1.jsx)(CircleMarker, { x: xScale(kp.k_distance), y: yScale(freq), size: F64_MARKER_SIZE, color: F64_COLOR, opacity: 0.7 }, `f64-${kIdx}-${bandIdx}`))) }, `f64-kp-${kIdx}`))), f32Data.map((kp, kIdx) => ((0, jsx_runtime_1.jsx)(group_1.Group, { children: kp.frequencies.map((freq, bandIdx) => ((0, jsx_runtime_1.jsx)(TriangleMarker, { x: xScale(kp.k_distance), y: yScale(freq), size: F32_MARKER_SIZE, color: F32_COLOR, opacity: 1 }, `f32-${kIdx}-${bandIdx}`))) }, `f32-kp-${kIdx}`))), (0, jsx_runtime_1.jsx)(axis_1.AxisLeft, { scale: yScale, stroke: BarChart_1.CHART_STYLES.axisColor, tickStroke: BarChart_1.CHART_STYLES.axisColor, tickFormat: (v) => v.toFixed(1), tickLabelProps: () => ({
                                    fill: BarChart_1.CHART_STYLES.labelColor,
                                    fontSize: 11,
                                    fontFamily: BarChart_1.CHART_STYLES.fontFamily,
                                    textAnchor: 'end',
                                    dx: -4,
                                    dy: 4,
                                }), numTicks: 6 }), (0, jsx_runtime_1.jsx)(axis_1.AxisBottom, { scale: xScale, top: innerHeight, stroke: BarChart_1.CHART_STYLES.axisColor, tickStroke: BarChart_1.CHART_STYLES.axisColor, tickValues: highSymmetryPoints.map(p => p.distance), tickFormat: (v) => {
                                    const pt = highSymmetryPoints.find(p => Math.abs(p.distance - v) < 0.01);
                                    return pt?.label || '';
                                }, tickLabelProps: () => ({
                                    fill: BarChart_1.CHART_STYLES.labelColor,
                                    fontSize: 11,
                                    fontFamily: BarChart_1.CHART_STYLES.fontFamily,
                                    textAnchor: 'middle',
                                    dy: 4,
                                }) }), (0, jsx_runtime_1.jsx)(text_1.Text, { x: -innerHeight / 2, y: -50, fontSize: 12, fontFamily: BarChart_1.CHART_STYLES.fontFamily, fill: BarChart_1.CHART_STYLES.labelColor, textAnchor: "middle", transform: "rotate(-90)", children: "\u03C9a / 2\u03C0c" }), (0, jsx_runtime_1.jsx)(text_1.Text, { x: innerWidth / 2, y: innerHeight + 45, fontSize: 12, fontFamily: BarChart_1.CHART_STYLES.fontFamily, fill: BarChart_1.CHART_STYLES.labelColor, textAnchor: "middle", children: "k-path" })] }), (0, jsx_runtime_1.jsxs)(group_1.Group, { top: margin.top - 22, left: margin.left, children: [(0, jsx_runtime_1.jsxs)(group_1.Group, { left: 0, children: [(0, jsx_runtime_1.jsx)("line", { x1: 0, y1: 6, x2: 20, y2: 6, stroke: MPB_COLOR, strokeWidth: 2 }), (0, jsx_runtime_1.jsx)(text_1.Text, { x: 26, y: 10, fontSize: 11, fontFamily: BarChart_1.CHART_STYLES.fontFamily, fill: BarChart_1.CHART_STYLES.labelColor, children: "MPB (reference)" })] }), (0, jsx_runtime_1.jsxs)(group_1.Group, { left: 130, children: [(0, jsx_runtime_1.jsx)(CircleMarker, { x: 6, y: 6, size: 4, color: F64_COLOR }), (0, jsx_runtime_1.jsx)(text_1.Text, { x: 18, y: 10, fontSize: 11, fontFamily: BarChart_1.CHART_STYLES.fontFamily, fill: BarChart_1.CHART_STYLES.labelColor, children: "Blaze f64" })] }), (0, jsx_runtime_1.jsxs)(group_1.Group, { left: 220, children: [(0, jsx_runtime_1.jsx)(TriangleMarker, { x: 6, y: 6, size: 3, color: F32_COLOR }), (0, jsx_runtime_1.jsx)(text_1.Text, { x: 18, y: 10, fontSize: 11, fontFamily: BarChart_1.CHART_STYLES.fontFamily, fill: BarChart_1.CHART_STYLES.labelColor, children: "Blaze f32" })] })] })] }), (0, jsx_runtime_1.jsx)("p", { style: {
                    marginTop: '1rem',
                    fontSize: '0.875rem',
                    color: BarChart_1.CHART_STYLES.captionColor,
                    lineHeight: 1.5,
                    fontFamily: BarChart_1.CHART_STYLES.fontFamily,
                    fontStyle: 'italic',
                    textAlign: 'left',
                } })] }));
}
