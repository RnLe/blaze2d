'use client';
"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.default = IterationBarChart;
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
function IterationBarChart({ width = 650, height = 380, }) {
    const { data: benchmarkData, loading } = (0, use_benchmarks_1.useSeries4Benchmarks)();
    // Transform data for the bar charts
    const { tmData, teData, kIndices, highSymmetryPoints } = (0, react_1.useMemo)(() => {
        const kPerSeg = benchmarkData.parameters.k_points_per_segment;
        // High symmetry points for square lattice: Γ→X→M→Γ
        const hsPoints = [
            { index: 0, label: 'Γ' },
            { index: kPerSeg, label: 'X' },
            { index: 2 * kPerSeg, label: 'M' },
            { index: 3 * kPerSeg, label: 'Γ' },
        ];
        const tm = {
            mpb: benchmarkData.TM.mpb?.k_points.map(kp => ({
                k_index: kp.k_index,
                iterations: kp.iterations,
            })) || [],
            blaze: benchmarkData.TM.blaze?.k_points.map(kp => ({
                k_index: kp.k_index,
                iterations: kp.iterations,
            })) || [],
        };
        const te = {
            mpb: benchmarkData.TE.mpb?.k_points.map(kp => ({
                k_index: kp.k_index,
                iterations: kp.iterations,
            })) || [],
            blaze: benchmarkData.TE.blaze?.k_points.map(kp => ({
                k_index: kp.k_index,
                iterations: kp.iterations,
            })) || [],
        };
        // Get all k-indices (use whichever solver has more points)
        const allIndices = tm.mpb.length >= tm.blaze.length
            ? tm.mpb.map(d => d.k_index)
            : tm.blaze.map(d => d.k_index);
        return { tmData: tm, teData: te, kIndices: allIndices, highSymmetryPoints: hsPoints };
    }, [benchmarkData]);
    if (loading) {
        return ((0, jsx_runtime_1.jsx)("div", { style: { width: width * 2 + 40, height, display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#888' }, children: "Loading benchmark data..." }));
    }
    const chartWidth = (width - 20) / 2;
    const margin = { top: 65, right: 20, bottom: 60, left: 70 };
    const innerWidth = chartWidth - margin.left - margin.right;
    const innerHeight = height - margin.top - margin.bottom;
    const renderChart = (data, polarization) => {
        const maxIter = Math.max(...data.mpb.map(d => d.iterations), ...data.blaze.map(d => d.iterations));
        const xScale = (0, scale_1.scaleBand)({
            domain: kIndices,
            range: [0, innerWidth],
            padding: 0.05,
        });
        const yScale = (0, scale_1.scaleLinear)({
            domain: [0, maxIter * 1.1],
            range: [innerHeight, 0],
        });
        const barWidth = Math.max(xScale.bandwidth() / 2 - 1, 1);
        // Create lookup maps for quick access
        const mpbMap = new Map(data.mpb.map(d => [d.k_index, d.iterations]));
        const blazeMap = new Map(data.blaze.map(d => [d.k_index, d.iterations]));
        // Show every Nth tick on x-axis
        const tickStep = kIndices.length > 40 ? 10 : 5;
        const xTickValues = kIndices.filter((_, i) => i % tickStep === 0);
        return ((0, jsx_runtime_1.jsxs)("svg", { width: chartWidth, height: height, style: { overflow: 'visible' }, children: [(0, jsx_runtime_1.jsx)(text_1.Text, { x: 0, y: 16, fontSize: 14, fontFamily: BarChart_1.CHART_STYLES.fontFamily, fill: BarChart_1.CHART_STYLES.labelColor, fontWeight: 700, children: `Iterations per K-Point (${polarization})` }), (0, jsx_runtime_1.jsxs)(group_1.Group, { left: margin.left, top: margin.top, children: [(0, jsx_runtime_1.jsx)(grid_1.GridRows, { scale: yScale, width: innerWidth, stroke: BarChart_1.CHART_STYLES.gridColor, strokeOpacity: 0.4, strokeDasharray: "3,3" }), highSymmetryPoints.map((hs) => {
                            const xPos = xScale(hs.index);
                            if (xPos === undefined)
                                return null;
                            return ((0, jsx_runtime_1.jsxs)("g", { children: [(0, jsx_runtime_1.jsx)("line", { x1: xPos + xScale.bandwidth() / 2, y1: 0, x2: xPos + xScale.bandwidth() / 2, y2: innerHeight, stroke: "#666", strokeWidth: 1, strokeDasharray: "2,2", strokeOpacity: 0.5 }), (0, jsx_runtime_1.jsx)(text_1.Text, { x: xPos + xScale.bandwidth() / 2, y: -8, fontSize: 11, fontFamily: BarChart_1.CHART_STYLES.fontFamily, fill: BarChart_1.CHART_STYLES.labelColor, textAnchor: "middle", children: hs.label })] }, hs.label + hs.index));
                        }), kIndices.map((kIdx) => {
                            const xPos = xScale(kIdx) || 0;
                            const mpbIter = mpbMap.get(kIdx) || 0;
                            const blazeIter = blazeMap.get(kIdx) || 0;
                            return ((0, jsx_runtime_1.jsxs)("g", { children: [mpbIter > 0 && ((0, jsx_runtime_1.jsx)(shape_1.Bar, { x: xPos, y: yScale(mpbIter), width: barWidth, height: innerHeight - yScale(mpbIter), fill: COLORS.mpb })), blazeIter > 0 && ((0, jsx_runtime_1.jsx)(shape_1.Bar, { x: xPos + barWidth, y: yScale(blazeIter), width: barWidth, height: innerHeight - yScale(blazeIter), fill: COLORS.blaze }))] }, kIdx));
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
                            }) }), (0, jsx_runtime_1.jsx)(text_1.Text, { x: -innerHeight / 2, y: -55, transform: "rotate(-90)", fontSize: 11, fontFamily: BarChart_1.CHART_STYLES.fontFamily, fill: BarChart_1.CHART_STYLES.labelColor, textAnchor: "middle", children: "Iterations to Convergence" }), (0, jsx_runtime_1.jsx)(text_1.Text, { x: innerWidth / 2, y: innerHeight + 45, fontSize: 11, fontFamily: BarChart_1.CHART_STYLES.fontFamily, fill: BarChart_1.CHART_STYLES.labelColor, textAnchor: "middle", children: "K-Point Index" })] }), (0, jsx_runtime_1.jsxs)(group_1.Group, { top: margin.top - 25, left: margin.left, children: [(0, jsx_runtime_1.jsx)("rect", { x: 0, y: -6, width: 12, height: 12, fill: COLORS.mpb }), (0, jsx_runtime_1.jsx)(text_1.Text, { x: 18, y: 4, fontSize: 11, fill: BarChart_1.CHART_STYLES.labelColor, fontFamily: BarChart_1.CHART_STYLES.fontFamily, children: "MPB" }), (0, jsx_runtime_1.jsx)("rect", { x: 60, y: -6, width: 12, height: 12, fill: COLORS.blaze }), (0, jsx_runtime_1.jsx)(text_1.Text, { x: 78, y: 4, fontSize: 11, fill: BarChart_1.CHART_STYLES.labelColor, fontFamily: BarChart_1.CHART_STYLES.fontFamily, children: "Blaze2D" })] })] }));
    };
    return ((0, jsx_runtime_1.jsxs)("div", { style: { display: 'flex', gap: '20px', flexWrap: 'wrap', justifyContent: 'center' }, children: [renderChart(tmData, 'TM'), renderChart(teData, 'TE')] }));
}
