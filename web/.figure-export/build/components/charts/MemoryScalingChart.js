'use client';
"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.default = MemoryScalingChart;
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
    frozen: '#7183ad', // Slightly lighter blue-gray than mpb
    fit: '#888', // Gray for fit lines
};
// Marker size
const MARKER_SIZE = 5;
function MemoryScalingChart({ width = 650, height = 380, }) {
    const { data: benchmarkData, loading } = (0, use_benchmarks_1.useSeries5Benchmarks)();
    // Transform data for the log-log plot
    const { tmData, teData, fitCoeffs } = (0, react_1.useMemo)(() => {
        const resData = benchmarkData.resolution;
        const tm = {
            mpb: resData.mpb.TM.map(d => ({ x: d.value, y: d.memory_mb, std: d.memory_mb_std || 0 })),
            blaze: resData.blaze.TM.map(d => ({ x: d.value, y: d.memory_mb, std: d.memory_mb_std || 0 })),
        };
        const te = {
            mpb: resData.mpb.TE.map(d => ({ x: d.value, y: d.memory_mb, std: d.memory_mb_std || 0 })),
            blaze: resData.blaze.TE.map(d => ({ x: d.value, y: d.memory_mb, std: d.memory_mb_std || 0 })),
        };
        // Compute power law fit coefficients: y = a * x^b
        // Using least squares on log-transformed data
        const computeFit = (data) => {
            const n = data.length;
            if (n < 2)
                return { a: 1, b: 0 };
            const logX = data.map(d => Math.log(d.x));
            const logY = data.map(d => Math.log(d.y));
            const sumLogX = logX.reduce((a, b) => a + b, 0);
            const sumLogY = logY.reduce((a, b) => a + b, 0);
            const sumLogXLogY = logX.reduce((sum, lx, i) => sum + lx * logY[i], 0);
            const sumLogX2 = logX.reduce((sum, lx) => sum + lx * lx, 0);
            const b = (n * sumLogXLogY - sumLogX * sumLogY) / (n * sumLogX2 - sumLogX * sumLogX);
            const logA = (sumLogY - b * sumLogX) / n;
            const a = Math.exp(logA);
            return { a, b };
        };
        const fitCoeffs = {
            tm: {
                mpb: computeFit(tm.mpb),
                blaze: computeFit(tm.blaze),
            },
            te: {
                mpb: computeFit(te.mpb),
                blaze: computeFit(te.blaze),
            },
        };
        return { tmData: tm, teData: te, fitCoeffs };
    }, [benchmarkData]);
    if (loading) {
        return ((0, jsx_runtime_1.jsx)("div", { style: { width: width * 2 + 40, height, display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#888' }, children: "Loading benchmark data..." }));
    }
    const chartWidth = (width - 20) / 2;
    const margin = { top: 65, right: 20, bottom: 60, left: 70 };
    const innerWidth = chartWidth - margin.left - margin.right;
    const innerHeight = height - margin.top - margin.bottom;
    // Common domain for both plots - extend x domain to cover fit line
    const xDomain = [10, 160];
    const yDomain = [3, 300];
    // Generate power-of-10 tick values for a given domain
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
    // Generate fit line points
    const generateFitLine = (coeffs) => {
        const points = [];
        // Start from xDomain[0] to ensure line reaches y-axis
        for (let x = xDomain[0]; x <= xDomain[1]; x *= 1.08) {
            const y = coeffs.a * Math.pow(x, coeffs.b);
            // Only include points within the y domain
            if (y >= yDomain[0] && y <= yDomain[1]) {
                points.push({ x, y });
            }
        }
        return points;
    };
    // Find a good position for the label along the fit line
    const getFitLabelPosition = (fitLine, offset = 0) => {
        if (fitLine.length < 2)
            return { x: 0, y: 0 };
        // Use a point around 70% along the line
        const idx = Math.min(Math.floor(fitLine.length * 0.2), fitLine.length - 1);
        return { x: fitLine[idx].x, y: fitLine[idx].y * (1 + offset) };
    };
    // Generate O(N²) reference line
    const generateN2RefLine = (refX, refY) => {
        const points = [];
        for (let x = xDomain[0]; x <= xDomain[1]; x *= 1.1) {
            const y = refY * Math.pow(x / refX, 2);
            if (y >= yDomain[0] && y <= yDomain[1]) {
                points.push({ x, y });
            }
        }
        return points;
    };
    const renderChart = (data, fits, polarization) => {
        const mpbFitLine = generateFitLine(fits.mpb);
        const blazeFitLine = generateFitLine(fits.blaze);
        // Generate N² reference line (use Blaze data as reference point, scaled down)
        const midIdx = Math.floor(data.blaze.length / 2);
        const refX = data.blaze[midIdx]?.x || 64;
        const refY = (data.blaze[midIdx]?.y || 20) / 5; // Divide by X to push line down
        const n2RefLine = generateN2RefLine(refX, refY);
        // Get label positions for fit lines
        const mpbLabelPos = getFitLabelPosition(mpbFitLine, 0.15);
        const blazeLabelPos = getFitLabelPosition(blazeFitLine, -0.25);
        return ((0, jsx_runtime_1.jsxs)("svg", { width: chartWidth, height: height, style: { overflow: 'visible' }, children: [(0, jsx_runtime_1.jsx)(text_1.Text, { x: 0, y: 16, fontSize: 14, fontFamily: BarChart_1.CHART_STYLES.fontFamily, fill: BarChart_1.CHART_STYLES.labelColor, fontWeight: 700, children: `Memory Scaling (${polarization})` }), (0, jsx_runtime_1.jsxs)(group_1.Group, { left: margin.left, top: margin.top, children: [(0, jsx_runtime_1.jsx)(grid_1.GridRows, { scale: yScale, width: innerWidth, stroke: BarChart_1.CHART_STYLES.gridColor, strokeOpacity: 0.2, strokeDasharray: "3,3" }), (0, jsx_runtime_1.jsx)(grid_1.GridColumns, { scale: xScale, height: innerHeight, stroke: BarChart_1.CHART_STYLES.gridColor, strokeOpacity: 0.2, strokeDasharray: "3,3" }), n2RefLine.length > 0 && ((0, jsx_runtime_1.jsxs)(jsx_runtime_1.Fragment, { children: [(0, jsx_runtime_1.jsx)(shape_1.LinePath, { data: n2RefLine, x: (d) => xScale(d.x), y: (d) => yScale(d.y), stroke: "#999", strokeWidth: 1, strokeDasharray: "4,4", strokeOpacity: 0.7 }), (0, jsx_runtime_1.jsxs)("text", { x: xScale(n2RefLine[n2RefLine.length - 1].x) - 30, y: yScale(n2RefLine[n2RefLine.length - 1].y) + 40, fontSize: 11, fontFamily: BarChart_1.CHART_STYLES.fontFamily, fill: "#888", children: ["O(N", (0, jsx_runtime_1.jsx)("tspan", { fontSize: 8, dy: -3, children: "2" }), (0, jsx_runtime_1.jsx)("tspan", { fontSize: 11, dy: 3, children: ")" })] })] })), (0, jsx_runtime_1.jsx)(shape_1.LinePath, { data: mpbFitLine, x: (d) => xScale(d.x), y: (d) => yScale(d.y), stroke: COLORS.mpb, strokeWidth: 1.5, strokeDasharray: "5,5", strokeOpacity: 0.6 }), mpbLabelPos.x > 0 && ((0, jsx_runtime_1.jsxs)("text", { x: xScale(mpbLabelPos.x), y: yScale(mpbLabelPos.y), fontSize: 13, fontFamily: BarChart_1.CHART_STYLES.fontFamily, fill: COLORS.frozen, children: ["\u221D N", (0, jsx_runtime_1.jsx)("tspan", { fontSize: 9, dy: -5, children: fits.mpb.b.toFixed(2) })] })), (0, jsx_runtime_1.jsx)(shape_1.LinePath, { data: blazeFitLine, x: (d) => xScale(d.x), y: (d) => yScale(d.y), stroke: COLORS.blaze, strokeWidth: 1.5, strokeDasharray: "5,5", strokeOpacity: 0.6 }), blazeLabelPos.x > 0 && ((0, jsx_runtime_1.jsxs)("text", { x: xScale(blazeLabelPos.x), y: yScale(blazeLabelPos.y), fontSize: 13, fontFamily: BarChart_1.CHART_STYLES.fontFamily, fill: COLORS.blaze, children: ["\u221D N", (0, jsx_runtime_1.jsx)("tspan", { fontSize: 9, dy: -5, children: fits.blaze.b.toFixed(2) })] })), data.mpb.map((d, i) => d.std > 0 && ((0, jsx_runtime_1.jsxs)("g", { children: [(0, jsx_runtime_1.jsx)("line", { x1: xScale(d.x), y1: yScale(d.y - d.std), x2: xScale(d.x), y2: yScale(d.y + d.std), stroke: COLORS.mpb, strokeWidth: 1.5, strokeOpacity: 0.6 }), (0, jsx_runtime_1.jsx)("line", { x1: xScale(d.x) - 3, y1: yScale(d.y - d.std), x2: xScale(d.x) + 3, y2: yScale(d.y - d.std), stroke: COLORS.mpb, strokeWidth: 1.5, strokeOpacity: 0.6 }), (0, jsx_runtime_1.jsx)("line", { x1: xScale(d.x) - 3, y1: yScale(d.y + d.std), x2: xScale(d.x) + 3, y2: yScale(d.y + d.std), stroke: COLORS.mpb, strokeWidth: 1.5, strokeOpacity: 0.6 })] }, `mpb-err-${i}`))), (0, jsx_runtime_1.jsx)(shape_1.LinePath, { data: data.mpb, x: (d) => xScale(d.x), y: (d) => yScale(d.y), stroke: COLORS.mpb, strokeWidth: 2 }), data.mpb.map((d, i) => ((0, jsx_runtime_1.jsx)("circle", { cx: xScale(d.x), cy: yScale(d.y), r: MARKER_SIZE, fill: COLORS.mpb }, `mpb-${i}`))), data.blaze.map((d, i) => d.std > 0 && ((0, jsx_runtime_1.jsxs)("g", { children: [(0, jsx_runtime_1.jsx)("line", { x1: xScale(d.x), y1: yScale(d.y - d.std), x2: xScale(d.x), y2: yScale(d.y + d.std), stroke: COLORS.blaze, strokeWidth: 1.5, strokeOpacity: 0.6 }), (0, jsx_runtime_1.jsx)("line", { x1: xScale(d.x) - 3, y1: yScale(d.y - d.std), x2: xScale(d.x) + 3, y2: yScale(d.y - d.std), stroke: COLORS.blaze, strokeWidth: 1.5, strokeOpacity: 0.6 }), (0, jsx_runtime_1.jsx)("line", { x1: xScale(d.x) - 3, y1: yScale(d.y + d.std), x2: xScale(d.x) + 3, y2: yScale(d.y + d.std), stroke: COLORS.blaze, strokeWidth: 1.5, strokeOpacity: 0.6 })] }, `blaze-err-${i}`))), (0, jsx_runtime_1.jsx)(shape_1.LinePath, { data: data.blaze, x: (d) => xScale(d.x), y: (d) => yScale(d.y), stroke: COLORS.blaze, strokeWidth: 2 }), data.blaze.map((d, i) => ((0, jsx_runtime_1.jsx)("rect", { x: xScale(d.x) - MARKER_SIZE, y: yScale(d.y) - MARKER_SIZE, width: MARKER_SIZE * 2, height: MARKER_SIZE * 2, fill: COLORS.blaze }, `blaze-${i}`))), (0, jsx_runtime_1.jsx)(axis_1.AxisLeft, { scale: yScale, stroke: BarChart_1.CHART_STYLES.axisColor, tickStroke: BarChart_1.CHART_STYLES.axisColor, tickFormat: (v) => `${v}`, tickLabelProps: () => ({
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
                            }), tickValues: xTicks }), (0, jsx_runtime_1.jsx)(text_1.Text, { x: -innerHeight / 2, y: -55, transform: "rotate(-90)", fontSize: 11, fontFamily: BarChart_1.CHART_STYLES.fontFamily, fill: BarChart_1.CHART_STYLES.labelColor, textAnchor: "middle", children: "Peak Memory (MB)" }), (0, jsx_runtime_1.jsx)(text_1.Text, { x: innerWidth / 2, y: innerHeight + 45, fontSize: 11, fontFamily: BarChart_1.CHART_STYLES.fontFamily, fill: BarChart_1.CHART_STYLES.labelColor, textAnchor: "middle", children: "Resolution (N)" })] }), (0, jsx_runtime_1.jsxs)(group_1.Group, { top: margin.top - 25, left: margin.left, children: [(0, jsx_runtime_1.jsx)("circle", { cx: 6, cy: 0, r: 5, fill: COLORS.mpb }), (0, jsx_runtime_1.jsx)(text_1.Text, { x: 16, y: 4, fontSize: 11, fill: BarChart_1.CHART_STYLES.labelColor, fontFamily: BarChart_1.CHART_STYLES.fontFamily, children: "MPB" }), (0, jsx_runtime_1.jsx)("rect", { x: 60, y: -5, width: 10, height: 10, fill: COLORS.blaze }), (0, jsx_runtime_1.jsx)(text_1.Text, { x: 76, y: 4, fontSize: 11, fill: BarChart_1.CHART_STYLES.labelColor, fontFamily: BarChart_1.CHART_STYLES.fontFamily, children: "Blaze2D" })] })] }));
    };
    return ((0, jsx_runtime_1.jsxs)("div", { style: { display: 'flex', flexDirection: 'column', gap: '0.5rem' }, children: [(0, jsx_runtime_1.jsxs)("div", { style: { display: 'flex', gap: '20px', justifyContent: 'center' }, children: [renderChart(tmData, fitCoeffs.tm, 'TM'), renderChart(teData, fitCoeffs.te, 'TE')] }), (0, jsx_runtime_1.jsx)("p", { style: {
                    marginTop: '1rem',
                    fontSize: '0.875rem',
                    color: '#888',
                    lineHeight: 1.5,
                    fontFamily: 'var(--font-sans), system-ui, sans-serif',
                    fontStyle: 'italic',
                    textAlign: 'left',
                }, children: "Dashed lines show power-law fits to the measured data; the lower-right dashed line indicates an O(N\u00B2) reference scaling." })] }));
}
