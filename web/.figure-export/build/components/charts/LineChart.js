'use client';
"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.default = LineChart;
const jsx_runtime_1 = require("react/jsx-runtime");
const react_1 = require("react");
const group_1 = require("@visx/group");
const shape_1 = require("@visx/shape");
const scale_1 = require("@visx/scale");
const axis_1 = require("@visx/axis");
const grid_1 = require("@visx/grid");
const text_1 = require("@visx/text");
const BarChart_1 = require("./BarChart");
const defaultMargin = { top: 60, right: 30, bottom: 60, left: 70 };
// Marker size
const MARKER_SIZE = 6;
/**
 * Render a marker shape at a given position
 */
function Marker({ x, y, shape, color, size = MARKER_SIZE }) {
    switch (shape) {
        case 'circle':
            return ((0, jsx_runtime_1.jsx)("circle", { cx: x, cy: y, r: size, fill: color }));
        case 'square':
            return ((0, jsx_runtime_1.jsx)("rect", { x: x - size, y: y - size, width: size * 2, height: size * 2, fill: color }));
        case 'triangle':
            const h = size * 1.7;
            const points = [
                [x, y - h * 0.6],
                [x - size, y + h * 0.4],
                [x + size, y + h * 0.4],
            ].map(p => p.join(',')).join(' ');
            return ((0, jsx_runtime_1.jsx)("polygon", { points: points, fill: color }));
        default:
            return null;
    }
}
function LineChart({ series, width = 600, height = 400, title, caption, xLabel, yLabel, xTickFormat = (v) => `${v}`, yTickFormat = (v) => `${v}`, showLegend = true, showGrid = true, showErrorBars = false, margin = defaultMargin, xDomain, yDomain, }) {
    // Calculate inner dimensions
    const innerWidth = width - margin.left - margin.right;
    const innerHeight = height - margin.top - margin.bottom;
    // Calculate domains from data if not provided
    const { computedXDomain, computedYDomain } = (0, react_1.useMemo)(() => {
        const allX = series.flatMap(s => s.data.map(d => d.x));
        const allY = series.flatMap(s => s.data.map(d => d.y + (showErrorBars && d.std ? d.std : 0)));
        const allYMin = series.flatMap(s => s.data.map(d => d.y - (showErrorBars && d.std ? d.std : 0)));
        const xMin = Math.min(...allX);
        const xMax = Math.max(...allX);
        const yMin = Math.min(0, ...allYMin);
        const yMax = Math.max(...allY) * 1.1;
        return {
            computedXDomain: xDomain || [xMin, xMax],
            computedYDomain: yDomain || [yMin, yMax],
        };
    }, [series, xDomain, yDomain, showErrorBars]);
    // Scales
    const xScale = (0, react_1.useMemo)(() => {
        return (0, scale_1.scaleLinear)({
            domain: computedXDomain,
            range: [0, innerWidth],
            nice: true,
        });
    }, [computedXDomain, innerWidth]);
    const yScale = (0, react_1.useMemo)(() => {
        return (0, scale_1.scaleLinear)({
            domain: computedYDomain,
            range: [innerHeight, 0],
            nice: true,
        });
    }, [computedYDomain, innerHeight]);
    // Legend layout - compute wrapping
    const legendItemWidths = (0, react_1.useMemo)(() => {
        // Estimate widths based on label lengths (rough approximation: 7px per char + marker space)
        return series.map(s => Math.max(70, s.label.length * 7 + 30));
    }, [series]);
    const legendLayout = (0, react_1.useMemo)(() => {
        const maxWidth = innerWidth;
        const items = [];
        let currentX = 0;
        let currentRow = 0;
        const rowHeight = 20;
        const itemGap = 16;
        series.forEach((s, i) => {
            const itemWidth = legendItemWidths[i];
            if (currentX + itemWidth > maxWidth && currentX > 0) {
                currentRow++;
                currentX = 0;
            }
            items.push({ seriesIdx: i, x: currentX, row: currentRow });
            currentX += itemWidth + itemGap;
        });
        return { items, totalRows: currentRow + 1, rowHeight };
    }, [series, legendItemWidths, innerWidth]);
    return ((0, jsx_runtime_1.jsxs)("div", { className: "line-chart-container", style: { width: '100%', maxWidth: width }, children: [(0, jsx_runtime_1.jsxs)("svg", { width: width, height: height, style: { overflow: 'visible' }, children: [title && ((0, jsx_runtime_1.jsx)(text_1.Text, { x: 0, y: 16, fontSize: 14, fontFamily: BarChart_1.CHART_STYLES.fontFamily, fill: BarChart_1.CHART_STYLES.labelColor, fontWeight: 700, children: title })), (0, jsx_runtime_1.jsxs)(group_1.Group, { left: margin.left, top: margin.top, children: [showGrid && ((0, jsx_runtime_1.jsx)(grid_1.GridRows, { scale: yScale, width: innerWidth, stroke: BarChart_1.CHART_STYLES.gridColor, strokeOpacity: 0.3, strokeDasharray: "3,3" })), series.map((s) => ((0, jsx_runtime_1.jsxs)(group_1.Group, { children: [(0, jsx_runtime_1.jsx)(shape_1.LinePath, { data: s.data, x: (d) => xScale(d.x), y: (d) => yScale(d.y), stroke: s.color, strokeWidth: s.isReference ? 1.5 : 2, strokeDasharray: s.isReference ? '6,4' : undefined, strokeOpacity: s.isReference ? 0.7 : 1 }), showErrorBars && !s.isReference && s.data.map((d, i) => {
                                        if (!d.std)
                                            return null;
                                        const x = xScale(d.x);
                                        const yTop = yScale(d.y + d.std);
                                        const yBottom = yScale(d.y - d.std);
                                        const capWidth = 4;
                                        return ((0, jsx_runtime_1.jsxs)(group_1.Group, { children: [(0, jsx_runtime_1.jsx)("line", { x1: x, y1: yTop, x2: x, y2: yBottom, stroke: s.color, strokeWidth: 1.5 }), (0, jsx_runtime_1.jsx)("line", { x1: x - capWidth, y1: yTop, x2: x + capWidth, y2: yTop, stroke: s.color, strokeWidth: 1.5 }), (0, jsx_runtime_1.jsx)("line", { x1: x - capWidth, y1: yBottom, x2: x + capWidth, y2: yBottom, stroke: s.color, strokeWidth: 1.5 })] }, `error-${s.id}-${i}`));
                                    }), !s.isReference && s.data.map((d, i) => ((0, jsx_runtime_1.jsx)(Marker, { x: xScale(d.x), y: yScale(d.y), shape: s.marker || 'circle', color: s.color }, `marker-${s.id}-${i}`)))] }, s.id))), (0, jsx_runtime_1.jsx)(axis_1.AxisLeft, { scale: yScale, stroke: BarChart_1.CHART_STYLES.axisColor, tickStroke: BarChart_1.CHART_STYLES.axisColor, tickFormat: (v) => yTickFormat(v), tickLabelProps: () => ({
                                    fill: BarChart_1.CHART_STYLES.labelColor,
                                    fontSize: 11,
                                    fontFamily: BarChart_1.CHART_STYLES.fontFamily,
                                    textAnchor: 'end',
                                    dx: -4,
                                    dy: 4,
                                }), numTicks: 5 }), (0, jsx_runtime_1.jsx)(axis_1.AxisBottom, { scale: xScale, top: innerHeight, stroke: BarChart_1.CHART_STYLES.axisColor, tickStroke: BarChart_1.CHART_STYLES.axisColor, tickFormat: (v) => xTickFormat(v), tickLabelProps: () => ({
                                    fill: BarChart_1.CHART_STYLES.labelColor,
                                    fontSize: 11,
                                    fontFamily: BarChart_1.CHART_STYLES.fontFamily,
                                    textAnchor: 'middle',
                                    dy: 4,
                                }), numTicks: 6 }), yLabel && ((0, jsx_runtime_1.jsx)(text_1.Text, { x: -innerHeight / 2, y: -50, fontSize: 12, fontFamily: BarChart_1.CHART_STYLES.fontFamily, fill: BarChart_1.CHART_STYLES.labelColor, textAnchor: "middle", transform: "rotate(-90)", children: yLabel })), xLabel && ((0, jsx_runtime_1.jsx)(text_1.Text, { x: innerWidth / 2, y: innerHeight + 45, fontSize: 12, fontFamily: BarChart_1.CHART_STYLES.fontFamily, fill: BarChart_1.CHART_STYLES.labelColor, textAnchor: "middle", children: xLabel }))] }), showLegend && ((0, jsx_runtime_1.jsx)(group_1.Group, { top: margin.top - 22, left: margin.left, children: legendLayout.items.map(({ seriesIdx, x, row }) => {
                            const s = series[seriesIdx];
                            const y = row * legendLayout.rowHeight;
                            if (s.isReference) {
                                return ((0, jsx_runtime_1.jsxs)(group_1.Group, { left: x, top: y, children: [(0, jsx_runtime_1.jsx)("line", { x1: 0, y1: 6, x2: 16, y2: 6, stroke: s.color, strokeWidth: 1.5, strokeDasharray: "4,3", strokeOpacity: 0.7 }), (0, jsx_runtime_1.jsx)(text_1.Text, { x: 20, y: 10, fontSize: 11, fontFamily: BarChart_1.CHART_STYLES.fontFamily, fill: BarChart_1.CHART_STYLES.labelColor, children: s.label })] }, `legend-${s.id}`));
                            }
                            return ((0, jsx_runtime_1.jsxs)(group_1.Group, { left: x, top: y, children: [(0, jsx_runtime_1.jsx)(Marker, { x: 6, y: 6, shape: s.marker || 'circle', color: s.color, size: 5 }), (0, jsx_runtime_1.jsx)(text_1.Text, { x: 18, y: 10, fontSize: 11, fontFamily: BarChart_1.CHART_STYLES.fontFamily, fill: BarChart_1.CHART_STYLES.labelColor, children: s.label })] }, `legend-${s.id}`));
                        }) }))] }), caption && ((0, jsx_runtime_1.jsx)("p", { style: {
                    marginTop: '1rem',
                    fontSize: '0.875rem',
                    color: BarChart_1.CHART_STYLES.captionColor,
                    lineHeight: 1.5,
                    fontFamily: BarChart_1.CHART_STYLES.fontFamily,
                    fontStyle: 'italic',
                    textAlign: 'left',
                }, children: caption }))] }));
}
