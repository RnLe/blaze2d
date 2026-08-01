'use client';
"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.default = BoxPlot;
const jsx_runtime_1 = require("react/jsx-runtime");
const react_1 = require("react");
const group_1 = require("@visx/group");
const scale_1 = require("@visx/scale");
const axis_1 = require("@visx/axis");
const grid_1 = require("@visx/grid");
const text_1 = require("@visx/text");
const BarChart_1 = require("./BarChart");
const defaultMargin = { top: 60, right: 30, bottom: 60, left: 70 };
function BoxPlot({ data, width = 600, height = 400, title, caption, yLabel, yTickFormat = (v) => v.toExponential(1), logScale = false, defaultBoxColor = '#3b82f6', margin = defaultMargin, boxWidth = 0.6, }) {
    // Calculate inner dimensions
    const innerWidth = width - margin.left - margin.right;
    const innerHeight = height - margin.top - margin.bottom;
    // X scale (categorical)
    const xScale = (0, react_1.useMemo)(() => {
        return (0, scale_1.scaleBand)({
            domain: data.map(d => d.label),
            range: [0, innerWidth],
            padding: 0.3,
        });
    }, [data, innerWidth]);
    // Y scale (linear or log)
    const yScale = (0, react_1.useMemo)(() => {
        const allValues = data.flatMap(d => [d.whiskerLow, d.whiskerHigh]);
        const minVal = Math.min(...allValues);
        const maxVal = Math.max(...allValues);
        // Add some padding
        const padding = (maxVal - minVal) * 0.1;
        if (logScale) {
            // For log scale, ensure minimum is positive
            const logMin = minVal > 0 ? minVal * 0.5 : 1e-10;
            const logMax = maxVal * 2;
            return (0, scale_1.scaleLinear)({
                domain: [Math.log10(logMin), Math.log10(logMax)],
                range: [innerHeight, 0],
                nice: true,
            });
        }
        return (0, scale_1.scaleLinear)({
            domain: [Math.max(0, minVal - padding), maxVal + padding],
            range: [innerHeight, 0],
            nice: true,
        });
    }, [data, innerHeight, logScale]);
    // Helper to convert value to y position
    const toY = (value) => {
        if (logScale) {
            return yScale(Math.log10(Math.max(value, 1e-10)));
        }
        return yScale(value);
    };
    // Box visual width
    const actualBoxWidth = xScale.bandwidth() * boxWidth;
    return ((0, jsx_runtime_1.jsxs)("div", { className: "boxplot-container", style: { width: '100%', maxWidth: width }, children: [(0, jsx_runtime_1.jsxs)("svg", { width: width, height: height, style: { overflow: 'visible' }, children: [title && ((0, jsx_runtime_1.jsx)(text_1.Text, { x: 0, y: 16, fontSize: 14, fontFamily: BarChart_1.CHART_STYLES.fontFamily, fill: BarChart_1.CHART_STYLES.labelColor, fontWeight: 700, children: title })), (0, jsx_runtime_1.jsxs)(group_1.Group, { left: margin.left, top: margin.top, children: [(0, jsx_runtime_1.jsx)(grid_1.GridRows, { scale: yScale, width: innerWidth, stroke: BarChart_1.CHART_STYLES.gridColor, strokeOpacity: 0.3, strokeDasharray: "3,3" }), data.map((d, i) => {
                                const x = (xScale(d.label) || 0) + (xScale.bandwidth() - actualBoxWidth) / 2;
                                const color = d.color || defaultBoxColor;
                                const whiskerLowY = toY(d.whiskerLow);
                                const q1Y = toY(d.q1);
                                const medianY = toY(d.median);
                                const q3Y = toY(d.q3);
                                const whiskerHighY = toY(d.whiskerHigh);
                                const boxHeight = q1Y - q3Y;
                                const centerX = x + actualBoxWidth / 2;
                                return ((0, jsx_runtime_1.jsxs)(group_1.Group, { children: [(0, jsx_runtime_1.jsx)("line", { x1: centerX, y1: whiskerLowY, x2: centerX, y2: whiskerHighY, stroke: color, strokeWidth: 1.5 }), (0, jsx_runtime_1.jsx)("line", { x1: x + actualBoxWidth * 0.25, y1: whiskerLowY, x2: x + actualBoxWidth * 0.75, y2: whiskerLowY, stroke: color, strokeWidth: 2 }), (0, jsx_runtime_1.jsx)("line", { x1: x + actualBoxWidth * 0.25, y1: whiskerHighY, x2: x + actualBoxWidth * 0.75, y2: whiskerHighY, stroke: color, strokeWidth: 2 }), (0, jsx_runtime_1.jsx)("rect", { x: x, y: q3Y, width: actualBoxWidth, height: boxHeight, fill: color, fillOpacity: 0.5, stroke: color, strokeWidth: 2, rx: 2 }), (0, jsx_runtime_1.jsx)("line", { x1: x, y1: medianY, x2: x + actualBoxWidth, y2: medianY, stroke: "#ffffff", strokeWidth: 2.5 }), d.mean !== undefined && ((0, jsx_runtime_1.jsx)("polygon", { points: `
                      ${centerX},${toY(d.mean) - 5}
                      ${centerX + 5},${toY(d.mean)}
                      ${centerX},${toY(d.mean) + 5}
                      ${centerX - 5},${toY(d.mean)}
                    `, fill: "#ffffff", stroke: color, strokeWidth: 1 }))] }, `box-${i}`));
                            }), (0, jsx_runtime_1.jsx)(axis_1.AxisLeft, { scale: yScale, stroke: BarChart_1.CHART_STYLES.axisColor, tickStroke: BarChart_1.CHART_STYLES.axisColor, tickLabelProps: {
                                    fill: BarChart_1.CHART_STYLES.labelColor,
                                    fontSize: 11,
                                    fontFamily: BarChart_1.CHART_STYLES.fontFamily,
                                    textAnchor: 'end',
                                    dy: '0.33em',
                                    dx: -4,
                                }, tickFormat: (v) => {
                                    if (logScale) {
                                        return yTickFormat(Math.pow(10, v));
                                    }
                                    return yTickFormat(v);
                                }, numTicks: 6 }), yLabel && ((0, jsx_runtime_1.jsx)(text_1.Text, { x: -innerHeight / 2, y: -50, transform: "rotate(-90)", fontSize: 12, fontFamily: BarChart_1.CHART_STYLES.fontFamily, fill: BarChart_1.CHART_STYLES.labelColor, textAnchor: "middle", children: yLabel })), (0, jsx_runtime_1.jsx)(axis_1.AxisBottom, { scale: xScale, top: innerHeight, stroke: BarChart_1.CHART_STYLES.axisColor, tickStroke: BarChart_1.CHART_STYLES.axisColor, tickLabelProps: {
                                    fill: BarChart_1.CHART_STYLES.labelColor,
                                    fontSize: 11,
                                    fontFamily: BarChart_1.CHART_STYLES.fontFamily,
                                    textAnchor: 'middle',
                                    dy: '0.5em',
                                } })] })] }), caption && ((0, jsx_runtime_1.jsx)("p", { style: {
                    marginTop: '1rem',
                    fontSize: '0.875rem',
                    color: BarChart_1.CHART_STYLES.captionColor,
                    lineHeight: 1.5,
                    maxWidth: width,
                    fontStyle: 'italic',
                }, children: caption }))] }));
}
