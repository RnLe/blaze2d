'use client';
"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.CHART_STYLES = void 0;
exports.default = BarChart;
const jsx_runtime_1 = require("react/jsx-runtime");
const react_1 = require("react");
const group_1 = require("@visx/group");
const shape_1 = require("@visx/shape");
const scale_1 = require("@visx/scale");
const axis_1 = require("@visx/axis");
const grid_1 = require("@visx/grid");
const text_1 = require("@visx/text");
const defaultMargin = { top: 60, right: 30, bottom: 100, left: 70 };
// Shared chart styling constants - export for use in other chart components
exports.CHART_STYLES = {
    labelColor: '#ffffff',
    gridColor: '#333',
    axisColor: '#555',
    captionColor: '#888',
    barBorderColor: '#eaf1fe',
    fontFamily: 'var(--font-sans), system-ui, sans-serif',
};
function BarChart({ data, width = 600, height = 400, title, caption, yLabel, yTickFormat = (v) => `${v}`, labelAngle = -45, showValues = true, valueFormat = (v) => v.toFixed(0), showStd = true, defaultBarColor = '#3b82f6', margin = defaultMargin, groupColors, categoryBrackets, showCategoryBrackets = false, bracketOffset = 50, }) {
    // Calculate inner dimensions
    const innerWidth = width - margin.left - margin.right;
    const innerHeight = height - margin.top - margin.bottom;
    // Determine if we have groups
    const hasGroups = data.some(d => d.group);
    const groups = hasGroups
        ? [...new Set(data.map(d => d.group).filter(Boolean))]
        : [];
    // Get unique data point identifiers (use id if present, otherwise label)
    const getDataId = (d) => d.id || d.label;
    // Get unique position IDs in order of first appearance
    // For grouped charts, multiple data points share the same position ID (one per group)
    const positionIds = (0, react_1.useMemo)(() => {
        const seen = new Set();
        const ids = [];
        data.forEach(d => {
            const id = getDataId(d);
            if (!seen.has(id)) {
                seen.add(id);
                ids.push(id);
            }
        });
        return ids;
    }, [data]);
    // Scales
    const xScale = (0, react_1.useMemo)(() => {
        if (hasGroups) {
            return (0, scale_1.scaleBand)({
                domain: positionIds,
                range: [0, innerWidth],
                padding: 0.2,
            });
        }
        return (0, scale_1.scaleBand)({
            domain: data.map(d => getDataId(d)),
            range: [0, innerWidth],
            padding: 0.3,
        });
    }, [data, positionIds, innerWidth, hasGroups]);
    const groupScale = (0, react_1.useMemo)(() => {
        if (!hasGroups)
            return null;
        return (0, scale_1.scaleBand)({
            domain: groups,
            range: [0, xScale.bandwidth()],
            padding: 0.1,
        });
    }, [groups, xScale, hasGroups]);
    const maxValue = (0, react_1.useMemo)(() => {
        return Math.max(...data.map(d => d.value + (d.std || 0))) * 1.15;
    }, [data]);
    const yScale = (0, react_1.useMemo)(() => {
        return (0, scale_1.scaleLinear)({
            domain: [0, maxValue],
            range: [innerHeight, 0],
            nice: true,
        });
    }, [maxValue, innerHeight]);
    // Color scale for groups
    const colorScale = (0, react_1.useMemo)(() => {
        if (groupColors) {
            return (group) => groupColors[group] || defaultBarColor;
        }
        const defaultGroupColors = ['#ef4444', '#3b82f6', '#22c55e', '#f59e0b'];
        return (0, scale_1.scaleOrdinal)({
            domain: groups,
            range: defaultGroupColors,
        });
    }, [groups, groupColors, defaultBarColor]);
    const getBarColor = (d) => {
        if (d.color)
            return d.color;
        if (d.group && hasGroups)
            return colorScale(d.group);
        return defaultBarColor;
    };
    return ((0, jsx_runtime_1.jsxs)("div", { className: "bar-chart-container", style: { width: '100%', maxWidth: width }, children: [(0, jsx_runtime_1.jsxs)("svg", { width: width, height: height, style: { overflow: 'visible' }, children: [title && ((0, jsx_runtime_1.jsx)(text_1.Text, { x: 0, y: 16, fontSize: 14, fontFamily: exports.CHART_STYLES.fontFamily, fill: exports.CHART_STYLES.labelColor, fontWeight: 700, children: title })), (0, jsx_runtime_1.jsxs)(group_1.Group, { left: margin.left, top: margin.top, children: [(0, jsx_runtime_1.jsx)(grid_1.GridRows, { scale: yScale, width: innerWidth, stroke: exports.CHART_STYLES.gridColor, strokeOpacity: 0.3, strokeDasharray: "3,3" }), hasGroups ? (
                            // Grouped bars - iterate over unique position IDs
                            positionIds.map((posId) => {
                                // Get all data points for this position (one per group)
                                const positionData = data.filter(d => getDataId(d) === posId);
                                const x0 = xScale(posId) || 0;
                                return ((0, jsx_runtime_1.jsx)(group_1.Group, { left: x0, children: positionData.map((d) => {
                                        const barWidth = groupScale?.bandwidth() || 0;
                                        const barX = groupScale?.(d.group || '') || 0;
                                        const barHeight = innerHeight - yScale(d.value);
                                        const barY = yScale(d.value);
                                        const color = getBarColor(d);
                                        return ((0, jsx_runtime_1.jsxs)(group_1.Group, { children: [(0, jsx_runtime_1.jsx)(shape_1.Bar, { x: barX, y: barY, width: barWidth, height: barHeight, fill: color, stroke: exports.CHART_STYLES.barBorderColor, strokeWidth: 1, rx: 2 }), showStd && d.std && d.std > 0 && ((0, jsx_runtime_1.jsxs)(jsx_runtime_1.Fragment, { children: [(0, jsx_runtime_1.jsx)("line", { x1: barX + barWidth / 2, x2: barX + barWidth / 2, y1: yScale(d.value - d.std), y2: yScale(d.value + d.std), stroke: "#fff", strokeWidth: 1.5, opacity: 0.8 }), (0, jsx_runtime_1.jsx)("line", { x1: barX + barWidth / 2 - 4, x2: barX + barWidth / 2 + 4, y1: yScale(d.value + d.std), y2: yScale(d.value + d.std), stroke: "#fff", strokeWidth: 1.5, opacity: 0.8 }), (0, jsx_runtime_1.jsx)("line", { x1: barX + barWidth / 2 - 4, x2: barX + barWidth / 2 + 4, y1: yScale(d.value - d.std), y2: yScale(d.value - d.std), stroke: "#fff", strokeWidth: 1.5, opacity: 0.8 })] })), showValues && ((0, jsx_runtime_1.jsx)(text_1.Text, { x: barX + barWidth / 2, y: barY - 8, fontSize: 11, fontFamily: exports.CHART_STYLES.fontFamily, fill: exports.CHART_STYLES.labelColor, textAnchor: "middle", children: valueFormat(d.value) }))] }, `${getDataId(d)}-${d.group}`));
                                    }) }, posId));
                            })) : (
                            // Simple bars
                            data.map((d) => {
                                const barWidth = xScale.bandwidth();
                                const barX = xScale(d.label) || 0;
                                const barHeight = innerHeight - yScale(d.value);
                                const barY = yScale(d.value);
                                const color = getBarColor(d);
                                return ((0, jsx_runtime_1.jsxs)(group_1.Group, { children: [(0, jsx_runtime_1.jsx)(shape_1.Bar, { x: barX, y: barY, width: barWidth, height: barHeight, fill: color, stroke: exports.CHART_STYLES.barBorderColor, strokeWidth: 1, rx: 2 }), showStd && d.std && d.std > 0 && ((0, jsx_runtime_1.jsxs)(jsx_runtime_1.Fragment, { children: [(0, jsx_runtime_1.jsx)("line", { x1: barX + barWidth / 2, x2: barX + barWidth / 2, y1: yScale(d.value - d.std), y2: yScale(d.value + d.std), stroke: "#fff", strokeWidth: 1.5, opacity: 0.8 }), (0, jsx_runtime_1.jsx)("line", { x1: barX + barWidth / 2 - 4, x2: barX + barWidth / 2 + 4, y1: yScale(d.value + d.std), y2: yScale(d.value + d.std), stroke: "#fff", strokeWidth: 1.5, opacity: 0.8 }), (0, jsx_runtime_1.jsx)("line", { x1: barX + barWidth / 2 - 4, x2: barX + barWidth / 2 + 4, y1: yScale(d.value - d.std), y2: yScale(d.value - d.std), stroke: "#fff", strokeWidth: 1.5, opacity: 0.8 })] })), showValues && ((0, jsx_runtime_1.jsx)(text_1.Text, { x: barX + barWidth / 2, y: barY - 8, fontSize: 11, fontFamily: exports.CHART_STYLES.fontFamily, fill: exports.CHART_STYLES.labelColor, textAnchor: "middle", children: valueFormat(d.value) }))] }, d.label));
                            })), (0, jsx_runtime_1.jsx)(axis_1.AxisLeft, { scale: yScale, stroke: exports.CHART_STYLES.axisColor, tickStroke: exports.CHART_STYLES.axisColor, tickLabelProps: {
                                    fill: exports.CHART_STYLES.labelColor,
                                    fontSize: 11,
                                    fontFamily: exports.CHART_STYLES.fontFamily,
                                    textAnchor: 'end',
                                    dy: '0.33em',
                                    dx: -4,
                                }, tickFormat: (v) => yTickFormat(v), numTicks: 6 }), yLabel && ((0, jsx_runtime_1.jsx)(text_1.Text, { x: -innerHeight / 2, y: -50, transform: "rotate(-90)", fontSize: 12, fontFamily: exports.CHART_STYLES.fontFamily, fill: exports.CHART_STYLES.labelColor, textAnchor: "middle", children: yLabel })), (0, jsx_runtime_1.jsx)(axis_1.AxisBottom, { scale: xScale, top: innerHeight, stroke: exports.CHART_STYLES.axisColor, tickStroke: exports.CHART_STYLES.axisColor, tickLabelProps: (value) => ({
                                    fill: exports.CHART_STYLES.labelColor,
                                    fontSize: 11,
                                    fontFamily: exports.CHART_STYLES.fontFamily,
                                    textAnchor: labelAngle < 0 ? 'end' : labelAngle > 0 ? 'start' : 'middle',
                                    dy: labelAngle !== 0 ? '0.25em' : '0.5em',
                                    dx: labelAngle < 0 ? -4 : labelAngle > 0 ? 4 : 0,
                                    angle: labelAngle,
                                }), tickFormat: (posId) => {
                                    // Find the display label for this position ID
                                    const dataPoint = data.find(d => getDataId(d) === posId);
                                    return dataPoint?.label || String(posId);
                                } }), showCategoryBrackets && categoryBrackets && categoryBrackets.map((bracket, i) => {
                                const [firstIdx, lastIdx] = bracket.labelIndices;
                                const firstPosId = positionIds[firstIdx];
                                const lastPosId = positionIds[lastIdx];
                                if (!firstPosId || !lastPosId)
                                    return null;
                                const x1 = (xScale(firstPosId) || 0) + xScale.bandwidth() * 0.1;
                                const x2 = (xScale(lastPosId) || 0) + xScale.bandwidth() * 0.9;
                                const bracketY = innerHeight + bracketOffset;
                                const bracketHeight = 8;
                                return ((0, jsx_runtime_1.jsxs)(group_1.Group, { children: [(0, jsx_runtime_1.jsx)("line", { x1: x1, x2: x1, y1: bracketY, y2: bracketY + bracketHeight, stroke: exports.CHART_STYLES.axisColor, strokeWidth: 1 }), (0, jsx_runtime_1.jsx)("line", { x1: x1, x2: x2, y1: bracketY + bracketHeight, y2: bracketY + bracketHeight, stroke: exports.CHART_STYLES.axisColor, strokeWidth: 1 }), (0, jsx_runtime_1.jsx)("line", { x1: x2, x2: x2, y1: bracketY, y2: bracketY + bracketHeight, stroke: exports.CHART_STYLES.axisColor, strokeWidth: 1 }), (0, jsx_runtime_1.jsx)(text_1.Text, { x: (x1 + x2) / 2, y: bracketY + bracketHeight + 14, fontSize: 11, fontFamily: exports.CHART_STYLES.fontFamily, fill: exports.CHART_STYLES.labelColor, textAnchor: "middle", children: bracket.category })] }, `bracket-${i}`));
                            })] }), hasGroups && groups.length > 0 && ((0, jsx_runtime_1.jsx)(group_1.Group, { top: margin.top - 22, left: margin.left, children: groups.map((group, i) => {
                            // Calculate dynamic x position based on previous labels' widths
                            // Adjusted to match original ~80px spacing for short labels while accommodating long ones
                            const CHAR_WIDTH = 6; // Improved estimate for font-size 11
                            const RECT_WIDTH = 12;
                            const TEXT_PADDING = 6; // Space between rect and text
                            const ITEM_GAP = 50; // Large gap to maintain uniform spacing similar to original 80px grid
                            let xOffset = 0;
                            for (let j = 0; j < i; j++) {
                                const prevLabelLen = groups[j].length;
                                const prevItemWidth = RECT_WIDTH + TEXT_PADDING + (prevLabelLen * CHAR_WIDTH);
                                xOffset += prevItemWidth + ITEM_GAP;
                            }
                            return ((0, jsx_runtime_1.jsxs)(group_1.Group, { left: xOffset, children: [(0, jsx_runtime_1.jsx)("rect", { width: RECT_WIDTH, height: 12, fill: colorScale(group), rx: 2 }), (0, jsx_runtime_1.jsx)(text_1.Text, { x: RECT_WIDTH + TEXT_PADDING, y: 10, fontSize: 11, fontFamily: exports.CHART_STYLES.fontFamily, fill: exports.CHART_STYLES.labelColor, children: group })] }, group));
                        }) }))] }), caption && ((0, jsx_runtime_1.jsx)("p", { style: {
                    marginTop: '1rem',
                    fontSize: '0.875rem',
                    color: exports.CHART_STYLES.captionColor,
                    lineHeight: 1.5,
                    maxWidth: width,
                    fontStyle: 'italic',
                }, children: caption }))] }));
}
