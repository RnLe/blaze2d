'use client';
"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.default = MemoryUsageChart;
const jsx_runtime_1 = require("react/jsx-runtime");
const react_1 = require("react");
const BarChart_1 = __importDefault(require("./BarChart"));
const use_benchmarks_1 = require("../../lib/use-benchmarks");
const COLORS = {
    MPB: '#5477c4', // Blue-gray
    Blaze: '#eaf1fe', // Light blue-white
};
function MemoryUsageChart({ width = 650, height = 420, sweep, }) {
    const { data: benchmarkData, loading } = (0, use_benchmarks_1.useSeries5Benchmarks)();
    const sweepData = benchmarkData[sweep];
    const sweepLabel = sweep === 'resolution' ? 'Resolution' : 'Number of Bands';
    // Transform data for grouped bar chart
    // Each value is a position on the x-axis, with TM and TE as pairs of MPB/Blaze
    const { tmData, teData } = (0, react_1.useMemo)(() => {
        const tm = [];
        const te = [];
        for (const val of sweepData.values) {
            const mpbTM = sweepData.mpb.TM.find(d => d.value === val);
            const blazeTM = sweepData.blaze.TM.find(d => d.value === val);
            const mpbTE = sweepData.mpb.TE.find(d => d.value === val);
            const blazeTE = sweepData.blaze.TE.find(d => d.value === val);
            if (mpbTM) {
                tm.push({
                    id: `${val}`,
                    label: `${val}`,
                    value: mpbTM.memory_mb,
                    std: mpbTM.memory_mb_std,
                    group: 'MPB',
                });
            }
            if (blazeTM) {
                tm.push({
                    id: `${val}`,
                    label: `${val}`,
                    value: blazeTM.memory_mb,
                    std: blazeTM.memory_mb_std,
                    group: 'Blaze',
                });
            }
            if (mpbTE) {
                te.push({
                    id: `${val}`,
                    label: `${val}`,
                    value: mpbTE.memory_mb,
                    std: mpbTE.memory_mb_std,
                    group: 'MPB',
                });
            }
            if (blazeTE) {
                te.push({
                    id: `${val}`,
                    label: `${val}`,
                    value: blazeTE.memory_mb,
                    std: blazeTE.memory_mb_std,
                    group: 'Blaze',
                });
            }
        }
        return { tmData: tm, teData: te };
    }, [sweepData]);
    // Calculate memory reduction for caption
    const avgReduction = (0, react_1.useMemo)(() => {
        let totalRatio = 0;
        let count = 0;
        for (const val of sweepData.values) {
            const mpbTM = sweepData.mpb.TM.find(d => d.value === val);
            const blazeTM = sweepData.blaze.TM.find(d => d.value === val);
            const mpbTE = sweepData.mpb.TE.find(d => d.value === val);
            const blazeTE = sweepData.blaze.TE.find(d => d.value === val);
            if (mpbTM && blazeTM) {
                totalRatio += mpbTM.memory_mb / blazeTM.memory_mb;
                count++;
            }
            if (mpbTE && blazeTE) {
                totalRatio += mpbTE.memory_mb / blazeTE.memory_mb;
                count++;
            }
        }
        return count > 0 ? (totalRatio / count).toFixed(0) : '?';
    }, [sweepData]);
    if (loading) {
        return ((0, jsx_runtime_1.jsx)("div", { style: { width: width * 2 + 40, height, display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#888' }, children: "Loading benchmark data..." }));
    }
    const chartWidth = (width - 20) / 2;
    return ((0, jsx_runtime_1.jsxs)("div", { style: { display: 'flex', flexDirection: 'column', gap: '0.5rem' }, children: [(0, jsx_runtime_1.jsxs)("div", { style: { display: 'flex', gap: '20px', justifyContent: 'center' }, children: [(0, jsx_runtime_1.jsx)(BarChart_1.default, { data: tmData, width: chartWidth, height: height, title: `Peak Memory vs ${sweepLabel} (TM)`, yLabel: "Peak Memory (MB)", yTickFormat: (v) => `${v.toFixed(0)}`, valueFormat: (v) => `${v.toFixed(0)}`, labelAngle: 0, showValues: false, showStd: true, groupColors: COLORS, caption: "", margin: { top: 60, right: 30, bottom: 50, left: 75 } }), (0, jsx_runtime_1.jsx)(BarChart_1.default, { data: teData, width: chartWidth, height: height, title: `Peak Memory vs ${sweepLabel} (TE)`, yLabel: "Peak Memory (MB)", yTickFormat: (v) => `${v.toFixed(0)}`, valueFormat: (v) => `${v.toFixed(0)}`, labelAngle: 0, showValues: false, showStd: true, groupColors: COLORS, caption: "", margin: { top: 60, right: 30, bottom: 50, left: 75 } })] }), (0, jsx_runtime_1.jsx)("p", { style: {
                    marginTop: '1rem',
                    fontSize: '0.875rem',
                    color: '#888',
                    lineHeight: 1.5,
                    fontFamily: 'var(--font-sans), system-ui, sans-serif',
                    fontStyle: 'italic',
                    textAlign: 'left',
                } })] }));
}
