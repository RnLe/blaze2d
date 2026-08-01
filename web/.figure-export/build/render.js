"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
const jsx_runtime_1 = require("react/jsx-runtime");
const server_1 = require("react-dom/server");
const node_fs_1 = require("node:fs");
const BandComparisonChart_1 = __importDefault(require("./components/charts/BandComparisonChart"));
const BandsBarChart_1 = __importDefault(require("./components/charts/BandsBarChart"));
const DeviationBoxPlotChart_1 = __importDefault(require("./components/charts/DeviationBoxPlotChart"));
const EpsilonBarChart_1 = __importDefault(require("./components/charts/EpsilonBarChart"));
const IterationBarChart_1 = __importDefault(require("./components/charts/IterationBarChart"));
const IterationDistributionChart_1 = __importDefault(require("./components/charts/IterationDistributionChart"));
const IterationTimeChart_1 = __importDefault(require("./components/charts/IterationTimeChart"));
const MemoryRatioChart_1 = __importDefault(require("./components/charts/MemoryRatioChart"));
const MemoryScalingChart_1 = __importDefault(require("./components/charts/MemoryScalingChart"));
const MemoryUsageChart_1 = __importDefault(require("./components/charts/MemoryUsageChart"));
const MultiCorePerformanceChart_1 = __importDefault(require("./components/charts/MultiCorePerformanceChart"));
const ResolutionBarChart_1 = __importDefault(require("./components/charts/ResolutionBarChart"));
const ResolutionScalingChart_1 = __importDefault(require("./components/charts/ResolutionScalingChart"));
const ResolutionSpeedupChart_1 = __importDefault(require("./components/charts/ResolutionSpeedupChart"));
const SingleCorePerformanceChart_1 = __importDefault(require("./components/charts/SingleCorePerformanceChart"));
const SpeedupScalingChart_1 = __importDefault(require("./components/charts/SpeedupScalingChart"));
const ThroughputScalingChart_1 = __importDefault(require("./components/charts/ThroughputScalingChart"));
const jobs = [
    { out: "single-core", element: (0, jsx_runtime_1.jsx)(SingleCorePerformanceChart_1.default, { "width": 650, "height": 420 }) },
    { out: "multi-core", element: (0, jsx_runtime_1.jsx)(MultiCorePerformanceChart_1.default, { "width": 650, "height": 420 }) },
    { out: "memory-resolution", element: (0, jsx_runtime_1.jsx)(MemoryUsageChart_1.default, { "width": 1000, "height": 380, "sweep": "resolution" }) },
    { out: "memory-bands", element: (0, jsx_runtime_1.jsx)(MemoryUsageChart_1.default, { "width": 1000, "height": 380, "sweep": "num_bands" }) },
    { out: "memory-ratio", element: (0, jsx_runtime_1.jsx)(MemoryRatioChart_1.default, { "width": 1000, "height": 380 }) },
    { out: "memory-scaling", element: (0, jsx_runtime_1.jsx)(MemoryScalingChart_1.default, { "width": 1000, "height": 380 }) },
    { out: "bands-tm", element: (0, jsx_runtime_1.jsx)(BandComparisonChart_1.default, { "width": 550, "height": 420, "polarization": "TM" }) },
    { out: "bands-te", element: (0, jsx_runtime_1.jsx)(BandComparisonChart_1.default, { "width": 550, "height": 420, "polarization": "TE" }) },
    { out: "deviation-tm", element: (0, jsx_runtime_1.jsx)(DeviationBoxPlotChart_1.default, { "width": 400, "height": 400, "polarization": "TM" }) },
    { out: "deviation-te", element: (0, jsx_runtime_1.jsx)(DeviationBoxPlotChart_1.default, { "width": 400, "height": 400, "polarization": "TE" }) },
    { out: "throughput-scaling", element: (0, jsx_runtime_1.jsx)(ThroughputScalingChart_1.default, { "width": 1000, "height": 380 }) },
    { out: "speedup-scaling", element: (0, jsx_runtime_1.jsx)(SpeedupScalingChart_1.default, { "width": 1000, "height": 380 }) },
    { out: "resolution-bar", element: (0, jsx_runtime_1.jsx)(ResolutionBarChart_1.default, { "width": 1000, "height": 380 }) },
    { out: "resolution-speedup", element: (0, jsx_runtime_1.jsx)(ResolutionSpeedupChart_1.default, { "width": 1000, "height": 380 }) },
    { out: "resolution-scaling", element: (0, jsx_runtime_1.jsx)(ResolutionScalingChart_1.default, { "width": 1000, "height": 380 }) },
    { out: "iteration-bar", element: (0, jsx_runtime_1.jsx)(IterationBarChart_1.default, { "width": 1000, "height": 380 }) },
    { out: "iteration-time", element: (0, jsx_runtime_1.jsx)(IterationTimeChart_1.default, { "width": 1000, "height": 380 }) },
    { out: "iteration-distribution", element: (0, jsx_runtime_1.jsx)(IterationDistributionChart_1.default, { "width": 1040, "height": 380 }) },
    { out: "epsilon-bar", element: (0, jsx_runtime_1.jsx)(EpsilonBarChart_1.default, { "width": 1000, "height": 380 }) },
    { out: "bands-bar", element: (0, jsx_runtime_1.jsx)(BandsBarChart_1.default, { "width": 1000, "height": 380 }) },
];
const rendered = {};
for (const job of jobs) {
    rendered[job.out] = (0, server_1.renderToStaticMarkup)(job.element);
}
(0, node_fs_1.writeFileSync)(process.argv[2], JSON.stringify(rendered));
