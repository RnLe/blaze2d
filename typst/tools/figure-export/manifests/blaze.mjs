// Figure manifest for reports/blaze (mirrors web/content/blaze.mdx).
//
// One entry per embedded visual, in the order it appears on the page, with the
// exact props the MDX passes. Keeping the props here rather than in the
// exporter is what lets a figure be re-exported identically after the page
// changes: update the props, re-run `make figures`.

export default {
  report: 'blaze',
  source: 'web/content/blaze.mdx',
  figures: [
    { out: 'single-core', component: 'SingleCorePerformanceChart', props: { width: 650, height: 420 } },
    { out: 'multi-core', component: 'MultiCorePerformanceChart', props: { width: 650, height: 420 } },

    { out: 'memory-resolution', component: 'MemoryUsageChart', props: { width: 1000, height: 380, sweep: 'resolution' } },
    { out: 'memory-bands', component: 'MemoryUsageChart', props: { width: 1000, height: 380, sweep: 'num_bands' } },
    { out: 'memory-ratio', component: 'MemoryRatioChart', props: { width: 1000, height: 380 } },
    { out: 'memory-scaling', component: 'MemoryScalingChart', props: { width: 1000, height: 380 } },

    // Replaces the interactive EpsilonGridViewer, which has no print form.
    { out: 'epsilon-grids', generator: 'epsilon-grids', options: { resolutions: [16, 32, 64, 128] } },

    { out: 'bands-tm', component: 'BandComparisonChart', props: { width: 550, height: 420, polarization: 'TM' } },
    { out: 'bands-te', component: 'BandComparisonChart', props: { width: 550, height: 420, polarization: 'TE' } },
    { out: 'deviation-tm', component: 'DeviationBoxPlotChart', props: { width: 400, height: 400, polarization: 'TM' } },
    { out: 'deviation-te', component: 'DeviationBoxPlotChart', props: { width: 400, height: 400, polarization: 'TE' } },

    { out: 'throughput-scaling', component: 'ThroughputScalingChart', props: { width: 1000, height: 380 } },
    { out: 'speedup-scaling', component: 'SpeedupScalingChart', props: { width: 1000, height: 380 } },

    { out: 'resolution-bar', component: 'ResolutionBarChart', props: { width: 1000, height: 380 } },
    { out: 'resolution-speedup', component: 'ResolutionSpeedupChart', props: { width: 1000, height: 380 } },
    { out: 'resolution-scaling', component: 'ResolutionScalingChart', props: { width: 1000, height: 380 } },

    { out: 'iteration-bar', component: 'IterationBarChart', props: { width: 1000, height: 380 } },
    { out: 'iteration-time', component: 'IterationTimeChart', props: { width: 1000, height: 380 } },
    { out: 'iteration-distribution', component: 'IterationDistributionChart', props: { width: 1040, height: 380 } },

    { out: 'epsilon-bar', component: 'EpsilonBarChart', props: { width: 1000, height: 380 } },
    { out: 'bands-bar', component: 'BandsBarChart', props: { width: 1000, height: 380 } },
  ],
};
