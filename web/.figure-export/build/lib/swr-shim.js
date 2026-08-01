"use strict";
// Shim for the `swr` package.
//
// BandComparisonChart and DeviationBoxPlotChart fetch series6-accuracy.json via
// useSWR. Under server rendering there is no revalidation cycle to wait for, so
// the hook resolves immediately from disk and reports a settled, successful
// state. The shape matches the three fields those two components destructure.
//
// The fetcher argument is accepted for signature compatibility and ignored: it
// would perform a network request, which is exactly what is being replaced.
Object.defineProperty(exports, "__esModule", { value: true });
exports.default = useSWR;
const data_source_1 = require("./data-source");
function useSWR(key, _fetcher) {
    if (!key)
        return { data: undefined, error: undefined, isLoading: false };
    return { data: (0, data_source_1.loadJson)(key), error: undefined, isLoading: false };
}
