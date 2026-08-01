// Shim for the `swr` package.
//
// BandComparisonChart and DeviationBoxPlotChart fetch series6-accuracy.json via
// useSWR. Under server rendering there is no revalidation cycle to wait for, so
// the hook resolves immediately from disk and reports a settled, successful
// state. The shape matches the three fields those two components destructure.
//
// The fetcher argument is accepted for signature compatibility and ignored: it
// would perform a network request, which is exactly what is being replaced.

import { loadJson } from './data-source';

export interface SWRResponse<T> {
  data: T | undefined;
  error: unknown;
  isLoading: boolean;
}

export default function useSWR<T>(
  key: string | null,
  _fetcher?: (key: string) => Promise<T>,
): SWRResponse<T> {
  if (!key) return { data: undefined, error: undefined, isLoading: false };
  return { data: loadJson<T>(key), error: undefined, isLoading: false };
}
