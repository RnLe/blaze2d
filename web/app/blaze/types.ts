/** Band diagram data loaded from binary format */
export interface BandData {
  bands: number[][]
  n_bands: number
  n_points: number
  symmetry_points: {
    labels: string[]
    indices: number[]
  }
  params: {
    lattice: string
    polarization: string
    eps_bg: number
    r_over_a: number
  }
}

/** Metadata for binary band data file */
export interface BandDataMeta {
  n_bands: number
  n_points: number
  params: {
    lattice: string
    polarization: string
    eps_bg: number
    r_over_a: number
  }
  symmetry_points: {
    labels: string[]
    indices: number[]
  }
  k_path: number[][]
  stats: {
    globalMin: number
    globalMax: number
    bands: Array<{
      index: number
      min: number
      max: number
    }>
  }
  binary: {
    format: 'float32'
    layout: 'band-major'
    totalBytes: number
    floatsPerBand: number
  }
}
