'use client';

import { useState, useEffect, useMemo, useRef } from 'react';
import { getAssetPath } from '../../lib/paths';

interface EpsilonGridData {
  resolution: number;
  nx: number;
  ny: number;
  epsilon_background: number;
  epsilon_hole: number;
  radius: number;
  grid: number[];  // Flat array, row-major
}

interface EpsilonGridProps {
  resolution: number;
  width?: number;
  height?: number;
  colormap?: 'viridis' | 'plasma' | 'inferno' | 'magma' | 'grayscale';
  showColorbar?: boolean;
}

// Pre-computed Viridis colormap (256 colors)
const VIRIDIS_COLORS: [number, number, number][] = [
  [68, 1, 84], [68, 2, 86], [69, 4, 87], [69, 5, 89], [70, 7, 90],
  [70, 8, 92], [70, 10, 93], [70, 11, 94], [71, 13, 96], [71, 14, 97],
  [71, 16, 99], [71, 17, 100], [71, 19, 101], [72, 20, 103], [72, 22, 104],
  [72, 23, 105], [72, 24, 106], [72, 26, 108], [72, 27, 109], [72, 28, 110],
  [72, 29, 111], [72, 31, 112], [72, 32, 113], [72, 33, 115], [72, 35, 116],
  [72, 36, 117], [72, 37, 118], [72, 38, 119], [72, 40, 120], [72, 41, 121],
  [71, 42, 122], [71, 44, 122], [71, 45, 123], [71, 46, 124], [71, 47, 125],
  [70, 48, 126], [70, 50, 126], [70, 51, 127], [69, 52, 128], [69, 53, 129],
  [69, 55, 129], [68, 56, 130], [68, 57, 131], [68, 58, 131], [67, 60, 132],
  [67, 61, 132], [66, 62, 133], [66, 63, 133], [66, 64, 134], [65, 66, 134],
  [65, 67, 135], [64, 68, 135], [64, 69, 136], [63, 71, 136], [63, 72, 137],
  [62, 73, 137], [62, 74, 137], [61, 75, 138], [61, 76, 138], [61, 78, 138],
  [60, 79, 139], [60, 80, 139], [59, 81, 139], [59, 82, 139], [58, 83, 140],
  [58, 84, 140], [57, 85, 140], [57, 86, 140], [56, 88, 141], [56, 89, 141],
  [55, 90, 141], [55, 91, 141], [54, 92, 141], [54, 93, 141], [53, 94, 141],
  [53, 95, 142], [52, 96, 142], [52, 97, 142], [51, 98, 142], [51, 99, 142],
  [51, 100, 142], [50, 101, 142], [50, 102, 142], [49, 103, 142], [49, 104, 142],
  [49, 105, 142], [48, 106, 142], [48, 107, 142], [47, 108, 142], [47, 109, 142],
  [46, 110, 142], [46, 111, 142], [46, 112, 142], [45, 113, 142], [45, 114, 142],
  [45, 115, 142], [44, 116, 142], [44, 117, 142], [44, 117, 142], [43, 118, 142],
  [43, 119, 142], [43, 120, 142], [42, 121, 142], [42, 122, 142], [42, 123, 141],
  [41, 124, 141], [41, 125, 141], [41, 126, 141], [40, 126, 141], [40, 127, 141],
  [40, 128, 141], [39, 129, 141], [39, 130, 141], [39, 131, 140], [38, 132, 140],
  [38, 133, 140], [38, 134, 140], [37, 135, 140], [37, 135, 139], [37, 136, 139],
  [36, 137, 139], [36, 138, 139], [36, 139, 138], [35, 140, 138], [35, 141, 138],
  [35, 142, 137], [34, 143, 137], [34, 143, 137], [34, 144, 136], [34, 145, 136],
  [33, 146, 135], [33, 147, 135], [33, 148, 135], [33, 149, 134], [32, 150, 134],
  [32, 150, 133], [32, 151, 133], [32, 152, 132], [31, 153, 132], [31, 154, 131],
  [31, 155, 131], [31, 156, 130], [31, 157, 130], [31, 157, 129], [31, 158, 128],
  [30, 159, 128], [30, 160, 127], [30, 161, 127], [30, 162, 126], [30, 163, 125],
  [30, 163, 125], [30, 164, 124], [30, 165, 123], [30, 166, 123], [30, 167, 122],
  [31, 168, 121], [31, 168, 120], [31, 169, 120], [31, 170, 119], [31, 171, 118],
  [32, 172, 117], [32, 172, 116], [32, 173, 116], [33, 174, 115], [33, 175, 114],
  [34, 176, 113], [34, 176, 112], [35, 177, 111], [35, 178, 110], [36, 179, 109],
  [37, 179, 108], [38, 180, 107], [38, 181, 106], [39, 182, 105], [40, 182, 104],
  [41, 183, 103], [42, 184, 102], [43, 184, 101], [44, 185, 100], [45, 186, 99],
  [46, 186, 98], [47, 187, 97], [49, 188, 96], [50, 188, 94], [51, 189, 93],
  [52, 189, 92], [54, 190, 91], [55, 191, 90], [57, 191, 88], [58, 192, 87],
  [60, 192, 86], [61, 193, 85], [63, 193, 83], [64, 194, 82], [66, 194, 81],
  [68, 195, 79], [70, 195, 78], [71, 196, 77], [73, 196, 75], [75, 197, 74],
  [77, 197, 73], [79, 198, 71], [81, 198, 70], [83, 198, 68], [85, 199, 67],
  [87, 199, 65], [89, 200, 64], [91, 200, 62], [93, 200, 61], [95, 201, 59],
  [97, 201, 58], [99, 201, 56], [102, 202, 55], [104, 202, 53], [106, 202, 52],
  [108, 203, 50], [111, 203, 48], [113, 203, 47], [115, 203, 45], [118, 204, 44],
  [120, 204, 42], [122, 204, 40], [125, 204, 39], [127, 205, 37], [130, 205, 35],
  [132, 205, 34], [135, 205, 32], [137, 205, 30], [140, 206, 29], [142, 206, 27],
  [145, 206, 25], [148, 206, 24], [150, 206, 22], [153, 206, 20], [155, 206, 19],
  [158, 206, 17], [161, 207, 16], [163, 207, 14], [166, 207, 13], [169, 207, 12],
  [171, 207, 11], [174, 207, 10], [177, 207, 10], [179, 206, 10], [182, 206, 11],
  [185, 206, 12], [187, 206, 14], [190, 206, 16], [193, 205, 19], [195, 205, 22],
  [198, 205, 25], [200, 204, 29], [203, 204, 33], [205, 203, 37], [207, 203, 41],
  [210, 202, 46], [212, 202, 51], [214, 201, 55], [216, 201, 60], [218, 200, 65],
  [220, 199, 70], [222, 199, 75], [224, 198, 80], [225, 197, 86], [227, 196, 91],
  [228, 196, 96], [230, 195, 102], [231, 194, 107], [232, 193, 112], [234, 193, 118],
  [235, 192, 123], [236, 191, 128], [237, 190, 134], [238, 189, 139], [239, 189, 144],
  [240, 188, 150], [240, 187, 155], [241, 186, 160], [242, 186, 166], [242, 185, 171],
  [243, 184, 176], [243, 184, 181], [244, 183, 186], [244, 183, 191], [244, 182, 197],
  [245, 182, 202], [245, 181, 207], [245, 181, 212], [246, 180, 217], [246, 180, 222],
  [246, 180, 227], [247, 180, 232], [247, 179, 237], [247, 179, 242], [248, 179, 247],
  [253, 231, 37]
];

// Grayscale colormap
function getGrayscaleColor(t: number): [number, number, number] {
  const v = Math.round(t * 255);
  return [v, v, v];
}

// Get color from viridis colormap
function getViridisColor(t: number): [number, number, number] {
  const idx = Math.min(255, Math.max(0, Math.round(t * 255)));
  return VIRIDIS_COLORS[idx];
}

function getColor(t: number, colormap: string): [number, number, number] {
  switch (colormap) {
    case 'grayscale':
      return getGrayscaleColor(t);
    default:
      return getViridisColor(t);
  }
}

export default function EpsilonGrid({
  resolution,
  width = 300,
  height = 300,
  colormap = 'viridis',
  showColorbar = true,
}: EpsilonGridProps) {
  const [data, setData] = useState<EpsilonGridData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Validate resolution (must be 16-256, step 16)
  const validResolution = useMemo(() => {
    const r = Math.round(resolution / 16) * 16;
    return Math.min(256, Math.max(16, r));
  }, [resolution]);

  useEffect(() => {
    setLoading(true);
    setError(null);
    
    fetch(getAssetPath(`/data/epsilon-grids/epsilon-res${validResolution}.json`))
      .then(res => {
        if (!res.ok) throw new Error(`Failed to load epsilon grid (res=${validResolution})`);
        return res.json();
      })
      .then((jsonData: EpsilonGridData) => {
        setData(jsonData);
        setLoading(false);
      })
      .catch(err => {
        setError(err.message);
        setLoading(false);
      });
  }, [validResolution]);

  // Generate image data URL from grid
  const imageDataUrl = useMemo(() => {
    if (!data) return null;
    
    const { nx, ny, grid, epsilon_background, epsilon_hole } = data;
    
    // Create canvas
    const canvas = document.createElement('canvas');
    canvas.width = nx;
    canvas.height = ny;
    const ctx = canvas.getContext('2d');
    if (!ctx) return null;
    
    const imageData = ctx.createImageData(nx, ny);
    
    // Find min/max for normalization
    const minEps = epsilon_hole;
    const maxEps = epsilon_background;
    const range = maxEps - minEps;
    
    for (let iy = 0; iy < ny; iy++) {
      for (let ix = 0; ix < nx; ix++) {
        const eps = grid[iy * nx + ix];
        const t = range > 0 ? (eps - minEps) / range : 0;
        const [r, g, b] = getColor(t, colormap);
        
        // Note: canvas y=0 is top, but our grid iy=0 is bottom
        // Flip vertically
        const canvasY = ny - 1 - iy;
        const idx = (canvasY * nx + ix) * 4;
        imageData.data[idx] = r;
        imageData.data[idx + 1] = g;
        imageData.data[idx + 2] = b;
        imageData.data[idx + 3] = 255;
      }
    }
    
    ctx.putImageData(imageData, 0, 0);
    return canvas.toDataURL();
  }, [data, colormap]);

  if (loading) {
    return (
      <div style={{ width, height, display: 'flex', alignItems: 'center', justifyContent: 'center', background: '#f0f0f0' }}>
        Loading...
      </div>
    );
  }

  if (error || !data || !imageDataUrl) {
    return (
      <div style={{ width, height, display: 'flex', alignItems: 'center', justifyContent: 'center', background: '#ffe0e0', color: '#800' }}>
        Error: {error || 'Failed to render'}
      </div>
    );
  }

  const colorbarWidth = showColorbar ? 60 : 0;
  const imageWidth = width - colorbarWidth;
  const imageHeight = height;

  return (
    <div style={{ display: 'flex', gap: '8px' }}>
      <div style={{ width: imageWidth, height: imageHeight }}>
        <img
          src={imageDataUrl}
          alt={`Smoothed epsilon grid (${data.resolution}×${data.resolution})`}
          style={{
            width: '100%',
            height: '100%',
            imageRendering: 'pixelated',
          }}
        />
      </div>
      {showColorbar && (
        <div style={{ width: colorbarWidth - 8, display: 'flex', flexDirection: 'column', fontSize: '11px' }}>
          <div style={{ flex: 1, display: 'flex', flexDirection: 'column' }}>
            <span style={{ textAlign: 'right', marginBottom: '2px' }}>{data.epsilon_background.toFixed(1)}</span>
            <div
              style={{
                flex: 1,
                background: `linear-gradient(to bottom, ${
                  Array.from({ length: 10 }, (_, i) => {
                    const t = 1 - i / 9;
                    const [r, g, b] = getColor(t, colormap);
                    return `rgb(${r},${g},${b})`;
                  }).join(', ')
                })`,
                borderRadius: '2px',
              }}
            />
            <span style={{ textAlign: 'right', marginTop: '2px' }}>{data.epsilon_hole.toFixed(1)}</span>
          </div>
          <div style={{ textAlign: 'center', marginTop: '4px', fontSize: '10px', color: '#666' }}>ε</div>
        </div>
      )}
    </div>
  );
}
