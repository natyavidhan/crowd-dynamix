/**
 * Utility functions for color manipulation.
 */

/**
 * Interpolate between two hex colors.
 */
export function lerpColor(color1: string, color2: string, t: number): string {
  const c1 = hexToRgb(color1);
  const c2 = hexToRgb(color2);
  
  if (!c1 || !c2) return color1;
  
  const r = Math.round(c1.r + (c2.r - c1.r) * t);
  const g = Math.round(c1.g + (c2.g - c1.g) * t);
  const b = Math.round(c1.b + (c2.b - c1.b) * t);
  
  return rgbToHex(r, g, b);
}

/**
 * Convert CII value to color (green -> yellow -> red).
 */
export function ciiToColor(cii: number): string {
  const GREEN = '#22c55e';
  const YELLOW = '#eab308';
  const RED = '#dc2626';
  
  if (cii < 0.35) {
    // Green to yellow
    const t = cii / 0.35;
    return lerpColor(GREEN, YELLOW, t);
  } else {
    // Yellow to red
    const t = (cii - 0.35) / 0.65;
    return lerpColor(YELLOW, RED, t);
  }
}

/**
 * Get risk level color.
 */
export function riskLevelColor(level: 'green' | 'yellow' | 'red'): string {
  switch (level) {
    case 'green':
      return '#22c55e';
    case 'yellow':
      return '#eab308';
    case 'red':
      return '#dc2626';
  }
}

/**
 * Get agent state color.
 */
export function agentStateColor(state: string): string {
  switch (state) {
    case 'moving':
      return '#4a90d9';
    case 'slowing':
      return '#f5a623';
    case 'stopped':
      return '#d0021b';
    case 'pushing':
      return '#9013fe';
    default:
      return '#4a90d9';
  }
}

// Helpers

function hexToRgb(hex: string): { r: number; g: number; b: number } | null {
  const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
  return result
    ? {
        r: parseInt(result[1], 16),
        g: parseInt(result[2], 16),
        b: parseInt(result[3], 16),
      }
    : null;
}

function rgbToHex(r: number, g: number, b: number): string {
  return '#' + [r, g, b].map(x => {
    const hex = x.toString(16);
    return hex.length === 1 ? '0' + hex : hex;
  }).join('');
}
