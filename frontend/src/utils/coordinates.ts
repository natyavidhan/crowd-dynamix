/**
 * Coordinate conversion utilities.
 * Converts between geographic (lat/lng) and simulation (x/y meters) coordinates.
 */

import type { GeoPoint, SimPoint } from '@/types';

const METERS_PER_DEG_LAT = 111320;

export interface CoordinateSystem {
  origin: GeoPoint;
  metersPerDegLng: number;
}

/**
 * Create a coordinate system for a given origin.
 */
export function createCoordinateSystem(origin: GeoPoint): CoordinateSystem {
  const latRad = (origin.lat * Math.PI) / 180;
  const metersPerDegLng = METERS_PER_DEG_LAT * Math.cos(latRad);
  
  return {
    origin,
    metersPerDegLng,
  };
}

/**
 * Convert geographic to simulation coordinates.
 */
export function geoToSim(geo: GeoPoint, coords: CoordinateSystem): SimPoint {
  return {
    x: (geo.lng - coords.origin.lng) * coords.metersPerDegLng,
    y: (geo.lat - coords.origin.lat) * METERS_PER_DEG_LAT,
  };
}

/**
 * Convert simulation to geographic coordinates.
 */
export function simToGeo(sim: SimPoint, coords: CoordinateSystem): GeoPoint {
  return {
    lat: coords.origin.lat + sim.y / METERS_PER_DEG_LAT,
    lng: coords.origin.lng + sim.x / coords.metersPerDegLng,
  };
}

/**
 * Convert simulation point to Leaflet LatLng format [lat, lng].
 */
export function simToLatLng(
  sim: SimPoint,
  coords: CoordinateSystem
): [number, number] {
  const geo = simToGeo(sim, coords);
  return [geo.lat, geo.lng];
}
