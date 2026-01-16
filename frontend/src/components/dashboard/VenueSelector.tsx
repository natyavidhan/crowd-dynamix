/**
 * VenueSelector component for choosing simulation venues.
 */

import { useEffect } from 'react';
import { useSimulationStore } from '@/store';

export function VenueSelector() {
  const {
    availableVenues,
    currentVenue,
    venueLoading,
    fetchVenues,
    loadVenue,
    fetchCurrentVenue,
  } = useSimulationStore();

  useEffect(() => {
    fetchVenues();
    fetchCurrentVenue();
  }, [fetchVenues, fetchCurrentVenue]);

  const handleVenueChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    const venueId = e.target.value;
    if (venueId) {
      loadVenue(venueId);
    }
  };

  return (
    <div className="bg-gray-800 rounded-lg p-4 space-y-3">
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-semibold text-gray-300 uppercase tracking-wider">
          Venue
        </h3>
        {venueLoading && (
          <div className="flex items-center space-x-2">
            <div className="w-4 h-4 border-2 border-blue-500 border-t-transparent rounded-full animate-spin" />
            <span className="text-xs text-gray-400">Loading...</span>
          </div>
        )}
      </div>

      <select
        value={currentVenue?.id || ''}
        onChange={handleVenueChange}
        disabled={venueLoading}
        className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-md text-white text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:opacity-50"
      >
        <option value="">Select a venue...</option>
        {availableVenues.map((venue) => (
          <option key={venue.id} value={venue.id}>
            {venue.name}
          </option>
        ))}
      </select>

      {currentVenue && (
        <div className="space-y-2 pt-2 border-t border-gray-700">
          <div>
            <p className="text-white font-medium">{currentVenue.name}</p>
            <p className="text-gray-400 text-xs">{currentVenue.location}</p>
          </div>
          
          {currentVenue.description && (
            <p className="text-gray-500 text-xs">{currentVenue.description}</p>
          )}
          
          <div className="flex space-x-4 text-xs">
            <div className="flex items-center space-x-1">
              <span className="text-blue-400">●</span>
              <span className="text-gray-400">
                {currentVenue.spawn_point_count} spawn points
              </span>
            </div>
            <div className="flex items-center space-x-1">
              <span className="text-yellow-400">●</span>
              <span className="text-gray-400">
                {currentVenue.choke_point_count} choke points
              </span>
            </div>
          </div>
          
          {currentVenue.roads && currentVenue.roads.length > 0 && (
            <div className="text-xs text-gray-500">
              {currentVenue.roads.length} roads defined
            </div>
          )}
        </div>
      )}
    </div>
  );
}
