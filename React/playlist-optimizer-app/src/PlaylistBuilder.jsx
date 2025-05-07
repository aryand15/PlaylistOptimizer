import React, { useState } from 'react';
import SongSearch from './SongSearch';
import OptimizationPieChart from './OptimizationPieChart';
import './PlaylistBuilder.css';

const PlaylistBuilder = () => {
  const [playlist, setPlaylist] = useState([]);
  const [optimizedPlaylist, setOptimizedPlaylist] = useState(null);
  
  // Slider values for optimization parameters
  const [sliders, setSliders] = useState({
    tempo_smoothness: 20,
    energy_continuity: 20,
    mood_consistency: 20,
    key_compatibility: 20,
    genre_jump_smoothness: 20
  });

  const slidersToDisplayName = {
    tempo_smoothness: "Tempo Smoothness",
    energy_continuity: "Energy Continuity",
    mood_consistency: "Mood Consistency",
    key_compatibility: "Key Compatibility",
    genre_jump_smoothness: "Genre Jump Smoothness"
  }

  const generateColors = (count) =>
    Array.from({ length: count }, (_, i) => `hsl(${(i * 360) / count}, 70%, 60%)`
  );

  const sliderColors = generateColors(Object.keys(sliders).length);

  const handleSongAdd = (song) => {
    // Check if song is already in playlist
    if (!playlist.some(item => item.track_id === song.track_id)) {
      setPlaylist([...playlist, song]);
    }
  };

  const handleSongRemove = (songId) => {
    setPlaylist(playlist.filter(song => song.track_id !== songId));
  };

  const handleSliderChange = (e) => {
    const { name, value } = e.target;
    setSliders({
      ...sliders,
      [name]: parseInt(value),
    });
  };

  const optimizePlaylist = async () => {
    if (playlist.length < 2) {
      alert("Please add at least 2 songs to optimize your playlist");
      return;
    }

    try {
      // Extract just the track IDs for the API request
      const trackIds = playlist.map(song => song.track_id);
      console.log("bouta fetch")
      const response = await fetch('http://127.0.0.1:8000/optimize-playlist', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          track_ids: trackIds,
          params: sliders
        }),
      });

      if (!response.ok) {
        throw new Error('Failed to optimize playlist');
      }
      console.log(response)

      const optimizedSongs = await response.json();
      setOptimizedPlaylist(optimizedSongs);
    } catch (error) {
      console.error('Error optimizing playlist:', error);
      alert('Failed to optimize playlist. Please try again.');
    }
  };

  return (
    <div className="playlist-builder">
      <h1>Playlist Optimizer</h1>
      
      <div className="search-section">
        <h2>Search and Add Songs</h2>
        <SongSearch onSongAdd={handleSongAdd} />
      </div>
      
      <div className="playlist-section">
        <h2>Your Playlist ({playlist.length} songs)</h2>
        {playlist.length === 0 ? (
          <p className="empty-playlist">Your playlist is empty. Search for songs to add them.</p>
        ) : (
          <ul className="playlist">
            {playlist.map((song, index) => (
              <li key={song.track_id} className="playlist-item">
                <div className="song-number">{index + 1}</div>
                <div className="song-details">
                  <div className="song-title">{song.track_name}</div>
                  <div className="song-artist">{song.artists}</div>
                </div>
                <button 
                  className="remove-song" 
                  onClick={() => handleSongRemove(song.track_id)}
                >
                  Remove
                </button>
              </li>
            ))}
          </ul>
        )}
      </div>
      
      <div className="optimization-section">
        <h2>Optimization Parameters</h2>
        <p>How important is each factor to the final playlist?</p>
        
        <div className="pie-chart-container">
          <OptimizationPieChart
            optimizationParams = {Object.values(slidersToDisplayName)} 
            percentageData = {Object.values(sliders)} 
            colors = {sliderColors}
          />
        </div>
        
        
        <div className="sliders">
          {
            Object.keys(sliders).map((p, i) => (
              <div key={p} className="slider-container">
                <label 
                  htmlFor={p} 
                >
                  {slidersToDisplayName[p]}
                </label>

                <input
                  type="range"
                  id={p}
                  name={p}
                  min="0"
                  max="100"
                  step="10"
                  value={sliders[p]}
                  onChange={handleSliderChange}
                  style={{accentColor: sliderColors[i]}}
                />
                <span>{sliders[p]}%</span>
            </div>
            ))
          }
        </div>
        
        <div>
          Total: {Object.values(sliders).reduce((acc, e) => acc + e, 0)}%
        </div>
        
        <button 
          className="optimize-button" 
          onClick={optimizePlaylist}
          disabled={playlist.length < 2}
        >
          Optimize Playlist
        </button>
      </div>
      
      {optimizedPlaylist && (
        <div className="optimized-section">
          <h2>Optimized Playlist</h2>
          <ul className="playlist optimized">
            {optimizedPlaylist.map((song, index) => (
              <li key={`optimized-${song.track_id}`} className="playlist-item">
                <div className="song-number">{index + 1}</div>
                <div className="song-details">
                  <div className="song-title">{song.track_name}</div>
                  <div className="song-artist">{song.artists}</div>
                </div>
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
};

export default PlaylistBuilder;