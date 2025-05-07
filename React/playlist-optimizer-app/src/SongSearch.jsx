import React, { useState, useEffect } from 'react';
import './SongSearch.css';

const SongSearch = ({ onSongAdd }) => {
  const [searchTerm, setSearchTerm] = useState('');
  const [searchResults, setSearchResults] = useState([]);
  const [isLoading, setIsLoading] = useState(false);

  useEffect(() => {
    // Only search if there's actually a search term
    if (!searchTerm) {
      setSearchResults([]);
      return;
    }

    // Set a small delay to avoid too many API calls while typing
    const delayDebounceFn = setTimeout(() => {
      setIsLoading(true);
      
      fetch(`http://127.0.0.1:8000/songs/search?q=${encodeURIComponent(searchTerm)}`)
        .then(response => {
          if (!response.ok) {
            throw new Error('Network response was not ok');
          }
          return response.json();
        })
        .then(data => {
          setSearchResults(data);
          setIsLoading(false);
        })
        .catch(error => {
          console.error('Error fetching search results:', error);
          setIsLoading(false);
        });
    }, 300); // 300ms delay after typing stops

    return () => clearTimeout(delayDebounceFn);
  }, [searchTerm]);

  const handleSongSelect = (song) => {
    onSongAdd(song);
    setSearchTerm('');
    setSearchResults([]);
  };

  return (
    <div className="song-search">
      <div className="search-container">
        <input
          type="text"
          placeholder="Search for songs..."
          value={searchTerm}
          onChange={(e) => setSearchTerm(e.target.value)}
          className="search-input"
        />
        {isLoading && <div className="loader">Loading...</div>}
      </div>

      {searchResults.length > 0 && (
        <ul className="search-results">
          {searchResults.map((song) => (
            <li 
              key={song.track_id} 
              className="search-result-item"
              onClick={() => handleSongSelect(song)}
            >
              <div className="song-info">
                <span className="song-name">{song.track_name}</span>
                <span className="song-artist">{song.artists}</span>
              </div>
              <span className="song-album">{song.album_name}</span>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
};

export default SongSearch;