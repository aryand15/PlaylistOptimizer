# Playlist Optimizer

A web application that intelligently reorders a playlist of songs based on audio characteristics and user preferences.

## Overview
Playlist Optimizer allows users to select songs and receive optimized playlist orderings based on multiple audio characteristics:

- Mood - emotional tone consistency between tracks
- Energy - smooth transitions in intensity levels
- Tempo - gradual BPM (beats per minute) changes
- Key - harmonic compatibility between consecutive songs
- Genre - stylistic consistency throughout the playlist

Users can customize the importance of each criterion using percentage sliders, allowing for personalized playlist optimization.

## Features
- Track Selection: Choose from a database of ~90,000 songs with complete audio analysis data
- Custom Weighting: Adjust importance of each optimization criterion with percentage sliders
- Interactive UI: User-friendly interface for building and optimizing playlists

## Technology Stack
- Frontend: React
- Backend: FastAPI
- Database: SQLite
- Data Processing: Pandas, NumPy
- ORM: SQLAlchemy
- Optimization Algorithm: Pymoo (multi-criterial optimization)

## Data Source
Originally planned to use Spotify's API, but due to the deprecation of their audio analysis endpoint in November 2024, the application now uses a comprehensive Kaggle dataset containing ~90,000 tracks (after cleaning) with their audio analysis features.

[Link to dataset](
https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset)

## How It Works
The optimization process uses the following workflow:

1) User selects tracks from the database to include in their playlist
2) User adjusts importance sliders for each optimization criterion
3) Backend algorithm (Pymoo) calculates multiple optimal orderings based on the weighted criteria
4) Ordering with the highest score gets displayed to the user

## First-time setup
1) Make sure you have Python 3.8+, Node.js 18+, and Git.
2) Clone the repository:
```
git clone https://github.com/aryand15/playlist-optimizer.git
cd playlist-optimizer
```
3) Download the Spotify Tracks Dataset as a CSV file [here](https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset), and unzip it.
4) Activate a Python virtual environment and download the necessary packages:
```
python3 -m venv env
cd FastAPI
pip install requirements.txt
```
5) Build the SQLite database from the dataset using the pre-made script:
```
python build_db.py -i /path/to/your_spotify_data.csv
```
6) Start the backend server:
```
uvicorn main:app --reload
```
6) In a different terminal window, install the required npm packages for the front end:
```
cd React/playlist-optimizer-project
npm install
```
7) Start the front end server:
```
npm start
```
8) View the web application at [localhost:3000](localhost:3000)

## General Usage
1) Navigate to the root folder:
```
cd playlist-optimizer
```
2) Activate the Python virtual environment:
```
# MacOS/Linux
source env/bin/activate
```
3) Start the backend server:
```
cd FastAPI
uvicorn main:app --reload
```
4) In a different terminal window, start the front end server:
```
cd React/playlist-optimizer-app
npm start
```
5) View the web application at [localhost:3000](localhost:3000).

## Work-in-progress items
- Enforcing the rule that slider percentages always sum up to 100%.
- Visualizing the algorithm running in real time.
- Displaying all optimal playlist orderings discovered by the algorithm before the weighted sum.

