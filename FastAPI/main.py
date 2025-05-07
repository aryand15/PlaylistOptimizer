from fastapi import FastAPI, HTTPException, Depends
from typing import Annotated, List
from sqlalchemy.orm import Session
from sqlalchemy import or_
from pydantic import BaseModel
from database import SessionLocal, engine
import models
from fastapi.middleware.cors import CORSMiddleware
import random
from playlist_optimizer import run_optimization

app = FastAPI()

origins = [
    'http://localhost:3000',
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class SongBase(BaseModel):
    track_id: str
    artists: str
    album_name: str
    track_name: str
    popularity: float
    duration_ms: float
    explicit: int
    danceability: float
    energy: float
    key: int
    loudness: float
    mode: int
    speechiness: float
    acousticness: float
    instrumentalness: float
    liveness: float
    valence: float
    tempo: float
    time_signature: int
    track_genre: str

class SongModel(SongBase):
    class Config:
        orm_mode = True

class OptimizationParams(BaseModel):
    tempo_smoothness: float
    energy_continuity: float
    mood_consistency: float
    key_compatibility: float

class OptimizationRequest(BaseModel):
    track_ids: List[str]
    params: OptimizationParams


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


db_dependency = Annotated[Session, Depends(get_db)]

models.Base.metadata.create_all(bind=engine)

@app.post("/songs", response_model=SongModel)
async def create_song(song: SongBase, db: db_dependency):
    db_song = models.Song(**song.dict())
    db.add(db_song)
    db.commit()
    db.refresh(db_song)
    return db_song

@app.get("/songs", response_model=List[SongModel])
async def read_transactions(db: db_dependency, skip: int = 0, limit: int = 100):
    transactions = db.query(models.Song).offset(skip).limit(limit).all()
    return transactions

@app.get("/songs/search")
async def search_songs(q: str, db: db_dependency, limit: int = 10):
    if not q:
        return []
    
    search_term = f"%{q}%"
    songs = db.query(models.Song).filter(
        or_(
            models.Song.track_name.ilike(search_term),
            models.Song.album_name.ilike(search_term),
            models.Song.artists.ilike(search_term)
        )
    ).limit(limit).all()
    return songs

class OptimizationParams(BaseModel):
    tempo_smoothness: float
    energy_continuity: float
    mood_consistency: float
    key_compatibility: float
    genre_jump_smoothness: float

class OptimizationRequest(BaseModel):
    track_ids: List[str]
    params: OptimizationParams

@app.post("/optimize-playlist")
async def optimize_playlist(request: OptimizationRequest, db: db_dependency):
    try:
        # Check if we have enough songs
        if len(request.track_ids) < 2:
            raise HTTPException(status_code=400, detail="Need at least 2 songs to optimize")
        
        print(f"Received track_ids: {request.track_ids}")
        print(f"Received params: {request.params}")
        
        # Get the songs from the database
        songs = db.query(models.Song).filter(
            models.Song.track_id.in_(request.track_ids)
        ).all()
        
        print(f"Found {len(songs)} songs")
        
        if not songs:
            raise HTTPException(status_code=404, detail="No songs found with the provided IDs")
        
        # Verify all songs were found
        if len(songs) != len(request.track_ids):
            print(f"Warning: Only found {len(songs)} out of {len(request.track_ids)} requested songs")
            missing_ids = set(request.track_ids) - set(song.track_id for song in songs)
            print(f"Missing IDs: {missing_ids}")
        
        # Convert the normalized 0-100 slider values to weights between 0 and 1
        weights = {
            "tempo": request.params.tempo_smoothness / 100,
            "energy": request.params.energy_continuity / 100,
            "mood": request.params.mood_consistency / 100,
            "key": request.params.key_compatibility / 100,
            "genre": request.params.genre_jump_smoothness / 100
        }
        
        print(f"Using weights: {weights}")
        
        # Set optimization parameters
        pop_size = min(100, max(20, len(songs) * 5))  # Scale population size based on playlist length
        n_gen = min(100, max(20, len(songs) * 3))     # Scale generations based on playlist length
        
        # Run the optimization
        optimized_playlist = run_optimization(songs, weights, pop_size, n_gen)
        
        return optimized_playlist
    
    except Exception as e:
            print(f"Error in optimize_playlist: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")