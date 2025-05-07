from database import Base
from sqlalchemy import Column, Integer, String, Float

class Song(Base):
    __tablename__ = "songs"
    
    track_id = Column(String, primary_key=True, index=True)
    artists = Column(String)
    album_name = Column(String)
    track_name = Column(String)
    popularity = Column(Float)
    duration_ms = Column(Float)
    explicit = int
    danceability = Column(Float)
    energy = Column(Float)
    key = Column(Integer)
    loudness = Column(Float)
    mode = Column(Integer)
    speechiness = Column(Float)
    acousticness = Column(Float)
    instrumentalness = Column(Float)
    liveness = Column(Float)
    valence = Column(Float)
    tempo = Column(Float)
    time_signature = Column(Integer)
    track_genre = Column(String)