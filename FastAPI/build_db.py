import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sqlalchemy.orm import Session
from database import SessionLocal, engine
import models
import argparse


def load_and_clean(filepath: str) -> pd.DataFrame:
    """
    Cleans a CSV dataset containing Spotify API music features, 
    returning the final pre-processed Pandas DataFrame.

    Parameters:
        filepath (str): The path to the CSV file.

    Returns:
        df (pd.DataFrame): The final pre-processed DataFrame
    """
    

    # Load the data, remove all rows with any N/A entries
    df = pd.read_csv(filepath)
    df = df.dropna()

    # In this dataset, songs with multiple genres are separated into almost duplicate entries with different genres
    # We need to collapse these duplicate entries into one entry, aggregating the genres together
    genre_df = (
        df.groupby('track_id')['track_genre']
        .agg(lambda x: ';'.join(sorted(set(x))))
        .reset_index()
    )

    df = (
        df.drop_duplicates(subset=["track_id"])
        .drop(columns=["track_genre"])
        .merge(genre_df, on="track_id")
    )

    # Type cast relevant features to int as required
    df["duration_ms"] = df["duration_ms"].astype(int)
    df["explicit"] = df["explicit"].astype(int)
    df["key"] = df["key"].astype(int)
    df["mode"] = df["mode"].astype(int)
    df["time_signature"] = df["time_signature"].astype(int)

    # Normalize any continuous numerical features to lie in [0,1]
    # This will be needed when we perform the multi-objective optimization
    # We wouldn't want features with naturally higher values to skew calculations
    numeric_cols = [
        "popularity","duration_ms","danceability","energy",
        "loudness","speechiness","acousticness","instrumentalness",
        "liveness","valence","tempo"
    ]
    scaler = MinMaxScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    
    # Return the final processed DataFrame
    return df


def dump_to_db(df: pd.DataFrame, batch_size: int = 500):
    """
    Takes the cleaned DataFrame and writes to the songs table via SQLAlchemy, 
    reusing the FastAPI models.

    Parameters:
        df (pd.DataFrame): the cleaned DataFrame
        batch_size (int): how many songs to save to the DB at once. Default is 500.

    """
    # Drop all tables
    models.Base.metadata.drop_all(bind=engine)
    # Recreate tables
    models.Base.metadata.create_all(bind=engine)

    # Open a session
    db: Session = SessionLocal()

    try:
        
        # Iterate in batches to avoid blowing out memory / transaction size
        for start in range(0, len(df), batch_size):
            batch = df.iloc[start : start + batch_size]
            to_add = []
            for _, row in batch.iterrows():
                
                # Build an ORM object
                song = models.Song(
                    track_id        = row["track_id"],
                    artists         = row["artists"],
                    album_name      = row["album_name"],
                    track_name      = row["track_name"],
                    popularity      = float(row["popularity"]),
                    duration_ms     = float(row["duration_ms"]),
                    explicit        = int(row["explicit"]),
                    danceability    = float(row["danceability"]),
                    energy          = float(row["energy"]),
                    key             = int(row["key"]),
                    loudness        = float(row["loudness"]),
                    mode            = int(row["mode"]),
                    speechiness     = float(row["speechiness"]),
                    acousticness    = float(row["acousticness"]),
                    instrumentalness= float(row["instrumentalness"]),
                    liveness        = float(row["liveness"]),
                    valence         = float(row["valence"]),
                    tempo           = float(row["tempo"]),
                    time_signature  = int(row["time_signature"]),
                    track_genre     = row["track_genre"]
                )
                to_add.append(song)

            # Bulk save & commit each batch
            db.bulk_save_objects(to_add)
            db.commit()
        print(f"Inserted/updated {len(df)} songs into the DB.")
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def main():
    p = argparse.ArgumentParser(
        description="Load a CSV of tracks, clean it, and populate song.db"
    )
    p.add_argument(
        "--input", "-i",
        required=True,
        help="Path to the raw CSV (e.g. ~/Downloads/your.csv)"
    )
    args = p.parse_args()

    print(f"Loading and cleaning data from {args.input} ...")
    df = load_and_clean(args.input)

    print(f"Dumping {len(df)} records into song.db ...")
    dump_to_db(df)

    print("All data saved.")

if __name__ == "__main__":
    main()