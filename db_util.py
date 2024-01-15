import numpy as np
import pandas as pd
import sqlite3

def get_tracks():
    db = sqlite3.connect('tracks.db')
    cursor = db.cursor()
    cursor.execute('SELECT * FROM tracks')
    columns = list(map(lambda x: x[0], cursor.description))
    tracks = cursor.fetchall()
    db.close()
    df = pd.DataFrame(tracks, columns=columns)
    df['embedding'] = df['embedding'].apply(lambda x: np.frombuffer(x))
    return df

def add_embeddings(track_ids, embeddings):
    if type(track_ids) == str:
        track_ids = [track_ids]
        embeddings = [embeddings]
    db = sqlite3.connect('tracks.db')
    cursor = db.cursor()
    for i in range(len(track_ids)):
        track_id = track_ids[i]
        embedding = embeddings[i]
        embedding_bytes = embedding.tobytes()
        cursor.execute('UPDATE tracks SET embedding=? WHERE id=?', (embedding, track_id))
        db.commit()
    db.close()