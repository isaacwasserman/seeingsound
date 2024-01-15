import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import os
import tqdm
import subprocess
from rich.progress import Progress
import concurrent.futures
import sqlite3
import glob
from pprint import pprint
import getpass

token = spotipy.util.prompt_for_user_token(
    getpass.getpass("Spotify Username:"),
    "user-library-read",
    client_id=getpass.getpass("Spotify ClientID:"),
    client_secret=getpass.getpass("Spotify ClientSecret:"),
    redirect_uri='https://isaacwasserman.com'
)


if token:
    sp = spotipy.Spotify(auth=token)
    print("Successfully logged in")
else:
    print("Can't get token")



def get_playlist_length(playlist_id):
    pl_id = 'spotify:playlist:' + playlist_id
    results = sp.playlist_items(pl_id,
                                    fields='total',
                                    additional_types=['track'],
                                    limit=1)
    return results["total"]

def get_playlist_tracks(playlist_id):
    pl_id = 'spotify:playlist:' + playlist_id
    offset = 0
    while True:
        response = sp.playlist_items(pl_id,
                                    offset=offset,
                                    fields='items.track.id, items.track.name, items.track.artists.name, items.track.external_urls, total',
                                    additional_types=['track'],
                                    limit=100,
                                    market='US')

        if len(response['items']) == 0:
            break

        for item in response['items']:
            yield item['track']

        offset = offset + len(response['items'])

def get_artist_albums(artist_id):
    results = sp.artist_albums(artist_id, album_type='album')
    albums = results['items']
    while results['next']:
        results = sp.next(results)
        albums.extend(results['items'])
    return albums

def get_album_tracks(album_id):
    results = sp.album_tracks(album_id, market='US')
    tracks = results['items']
    while results['next']:
        results = sp.next(results)
        tracks.extend(results['items'])
    return tracks

def download_track(track, output_dir="tracks", bar=None, task=None, spotdl_path="/Users/isaac/miniforge3/envs/pytorch/bin/spotdl"):
    track_id = track['id']
    if not os.path.exists(os.path.join(output_dir, track_id + ".mp3")):
        # print("running:", [spotdl_path, "download", track['external_urls']['spotify'],  "--output", output_dir + "/{track-id}"])
        subprocess.run([spotdl_path, "download", track['external_urls']['spotify'],  "--output", output_dir + "/{track-id}"], stdout=subprocess.DEVNULL)
    if bar is not None:
        bar.update(1)

def download_playlist(playlist_id, output_dir="tracks", spotdl_path="/Users/isaac/miniforge3/envs/pytorch/bin/spotdl"):
    num_tracks = get_playlist_length(playlist_id)
    tracks = get_playlist_tracks(playlist_id)
    tracks = [track for track in tracks]
    with tqdm.tqdm(total=num_tracks) as bar:
        for track in tracks:
            download_track(track, output_dir=output_dir, spotdl_path=spotdl_path, bar=bar)
    return tracks

def download_album(album_id, bar=None, task=None):
    tracks = get_album_tracks(album_id)
    # Download tracks in parallel
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=10)
    for track in tracks:
        pool.submit(download_track, track, "tracks", bar, task)
    pool.shutdown(wait=True)

def download_artist(artist_id):
    albums = get_artist_albums(artist_id)
    num_tracks = [album["total_tracks"] for album in albums]
    with Progress() as bar:
        albums_task = bar.add_task("Downloading albums...", total=len(albums))
        songs_task = bar.add_task("Downloading tracks...", total=sum(num_tracks))
        for album in albums:
            download_album(album['id'], bar=bar, task=songs_task)
            bar.update(albums_task, advance=1)

def build_db(tracks_dir="tracks"):
    with sqlite3.connect('tracks.db') as db:
        cursor = db.cursor()
        paths = list(glob.glob(f"{tracks_dir}/*.mp3"))
        for path in tqdm.tqdm(paths, total=len(paths)):
            track_id = os.path.basename(path).split(".")[0]
            track = sp.track(track_id)
            audio_path = os.path.join(os.getcwd(), "tracks", track_id + ".mp3")
            new_row_data = (track_id, track["name"], ", ".join([artist["name"] for artist in track["artists"]]), audio_path, b'')
            insert_query = '''INSERT OR IGNORE INTO tracks (id, title, artist, audio_path, embedding) VALUES (?, ?, ?, ?, ?)'''
            cursor.execute(insert_query, new_row_data)
            db.commit()