import spotipy
from spotipy import util
from spotipy.oauth2 import SpotifyClientCredentials
import pandas as pd
from pandas import read_csv
import json

client_id = "25de10f8c435462cb6a5aab079cea2a9"
client_secret = "8fe6009a4faa4b6e891ad39e05488c9d"
username = "21tqxjahbydtkdwzchxyhuvhy"
scope = 'user-library-read playlist-read-private'
redirect_uri = "http://localhost:8888/callback/"
client_credentials_manager = SpotifyClientCredentials(client_id=client_id,
                                                      client_secret=client_secret)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)
token = util.prompt_for_user_token(username, scope, client_id, client_secret, redirect_uri)
sp = spotipy.Spotify(auth=token)

def fetch_playlists():
    results = sp.search('happy', type='playlist')
    items = results['playlists']['items']
    i = 0
    playlists = []

    while i < 10:
        if len(items) > 0:
            playlist = items[i]
            playlists.append(playlist['uri'])
            i += 1

    track_ids = []
    track_names = []
    artist_name = []
    energy = []
    danceability = []
    loudness = []
    mode = []
    valence = []
    tempo = []

    for j in range(len(playlists)):
        pl_id = str(playlists[j])
        offset = 0
        response = sp.playlist_tracks(pl_id, offset=offset, fields='items.track.id,total')
        plist = sp.playlist(pl_id)
        tracks = plist["tracks"]
        songs = tracks["items"]
        offset = offset + len(response['items'])
        if len(response['items']) == 0:
            break
        for i in range(0, len(songs)):
            if songs[i]['track']['id'] is not None:  # Removes  local tracks, if any
                track_ids.append(songs[i]['track']['id'])
                track_names.append(songs[i]['track']['name'])
                artist_name.append(songs[i]['track']['artists'][0]['name'])
                feature = sp.audio_features(songs[i]['track']['id'])
                energy.append(feature[0]['energy'])
                danceability.append(feature[0]['danceability'])
                loudness.append(feature[0]['loudness'])
                mode.append(feature[0]['mode'])
                valence.append(feature[0]['valence'])
                tempo.append(feature[0]['tempo'])


    track_dataframe = pd.DataFrame({'artist_name': artist_name, 'track_names': track_names, 'track_ids': track_ids, 'energy': energy, 'danceability': danceability, 'loudness': loudness, 'mode': mode, 'valence': valence, 'tempo': tempo, 'target': 'happy'})
    print(track_dataframe.to_string())

def data_cleanup():
    happy = read_csv(r"C:\Users\xatom\Desktop\MusicWeb\happy.csv")
    happy = happy.drop_duplicates(subset=['track_names'])
    happy = happy.drop('Unnamed: 0', 1)
    sad = read_csv(r"C:\Users\xatom\Desktop\MusicWeb\sad.csv")
    sad = sad.drop_duplicates(subset=['track_names'])
    sad = sad.drop('Unnamed: 0', 1)

    #happy.to_csv(r"C:\Users\xatom\Desktop\MusicWeb\happydata.csv", index = False)
    #sad.to_csv(r"C:\Users\xatom\Desktop\MusicWeb\saddata.csv", index = False)

    mood = pd.concat([happy, sad])
    mood = mood.drop_duplicates(subset=['track_names'], keep=False)

#track_dataframe.to_csv(r"C:\Users\xatom\Desktop\MusicWeb\happy.csv", mode='a')

