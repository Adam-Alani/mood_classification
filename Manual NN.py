import spotipy
from spotipy import util
from spotipy.oauth2 import SpotifyClientCredentials
import pandas as pd
import numpy as np
from pandas import read_csv
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import tensorflow as tf


# Set up Connection
client_id = "25de10f8c435462cb6a5aab079cea2a9"  # Need to create developer profile
client_secret = "8fe6009a4faa4b6e891ad39e05488c9d"
username = "21tqxjahbydtkdwzchxyhuvhy"
scope = 'user-library-read playlist-read-private'
redirect_uri = "http://localhost:8888/callback/"
client_credentials_manager = SpotifyClientCredentials(client_id=client_id,
                                                      client_secret=client_secret)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)
token = util.prompt_for_user_token(username, scope, client_id, client_secret, redirect_uri)
sp = spotipy.Spotify(auth=token)


# Imports
happy = read_csv(r"C:\Users\xatom\Desktop\MusicWeb\happydata.csv")
happy = happy[happy.artist_name != 'artist_name']

sad = read_csv(r"C:\Users\xatom\Desktop\MusicWeb\saddata.csv")
sad = sad[sad.artist_name != 'artist_name']

chill = read_csv(r"C:\Users\xatom\Desktop\MusicWeb\chill.csv")
chill = chill[chill.artist_name != 'artist_name']

excited = read_csv(r"C:\Users\xatom\Desktop\MusicWeb\excited.csv")
excited = excited[excited.artist_name != 'artist_name']

mood = pd.concat([happy, sad])
mood = mood.drop_duplicates(subset=['track_names'], keep=False)


# Data Cleanup
X = mood[['energy', 'valence', 'danceability']]
target_list = mood.iloc[:, -1]
encoder = LabelEncoder()
y = encoder.fit_transform(target_list)


scaler = MinMaxScaler()
scaler.fit(X)
scaled_data = scaler.transform(X)
scaled_df = pd.DataFrame(scaled_data, columns=X.columns)

X_train, X_test, y_train, y_test = train_test_split(scaled_df, y, test_size=0.2, random_state = 42)

# Neural Network
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(3, activation='relu', input_shape=(X_train.shape[1],)))

model.add(tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)))
model.add(tf.keras.layers.Dropout(0.3))

model.add(tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)))
model.add(tf.keras.layers.Dropout(0.3))


model.add(tf.keras.layers.Dense(2, activation='softmax'))


optimiser = tf.keras.optimizers.Adam(learning_rate=0.0001)
model.compile(optimizer=optimiser, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

history = model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=8, epochs=10)
test_loss, test_acc = model.evaluate(X_test, y_test)
test_loss, test_acc = model.evaluate(X_test, y_test)

print(model.summary())
print(test_loss, test_acc)

# Plot Data
def plot_history(history):

    fig, axs = plt.subplots(2)

    # create accuracy sublpot
    axs[0].plot(history.history["accuracy"], label="train accuracy")
    axs[0].plot(history.history["val_accuracy"], label="test accuracy")
    axs[0].set_ylabel("Accuracy")
    axs[0].legend(loc="lower right")
    axs[0].set_title("Accuracy eval")

    # create error sublpot
    axs[1].plot(history.history["loss"], label="train error")
    axs[1].plot(history.history["val_loss"], label="test error")
    axs[1].set_ylabel("Error")
    axs[1].set_xlabel("Epoch")
    axs[1].legend(loc="upper right")
    axs[1].set_title("Error eval")

    plt.show()
plot_history(history)

playlist_id = 'spotify:playlist:37i9dQZF1DX64Y3du11rR1'
playlist = sp.playlist(playlist_id)

tracks = playlist["tracks"]
songs = tracks["items"]

track_ids = []
track_names = []
artist_name = []
energy = []
danceability = []
loudness = []
mode = []
valence = []
tempo = []

for i in range(0, len(songs)):
    if songs[i]['track']['id'] != None: # Removes  local tracks, if any
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
track_dataframe = pd.DataFrame({'artist_name' : artist_name, 'track_names' : track_names, 'track_ids' : track_ids, 'energy' : energy,
                                'danceability' : danceability, 'loudness' : loudness, 'mode' : mode, 'valence' : valence, 'tempo' : tempo})

Xnew = track_dataframe[['energy', 'valence', 'danceability']]
scaler = MinMaxScaler()
scaler.fit(Xnew)
scaled_data = scaler.transform(Xnew)
Xnew = pd.DataFrame(scaled_data, columns=X.columns)

probability_model = tf.keras.Sequential([model,
                                         tf.keras.layers.Softmax()])

preds = probability_model.predict(Xnew)
song = track_dataframe['track_names'].tolist()
i = 0
while i < len(preds):
    np.sum(preds[i])
    labels = np.argmax(preds[i])
    if labels == 0:
        print(song[i] + ": Happy")
    else:
        print(song[i] + ": Sad")
    i += 1


# 0 = Happy, 1 = Sad or :        0 = 'Chill" , 1= 'excited, 2= 'happy' 3= 'sad'