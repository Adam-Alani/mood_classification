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
from tensorflow.python.saved_model import saved_model

client_id = ""  # Need to create developer profile
client_secret = ""
username = ""
scope = 'user-library-read playlist-read-private'
redirect_uri = "http://localhost:8888/callback/"
client_credentials_manager = SpotifyClientCredentials(client_id=client_id,
                                                      client_secret=client_secret)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)
token = util.prompt_for_user_token(username, scope, client_id, client_secret, redirect_uri)
sp = spotipy.Spotify(auth=token)

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


X = mood[['energy', 'valence', 'danceability']]
target_list = mood.iloc[:, -1]
encoder = LabelEncoder()
y = encoder.fit_transform(target_list)


scaler = MinMaxScaler()
scaler.fit(X)
scaled_data = scaler.transform(X)
scaled_df = pd.DataFrame(scaled_data, columns=X.columns)

X_train, X_test, y_train, y_test = train_test_split(scaled_df, y, test_size=0.2, random_state = 42)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(3, activation='relu', input_shape=(X_train.shape[1],)))

model.add(tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)))
model.add(tf.keras.layers.Dropout(0.3))

model.add(tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)))
model.add(tf.keras.layers.Dropout(0.3))


model.add(tf.keras.layers.Dense(2, activation='softmax'))

optimiser = tf.keras.optimizers.Adam(learning_rate=0.0001)
model.compile(optimizer=optimiser, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

history = model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=1, epochs=200)
test_loss, test_acc = model.evaluate(X_test, y_test)


model.save('saved_model/MusicModel')


print(model.summary())







# 0 = Happy, 1 = Sad or :        0 = 'Chill" , 1= 'excited, 2= 'happy' 3= 'sad'
