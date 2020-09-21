#Imports
from pychord import Chord
import pandas as pd

import random

#Files (at a certain point we're gonna ditch this)
Chords1 = pd.read_csv(r"C:\Users\xatom\Desktop\MusicWeb\chord1.csv", encoding='cp1252')

#ChordPicker

print("Root? " "['c','c#','d','d#','e','f','f#','g','g#','a','a#','b']" )
root = input(str()).upper()
print("Quality? (m7/Maj7/Dim etc..)")
quality = input(str())
I = Chord(root + quality)
print(I.info())


#ChordGenerator
#Pseudo Code:
#If quality = m7 , and mood = sad, print a chord progression where first chord is a minor chord
def ChordProg_Gen(I):
    major = ('maj7', 'maj', 'maj9')
    minor = ('m7', 'min', 'm6', 'm9')
    dom = ('7')
    print("How ya feelin?")
    mood = input(str())
    if quality in minor :
        if Chords1["Style"].str.contains(mood).any():
            print(Chords1['Progression'][Chords1['Style'].str.contains(mood)].sample(n=1).to_string(index=False))
    if quality in major :
        if Chords1["Style"].str.contains(mood).any():
            print(Chords1['Progression'][Chords1['Style'].str.contains(mood)].sample(n=1).to_string(index=False))
    if quality in dom :
        if Chords1["Style"].str.contains(mood).any():
            print(Chords1['Progression'][Chords1['Style'].str.contains(mood)].sample(n=1).to_string(index=False))



#important things we need but dont know where to put yet
    ROMAN_NUM = {"I": 0,
                       "i": 0,
                       "ii": 2,
                       "ii°": 2,
                       "III": 4,
                       "iii": 3,
                       "IV": 5,
                       "iv": 5,
                       "V": 7,
                       "v": 7,
                       "vi": 9,
                       "VI": 9,
                       "VII": 11,
                       "vii°": 10}


print(ChordProg_Gen(I))



