from flask import Flask, request, jsonify, render_template
from flask_socketio import SocketIO 
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os
import pathlib
from PIL import Image
import csv
import pandas as pd
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
import tensorflow.keras as keras
import warnings
from sklearn.preprocessing import StandardScaler
import pan
warnings.filterwarnings('ignore')


app = Flask(__name__)
stfl=keras.models.load_model('F:\\Sem-5\\SGP\\source\\my_model2.h5')
@app.route('/')
def home():
    return render_template('index.html')
#@app.route('/predict',methods=['POST'])

@app.route('/predict',methods=['POST','GET'])
def predict():
    if request.method=='POST':
        result = request.form
        new=[]

        song_path=result["path"]
    #return (song_path)
        data=pd.read_csv('F:\\Sem-5\\SGP\\source\\data2.csv')

        x=data.iloc[:,1:-1].values
        sc=StandardScaler()

        x=sc.fit_transform(x)

        #print(x)
        #print(x.shape)








        new_path="F:\\Sem-5\\SGP\\Data\\genres_original\\blues\\"+song_path
        
        new_song=new_path
        new_signal,sr=librosa.load(new_song,duration=60)
        f1 = librosa.feature.mfcc(new_signal,sr=sr)
        f2 = librosa.feature.rms(y=new_signal)
        f3 = librosa.feature.spectral_centroid(y=new_signal, sr=sr)
        f4 = librosa.feature.spectral_bandwidth(y=new_signal, sr=sr)
        f5 = librosa.feature.spectral_rolloff(y=new_signal, sr=sr)
        f6 = librosa.feature.zero_crossing_rate(new_signal)
        f7 = librosa.feature.chroma_stft(y=new_signal, sr=sr)

    ##print(mfccs)
    ##print(mfccs.shape)

    ##librosa.display.specshow(mfccs, sr=sr,x_axis='time')
    ##plt.xlabel("Time") 
    ##plt.ylabel("MFCC")
    ##plt.colorbar()
    ##plt.show()
        
        new.append(np.mean(f7))
        new.append(np.mean(f2))
        new.append(np.mean(f3))
        new.append(np.mean(f4))
        new.append(np.mean(f5))
        new.append(np.mean(f6))



        for e in f1:

            new.append(np.mean(e))

    ##print(to)
        final_new=[new]
        final_new=np.array(final_new)
        print(final_new)
        #sc=StandardScaler()
        #final=sc.fit_transform(final_new)
        final_pree= stfl.predict(sc.transform(final_new))
        r=np.argmax(final_pree)
    #print(final_pre)
    #print(r)
        if r==0:
            r=("The genre for this song is Blues")
        elif r==1:
            r=("The genere for this song is Classical")
        elif r==2:
            r=("The genere for this song is Country")
        elif r==3:
            r=("The genere for this song is Disco")
        elif r==4:
            r=("The genere for this song is HipHop")
        elif r==5:
            r=("The genere for this song is Jazz")
        elif r==6:
            r=("The genere for this song is Mental")
        elif r==7:
            r=("The genere for this song is Pop")
        elif r==8:
            r=("The genere for this song is Reggae")
        elif r==9:
            r=("The genere for this song is Rock")
        print(r)


        tags = ['Blues', 'Classical', 'Country', 'Disco', 'HipHop', 'Jazz', 'Mental', 'Pop', 'Reggae', 'Rock']
        tags = np.array(tags)

        colors = ['b','g','c','r','m','k','y','#ff1122','#5511ff','#44ff22']
        fig, ax = plt.subplots()
        index = np.arange(tags.shape[0])
        opacity = 1
        bar_width = 0.2
        mean=final_pree.flatten() 
    #for g in rini_array1.flatten() ange(0, tags.shape[0]):
        plt.bar(x=index, height=mean, width=bar_width, alpha=opacity, color=colors)
        plt.rcParams["figure.figsize"] = (10, 6)

        plt.xlabel('Genres')
        plt.ylabel('Percentage')
        plt.title('Scores by genre')
        plt.xticks(index + bar_width / 2, tags)
        plt.tight_layout()
        fig.autofmt_xdate()
        plt.savefig("F:\\Sem-5\\SGP\\source\\static\\plot.png")
        return render_template("index.html",output="{}".format(r))











if __name__ == "__main__":
    app.run(debug=True)