import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import Sequential
from keras.layers.core import Dense
from matplotlib import pyplot as plt
from tensorflow import keras
import requests
import json

class NeuralNetwork:
    def Learn(self, filename):
        df = pd.read_csv(filename)
        train, validate, test = np.split(df.sample(frac=1, random_state=42), [int(.6*len(df)), int(.8*len(df))])
        df.head()
        abalone_features = train.copy()
        abalone_labels = abalone_features.pop('y')
        self.abalone_test = test.copy()
        abalone_y_test = self.abalone_test.pop('y')
        self.model = Sequential()
        self.model.trainable = True
        self.model.add(Dense(70, input_dim=10, activation='relu'))
        self.model.add(Dense(70, activation='relu'))
        self.model.add(Dense(1, activation='relu'))
        self.model.compile(optimizer='adam', loss='mae' , metrics=['accuracy']) 
        self.losses = self.model.fit(abalone_features, abalone_labels, epochs=30, )
        V = self.model.predict(self.abalone_test).round()
       # print(V)
    def SaveModel(self , path): self.model.save(path)
    def LoadModel(self , path): 
        self.model = keras.models.load_model(path)
    def NeuralWork(self , data): return self.model.predict(data , batch_size = 1).round()
    def ShowNeuralData(self):
        fig = plt.figure()
        ax = fig.add_subplot()
        fig.subplots_adjust(top=0.85)
        ax.set_title('обучение нейронной сети' , fontsize=14)
        ax.set_xlabel('эпохи' , fontsize=14)
        ax.set_ylabel('ошибка' , fontsize=14)   
        plt.plot(self.losses.history["loss"])
        plt.show()

def get_request(_url):
    get_param = {'index':'10'}
    get_response = requests.get(url=_url , params=get_param)
    print(get_response)
    return get_response.content
def post_request(_url , data):
    post_param = {'winChance':data}
    post_response = requests.post(url=_url , data=post_param)
def start_request(url):
    while(True):
        DATA = []
        data = json.loads(get_request(url))
        d1 = dict(list(data.items())[:len(data)-5])
        for key, value in d1.items(): DATA.append(value)
        df=pd.DataFrame(DATA )
        normalized_df=(np.transpose(df)-DFD.mean().to_numpy())/DFD.std().to_numpy()
        accuary = str(NN.NeuralWork(df)[0])
        if(accuary == '0'):  accuary = 'draw'
        if(accuary == '1'):  accuary = 'lose'
        if(accuary == '2'):  accuary = 'win'
        post_request(url , accuary)



Df = pd.read_csv("D:\\GIT_RPS\LoLNeural\\data\\YDataset.csv")
Df = Df.drop("y", axis = 1)
x = Df.values
DFD = pd.DataFrame(x)
NN = NeuralNetwork()
#N.Learn("D:\\GIT_RPS\LoLNeural\\data\\N_YDataset.csv")
NN.LoadModel("D:\\GIT_RPS\LoLNeural\\model")
#NN.SaveModel("D:\\GIT_RPS\LoLNeural\\model")
#dd = {'kill': 13, 'abilityPower': 178, 'armor': 193, 'attackDamage': 104, 'attackSpeed': 79, 'healthMax': 2126, 'lifesteal': 31, 'magicResist': 115, 'movementSpeed': 41, 'powerMax':1688}
start_request("http://25.47.99.103:8080/local/status")