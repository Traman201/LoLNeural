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
        abalone_test = test.copy()
        abalone_y_test = abalone_test.pop('y')
        self.model = Sequential()
        self.model.add(Dense(56, input_dim=9, activation='relu'))
        self.model.add(Dense(56, activation='relu'))
        self.model.add(Dense(1, activation='relu'))
        self.model.compile(optimizer='adam', loss='mae') 
        self.losses = self.model.fit(abalone_features, abalone_labels, epochs=30, )
        V = self.model.predict(abalone_test).round()
    def SaveModel(self , path):
        self.model.save(path)

    def LoadModel(self , path): self.model = keras.models.load_model(path)

    def NeuralWork(self , data): return self.model.predict(data)
    def Normaliation(self): 
        return 0
    def ShowNeuralData(self):
        fig = plt.figure()
        ax = fig.add_subplot()
        fig.subplots_adjust(top=0.85)
        ax.set_title('обучение нейронной сети' , fontsize=14)
        ax.set_xlabel('эпохи' , fontsize=14)
        ax.set_ylabel('ошибка' , fontsize=14)   
        plt.plot(self.losses.history["loss"])
        plt.show()


    #def __init__(self, filename):



get_param = {'index':'10'}
post_param = {'winChance':'34'}
def get_request():
    get_response = requests.get(url="http://25.66.210.56:8080/local/status" , params=get_param)
    post_response = requests.post(url="http://25.66.210.56:8080/local/status" , data=post_param)
    print(post_response)
    return get_response.content
  #  print(post_response.content)

NN = NeuralNetwork()
NN.LoadModel("D:\\GIT_RPS\LoLNeural\\model")

while(True):
    DATA = []
    data = json.loads(get_request())
    d1 = dict(list(data.items())[:len(data)-3])
    for key, value in d1.items(): DATA.append(value)
    df=pd.DataFrame(DATA)
    normalized_df=(df-df.mean())/df.std()
    #print(df[0].to_numpy())
    print(d1)
    #print(NN.NeuralWork(normalized_df.to_numpy()))





"""
NN = NeuralNetwork()
NN.Learn("D:\\GIT_RPS\LoLNeural\\data\\N_YDataset.csv")
NN.ShowNeuralData()
NN.SaveModel("D:\\GIT_RPS\LoLNeural\\model")
"""

"""
filename = "C:\\neural\\N_YDataset.csv"
df = pd.read_csv(filename)
train, validate, test = np.split(df.sample(frac=1, random_state=42), [int(.6*len(df)), int(.8*len(df))])
df.head()
abalone_features = train.copy()
abalone_labels = abalone_features.pop('y')

abalone_test = test.copy()
abalone_y_test = abalone_test.pop('y')

np.savetxt('TYdata.txt', abalone_y_test, delimiter=',')
model = Sequential()
model.add(Dense(56, input_dim=28, activation='relu'))
model.add(Dense(56, activation='relu'))
model.add(Dense(1, activation='relu'))
model.compile(optimizer='adam', loss='mae') 
losses = model.fit(abalone_features, abalone_labels, epochs=30, )
V = model.predict(abalone_test).round()

print(V[0])
np.savetxt('Ydata.txt', V, delimiter=',')
Vnp = np.array(V)
abalone_y_testnp = abalone_y_test.to_numpy()
er = 0
for i in range(0,len(Vnp)):
    if(int(Vnp[i]) != abalone_y_testnp[i] ): er +=1
print("error is:" , er/len(Vnp) * 100)

fig = plt.figure()
ax = fig.add_subplot()
fig.subplots_adjust(top=0.85)

# Set titles for the figure and the subplot respectively
ax.set_title('обучение нейронной сети' , fontsize=14)

ax.set_xlabel('эпохи' , fontsize=14)
ax.set_ylabel('ошибка' , fontsize=14)

print(losses.history)
plt.plot(losses.history["loss"])
plt.show()
"""
#plt.plot(losses.history['val_acc'])