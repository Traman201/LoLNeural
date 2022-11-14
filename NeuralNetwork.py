import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import Sequential
from keras.layers.core import Dense
from matplotlib import pyplot as plt
from tensorflow import keras
import requests
import json
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import SpectralClustering
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics.pairwise import pairwise_distances

class NeuralNetwork:
    def Learn(self, filename):
        df = pd.read_csv(filename)
        train, validate, test = np.split(df.sample(frac=1, random_state=42), [int(.6*len(df)), int(.8*len(df))])
        df.head()
        abalone_features = train.copy()
        abalone_labels = abalone_features.pop('y')
        self.abalone_test = test.copy()
        abalone_y_test = self.abalone_test.pop('y')
        print(self.abalone_test)
        self.model = Sequential()
        self.model.trainable = True
        self.model.add(Dense(70, input_dim=10, activation='relu'))
        self.model.add(Dense(70, activation='relu'))
        self.model.add(Dense(1, activation='relu'))
        self.model.compile(optimizer='adam', loss='mae' , metrics=['accuracy']) 
        self.losses = self.model.fit(abalone_features, abalone_labels, epochs=20, )
        V = self.model.predict(self.abalone_test).round()
        print(V)
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


    #def __init__(self, filename):



Df = pd.read_csv("D:\\GIT_RPS\LoLNeural\\data\\YDataset.csv")
Df = Df.drop("y", axis = 1)
x = Df.values
DFD = pd.DataFrame(x)
print(DFD)

def get_request():
    get_param = {'index':'10'}
    get_response = requests.get(url="http://25.47.99.103:8080/local/status" , params=get_param)
    print(get_response)
    return get_response.content
def post_request(data):
    post_param = {'winChance':data}
    post_response = requests.post(url="http://25.47.99.103:8080/local/status" , data=post_param)

NN = NeuralNetwork()
NN.Learn("D:\\GIT_RPS\LoLNeural\\data\\N_YDataset.csv")
#NN.LoadModel("D:\\GIT_RPS\LoLNeural\\model")
#NN.SaveModel("D:\\GIT_RPS\LoLNeural\\model")



dd = {'kill': 0.6842105263157894, 'abilityPower': 0.56565402962591, 'armor': 0.5770452740270056, 'attackDamage': 0.5154098360655738, 'attackSpeed': 0.7226289800783245, 'healthMax': 0.6219512195121952, 'lifesteal': 0.5788216560509554, 'magicResist': 0.5779100037188546, 'movementSpeed': 0.5779100037188546, 'powerMax': 0.5678027280671758}
min_max_scaler = preprocessing.MinMaxScaler()
while(True):
    DATA = []
    data = json.loads(get_request())
    d1 = dict(list(data.items())[:len(data)-4])
    for key, value in dd.items(): DATA.append(value)
    df=pd.DataFrame(DATA )
    x_scaled = min_max_scaler.fit_transform(DFD.values)
    normalized_df=(np.transpose(df)-np.transpose(DFD.mean()))/np.transpose(DFD.std())
    print(np.transpose(df))
    #accuary = str(NN.NeuralWork(df)[0])
    print(NN.NeuralWork(np.transpose(df)))
   # post_request(accuary)



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