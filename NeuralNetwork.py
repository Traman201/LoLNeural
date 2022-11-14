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
        #print(self.abalone_test)
        self.model = Sequential()
        self.model.add(Dense(56, input_dim=10, activation='relu'))
        self.model.add(Dense(56, activation='relu'))
        self.model.add(Dense(1, activation='relu'))
        self.model.compile(optimizer='adam', loss='mae') 
        self.losses = self.model.fit(abalone_features, abalone_labels, epochs=30, )
       # V = self.model.predict(self.abalone_test).round()
    def SaveModel(self , path): self.model.save(path)
    def LoadModel(self , path): self.model = keras.models.load_model(path)
    def NeuralWork(self , data): return self.model.predict(data)
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
#NN.Learn("D:\\GIT_RPS\LoLNeural\\data\\N_YDataset.csv")
#NN.SaveModel("D:\\GIT_RPS\LoLNeural\\model")
NN.LoadModel("D:\\GIT_RPS\LoLNeural\\model")


dd = {'kill': 100.0, 'abilityPower': 5.0, 'armor': 13.149999999999999, 'attackDamage': -22.740000000000002, 'attackSpeed': -0.19700000000000006, 'healthMax': 27.0, 'lifesteal': 0.0, 'magicResist': 2.1999999999999957, 'movementSpeed': -5.0, 'powerMax': -379.0}

while(True):
    DATA = []
    data = json.loads(get_request())
    d1 = dict(list(data.items())[:len(data)-3])
    for key, value in dd.items(): DATA.append(value)
    df=pd.DataFrame(DATA )
    #df.head()
    #df.transpose()
    normalized_df=(df-df.mean())/df.std()
    #print(df[0].to_numpy())
    #print(d1.to_numpy())
    #test_input = np.random.random((1, 10))
    #print(test_input)
    print(np.transpose(normalized_df))
    #print(np.transpose(normalized_df))
    print(NN.NeuralWork(np.transpose(normalized_df)))



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