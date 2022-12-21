import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import Sequential
from keras.layers.core import Dense
from matplotlib import pyplot as plt
from tensorflow import keras
import requests
import json
from keras.layers import Dense, Dropout
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from keras.optimizers import RMSprop
def get_layer(model,x):
    from keras import backend as K

    get_3rd_layer_output = K.function([model.layers[0].input],
                                      [model.layers[2].output])
    layer_output = get_3rd_layer_output([x])[0]
    print(layer_output.shape)
    return layer_output
class NeuralNetwork:
    def Learn(self, filename):
        df = pd.read_csv(filename)
        train, validate, test = np.split(df.sample(frac=1, random_state=42), [int(.6*len(df)), int(.8*len(df))])
        df.head()
        abalone_features = train.copy()
        abalone_labels = abalone_features.pop('y')
        self.abalone_test = test.copy()
        self.abalone_test =  self.abalone_test.astype('float32')
        abalone_y_test = self.abalone_test.pop('y')
        self.model = Sequential()
        self.model.trainable = True
        self.model.add(Dense(70, input_dim=10, activation='relu'))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(70, activation='relu'))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(3, activation='softmax'))
        self.model.compile(loss='sparse_categorical_crossentropy',
          optimizer="rmsprop", metrics=['accuracy']) 
        self.losses = self.model.fit(abalone_features, abalone_labels, epochs=10, )
        V = self.model.predict(self.abalone_test)
        print(V)
    def SaveModel(self , path): self.model.save(path)
    def LoadModel(self , path): 
        self.model = keras.models.load_model(path)
    def NeuralWork(self , data):
        data = data.astype('float32')
        #print("data is" , get_layer(self.model , data))
        out = self.model.predict(data , batch_size = 1)
        return out#self.model.get_layer('dense_2').output
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
    mmscaler = MinMaxScaler()
    while(True):
        DATA = []
        data = json.loads(get_request(url))
        d1 = dict(list(data.items())[:len(data)-5])
        for key, value in d1.items(): DATA.append(value)
        df=pd.DataFrame(DATA )
        normalized_df=(np.transpose(df)-DFD.mean().to_numpy())/DFD.std().to_numpy()
       # print(normalized_df)
        #
        X_train_norm = mmscaler.fit_transform(DFD)
        normalized_df = mmscaler.transform(np.transpose(df))
        acc = NN.NeuralWork(normalized_df)[0]
      #  for i in range(0  , 3):
       #    print("{:.12f}".format(float(acc[i])) , " ")
        accuary = str(acc)
       # print("{:.12f}".format(float(acc)))
        #print(accuary)
        procent = (acc[2] - acc[0] + 1) * 50
        if(accuary == '[0.]'):  accuary = 'Поражение'
        if(accuary == '[1.]'):  accuary = 'Ничья'
        if(accuary == '[2.]'):  accuary = 'Победа'
        print(procent)

        pr_str = str(round(procent, 2))

        post_request(url , pr_str)

Df = pd.read_csv("D:\\GIT_RPS\LoLNeural\\data\\YDataset.csv")
Df = Df.drop("y", axis = 1)
x = Df.values
DFD = pd.DataFrame(x)
NN = NeuralNetwork()
#NN.Learn("D:\\GIT_RPS\LoLNeural\\data\\N_YDataset.csv")
NN.LoadModel("D:\\GIT_RPS\LoLNeural\\model")
#NN.SaveModel("D:\\GIT_RPS\LoLNeural\\model")
dd = {'kill': 13, 'abilityPower': 178, 'armor': 193, 'attackDamage': 104, 'attackSpeed': 79, 'healthMax': 2126, 'lifesteal': 31, 'magicResist': 115, 'movementSpeed': 41, 'powerMax':1688}


start_request("http://25.66.210.56:8080/local/status")
