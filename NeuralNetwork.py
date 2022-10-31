import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import Sequential
from keras.layers.core import Dense

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
losses = model.fit(abalone_features, abalone_labels, epochs=10, )
V = model.predict(abalone_test).round()
print(V[0])
np.savetxt('Ydata.txt', V, delimiter=',')
Vnp = np.array(V)
abalone_y_testnp = abalone_y_test.to_numpy()
er = 0
for i in range(0,len(Vnp)):
    if(int(Vnp[i]) != abalone_y_testnp[i] ): er +=1
print("error is:" , er/len(Vnp) * 100)