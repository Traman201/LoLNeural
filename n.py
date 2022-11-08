import matplotlib.pyplot as plt 
import numpy as np 
from sklearn.cluster import KMeans
import json
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import SpectralClustering
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics.pairwise import pairwise_distances


'''
def distvec(vec_x , vec_y):
 dist = 0
 for i in range(0 , len(vec_x)-1):
        dist += (vec_y[i] - vec_x[i])**2
 return dist**0.5


def find_max_vec(vecdata , ydata , center):
        max_vec_len = [0,0,0]
        for i in range(0 , len(ydata)):
                if(max_vec_len[int(ydata[i])] < vecdata[i]): max_vec_len[int(ydata[i])] = vecdata[i]
        return max_vec_len
        
def find_procent(max_vec_len ,ydata, vecdata , center):
        procent = []
        for i in range(0 , len(ydata)):

'''


fig = plt.figure(figsize=(15, 8))
ax1 = fig.add_subplot(121)
colnames = ["time",
        "kill",
        "death",
        "assist",
        "abilityHaste",
        "abilityPower",
        "armor",
        "armorPen",
        "armorPenPercent",
        "attackDamage",
        "attackSpeed",
        "bonusArmorPenPercent",
        "bonusMagicPenPercent",
        "ccReduction",
        "cooldownReduction",
        "health",
        "healthMax",
        "healthRegen",
        "lifesteal",
        "magicPen",
        "magicPenPercent",
        "magicResist",
        "movementSpeed",
        "omnivamp",
        "physicalVamp",
        "power",
        "powerMax",
        "powerRegen",
        "spellVamp"]
p = "C:\\neural\\dataset.csv"
Df = pd.read_csv(p,names=colnames)
df = Df.drop("time", axis=1)
df = df.drop('abilityHaste', axis=1)
df = df.drop('armorPen', axis=1)
df = df.drop('bonusArmorPenPercent', axis=1)
df = df.drop('bonusMagicPenPercent', axis=1)
df = df.drop('cooldownReduction', axis=1)
df = df.drop('magicPenPercent', axis=1)
df = df.drop('physicalVamp', axis=1)
df = df.drop('spellVamp', axis=1)
normalized_df=(df-df.mean())/df.std()
X = df.values[:, :]
X = np.nan_to_num(normalized_df)
clusDataSet = StandardScaler().fit_transform(X)
clusterNum = 3
kMeans = KMeans(init="k-means++", n_clusters=clusterNum, n_init=12)
kMeans.fit(X)
labels = kMeans.fit_predict(X)
#df[TypeClaster] = labels

c_x = kMeans.cluster_centers_[:,14]
c_y = kMeans.cluster_centers_[:,15]
print(kMeans.cluster_centers_[0])


ax1.scatter(X[:,14], X[:,15], c=labels.astype(np.float), alpha=0.5)
ax1.scatter(c_x, c_y, s = 150, c = ["red"], marker = '*', label = 'Centroids')
ax1.set(xlabel='armor', ylabel='magic Resist', title="Clustering data")
Dfy = pd.read_csv(p , names=colnames)
Dfy = Dfy.drop('time', axis=1)
Dfy = Dfy.drop('abilityHaste', axis=1)
Dfy = Dfy.drop('armorPen', axis=1)
Dfy = Dfy.drop('bonusArmorPenPercent', axis=1)
Dfy = Dfy.drop('bonusMagicPenPercent', axis=1)
Dfy = Dfy.drop('cooldownReduction', axis=1)
Dfy = Dfy.drop('magicPenPercent', axis=1)
Dfy = Dfy.drop('physicalVamp', axis=1)
Dfy = Dfy.drop('spellVamp', axis=1)
x = df.values #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
normalized_dfy = pd.DataFrame(x_scaled)

#normalized_dfy=float((Dfy-Dfy.mean())/Dfy.std())
normalized_dfy['y'] = pd.Series(labels, index=normalized_dfy.index)
Dfy['y'] = pd.Series(labels, index=Dfy.index)

#print(normalized_dfy.sort_values(['y'], ascending=[False])['y'])
'''
Dists = []
vecdist = normalized_dfy
for index, row in vecdist.iterrows(): Dists.append(distvec(row , kMeans.cluster_centers_[int(row['y'])]))
maxlenvec = find_max_vec(Dists ,labels ,kMeans.cluster_centers_)
print(maxlenvec)
'''
#Dfy.to_csv("YDataset.csv", encoding='utf-8', index=False)
#normalized_dfy.to_csv("N_YDataset.csv", encoding='utf-8', index=False)
plt.show()
