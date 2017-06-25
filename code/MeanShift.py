import pandas as pd
import zipfile
import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.datasets.samples_generator import make_blobs
from numpy import vstack, shape

zf = zipfile.ZipFile('../train.csv.zip')
data=pd.read_csv(zf.open('train.csv'))

print("Total rows: {0}".format(len(data)))

print(list(data))

Z=  np.zeros((1,1))
X=  np.zeros((1,2))
order = np.arange(len(data))
np.random.shuffle(order)
i=1
for k in np.arange(len(data)):
    if data.loc[k].get('MISSING_DATA') == False :
        y=data.loc[k].get('POLYLINE')
        y=eval(y)
        y=np.array(y)
        
        if y.size != 0 :
            y=y[-1]
            X=vstack((X,y))
            z=np.array([data.loc[k].get('TRIP_ID')])
            Z=vstack((Z,z))
    i=i+1
    print("vong lap thu %d" %i)
            
X=np.delete(X, 0, 0)      
Z=np.delete(Z, 0, 0)

print("so hang cua X la %d" % shape(X)[0])

bandwidth=0.002
print("band_width la :")
print(bandwidth)
ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
ms.fit(X)

labels = ms.labels_
cluster_centers = ms.cluster_centers_

labels_unique = np.unique(labels)
n_clusters_ = len(labels_unique)
print("number of estimated clusters : %d" % n_clusters_)

DF1= np.zeros((1,2))
for k in range(np.shape(Z)[0]):
    y=Z[k][0]
    z=labels[k]
    y=np.array([y,z])
    DF1=vstack((DF1,y))
DF1=np.delete(DF1, 0, 0) 
df1 = pd.DataFrame(DF1, index=range(np.shape(Z)[0]), columns=np.array(['TRIP_ID','CLUSTER_ID']))
df1.to_csv("clustering_result3.csv",index=None)
print("in xong file clustering_result3")

DF2= np.zeros((1,2))
for k in range(np.shape(cluster_centers)[0]):
    y=cluster_centers[k]
    y=np.array_str(y)
    y=np.array([k,y])
    DF2=vstack((DF2,y))
DF2=np.delete(DF2, 0, 0) 
df2 = pd.DataFrame(DF2, index=range(np.shape(cluster_centers)[0]), columns=np.array(['CLUSTER_ID','COORDINATES']))
df2.to_csv("cluster_coordinates3.csv",index=None)
print("in xong file cluster_coordinates3")

import matplotlib.pyplot as plt
from itertools import cycle
plt.figure(1)
plt.clf()
colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
for k, col in zip(range(n_clusters_), colors):
    my_members = labels == k
    cluster_center = cluster_centers[k]
    plt.plot(X[my_members, 0], X[my_members, 1], col + '.')
    plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
             markeredgecolor='k', markersize=14)
plt.title('Estimated number of clusters: %d' % n_clusters_)
plt.show()

