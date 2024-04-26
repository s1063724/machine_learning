# -*- coding: utf-8 -*-
"""
Created on Sat Apr 20 22:09:10 2024

@author: 10305
"""
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

#  載入scikit-learn資料集範例資料
X, _ = make_blobs(n_samples=200, centers=4, cluster_std=0.5, random_state=0)

#inertia_集群內誤差平方和，做轉折判斷法的依據
for i in range(1,10):
    kmeans = KMeans(n_clusters=i,
                    init='k-means++',
                    n_init=15,
                    random_state=0,
                    max_iter=200).fit(X)
    if kmeans.inertia_ >= 90:
        print(f'n_clusters: {i} => {kmeans.inertia_:.4f}')
kmeans = KMeans(n_clusters=4,
                init='k-means++',
                n_init=15,
                random_state=0,
                max_iter=200).fit(X)
print("cluster_centers=\n",kmeans.cluster_centers_)