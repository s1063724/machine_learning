# -*- coding: utf-8 -*-
"""
Created on Sun Apr 21 12:45:33 2024

@author: 10305
"""
import numpy as np
import pandas as pd
# Load data 載入資料
data = pd.read_csv('dataset/data_perf_add.txt', header=None)
X = data.copy()
# Find the best epsilon 
eps_grid = np.linspace(0.3, 1.2, num=10)
    # Train DBSCAN clustering model 訓練DBSCAN分群模型
    # ################
    # min_samples = 5
    # ################
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
best_silhouette_scores = float('-inf')
for eps in eps_grid:
    model = DBSCAN(eps=eps, min_samples=5).fit(X)
    scores = silhouette_score(X, model.labels_)
    if scores >= best_silhouette_scores:
        best_silhouette_scores = scores
        best_eps = eps
        best_model = model
        best_labels = model.labels_
    # print(f"Epsilon: {eps:.1f} --> silhouette score: {scores:.4f}")
print(f"Best epsilon= {best_eps:.1f}")
print(f"Best silhouette score= {best_silhouette_scores:.4f}")
if -1 in best_labels:
    count = len(set(best_labels))-1
else:
    count = len(set(best_labels))
print(f"Estimated number of clusters= {count}")