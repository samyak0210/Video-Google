import cv2
import os
import numpy as np
import math
import json
from sklearn.cluster import KMeans

def tf(frameNo, blob):
	return blob.count(frameNo) / len(blob)

def n_containing(frameNo, blobList):
	count=0
	for i in blobList:
		if frameNo in i:
			count+=1
	return count

def idf(frameNo, blobList):
	return math.log(len(blobList) / (n_containing(frameNo, blobList)))

def tfidf(frameNo, blob, blobList):
	return tf(frameNo, blob) * idf(frameNo, blobList)

num = len(os.listdir('shots/'))
num_clusters = 800
shots = ['./shots/frame'+str(i)+'.png' for i in range(1,num+1)]

features = np.empty((0,128), int)
index = np.empty((0,2), int)

for i, l in enumerate(shots):
	frame = cv2.imread(l,0)
	sift = cv2.xfeatures2d.SIFT_create()
	kp, des = sift.detectAndCompute(frame,None)
	if des is None:
		continue
	index = np.append(index, np.array([[i+1,len(des)]]), axis=0)
	features = np.append(features, des, axis=0)

kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(features)
centres = (kmeans.cluster_centers_).tolist()
prediction = [[] for i in range(num_clusters)]
a = 0
for i in index:
	predict = kmeans.predict(features[a:a+i[1] , :])
	a = i[1]
	for j in predict:
		prediction[j].append(i[0])

prediction_unique = [[] for i in range(num_clusters)]
weights = [[] for i in range(num_clusters)]
for i, l in enumerate(prediction):
	prediction_unique[i] = np.unique(l).tolist()

for i, l in enumerate(prediction_unique):
	for j in l:
		ans = tfidf(j,prediction[i],prediction)
		weights[i].append(ans)

if 'inverted_indices.txt' in os.listdir('./'):
	os.system("rm inverted_indices.txt")


object = {
	"predict_matrix" : prediction_unique,
	"weights" : weights,
	"centres" : centres
}

with open("inverted_indices.txt","w") as f:
	json.dump(object, f)