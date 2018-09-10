import cv2
import json
import numpy as np
import os

refPt = []

def idf(frameNo, blob):
    return blob.count(frameNo) / len(blob)

def predict(centroids, des):
    diff = centroids - des
    dist = np.sqrt(np.sum(diff**2, axis=-1))
    return np.argmin(dist) + 1


def click_and_crop(event, x, y, flags, params):
    global refPt
    if event == cv2.EVENT_LBUTTONDOWN:
        refPt = [(x, y)]

    elif event == cv2.EVENT_LBUTTONUP:
        refPt.append((x, y))
        cv2.rectangle(image, refPt[0], refPt[1], (0, 255, 0), 2)



image = cv2.imread("image.png")
clone = image.copy()
cv2.namedWindow("image")
cv2.setMouseCallback("image", click_and_crop)

while True:
    cv2.imshow("image", image)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("r"):
        image = clone.copy()

    elif key == ord("c"):
        break

if len(refPt) == 2:
    cropped_image = clone[refPt[0][1]:refPt[1][1], refPt[0][0]:refPt[1][0]]
# cv2.imwrite('img.png',cropped_image)
cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_RGB2GRAY)

with open("inverted_indices.txt") as f:
    object = json.load(f)

weights = object["weights"]
inverted_indices = object["predict_matrix"]
centroids = np.array(object["centres"])


sift = cv2.xfeatures2d.SIFT_create()
kp, des = sift.detectAndCompute(cropped_image, None)

cluster = np.zeros(len(inverted_indices))
b = []

for i in des:
    b.append(predict(centroids, i))

uni = np.unique(b)

for i in uni:
    cluster[i-1] = idf(i,b)

num_shots = len(os.listdir('./shots/'))
mat = np.zeros((len(cluster) , num_shots))

for i , l in enumerate(inverted_indices):
    for j in l:
        mat[i][j-1] = weights[i][l.index(j)]

arr = np.matmul(cluster , mat)
buck = []
for i , l in enumerate(arr):
    buck.append([l,i+1])

buck.sort(reverse=True)
for i in buck:
    print(i[1])
    img = cv2.imread('./shots/frame'+str(i[1])+'.png')
    cv2.imshow('image',img)
    c = cv2.waitKey(10000)
    if c == ord('q'):
        break