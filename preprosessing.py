import os
import re
import nibabel as nib
import numpy as np

for dirpath, dirname, filename in os.walk("data/collect/"):
    filenames = filename
img_files = list(filter(lambda x: re.search(r'.*\.img$',x), filenames))


def thresholding(img, p=0.01):
    img = img/np.max(img)
    for t in range(1,10000):
        if np.mean(img > 0.0001*t) < p:
            break
    img[img <= 0.0001*t] = 0
    print(np.mean(img > 0.0001*t))
    return img

def find_points(img):
    points = []
    weights = []
    for i in range(128):
        for j in range(128):
            for k in range(128):
                if img[i,j,k] > 0:
                    points.append([i,j,k])
                    weights.append(img[i,j,k])
    return np.array(points,dtype=np.float32), \
           np.array(weights,dtype=np.float32)

MAX_POINTS = 20941
data = np.zeros([43,128,128,128,1],dtype=np.float32)
points = np.zeros([43,MAX_POINTS,3],dtype=np.float32)
weights = np.zeros([43,MAX_POINTS],dtype=np.float32)

for i, fm in enumerate(img_files):
    print("it: ", i)
    cur = nib.load("data/collect/"+fm)
    cur = np.asarray(cur.get_data(),dtype=np.float32)
    cur = thresholding(cur)
    p, w = find_points(cur)
    data[i,:,:,:,0] = cur
    points[i,:p.shape[0],:] = p
    weights[i,:w.shape[0]] = w
    
np.savez("/data/data.npz",img=data,points=points,weights=weights)