import numpy as np
import pandas
import random
from ica import ica1
import matplotlib.pyplot as plt
from sklearn import datasets
#mnist = datasets.fetch_openml('mnist_784',version=1,return_X_y=True)
#X, y = mnist

from sklearn.datasets import fetch_lfw_people
lfw_people = fetch_lfw_people(return_X_y=True)
X,y = lfw_people

#n = int(np.ceil(np.sqrt(X.shape[1])))

df = pandas.DataFrame(X)
rows = np.random.choice(df.index.values, 115)
sampled_df = df.iloc[rows]
X = np.array(sampled_df)

n_components = 10
A,S,W = ica1(np.array(X), n_components)

res = np.outer(W[3].T,A[8])@S

fig, ax = plt.subplots(nrows=2, ncols=5, sharex=True, sharey=True,)
ix = 0
for i in range(2):
  for j in range(5):
    digit = S[ix]
    ix += 1
    digit_pixels = np.array(digit).reshape(62,47)
    ax[i,j].imshow(digit_pixels)

fig.savefig("ica_comps.png")
