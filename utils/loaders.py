import os
from os import walk

from PIL import Image

import numpy as np

from tensorflow.keras.datasets import mnist

# Define load_mnist
def load_mnist():
  (x_train, y_train), (x_test, y_test) = mnist.load_data()

  x_train = x_train.astype("float32") / 255.
  x_train = x_train.reshape(x_train.shape + (1,))
  x_test = x_test.astype("float32") / 255.
  x_test = x_test.reshape(x_test.shape + (1,))

  return (x_train, y_train), (x_test, y_test)

def load_safari(folder):
  mypath = os.path.join("./data", folder)
  txt_name_list = []
  for (_, _, filenames) in walk(mypath):
    for f in filenames:
      if f != ".DS_Store":
        txt_name_list.append(f)
        break

  slice_train = int(80000/len(txt_name_list))
  i = 0
  seed = np.random.randint(1, 10e6)

  for txt_name in txt_name_list:
    txt_path = os.path.join(mypath, txt_name)
    x = np.load(txt_path)
    x = (x.astype("float32") - 127.5 / 127.5)

    x = x.reshape(x.shape[0], 28, 28, 1)

    y = [i] * len(x)
    np.random.seed(seed)
    np.random.shuffle(x)
    np.random.seed(seed)
    np.random.shuffle(y)
    x = x[:slice_train]
    y = y[:slice_train]
    if i != 0:
      xtotal = np.concatenate((x, xtotal), axis=0)
      ytotal = np.concatenate((y, ytotal), axis=0)
    else:
      xtotal = x
      ytotal = y
    i += 1

  return xtotal, ytotal

def load_horizontal_line():
  datasets = []
  path = "data/horizontal_line/train"
  for (_, _, filenames) in walk(path):
    for f in filenames:
      image = Image.open(os.path.join(path, f))
      image = np.asarray(image)

      empty = np.zeros_like(image, dtype=int)
      empty[image] = 255
      empty = np.expand_dims(empty, axis=2)
      datasets.append(empty)

  return np.array(datasets)