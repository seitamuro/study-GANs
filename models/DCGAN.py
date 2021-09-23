import os
from pickle import load

from utils.loaders import load_horizontal_line

from tensorflow.keras.datasets import mnist
from tensorflow.keras import activations
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout, BatchNormalization, Activation, ZeroPadding2D
from tensorflow.keras.layers import LeakyReLU, UpSampling2D, Conv2D, Conv2DTranspose, MaxPooling2D
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam, SGD

import matplotlib.pyplot as plt

import sys

import numpy as np
from tensorflow.python.keras.engine import input_layer
from tensorflow.python.keras.layers.convolutional import Conv
from tensorflow.python.ops.gen_batch_ops import Batch, batch
from tensorflow.python.util.nest import _INPUT_TREE_SMALLER_THAN_SHALLOW_TREE

class DCGAN():
    def __init__(self, run_folder):
        self.run_folder = run_folder

        self.build_generator()
        self.build_discriminator()
        self.build_adversarial()

        self.generator.compile(
            loss="binary_crossentropy",
            optimizer=SGD(lr=0.0005, momentum=0.9, nesterov=True),
            metrics=["accuracy"]
        )

        self.discriminator.trainable = False
        self.model.compile(
            loss="binary_crossentropy",
            optimizer=SGD(lr=0.0005, momentum=0.9, nesterov=True),
            metrics=["accuracy"]
        )
        self.discriminator.trainable = True

        self.discriminator.compile(
            loss="binary_crossentropy",
            optimizer=SGD(lr=0.0005, momentum=0.9, nesterov=True),
            metrics=["accuracy"]
        )

    def build_generator(self):
        model = Sequential()
        model.add(Input(100))
        model.add(Dense(1024))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(Dense(256*7*7))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(Reshape((7, 7, 256)))
        model.add(UpSampling2D(size=(2, 2)))
        model.add(Conv2D(128, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(Conv2D(128, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(Conv2D(64, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(Conv2D(64, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(UpSampling2D(size=(2, 2)))
        model.add(Conv2D(1, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(Conv2D(1, (3, 3), padding="same"))
        model.add(BatchNormalization())
        model.add(Activation("tanh"))
        model.summary()

        self.generator = model

    def build_discriminator(self):
        model = Sequential()
        model.add(Conv2D(64, (3, 3), padding="same", input_shape=(28, 28, 1)))
        model.add(LeakyReLU())
        model.add(BatchNormalization())
        model.add(Conv2D(64, (3, 3), padding="same"))
        model.add(LeakyReLU())
        model.add(BatchNormalization())
        model.add(Conv2D(128, (3, 3), strides=2, padding="same"))
        model.add(LeakyReLU())
        model.add(BatchNormalization())
        model.add(Conv2D(128, (3, 3), strides=1, padding="same"))
        model.add(LeakyReLU())
        model.add(BatchNormalization())
        model.add(Conv2D(256, (3, 3), strides=2, padding="same"))
        model.add(LeakyReLU())
        model.add(BatchNormalization())
        model.add(Conv2D(256 , (3, 3), strides=1, padding="same"))
        model.add(LeakyReLU())
        model.add(BatchNormalization())
        model.add(Flatten())
        model.add(Dense(1024))
        model.add(LeakyReLU())
        model.add(BatchNormalization())
        model.add(Dense(1))
        model.add(BatchNormalization())
        model.add(Activation("sigmoid"))

        model.summary()

        self.discriminator = model

    def build_adversarial(self):
        input_layer = Input(shape=(100,))
        output_layer = self.discriminator(self.generator(input_layer))
        model = Model(input_layer, output_layer)

        model.summary()

        self.model = model

    def train(self, x_train, epochs=101, batch_size=128, print_interval=20):
        #(x_train, _), (_, _) = mnist.load_data()
        #x_train = np.expand_dims(x_train, axis=3)
        #x_train = x_train.astype(np.float32) / 127.5 - 1

        for epoch in range(epochs):
            g_loss, g_acc = self.train_generator(batch_size)
            d_loss, d_acc = self.train_discriminator(x_train, batch_size)

            print("epoch: %d d_loss: %.3f d_acc: %.3f g_loss: %.3f g_acc: %.3f" % (epoch, d_loss, d_acc, g_loss, g_acc))

            if epoch % print_interval == 0:
                self.save_images(epoch)


    def train_generator(self, batch_size):
        noise = np.random.normal(0, 1, (batch_size, 100))
        labels = np.ones((batch_size, 1))
        self.discriminator.trainable = False
        g_loss, g_acc = self.model.train_on_batch(noise, labels)
        self.discriminator.trainable = True

        return g_loss, g_acc

    def train_discriminator(self, x_train, batch_size):
        idx = np.random.randint(0, len(x_train), int(batch_size/2))

        true_image = x_train[idx]
        true_label = [1] * int(batch_size / 2)

        noise = np.random.normal(0, 1, (batch_size // 2, 100))
        gen_image = self.generator.predict(noise)
        gen_label = [0] * int(batch_size / 2)

        images = np.concatenate((true_image, gen_image))
        labels = np.concatenate((true_label, gen_label))
        d_loss, d_acc = self.discriminator.train_on_batch(images, labels)

        return d_loss, d_acc

    def save_images(self, epoch, r=5, c=5):
        noise = np.random.normal(0, 1, (r*c, 100))
        gen_imgs = self.generator.predict(noise)

        fig, axs = plt.subplots(r, c, figsize=(15, 15))
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i, j].imshow(gen_imgs[cnt, :, :, 0], cmap="gray")
                axs[i, j].axis("off")
                cnt += 1
        fig.savefig(os.path.join(self.run_folder, "images/images_%d.png" % epoch))
        plt.close()

if __name__ == "__main__":
    dcgan = DCGAN("run/dcgan/0002_mnist")
    x_train = load_horizontal_line()
    x_train = x_train / 127.5 - 1.
    dcgan.train(x_train=x_train, epochs=60001, print_interval=100)