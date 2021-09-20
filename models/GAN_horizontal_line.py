import os

#from __future__ import print_function, division
from keras import optimizers
from keras.engine.base_layer import disable_tracking

from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam

import matplotlib.pyplot as plt

import sys

import numpy as np
from tensorflow.python.keras.engine import input_layer
from tensorflow.python.ops.gen_batch_ops import Batch, batch
from tensorflow.python.util.nest import _INPUT_TREE_SMALLER_THAN_SHALLOW_TREE

class GAN():
    def __init__(self,
        input_shape,
        z_dim,
        optimizer,
        run_folder,
        generator_activation="leaky_relu",
        generator_dense=[256, 512, 1024],
        discriminator_dense=[256, 512, 1024],
        discriminator_activation="leaky_relu"
    ):
        self.img_rows = input_shape[0]
        self.img_cols = input_shape[1]
        self.img_channels = input_shape[2]
        self.input_shape = input_shape
        self.z_dim = z_dim
        self.generator_activation=generator_activation
        self.generator_dense = generator_dense
        self.discriminator_dense = discriminator_dense
        self.discriminator_activation = discriminator_activation
        self.run_folder = run_folder

        if optimizer == "adam":
            self.optimizer = Adam(0.0002, 0.5)
        else:
            print("Undefined optimizer '%s'" % (optimizer))
            exit()

        self.discriminator = self._build_discriminator()
        self.discriminator.compile(
            loss="binary_crossentropy",
            optimizer = self.optimizer,
            metrics=["accuracy"]
        )

        self.generator = self._build_generator()

        input_layer = Input(shape=(self.z_dim,))
        img = self.generator(input_layer)

        self.discriminator.trainable = False
        
        output_layer = self.discriminator(img)

        self.model = Model(input_layer, output_layer)
        self.model.compile(
            loss="binary_crossentropy",
            optimizer=self.optimizer
        )

    def _build_generator(self):

        input_layer = Input(shape=(self.z_dim,))

        x = input_layer

        for i in range(len(self.generator_dense)):
            x = Dense(self.generator_dense[i])(x)
            if self.generator_activation == "leaky_relu":
                x = LeakyReLU(alpha=0.2)(x)
            else:
                x = Activation(self.generator_activation)(x)
            x = BatchNormalization(momentum=0.8)(x)

        x = Dense(np.prod(self.input_shape), activation="tanh")(x)
        output_layer = Reshape(self.input_shape)(x)

        return Model(input_layer, output_layer)

    def _build_discriminator(self):

        # Input Layer
        input_layer = Input(shape=self.input_shape)

        x = input_layer

        # Middle Layer
        x = Flatten()(x)
        for i in range(len(self.discriminator_dense)):
            x = Dense(self.discriminator_dense[i])(x)
            if self.discriminator_activation == "leaky_relu":
                x = LeakyReLU(alpha=0.2)(x)
            else:
                x = Activation(self.discriminator_activation)(x)

        # Output Layer
        output_layer = Dense(1, activation="sigmoid")(x)

        return Model(input_layer, output_layer)

    def train(self, x_train, epochs, batch_size, sample_interval):
        for epoch in range(epochs):
            d_loss, d_acc_real, d_acc_fake = self._train_discriminator(x_train, batch_size)
            g_loss = self._train_generator(batch_size)

            print("%d [D loss: %f acc_real: %.2f acc_fake: %.2f] [G loss: %f]" % (epoch, d_loss, d_acc_real, d_acc_fake, g_loss))

            if epoch % sample_interval == 0:
                self.epoch = epoch
                self.sample_images(self.run_folder)

    def _train_discriminator(self, x_train, batch_size):
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        idx = np.random.randint(0, x_train.shape[0], batch_size)
        imgs = x_train[idx]

        noise = np.random.normal(0, 1, (batch_size, self.z_dim))

        gen_imgs = self.generator.predict(noise)

        d_loss_real, d_acc_real = self.discriminator.train_on_batch(imgs, valid)
        d_loss_fake, d_acc_fake = self.discriminator.train_on_batch(gen_imgs, fake)
        d_loss = 0.5 * (d_loss_real, d_loss_fake)

        return [d_loss, d_acc_real, d_acc_fake]

    def _train_generator(self, batch_size):
        valid = np.ones((batch_size, 1))
        noise = np.random.normal(0, 1, (batch_size, self.z_dim))

        g_loss = self.model.train_on_batch(noise, valid)

        return g_loss

    def sample_images(self, run_folder):
        r, c = 5, 5
        noise = np.random.normal(0, 1, (r*c, self.z_dim))
        gen_imgs = self.generator.predict(noise)

        gen_imgs = gen_imgs * 255

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i, j].imshow(gen_imgs[cnt, :, :, 0], cmap="gray")
                axs[i, j].axis("off")
                cnt += 1
        fig.savefig(os.path.join(run_folder, "images/%d.png" % self.epoch))
        plt.close()