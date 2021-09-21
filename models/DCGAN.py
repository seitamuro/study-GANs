import os

from tensorflow.keras.datasets import mnist
from tensorflow.keras import activations
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout, BatchNormalization, Activation, ZeroPadding2D
from tensorflow.keras.layers import LeakyReLU, UpSampling2D, Conv2D, Conv2DTranspose
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam

import matplotlib.pyplot as plt

import sys

import numpy as np
from tensorflow.python.keras.layers.convolutional import Conv
from tensorflow.python.ops.gen_batch_ops import Batch

class DCGAN():
    def __init__(self,
        input_shape,
        z_dim,
        run_folder,
        optimizer="adam",
        generaotr_initial_dense_layer_size=(224, 224, 64),
        generator_conv = "conv",
        generator_conv_upsampling=[2, 2, 2],
        generator_conv_strides=[2, 2, 2],
        generator_conv_filters=[64, 32, 1],
        generator_conv_kernels=[2, 2, 2,],
        discriminator_conv_filters=[32, 64, 128, 256],
        discriminator_conv_kernels=[3, 3, 3, 3],
        discriminator_conv_strides=[2, 2, 2, 1],
        dropout_rate=0.25,
        normalize_momentum=0.5,
        learning_rate=0.0001
    ):
        self.input_shape = input_shape
        self.z_dim = z_dim
        self.run_folder = run_folder
        self.generator_conv = generator_conv
        self.generaotr_initial_dense_layer_size = generaotr_initial_dense_layer_size
        self.generator_conv_filters = generator_conv_filters
        self.generator_conv_kernels = generator_conv_kernels
        self.generator_conv_strides = generator_conv_strides
        self.normalize_momentum = normalize_momentum
        self.discriminator_conv_filters = discriminator_conv_filters
        self.discriminator_conv_kernels = discriminator_conv_kernels
        self.discriminator_conv_strides = discriminator_conv_strides
        self.dropout_rate = dropout_rate

        if optimizer == "adam":
            self.optimizer = Adam(learning_rate, 0.5)
        else:
            raise Exception("optimizer %s is not defined." % optimizer)

        self.discriminator = self.build_discriminator()
        self.discriminator.compile(
            loss="binary_crossentropy",
            optimizer=optimizer,
            metrics=["accuracy"]
        )

        # Build the generator

        self.generator = self.build_generator()

        z = Input(shape=(self.z_dim,))
        img = self.generator(z)

        self.discriminator.trainable = False

        valid = self.discriminator(img)

        self.combined = Model(z, valid)
        self.combined.compile(loss="binary_crossentropy", optimizer=optimizer)

    def build_generator(self):

        model = Sequential()

        input_layer = Input(shape=self.z_dim)

        model.add(Dense(np.prod(self.generaotr_initial_dense_layer_size), activation="relu", input_dim=self.z_dim))
        model.add(Reshape(self.generaotr_initial_dense_layer_size))
        #model.add(UpSampling2D())

        for i in range(len(self.generator_conv_filters)):
            if self.generator_conv == "conv":
                model.add(Conv2D(
                    self.generator_conv_filters[i],
                    kernel_size=self.generator_conv_kernels[i],
                    strides=self.generator_conv_strides[i],
                    padding="same")
                )
            elif self.generator_conv == "convT":
                model.add(Conv2DTranspose(
                    self.generator_conv_filters[i],
                    kernel_size=self.generator_conv_kernels[i],
                    strides=self.generator_conv_strides[i],
                    padding="same")
                )
            model.add(BatchNormalization(momentum=self.normalize_momentum))
            if i == len(self.generator_conv_filters) - 1:
                model.add(Activation("tanh"))
            else:
                model.add(Activation("relu"))

            #if model.layers[-1].output_shape[-2] != self.input_shape[-2]:
                #model.add(UpSampling2D())
            #if i == 0:
                #model.add(ZeroPadding2D(padding=(3, 3)))

        model.summary()

        output_layer = model(input_layer)

        return Model(input_layer, output_layer)

    def build_discriminator(self):

        model = Sequential()

        input_layer = Input(shape=self.input_shape)

        model.add(Conv2D(
            self.discriminator_conv_filters[0],
            kernel_size=self.discriminator_conv_kernels[0],
            strides=self.discriminator_conv_strides[0],
            input_shape=self.input_shape,
            padding="same")
        )
        model.add(BatchNormalization(momentum=self.normalize_momentum))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(self.dropout_rate))
        model.add(ZeroPadding2D(padding=((0, 1), (0, 1))))
        for i in range(1, len(self.discriminator_conv_filters)):
            model.add(Conv2D(
                self.discriminator_conv_filters[i],
                kernel_size=self.discriminator_conv_kernels[i],
                strides=self.discriminator_conv_strides[i],
                padding="same")
            )
            model.add(BatchNormalization(momentum=self.normalize_momentum))
            model.add(LeakyReLU(alpha=0.2))
            model.add(Dropout(self.dropout_rate))
        model.add(Flatten())
        model.add(Dense(1, activation="sigmoid"))
        
        model.summary()

        output_layer = model(input_layer)

        return Model(input_layer, output_layer)

    def train(self, epochs, batch_size=128, save_interval=50):
        (x_train, _), (_, _) = mnist.load_data()

        x_train = x_train / 127.5 - 1.
        x_train = np.expand_dims(x_train, axis=3)

        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in range(epochs):
            # train discriminator
            idx = np.random.randint(0, x_train.shape[0], batch_size)
            imgs = x_train[idx]

            noise = np.random.normal(0, 1, (batch_size, self.z_dim))
            gen_imgs = self.generator.predict(noise)

            d_loss_real = self.discriminator.train_on_batch(imgs, valid)
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # train generator
            g_loss = self.combined.train_on_batch(noise, valid)

            print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))

            if epoch % save_interval == 0:
                self.save_imgs(epoch)

    def save_imgs(self, epoch):
        r, c = 5, 5
        noise = np.random.normal(0, 1, (r*c, self.z_dim))
        gen_imgs = self.generator.predict(noise)

        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i, j].imshow(gen_imgs[cnt, :, :, 0], cmap="gray")
                axs[i, j].axis("off")
                cnt += 1
        fig.savefig(os.path.join(self.run_folder, "images/mnist_%d.png" % epoch))
        plt.close()