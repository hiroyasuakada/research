import numpy as np
import tensorflow as tf
from tensorflow.python.keras import layers


class Generator(object):
    def __init__(self, latent_dim):
        generator_input = tf.keras.Input(shape=(latent_dim,))

        x = layers.Dense(1024)(generator_input)
        x = layers.Activation('tanh')(x)
        x = layers.Dense(128*7*7)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('tanh')(x)
        x = layers.Reshape((7, 7, 128))(x)
        x = layers.UpSampling2D(size=(2, 2))(x)
        x = layers.Conv2D(64, 5, padding='same')(x)
        x = layers.Activation('tanh')(x)
        x = layers.UpSampling2D(size=(2, 2))(x)
        x = layers.Conv2D(1, 5, padding='same')(x)
        x = layers.Activation('tanh')(x)

        self.generator = tf.keras.models.Model(generator_input, x)

    def get_model(self):
        return self.generator


class Discriminator(object):
    def __init__(self, height, width, channels):
        discriminator_input = layers.Input(shape=(height, width, channels))

        x = layers.Conv2D(64, 5, padding='same')(discriminator_input)
        x = layers.Activation('tanh')(x)
        x = layers.MaxPool2D()(x)
        x = layers.Conv2D(128, 5)(x)
        x = layers.Activation('tanh')(x)
        x = layers.MaxPool2D()(x)
        x = layers.Flatten()(x)
        x = layers.Dense(1024)(x)
        x = layers.Activation('tanh')(x)
        x = layers.Dense(1, activation='sigmoid')(x)

        self.discriminator = tf.keras.models.Model(discriminator_input, x)

        # compile discriminator
        discriminator_optimizer = tf.keras.optimizers.SGD(lr=0.0005, momentum=0.9, nesterov=True)
        self.discriminator.compile(optimizer=discriminator_optimizer, loss='binary_crossentropy')

    def get_model(self):
        return self.discriminator


class DCGAN(object):
    def __init__(self, latent_dim, height, width, channels):
        # set generator
        self._latent_dim = latent_dim
        g = Generator(latent_dim)
        self._generator = g.get_model()

        # set discriminator
        d = Discriminator(height, width, channels)
        self._discriminator = d.get_model()

        # disable training when combined with generator
        self._discriminator.trainable = False

        # set DCGAN
        dcgan_input = tf.keras.Input(shape=(latent_dim,))
        dcgan_output = self._discriminator(self._generator(dcgan_input))
        self._dcgan = tf.keras.models.Model(dcgan_input, dcgan_output)

        # compile DCGAN
        dcgan_optimizer = tf.keras.optimizers.SGD(lr=0.0005, momentum=0.9, nesterov=True)
        self._dcgan.compile(optimizer=dcgan_optimizer, loss='binary_crossentropy')

    def train(self, real_images, batch_size):
        # Train so discriminator can detect fake
        random_latent_vectors = np.random.normal(size=(batch_size, self._latent_dim))
        generated_images = self._generator.predict(random_latent_vectors)
        labels = np.ones((batch_size, 1))
        labels += 0.05 * np.random.random(labels.shape)
        d_loss1 = self._discriminator.train_on_batch(generated_images, labels)

        # Train so discriminator can detect real
        labels = np.zeros((batch_size, 1))
        labels += 0.05 * np.random.random(labels.shape)
        d_loss2 = self._discriminator.train_on_batch(real_images, labels)
        d_loss = (d_loss1 + d_loss2)/2.0

        # Train so generator can fool discriminator
        random_latent_vectors = np.random.normal(size=(batch_size, self._latent_dim))
        misleading_targets = np.zeros((batch_size, 1))
        g_loss = self._dcgan.train_on_batch(random_latent_vectors, misleading_targets)
        return d_loss, g_loss

    def predict(self, latent_vector):
        return self._generator.predict(latent_vector)

    def load_weights(self, file_path, by_name=False):
        self._dcgan.load_weights(file_path, by_name)

    def save_weights(self, file_path, overwrite=True):
        self._dcgan.save_weights(file_path, overwrite)
