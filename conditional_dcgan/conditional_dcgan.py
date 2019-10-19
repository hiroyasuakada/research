import numpy as np
import tensorflow as tf
from tensorflow.python.keras import layers


class Generator(object):
    def __init__(self, latent_dim, condition_dim):
        # prepare latent vector (noise) input
        generator_input1 = layers.Input(shape=(latent_dim, ))

        x1 = layers.Dense(1024)(generator_input1)
        x1 = layers.Activation('tanh')(x1)
        x1 = layers.Dense(128 * 7 * 7)(x1)
        x1 = layers.BatchNormalization()(x1)
        x1 = layers.Activation('tanh')(x1)
        x1 = layers.Reshape((7, 7, 128))(x1)

        # prepare conditional input
        generator_input2 = layers.Input(shape=(condition_dim, ))

        x2 = layers.Dense(1024)(generator_input2)
        x2 = layers.Activation('tanh')(x2)
        x2 = layers.Dense(128 * 7 * 7)(x2)
        x2 = layers.BatchNormalization()(x2)
        x2 = layers.Activation('tanh')(x2)
        x2 = layers.Reshape((7, 7, 128))(x2)

        # concatenate 2 inputs
        generator_input = layers.Concatenate()([x1, x2])

        x = layers.UpSampling2D(size=(2, 2))(generator_input)
        x = layers.Conv2D(64, 5, padding='same')(x)
        x = layers.Activation('tanh')(x)
        x = layers.UpSampling2D(size=(2, 2))(x)
        x = layers.Conv2D(1, 5, padding='same')(x)
        x = layers.Activation('tanh')(x)

        self.generator = tf.keras.models.Model(inputs=[generator_input1, generator_input2], outputs=x)

    def get_model(self):
        return self.generator


class Discriminator(object):
    def __init__(self, height, width, channels, condition_dim):
        # prepare real images
        discriminator_input1 = layers.Input(shape=(height, width, channels))

        x1 = layers.Conv2D(64, 5, padding='same')(discriminator_input1)
        x1 = layers.Activation('tanh')(x1)
        x1 = layers.MaxPooling2D(pool_size=(2, 2))(x1)
        x1 = layers.Conv2D(128, 5)(x1)
        x1 = layers.Activation('tanh')(x1)
        x1 = layers.MaxPooling2D(pool_size=(2, 2))(x1)

        # condition input from generator
        discriminator_input2 = layers.Input(shape=(condition_dim, ))

        x2 = layers.Dense(1024)(discriminator_input2)
        x2 = layers.Activation('tanh')(x2)
        x2 = layers.Dense(5 * 5 * 128)(x2)
        x2 = layers.BatchNormalization()(x2)
        x2 = layers.Activation('tanh')(x2)
        x2 = layers.Reshape((5, 5, 128))(x2)

        # concatenate 2 inputs
        discriminator_input = layers.concatenate([x1, x2])

        x = layers.Flatten()(discriminator_input)
        x = layers.Dense(1024)(x)
        x = layers.Activation('tanh')(x)
        x = layers.Dense(1, activation='sigmoid')(x)

        self.discriminator = tf.keras.models.Model(inputs=[discriminator_input1, discriminator_input2], outputs=x)

        # # compile discriminator
        # discriminator_optimizer = tf.keras.optimizers.SGD(lr=0.0005, momentum=0.9, nesterov=True)
        # self.discriminator.compile(optimizer=discriminator_optimizer, loss='binary_crossentropy')

    def get_model(self):
        return self.discriminator


class ConditionalDCGAN(object):
    def __init__(self, latent_dim, height, width, channels, condition_dim):
        self._latent_dim = latent_dim
        self._condition_dim = condition_dim

        # set generator
        g = Generator(latent_dim, condition_dim)
        self._generator = g.get_model()
        print('Generator:')
        self._generator.summary()

        # set discriminator
        d = Discriminator(height, width, channels, condition_dim)
        self._discriminator = d.get_model()
        print('Discriminator:')
        self._discriminator.summary()

        # compile discriminator
        discriminator_optimizer = tf.keras.optimizers.SGD(lr=0.0005, momentum=0.9, nesterov=True)
        self._discriminator.compile(optimizer=discriminator_optimizer, loss='binary_crossentropy')

        # disable training when combined with generator
        self._discriminator.trainable = False

        # set DCGAN
        dcgan_input1 = layers.Input(shape=(latent_dim, ))
        dcgan_input2 = layers.Input(shape=(condition_dim, ))
        generated_images = self._generator([dcgan_input1, dcgan_input2])
        dcgan_output_is_real = self._discriminator([generated_images, dcgan_input2])
        self.dcgan = tf.keras.models.Model([dcgan_input1, dcgan_input2], dcgan_output_is_real)
        print('ConditionalDCGAN:')
        self.dcgan.summary()

        # compile CDCGAN
        dcgan_optimizer = tf.keras.optimizers.SGD(lr=0.0005, momentum=0.9, nesterov=True)
        self.dcgan.compile(optimizer=dcgan_optimizer, loss='binary_crossentropy')

    def train(self, real_images, conditions, batch_size):
        # set adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        # train discriminator so it can detect real or fake
        random_latent_vectors = np.random.normal(size=(batch_size, self._latent_dim))
        generated_images = self._generator.predict([random_latent_vectors, conditions])
        combined_conditions = np.concatenate([conditions, conditions])
        combined_images = np.concatenate([real_images, generated_images])
        real_labels = np.concatenate([valid, fake])
        real_labels += 0.05 * np.random.random(real_labels.shape)  # add noise
        d_loss = self._discriminator.train_on_batch([combined_images, combined_conditions], real_labels)

        # train generator so it can fool discriminator
        random_latent_vectors = np.random.normal(size=(batch_size, self._latent_dim))
        misleading_targets = valid
        g_loss = self.dcgan.train_on_batch([random_latent_vectors, conditions], misleading_targets)
        return d_loss, g_loss

    def predict(self, latent_vector, condition):
        # return only image (remember generator returns condition too)
        return self._generator.predict([latent_vector, condition])

    def load_weights(self, file_path, by_name=False):
        # load weights saved by save_weights() before
        self.dcgan.load_weights(file_path, by_name)

    def save_weights(self, file_path, overwrite=True):
        # save weights as HDF5 configuration
        self.dcgan.save_weights(file_path, overwrite)
