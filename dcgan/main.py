import os
import numpy as np
import tensorflow as tf
import PIL
from tensorflow.python.keras.preprocessing import image
from dcgan import DCGAN


# Normalize image from 0 - 255 to -1 - 1
def normalize(X):
    return (X - 127.5) / 127.5


# Denormalize from -1 - 1 to 0 - 255
def denormalize(X):
    return (X + 1.0) * 127.5


def train(latent_dim, height, width, channels):
    (X_train, Y_train), (_, _) = tf.keras.datasets.mnist.load_data()
    # X_train = X_train[0: 2000]
    X_train = X_train.reshape((X_train.shape[0],) + (height, width, channels)).astype('float32')
    X_train = normalize(X_train)
    epochs = 20
    # epochs = 2
    batch_size = 128
    iterations = X_train.shape[0]//batch_size
    dcgan = DCGAN(latent_dim, height, width, channels)

    for epoch in range(epochs):
        for iteration in range(iterations):
            real_images = X_train[iteration*batch_size:(iteration+1)*batch_size]
            d_loss, g_loss = dcgan.train(real_images, batch_size)
            if (iteration + 1) % 10 == 0:
                print('{} / {}'.format(iteration + 1, iterations))
                print('discriminator loss: {}'.format(d_loss))
                print('generator loss: {}'.format(g_loss))
                print()
                with open('loss.txt', 'a') as f:
                    f.write(str(d_loss) + ',' + str(g_loss) + '\r')
        dcgan.save_weights('gan' + '_epoch' + str(epoch + 1) + '.h5')
        print('epoch' + str(epoch) + ' end')
        print()


def predict(latent_dim, height, width, channels):
    random_latent_vectors = np.random.normal(size=(100, latent_dim))
    dcgan = DCGAN(latent_dim, height, width, channels)
    dcgan.load_weights('gan_epoch20.h5')
    # dcgan.load_weights('gan_epoch1.h5')
    generated_images = dcgan.predict(random_latent_vectors)
    for i, generated_image in enumerate(generated_images):
        img = image.array_to_img(denormalize(generated_image), scale=False)
        img.save(os.path.join('generated', str(i) + '.png'))


if __name__ == '__main__':
    latent_dim = 100
    height = 28
    width = 28
    channels = 1
    train(latent_dim, height, width, channels)
    predict(latent_dim, height, width, channels)
