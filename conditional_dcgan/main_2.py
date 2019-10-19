import os
import numpy as np
from tensorflow import keras
import PIL
from tensorflow.python.keras.preprocessing import image
from tensorflow.python.keras.utils.np_utils import to_categorical
from conditional_dcgan_2 import Generator, Discriminator, ConditionalDCGAN


# Normalize image from [0, 255] to [-1, 1]
def normalize(X):
    return (X - 127.5) / 127.5


# Denormalize from [-1, 1] to [0, 255]
def denormalize(X):
    return (X + 1.0) * 127.5


def train(latent_dim, height, width, channels, num_class):
    (X_train, Y_train), (_, _) = keras.datasets.mnist.load_data()
    X_train = X_train[0:500]
    Y_train = Y_train[0:500]
    Y_train = to_categorical(Y_train, num_class)  # convert data into one-hot vectors
    # X_train = X_train.reshape((X_train.shape[0], ) + (height, width, channels)).astype('float32')
    X_train = X_train.astype('float32')
    X_train = X_train[:, :, :, None]
    X_train = normalize(X_train)

    # EPOCHS = 50
    # BATCH_SIZE = 128
    epochs = 3
    batch_size = 128
    iterations = int(X_train.shape[0] // batch_size)

    class_generator = Generator(latent_dim, num_class)
    generator = class_generator.get_model()
    class_discriminator = Discriminator(height, width, channels, num_class)
    discriminator = class_discriminator.get_model()
    class_dcgan = ConditionalDCGAN(latent_dim, height, width, channels, num_class)
    dcgan = class_dcgan.get_model()

    # set adversarial ground truths
    valid = np.ones((batch_size, 1))
    fake = np.zeros((batch_size, 1))

    # show networks
    generator.summary()
    discriminator.summary()
    dcgan.summary()

    for epoch in range(epochs):
        for iteration in range(iterations):
            real_images = X_train[iteration * batch_size: (iteration + 1) * batch_size]
            conditions = Y_train[iteration * batch_size: (iteration + 1) * batch_size]

            # train discriminator so it can detect real or fake
            random_latent_vectors = np.random.normal(size=(batch_size, latent_dim))
            generated_images = class_dcgan.predict(random_latent_vectors, conditions)
            combined_conditions = np.concatenate([conditions, conditions])
            combined_images = np.concatenate([real_images, generated_images])
            real_labels = np.concatenate([valid, fake])
            real_labels += 0.05 * np.random.random(real_labels.shape)  # add noise
            d_loss = discriminator.train_on_batch([combined_images, combined_conditions], real_labels)

            # train generator so it can fool discriminator
            random_latent_vectors = np.random.normal(size=(batch_size, latent_dim))
            misleading_targets = valid
            g_loss = dcgan.train_on_batch([random_latent_vectors, conditions], misleading_targets)

            # d_loss, g_loss = dcgan.train(real_images, conditions, batch_size)
            # show the progress of learning
            if (iteration + 1) % 2 == 0:
                print('{} / {}'.format(iteration + 1, iterations))
                print('discriminator loss: {}'.format(d_loss))
                print('generator loss: {}'.format(g_loss))
                print()
                with open('loss.txt', 'a') as f:
                    f.write(str(d_loss) + ',' + str(g_loss) + '\r')
        if (epoch + 1) % 2 == 0:
            class_dcgan.save_weights('gan' + '_epoch' + str(epoch + 1) + '.h5')
            random_latent_vectors = np.random.normal(size=(batch_size, latent_dim))
            generated_images = class_dcgan.predict(random_latent_vectors, conditions)
            # save generated images
            for i, generated_image in enumerate(generated_images):
                img = denormalize(generated_image)
                img = image.array_to_img(img, scale=False)
                condition = np.argmax(conditions[1])
                img.save(os.path.join('generated_images', str(epoch) + '_' + str(condition) + '.png'))
        print('epoch' + str(epoch) + ' end')
        print()


def predict(latent_dim, height, width, channels, num_class):
    class_dcgan = ConditionalDCGAN(latent_dim, height, width, channels, num_class)
    # dcgan.load_weights('gan_epoch50.h5')  # load weights after 50 times learning
    class_dcgan.load_weights('gan_epoch2.h5')  # load weights after 1 times learning
    for num in range(num_class):
        for id in range(10):
            random_latent_vectors = np.random.normal(size=(1, latent_dim))
            # create conditions like [0, 0, 1, 0, 0, 0, 0, 0, 0, 0] = 2
            condition = np.zeros((1, num_class), dtype=np.float32)
            condition[0, num] = 1
            generated_images = class_dcgan.predict(random_latent_vectors, condition)
            img = image.array_to_img(denormalize(generated_images[0]), scale=False)
            img.save(os.path.join('generated_images', str(num) + '_' + str(id) + '.png'))


if __name__ == '__main__':
    latent_dim = 100
    height = 28
    width = 28
    channels = 1
    num_class = 10
    train(latent_dim, height, width, channels, num_class)
    predict(latent_dim, height, width, channels, num_class)









