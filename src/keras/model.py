"""
Guide: https://www.tensorflow.org/guide/keras/customizing_what_happens_in_fit
"""

from tensorflow.keras import layers
import tensorflow as tf


class GAN(tf.keras.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.noise_dim = 28*28     # noise input dimension
        self.generator = self.init_generator(self.noise_dim)
        self.discriminator = self.init_discriminator()
        print("Hello 3")

    @staticmethod
    def init_generator(noise_dim):
        input = layers.Input(shape=(noise_dim, ))
        h = layers.Dense(2048, activation='relu')(input)
        h = layers.Dense(1024, activation='relu')(h)
        output = layers.Dense(28*28, activation='sigmoid')(h)       # generated image has values between 0-1
        model = tf.keras.Model(inputs=input, outputs=output, name='generator')
        return model

    @staticmethod
    def init_discriminator():
        input = layers.Input(shape=(28*28, ))
        h = layers.Dense(256, activation='relu')(input)
        h = layers.Dense(128, activation='relu')(h)
        output = layers.Dense(1, activation='sigmoid')(h)           # P(true image) : 1 - true image, 0 - fake image
        model = tf.keras.Model(inputs=input, outputs=output, name='discriminator')
        return model

    def sample_random_noise(self, batch_size):
        random_noise = tf.random.normal(shape=(batch_size, self.noise_dim))
        return random_noise

    def compile(self, d_optimizer, g_optimizer, loss_fn):
        super().compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.loss_fn = loss_fn

    def train_step(self, real_images):
        #real_images are array of flattened images.
        if isinstance(real_images, tuple):
            real_images = real_images[0]
        batch_size = tf.shape(real_images)[0]

        # Discriminator training
        random_noise = self.sample_random_noise(batch_size)
        fake_images = self.generator(random_noise)
        combine_images = tf.concat([real_images, fake_images], axis=0)
        # true image : 1 / fake image: 0
        labels = tf.concat([tf.ones((batch_size, 1)), tf.zeros((batch_size, 1))], axis=0)

        # label smoothening
        # Add random noise to the labels - important trick!
        # labels += 0.05 * tf.random.uniform(tf.shape(labels))    # <--- makes d_loss negative which should not be possible??

        with tf.GradientTape() as tape:
            predictions = self.discriminator(combine_images)
            d_loss = self.loss_fn(labels, predictions)

        grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
        self.d_optimizer.apply_gradients(
            zip(grads, self.discriminator.trainable_weights)
        )

        # Generator training
        random_noise = self.sample_random_noise(batch_size)
        # So that we can use the same loss fn of D, we just flip the labels of the data
        misleading_labels = tf.ones((batch_size, 1))
        with tf.GradientTape() as tape:
            fake_images = self.generator(random_noise)
            predictions = self.discriminator(fake_images)
            g_loss = self.loss_fn(misleading_labels, predictions)

        grads = tape.gradient(g_loss, self.generator.trainable_weights)
        self.g_optimizer.apply_gradients(
            zip(grads, self.generator.trainable_weights)
        )

        return {'d_loss': d_loss, 'g_loss': g_loss}


class GAN_CNN(GAN):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.noise_dim = 128     # noise input dimension
        self.generator = self.init_generator(self.noise_dim)
        self.discriminator = self.init_discriminator()
        print("Hello cnn")

    @staticmethod
    def init_generator(noise_dim):
        generator = layers.Sequential(
            [
                layers.Input(shape=(noise_dim,)),
                # We want to generate 128 coefficients to reshape into a 7x7x128 map
                layers.Dense(7 * 7 * 128),
                layers.LeakyReLU(alpha=0.2),
                layers.Reshape((7, 7, 128)),
                layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding="same"),
                layers.LeakyReLU(alpha=0.2),
                layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding="same"),
                layers.LeakyReLU(alpha=0.2),
                layers.Conv2D(1, (7, 7), padding="same", activation="sigmoid"),
            ],
            name="generator",
        )
        return generator

    @staticmethod
    def init_discriminator():
        discriminator = tf.keras.Sequential(
            [
                layers.Input(shape=(28, 28, 1)),
                layers.Conv2D(64, (3, 3), strides=(2, 2), padding="same"),
                layers.LeakyReLU(alpha=0.2),
                layers.Conv2D(128, (3, 3), strides=(2, 2), padding="same"),
                layers.LeakyReLU(alpha=0.2),
                layers.GlobalMaxPooling2D(),
                layers.Dense(1),
            ],
            name="discriminator",
        )
        return discriminator

