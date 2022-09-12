from tensorflow.keras import layers
import tensorflow as tf


class GAN(tf.keras.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.generator = self.init_generator()
        self.discriminator = self.init_discriminator()
        self.latent_dim = 28*28

    @staticmethod
    def init_generator():
        input = layers.Input(shape=(28*28, ))
        h = layers.Dense(256, activation='relu')(input)
        h = layers.Dense(128, activation='relu')(h)
        output = layers.Dense(28*28, activation='sigmoid')(h)
        model = tf.keras.Model(inputs=input, outputs=output, name='generator')
        return model

    @staticmethod
    def init_discriminator():
        input = layers.Input(shape=(28*28, ))
        h = layers.Dense(256, activation='relu')(input)
        h = layers.Dense(128, activation='relu')(h)
        output = layers.Dense(1, activation='sigmoid')(h)
        model = tf.keras.Model(inputs=input, outputs=output, name='discriminator')
        return model

    def sample_random_noise(self, batch_size):
        random_noise = tf.random.normal(shape=(batch_size, self.latent_dim))
        return random_noise

    def compile(self,d_optimizer, g_optimizer, loss_fn):
        super().compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.loss_fn = loss_fn

    def train_step(self, real_images):
        images = real_images[0]
        batch_size = len(images)
        images_flat = images.reshape((28 * 28), axis=1)

        # Discriminator training
        random_noise = self.sample_random_noise(batch_size)
        fake_images = self.generator(random_noise)
        combine_images = tf.concat([images_flat, fake_images], axis=0)
        labels = tf.concat([tf.ones((batch_size, 1)), tf.zeros((batch_size, 1))], axis=0)
        with tf.GradientTape() as tape:
            predictions = self.discriminator(combine_images)
            d_loss = self.loss_fn(labels, predictions)

        grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
        self.d_optimizer.apply_gradients(
            zip(grads, self.discriminator.trainable_weights)
        )

        # Generator training
        random_noise = self.sample_random_noise(batch_size)
        labels = tf.zeros((batch_size, 1))
        with tf.GradientTape() as tape:
            fake_images = self.generator(random_noise)
            predictions = self.discriminator(fake_images)
            g_loss = -1 * self.loss_fn(labels, predictions)

        grads = tape.gradient(g_loss, self.generator.trainable_weights)
        self.g_optimizer.apply_gradients(
            zip(grads, self.generator.trainable_weights)
        )

        return {'d_loss': d_loss, 'g_loss': g_loss}

