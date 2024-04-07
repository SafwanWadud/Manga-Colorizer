import tensorflow as tf
from tensorflow import keras
from keras import initializers, layers, Model, callbacks
from keras.losses import BinaryCrossentropy, MeanAbsoluteError
from matplotlib import pyplot as plt
import os

LAMBDA = 100

def downsample_layer(input, filters, batch_norm=True):
    weights_initializer = initializers.RandomNormal(stddev=0.02)
    output = layers.Conv2D(filters=filters, kernel_size=4, strides=2, padding='same', kernel_initializer=weights_initializer, use_bias=False)(input)
    if (batch_norm):
        output = layers.BatchNormalization()(output, training=True)
    output = layers.LeakyReLU(alpha=0.2)(output)
    return output

def upsample_layer(input, skipped_input, filters, dropout=True):
    weights_initializer = initializers.RandomNormal(stddev=0.02)
    output = layers.Conv2DTranspose(filters=filters, kernel_size=4, strides=2, padding='same', kernel_initializer=weights_initializer, use_bias=False)(input)
    output = layers.BatchNormalization()(output, training=True)
    if (dropout):
        output = layers.Dropout(rate=0.5)(output, training=True)
    output = layers.ReLU()(output)
    output = layers.Concatenate()([output, skipped_input])
    return output

def generator():
    input = layers.Input(shape=(256, 256, 1))

    # encoder
    encoder_l1 = downsample_layer(input, 64, batch_norm=False)
    encoder_l2 = downsample_layer(encoder_l1, 128)
    encoder_l3 = downsample_layer(encoder_l2, 256)
    encoder_l4 = downsample_layer(encoder_l3, 512)
    encoder_l5 = downsample_layer(encoder_l4, 512)
    encoder_l6 = downsample_layer(encoder_l5, 512)
    encoder_l7 = downsample_layer(encoder_l6, 512)
    
    bottleneck = downsample_layer(encoder_l7, 512)

    # decoder
    decoder_l1 = upsample_layer(bottleneck, encoder_l7, 512, dropout=True)
    decoder_l2 = upsample_layer(decoder_l1, encoder_l6, 512, dropout=True)
    decoder_l3 = upsample_layer(decoder_l2, encoder_l5, 512, dropout=True)
    decoder_l4 = upsample_layer(decoder_l3, encoder_l4, 512)
    decoder_l5 = upsample_layer(decoder_l4, encoder_l3, 256)
    decoder_l6 = upsample_layer(decoder_l5, encoder_l2, 128)
    decoder_l7 = upsample_layer(decoder_l6, encoder_l1, 64)

    weights_initializer = initializers.RandomNormal(stddev=0.02)
    output = layers.Conv2DTranspose(filters=3, kernel_size=4, strides=2, padding='same', activation='tanh', kernel_initializer=weights_initializer)(decoder_l7)
    return Model(inputs=input, outputs=output, name="generator")

def discriminator():
    input = layers.Input(shape=(256, 256, 1)) # uncoloured image
    target = layers.Input(shape=(256, 256, 3)) # actual or generated coloured image

    weights_initializer = initializers.RandomNormal(stddev=0.02)

    output = layers.Concatenate()([input, target])
    output = downsample_layer(output, 64, batch_norm=False)
    output = downsample_layer(output, 128)
    output = downsample_layer(output, 256)
    
    output = layers.ZeroPadding2D(padding=1)(output)
    output = layers.Conv2D(filters=512, kernel_size=4, strides=1, kernel_initializer=weights_initializer, use_bias=False)(output)
    output = layers.BatchNormalization()(output, training=True)
    output = layers.LeakyReLU(alpha=0.2)(output)

    output = layers.ZeroPadding2D(padding=1)(output)
    output = layers.Conv2D(filters=1, kernel_size=4, strides=1, kernel_initializer=weights_initializer)(output)

    return Model(inputs=[input, target], outputs=output, name="discriminator")

def discriminator_loss(gen_logits, actual_logits):
    gen_loss = BinaryCrossentropy(from_logits=True)(tf.zeros_like(gen_logits), gen_logits)
    actual_loss = BinaryCrossentropy(from_logits=True)(tf.ones_like(actual_logits), actual_logits)
    loss = (gen_loss + actual_loss)*0.5
    return loss

def generator_loss(gen_logits, gen_images, actual_images):
    cgan_loss = BinaryCrossentropy(from_logits=True)(tf.ones_like(gen_logits), gen_logits)
    l1_loss = MeanAbsoluteError()(actual_images, gen_images)
    loss = cgan_loss + LAMBDA * l1_loss
    return loss, cgan_loss, l1_loss

class cGAN(keras.Model):
    def __init__(self):
        super().__init__()
        self.generator = generator()
        self.discriminator = discriminator()

    def compile(self, disc_optimizer, gen_optimizer, disc_loss, gen_loss, metrics=None):
        super().compile(metrics=metrics)
        self.disc_optimizer = disc_optimizer
        self.gen_optimizer = gen_optimizer
        self.disc_loss = disc_loss
        self.gen_loss = gen_loss

    def train_step(self, data):
        input_images, actual_images = data

        # train discriminator
        with tf.GradientTape() as tape:
            gen_images = self.generator(input_images, training=True)
            gen_logits = self.discriminator([input_images, gen_images], training=True)
            actual_logits = self.discriminator([input_images, actual_images], training=True)

            d_loss = self.disc_loss(gen_logits, actual_logits)
            
        gradients = tape.gradient(d_loss, self.discriminator.trainable_variables)
        self.disc_optimizer.apply_gradients(zip(gradients, self.discriminator.trainable_variables))

        # train generator
        with tf.GradientTape() as tape:
            gen_images = self.generator(input_images, training=True)
            gen_logits = self.discriminator([input_images, gen_images], training=True)

            g_total_loss, g_loss, l1_loss  = self.gen_loss(gen_logits, gen_images, actual_images)
            
        gradients = tape.gradient(g_total_loss, self.generator.trainable_variables)
        self.gen_optimizer.apply_gradients(zip(gradients, self.generator.trainable_variables))

        # # update metrics
        # for metric in self.metrics:
        #     metric.update_state(y, y_pred)

        return {"d_loss": d_loss*2, "g_loss": g_loss, "l1_loss": l1_loss}
        # return {"d_loss": d_loss, "g_loss": g_loss, **{metric.name : metric.result() for metric in self.metrics}}
    
    def test_step(self, data):
        input_images, actual_images = data

        gen_images = self.generator(input_images, training=False)
        gen_logits = self.discriminator([input_images, gen_images], training=False)
        actual_logits = self.discriminator([input_images, actual_images], training=False)
        d_loss = self.disc_loss(gen_logits, actual_logits)

        gen_images = self.generator(input_images, training=False)
        gen_logits = self.discriminator([input_images, gen_images], training=False)
        g_total_loss, g_loss, l1_loss  = self.gen_loss(gen_logits, gen_images, actual_images)

        # # update metrics
        # for metric in self.metrics:
        #     metric.update_state(y, y_pred)

        return {"d_loss": d_loss*2, "g_loss": g_loss, "l1_loss": l1_loss}
    
def unnormalize(input, generated, actual):
    input = (input * 0.5) + 0.5
    generated = (generated * 0.5) + 0.5
    actual = (actual * 0.5) + 0.5
    return input, generated, actual

class GenerateImagesCallback(callbacks.Callback):

    def __init__(self, dataset, dataset_name, dir):
        super().__init__()
        self.dataset = dataset
        self.dataset_name = dataset_name
        self.image_dir = dir 
        self.test_mode = dataset_name == 'test'

    def on_epoch_end(self, epoch, logs=None):
        
        subset = self.dataset.take(1)

        for input_images, actual_images in subset:
            gen_images = self.model.generator(input_images, training=False)
            input_images, gen_images, actual_images = unnormalize(input_images, gen_images, actual_images)
            
            fig, axs = plt.subplots(4, 3, figsize=(30, 30))
            plt.subplots_adjust(wspace=0.05)
            axs[0, 0].set_title("Input", fontsize=30)
            axs[0, 1].set_title("Generated", fontsize=30)
            axs[0, 2].set_title("Actual", fontsize=30)
            for i in range(input_images.shape[0]):
                axs[i, 0].imshow(input_images[i], cmap='gray')
                axs[i, 0].axis("off")
                axs[i, 1].imshow(gen_images[i])
                axs[i, 1].axis("off")
                axs[i, 2].imshow(actual_images[i])
                axs[i, 2].axis("off")
            fig.savefig(os.path.join(self.image_dir, f"{self.dataset_name}-epoch-{epoch+1}.png"))
            plt.close()
    
    def on_test_end(self, logs=None):
        if (self.test_mode):
            self.on_epoch_end(0)        









