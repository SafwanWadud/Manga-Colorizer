import tensorflow as tf
from tensorflow import keras
import os
from keras.optimizers import Adam
from keras.utils import image_dataset_from_directory
from keras.layers import RandomFlip
from keras.callbacks import TensorBoard
import datetime

from network import cGAN, GenerateImagesCallback, discriminator_loss, generator_loss

BATCH_SIZE = 4
IMG_WIDTH = IMG_HEIGHT = 256
SEED = 42
EPOCHS = 30

DATASET_FOLDER = r'C:\Users\safwa\Desktop\dataset'

def train_model(train_dataset, val_dataset):
    cgan_model = cGAN()
    cgan_model.compile(disc_optimizer=Adam(learning_rate=2e-4, beta_1=0.5), 
                       gen_optimizer=Adam(learning_rate=2e-4, beta_1=0.5),
                       disc_loss=discriminator_loss,
                       gen_loss=generator_loss) # AUC, IOU
    
    run = datetime.datetime.now().strftime("%Y%m%d-%H%M")
    os.makedirs(f"./images/val/{run}", exist_ok=True)
    callbacks = [GenerateImagesCallback(val_dataset, 'val', f'./images/val/{run}'), TensorBoard(log_dir="./logs/"+run)]
    cgan_model.fit(x=train_dataset, validation_data=val_dataset, epochs=EPOCHS, steps_per_epoch=500, validation_steps=100, callbacks=callbacks)

    return cgan_model

def normalize(u_image, c_image): # normalize to [-1, 1]
    u_image, c_image = (u_image / 127.5 - 1), (c_image / 127.5 - 1)
    return u_image, c_image

def augment(u_image, c_image):
    u_image, c_image = RandomFlip(seed=SEED)(u_image), RandomFlip(seed=SEED)(c_image)
    return u_image, c_image

def preprocess_train(c_image):
    u_image = tf.image.rgb_to_grayscale(c_image)
    u_image, c_image = augment(u_image, c_image)
    u_image, c_image = normalize(u_image, c_image)
    return u_image, c_image

def preprocess_eval(c_image):
    u_image = tf.image.rgb_to_grayscale(c_image)
    u_image, c_image = normalize(u_image, c_image)
    return u_image, c_image

def build_datasets():
    coloured_dataset = image_dataset_from_directory(os.path.join(DATASET_FOLDER, "coloured"), 
                                                      labels=None, 
                                                      color_mode='rgb',
                                                      shuffle=True,
                                                      seed=SEED,
                                                      batch_size=None)
    
    train_size = int(coloured_dataset.cardinality().numpy()*0.8)
    train_dataset = coloured_dataset.take(train_size)
    
    train_size = int(train_dataset.cardinality().numpy()*0.75)
    val_dataset = train_dataset.skip(train_size)
    train_dataset = train_dataset.take(train_size)

    train_dataset = train_dataset.shuffle(buffer_size=1000, seed=SEED).map(preprocess_train, num_parallel_calls=tf.data.AUTOTUNE).repeat().batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    val_dataset = val_dataset.map(preprocess_eval, num_parallel_calls=tf.data.AUTOTUNE).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    return train_dataset, val_dataset

def main():
    train_dataset, val_dataset = build_datasets()

    model = train_model(train_dataset, val_dataset)

    model.generator.save_weights("./model/generator.weights.h5")
    model.discriminator.save_weights("./model/discriminator.weights.h5")
    

# TODO
# GPU
# - dataset shuffling, seed?
# aspect ratio stuff, or do L->AB, Gray->RGB
# metrics
# validation/testing eval
# callbacks
# saving/loading models
#

if __name__ == "__main__":
    main()


    