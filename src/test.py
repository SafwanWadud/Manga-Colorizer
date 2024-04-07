import tensorflow as tf
from tensorflow import keras
from keras.utils import image_dataset_from_directory
from keras.optimizers import Adam
import datetime


import os
from network import cGAN, GenerateImagesCallback, discriminator_loss, generator_loss

BATCH_SIZE = 4
SEED = 42

DATASET_FOLDER = r'C:\Users\safwa\Desktop\dataset'

def normalize(u_image, c_image): # normalize to [-1, 1]
    u_image, c_image = (u_image / 127.5 - 1), (c_image / 127.5 - 1)
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
    test_dataset = coloured_dataset.skip(train_size)
    
    test_dataset = test_dataset.map(preprocess_eval, num_parallel_calls=tf.data.AUTOTUNE).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    return test_dataset

def test_model(test_dataset):
    model = cGAN()

    model.compile(disc_optimizer=Adam(learning_rate=2e-4, beta_1=0.5), 
                       gen_optimizer=Adam(learning_rate=2e-4, beta_1=0.5),
                       disc_loss=discriminator_loss,
                       gen_loss=generator_loss)

    model.generator.build(input_shape=(256, 256, 1))
    model.generator.load_weights("./model/generator.weights.h5")

    # model.discriminator.build(input_shape=[(256, 256, 1), (256, 256, 3)])
    model.discriminator.load_weights("./model/discriminator.weights.h5")

    run = datetime.datetime.now().strftime("%Y%m%d-%H%M")
    os.makedirs(f"./images/test/{run}", exist_ok=True)
    
    history = model.evaluate(x=test_dataset, callbacks=[GenerateImagesCallback(test_dataset, 'test', f'./images/test/{run}')])

def main():
    test_dataset = build_datasets()

    test_model(test_dataset)
    

if __name__ == "__main__":
    main()