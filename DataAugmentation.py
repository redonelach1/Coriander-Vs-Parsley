import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os
from pathlib import Path
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import array_to_img


output_dir = "C:/Users/RedNa/OneDrive/Bureau/Coriander-vs-Parsley-master/AugDataSet/"


train_dataset = image_dataset_from_directory(
    "C:/Users/RedNa/OneDrive/Bureau/Coriander-vs-Parsley-master/train/",
    image_size=(150,150),
    batch_size=1,
    shuffle=True)


data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.2),
    ])



os.makedirs(output_dir, exist_ok=True)

for i, (images,labels) in enumerate(train_dataset):
    for j, image in enumerate(images):
        image = tf.expand_dims(image,axis=0)

        aug_img = data_augmentation(image)
        
        aug_img = tf.squeeze(aug_img, axis=0)
        
        print(f"saving image {i}_{j} shape : {aug_img.shape}")

        label = labels.numpy()[j]
        label_dir = Path(output_dir) / str(label)
        label_dir.mkdir(exist_ok=True,parents=True)

        image_path = label_dir / f"aug_{i}_{j}.jpg"
        array_to_img(aug_img).save(image_path)

