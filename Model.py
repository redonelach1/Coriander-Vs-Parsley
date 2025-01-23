import numpy as np
import tensorflow as tf
import os
from pathlib import Path
from tensorflow.keras import layers, models
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import VGG16


import matplotlib.pyplot as plt

train_dir = "./train/"

batch_size = 4
img_size = (150, 150)



#data loading and preprocessing
train_dataset = image_dataset_from_directory(
    train_dir,
    validation_split=0.2,
    label_mode="int",
    subset="training",
    seed=1337,
    image_size=img_size,
    batch_size=batch_size
    )

val_dataset = image_dataset_from_directory(
    train_dir,
    validation_split=0.2,
    label_mode="int",
    subset="validation",
    seed=1337,
    image_size=img_size,
    batch_size=batch_size
    )


#function for normalizing images
def normalize_images(image,label):
    image = tf.cast(image,tf.float32)/255.0
    return image, label


print(train_dataset.class_names)

for images, labels in train_dataset.take(1):
    print(labels.numpy())




#applying normalization to the datasets
train_dataset = train_dataset.map(normalize_images)
val_dataset = val_dataset.map(normalize_images)



#preloads step s+1 data while processing step s data, for optimized performance
train_dataset = train_dataset.prefetch(buffer_size=10)
val_dataset = val_dataset.prefetch(buffer_size=10)



base_model = VGG16(weights="imagenet", include_top=False, input_shape=(150,150,3))
base_model.trainable = False

model = models.Sequential(
    [
        base_model,
        layers.Flatten(),

        layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        layers.Dropout(0.5),

        
        layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        layers.Dropout(0.5),


        
        layers.Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        layers.Dropout(0.5),

        layers.Dense(1, activation='sigmoid', kernel_regularizer=tf.keras.regularizers.l2(0.01))
    ] 
    )

model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss="binary_crossentropy",
    metrics=["accuracy"]
    )


history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=15,
    verbose=1
)

val_loss, val_acc = model.evaluate(val_dataset)

print(f"Validation loss: {val_loss}")
print(f"Validation accuracy: {val_acc}")



# Plot training & validation accuracy values
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()
