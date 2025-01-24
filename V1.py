import numpy as np
import tensorflow as tf
import os
from pathlib import Path
from tensorflow.keras import layers, models
from tensorflow.keras.utils import image_dataset_from_directory
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



model = models.Sequential(
    [
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2,2)),

        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),

        layers.Flatten(),

        layers.Dense(128, activation='relu'),
        layers.Dropout(0.4),

        layers.Dense(1, activation='sigmoid')
    ] 
    )

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
    )


history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=10,
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


model.save("visuals.h5")