import tensorflow as tf
from tensorflow.keras import layers, models, optimizers

import numpy as np
from tensorflow.keras.preprocessing import image
import pathlib


# Specify the data set path
data_dir = "/content/drive/MyDrive/App/Mush2"

# Configure data loading()
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,     #We provide our own validation class 'validation_split=0.2' with this code. You can change this '0.2' ratio to suit yourself.
  subset="training",
  seed=123,
  image_size=(180, 180),
  batch_size=32)

class_names = train_ds.class_names
num_classes = len(class_names)

# Installing the pre-trained ResNet50 model
pretrained_model= tf.keras.applications.ResNet50(include_top=False,
                   input_shape=(180,180,3),
                   pooling='avg',
                   weights='imagenet'
)

# Freezing pre-trained weights
for layer in pretrained_model.layers:
        layer.trainable=False

# Create the model
model = models.Sequential([
    pretrained_model,
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dense(num_classes, activation='softmax')
])

# Model compilation
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(
  train_ds,
  epochs=10
)




# File path of the photo to be tested
test_image_path = "/content/drive/MyDrive/App/TestMussssh/1655108.jpg"

# Upload the photo and adjust its size and number of channels (RGB)
img = image.load_img(test_image_path, target_size=(180, 180))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)  # Batch boyutu ekleyin

# Make the model predict
predictions = model.predict(img_array)
predicted_class_index = np.argmax(predictions[0])
confidence = predictions[0][predicted_class_index]

# Get the class label
class_names = train_ds.class_names
predicted_class = class_names[predicted_class_index]

print("Tahmin edilen sınıf:", predicted_class)
print("Güven seviyesi:", confidence)
