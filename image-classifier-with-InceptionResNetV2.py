import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Specify the data set path
data_dir = "/your_data_dir/Mush"

# Apply data augmentation techniques
datagen = ImageDataGenerator(
    validation_split=0.2, 
    rescale=1./255, 
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Load training and validation data sets
train_ds = datagen.flow_from_directory(
    data_dir,
    subset="training",
    seed=123,
    target_size=(180, 180),
    batch_size=32
)

val_ds = datagen.flow_from_directory(
    data_dir,
    subset="validation",
    seed=123,
    target_size=(180, 180),
    batch_size=32
)

class_names = train_ds.class_indices.keys()
num_classes = len(class_names)

# Load the pre-trained InceptionResNetV2 model
pretrained_model = tf.keras.applications.InceptionResNetV2(include_top=False,
                       input_shape=(180,180,3),
                       pooling='avg',
                       weights='imagenet')

# Freeze pre-trained weights
for layer in pretrained_model.layers:
    layer.trainable = False

# Create the model
model = models.Sequential([
    pretrained_model,
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.5),  # Add Dropout layer to reduce overfitting
    layers.Dense(num_classes, activation='softmax')
])

# Add callback for learning rate scheduling
lr_schedule = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 1e-3 * 10**(-epoch / 20))

# Modeli derleyin
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=10,
  callbacks=[lr_schedule]
)



import numpy as np
from tensorflow.keras.preprocessing import image

# Specify the file path of the image to test
img_path = '/content/drive/MyDrive/App/TestMussssh/download (11).jpg'

# Upload and process the image
img = image.load_img(img_path, target_size=(180, 180))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array /= 255.0  # Normalizasyon

# Make a forecast
predictions = model.predict(img_array)
top_3_indices = predictions[0].argsort()[-3:][::-1]

# Get class names
class_names = list(train_ds.class_indices.keys())


# Print the first three forecasts
for i in top_3_indices:
    print(f"Sınıf: {class_names[i]}, Olasılık: {predictions[0][i]*100:.2f}%")
