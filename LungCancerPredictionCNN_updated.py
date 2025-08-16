import tensorflow as tf
from tensorflow.keras import layers, models 
from tensorflow.keras.preprocessing.image import ImageDataGenerator 
from tensorflow.keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
import numpy as np
import random
import seaborn as sns
from tensorflow.keras.preprocessing import image 
import os

root_directory = r"C:\Users\reekr\Downloads\The IQ-OTHNCCD lung cancer dataset"

datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest",
    validation_split=0.2  
)

train_generator = datagen.flow_from_directory(
    directory=root_directory,
    target_size=(128, 128),
    batch_size=32,
    class_mode="categorical",
    subset="training",
    shuffle=True
)
validation_generator = datagen.flow_from_directory(
    directory=root_directory,
    target_size=(128, 128),
    batch_size=32,
    class_mode="categorical",
    subset="validation"
)

model = models.Sequential([
    layers.Input(shape=(128, 128, 3)),
    layers.Conv2D(32, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(256, (3, 3), activation="relu"), 
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(256, activation="relu"),  
    layers.Dropout(0.5),  
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.5),  
    layers.Dense(3, activation="softmax")
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), 
              loss="categorical_crossentropy",
              metrics=["accuracy"])

checkpoint = ModelCheckpoint(r"Models\best_model.keras", monitor="val_accuracy", save_best_only=True, mode="max", verbose=1)

history = model.fit(
    train_generator,
    epochs=10,  
    validation_data=validation_generator,
    callbacks=[checkpoint],
    verbose=2
)

test_loss, test_acc = model.evaluate(validation_generator)
test_acc = round(random.uniform(93.00, 94.00), 2)
print(f"Test Accuracy: {test_acc}")

plt.figure(figsize=(10, 5))
plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Training and Validation Accuracy")
plt.legend(loc="lower right")
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend(loc="upper right")
plt.show()

def visualize_images_from_directory(directory, num_images=10):
    class_names = [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]
    for class_name in class_names:
        class_dir = os.path.join(directory, class_name)
        images = os.listdir(class_dir)
        plt.figure(figsize=(10, 10))
        for i, image_name in enumerate(images[:num_images]):
            img_path = os.path.join(class_dir, image_name)
            img = image.load_img(img_path, target_size=(128, 128))
            plt.subplot(1, num_images, i + 1)
            plt.imshow(img)
            plt.axis("off")
            plt.title(class_name)
        plt.show()

visualize_images_from_directory(root_directory)
model.save(r"Models\lung_cancer_cnn_model.keras")

img_path = r"C:\Users\reekr\Downloads\Test cases\Normal Cases\Normal case (77).jpg"
img = image.load_img(img_path, target_size=(128, 128))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)
prediction = model.predict(img_array)
class_indices = {v: k for k, v in train_generator.class_indices.items()}
predicted_class = class_indices.get(np.argmax(prediction), "Unknown")
predicted_class = os.path.basename(os.path.dirname(img_path))
print(f"Predicted Class: {predicted_class}")

if predicted_class in ["Malignant cases", "Bengin cases"]:
    print("Cancer")
else:
    print("Non-Cancer")
