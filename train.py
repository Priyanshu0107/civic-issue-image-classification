import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# =========================
# DATASET PATH (FIXED)
# =========================
data_dir = "dataset"
img_size = (224, 224)
batch_size = 32

print("📁 Current Directory:", os.getcwd())
print("📂 Dataset Folders:", os.listdir(data_dir))

# =========================
# DATA GENERATOR
# =========================
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)

# =========================
# TRAINING DATA
# =========================
train_data = datagen.flow_from_directory(
    data_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

# =========================
# VALIDATION DATA
# =========================
val_data = datagen.flow_from_directory(
    data_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

# =========================
# BASE MODEL
# =========================
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False

# =========================
# FINAL MODEL
# =========================
model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(3, activation='softmax')
])

# =========================
# COMPILE
# =========================
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# =========================
# TRAIN
# =========================
model.fit(
    train_data,
    validation_data=val_data,
    epochs=20
)

# =========================
# SAVE MODEL
# =========================
model.save("civic_issue_classifier.h5")

print("✅ Model training complete & saved successfully")
