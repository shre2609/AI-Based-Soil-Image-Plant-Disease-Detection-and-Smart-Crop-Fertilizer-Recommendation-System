import os, imghdr, hashlib
import numpy as np
import tensorflow as tf
from keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from keras import layers, models
from keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from keras.utils import load_img, img_to_array

print("✅ TensorFlow:", tf.__version__)

train_dir = r"D:\\Soil Project\\Dataset\\Train"
test_dir  = r"D:\\Soil Project\\Dataset\\Test"

# Clean Dataset 
def clean_dataset(directory):
    valid_exts = ["jpeg", "jpg", "png", "bmp", "gif"]
    for root, _, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            ext = imghdr.what(file_path)
            if ext not in valid_exts:
                os.remove(file_path)
                print("❌ Removed:", file_path)

clean_dataset(train_dir)
clean_dataset(test_dir)

# Detect Duplicate Images 
def get_file_hash(filepath):
    """Return MD5 hash of a file."""
    with open(filepath, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()

def check_duplicates(train_dir, test_dir):
    train_hashes, test_hashes = {}, {}
    
    # Hash train files
    for root, _, files in os.walk(train_dir):
        for file in files:
            path = os.path.join(root, file)
            train_hashes[get_file_hash(path)] = path
    
    # Hash test files
    duplicates = []
    for root, _, files in os.walk(test_dir):
        for file in files:
            path = os.path.join(root, file)
            h = get_file_hash(path)
            if h in train_hashes:
                duplicates.append((path, train_hashes[h]))
    
    if duplicates:
        print(f"⚠ Found {len(duplicates)} duplicate(s) between Train and Test sets:")
        for test_file, train_file in duplicates[:10]:  # show first 10
            print("Test:", test_file)
            print("Train:", train_file, "\n")
    else:
        print("✅ No duplicates found between Train and Test sets.")

check_duplicates(train_dir, test_dir)

# Load Datasets
batch_size = 32
img_size = (224, 224)

train_ds = tf.keras.utils.image_dataset_from_directory(
    train_dir, image_size=img_size, batch_size=batch_size, label_mode="categorical"
)
val_ds = tf.keras.utils.image_dataset_from_directory(
    test_dir, image_size=img_size, batch_size=batch_size, label_mode="categorical"
)

class_names = train_ds.class_names
print("Detected classes:", class_names)

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.map(lambda x, y: (preprocess_input(tf.cast(x, tf.float32)), y)).prefetch(AUTOTUNE)
val_ds   = val_ds.map(lambda x, y: (preprocess_input(tf.cast(x, tf.float32)), y)).prefetch(AUTOTUNE)

# 5Build Model (MobileNetV2)
base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False   # freeze backbone

x = layers.GlobalAveragePooling2D()(base_model.output)
x = layers.Dense(128, activation="relu")(x)
x = layers.Dropout(0.3)(x)
out = layers.Dense(len(class_names), activation="softmax")(x)

model = models.Model(inputs=base_model.input, outputs=out)
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.summary()

# Train
callbacks = [
    ModelCheckpoint("soil_model.keras", save_best_only=True, monitor="val_accuracy"),
    EarlyStopping(patience=3, restore_best_weights=True)
]

history = model.fit(train_ds, validation_data=val_ds, epochs=15, callbacks=callbacks)

# Evaluate
loss, acc = model.evaluate(val_ds)
print(f"📊 Test Accuracy: {acc*100:.2f}%")

y_true, y_pred = [], []
for imgs, labels in val_ds:
    preds = model.predict(imgs)
    y_true.extend(np.argmax(labels.numpy(), axis=1))
    y_pred.extend(np.argmax(preds, axis=1))

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.title("Confusion Matrix"); plt.show()

print("\n📌 Classification Report:\n", classification_report(y_true, y_pred, target_names=class_names))

# Predict Single Image
def predict_soil(image_path):
    img = load_img(image_path, target_size=(224, 224))
    arr = img_to_array(img)
    arr = np.expand_dims(arr, axis=0)
    arr = preprocess_input(arr)
    pred = model.predict(arr)
    idx = np.argmax(pred)
    print(f"🖼 {image_path}")
    print(f"🔍 Predicted Soil: {class_names[idx]} ({pred[0][idx]*100:.2f}%)")
