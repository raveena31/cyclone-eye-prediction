# import numpy as np
# import h5py
# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, GlobalAveragePooling2D
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.callbacks import EarlyStopping
# from sklearn.model_selection import train_test_split
# import matplotlib.pyplot as plt

# # -----------------------
# # Load HDF5 dataset
# # -----------------------
# data_path = 'TCIR-ALL_2017.h5'

# with h5py.File(data_path, 'r') as hf:
#     images = hf['matrix'][:]
#     info_group = hf['info']
#     block_items = info_group['block0_items'][:]
#     block_values = info_group['block0_values'][:]

#     lat_idx = np.where(block_items == b'lat')[0][0]
#     lon_idx = np.where(block_items == b'lon')[0][0]
#     labels = block_values[:, [lon_idx, lat_idx]]  # absolute lat/lon

# # -----------------------
# # Compute offsets relative to image center
# # -----------------------
# # Each image covers ±7° lat/lon
# radius_deg = 7.0
# # For all images, center = true lat/lon → offsets are 0
# # Optionally, if your labels have small shifts, compute offsets:
# # offsets = labels - labels_center (here, center = true lat/lon of each image)
# labels_offset = np.zeros_like(labels)  # shape (N,2)

# # -----------------------
# # Preprocess images
# # -----------------------
# images = images.astype('float32') / 255.0
# if len(images.shape) == 3:
#     images = np.expand_dims(images, axis=-1)

# # -----------------------
# # Train/test split
# # -----------------------
# X_train, X_test, y_train, y_test = train_test_split(
#     images, labels_offset, test_size=0.2, random_state=42
# )

# # -----------------------
# # Build CNN model
# # -----------------------
# model = Sequential([
#     Conv2D(32, (3,3), activation='relu', input_shape=X_train.shape[1:]),
#     MaxPooling2D((2,2)),

#     Conv2D(64, (3,3), activation='relu'),
#     MaxPooling2D((2,2)),

#     Conv2D(128, (3,3), activation='relu'),
#     GlobalAveragePooling2D(),

#     Dense(128, activation='relu'),
#     Dropout(0.5),
#     Dense(2, activation='linear')  # predict [dx, dy] offsets in degrees
# ])

# model.compile(optimizer=Adam(1e-3), loss='mse', metrics=['mae'])
# model.summary()

# # -----------------------
# # Train model
# # -----------------------
# es = EarlyStopping(patience=5, restore_best_weights=True)

# history = model.fit(
#     X_train, y_train,
#     validation_data=(X_test, y_test),
#     epochs=50,
#     batch_size=32,
#     callbacks=[es]
# )

# # -----------------------
# # Save model
# # -----------------------
# model.save('cyclone_center_offset_model.h5')
# print("✅ Model trained and saved as 'cyclone_center_offset_model.h5'")


import numpy as np
import h5py
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split

# -----------------------
# Load dataset
# -----------------------
data_path = 'TCIR-ALL_2017.h5'
with h5py.File(data_path, 'r') as hf:
    images = hf['matrix'][:]

images = images.astype('float32') / 255.0
if len(images.shape) == 3:
    images = np.expand_dims(images, axis=-1)

# -----------------------
# Image parameters
# -----------------------
img_size = images.shape[1]  # 201
center_pixel = img_size // 2
max_shift = 20  # max pixels to shift for synthetic eye positions

# -----------------------
# Apply random shifts to simulate cyclones anywhere
# -----------------------
shifted_images = []
eye_labels = []

for img in images:
    dx = np.random.randint(-max_shift, max_shift+1)
    dy = np.random.randint(-max_shift, max_shift+1)

    # Shift image
    shifted_img = np.roll(img, shift=dx, axis=1)  # horizontal
    shifted_img = np.roll(shifted_img, shift=dy, axis=0)  # vertical

    shifted_images.append(shifted_img)

    # New eye position in pixels
    eye_x = center_pixel + dx
    eye_y = center_pixel + dy
    eye_labels.append([eye_x, eye_y])

shifted_images = np.array(shifted_images)
eye_labels = np.array(eye_labels, dtype=np.float32)

# -----------------------
# Train/test split
# -----------------------
X_train, X_test, y_train, y_test = train_test_split(
    shifted_images, eye_labels, test_size=0.2, random_state=42
)

# -----------------------
# Build CNN
# -----------------------
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=X_train.shape[1:]),
    MaxPooling2D((2,2)),

    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D((2,2)),

    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D((2,2)),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(2, activation='linear')  # predict [x_pixel, y_pixel]
])

model.compile(optimizer=Adam(1e-3), loss='mse', metrics=['mae'])
model.summary()

# -----------------------
# Train
# -----------------------
es = EarlyStopping(patience=5, restore_best_weights=True)
model.fit(X_train, y_train,
          validation_data=(X_test, y_test),
          epochs=50,
          batch_size=32,
          callbacks=[es])

# -----------------------
# Save model
# -----------------------
model.save('cyclone_eye_detector.h5')
print("✅ Real cyclone eye detector trained and saved!")
