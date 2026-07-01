

# import numpy as np
# import h5py
# from tensorflow.keras.models import load_model
# import matplotlib.pyplot as plt

# data_path = 'TCIR-ALL_2017.h5'
# model_path = 'cyclone_center_offset_model.h5'

# # Load model without compilation
# model = load_model(model_path, compile=False)

# # Load images
# with h5py.File(data_path, 'r') as hf:
#     images = hf['matrix'][:]

# images = images.astype('float32') / 255.0
# if len(images.shape) == 3:
#     images = np.expand_dims(images, axis=-1)

# # Predict offsets
# pred_offsets = model.predict(images)

# # Image parameters
# img_size = 201
# radius_deg = 7.0
# pixel_per_deg = img_size / (2 * radius_deg)
# center_pixel = img_size // 2

# # Visualize first N images
# N = 20
# for i in range(N):
#     img = images[i]
#     img_disp = img[...,0] if img.shape[-1] > 1 else img

#     plt.imshow(img_disp, cmap='gray')

#     # True center: always middle of the image
#     plt.scatter(center_pixel, center_pixel, c='red', s=100, marker='o', edgecolors='black', label='True Center' if i==0 else "")

#     # Predicted offset
#     dx_deg, dy_deg = pred_offsets[i]
#     pred_x = center_pixel + dx_deg * pixel_per_deg
#     pred_y = center_pixel - dy_deg * pixel_per_deg  # invert y-axis
#     plt.scatter(pred_x, pred_y, c='green', s=100, marker='x', label='Predicted Center' if i==0 else "")

#     plt.title(f'Frame {i}')
#     plt.axis('off')
#     if i==0:
#         plt.legend()
#     plt.show()


import numpy as np
import h5py
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# -----------------------
# Paths
# -----------------------
data_path = 'TCIR-ALL_2017.h5'
model_path = 'cyclone_eye_detector.h5'

# -----------------------
# Load model
# -----------------------
model = load_model(model_path, compile=False)
print("✅ Model loaded.")

# -----------------------
# Load images
# -----------------------
with h5py.File(data_path, 'r') as hf:
    images = hf['matrix'][:]

images = images.astype('float32') / 255.0
if len(images.shape) == 3:
    images = np.expand_dims(images, axis=-1)

img_size = images.shape[1]
center_pixel = img_size // 2
max_shift = 20  # must match train.py

# -----------------------
# Apply same random shifts for testing
# -----------------------
shifted_images = []
true_eye_positions = []

for img in images[:20]:  # first 20 frames
    dx = np.random.randint(-max_shift, max_shift+1)
    dy = np.random.randint(-max_shift, max_shift+1)
    shifted_img = np.roll(img, shift=dx, axis=1)
    shifted_img = np.roll(shifted_img, shift=dy, axis=0)
    shifted_images.append(shifted_img)
    true_eye_positions.append([center_pixel + dx, center_pixel + dy])

shifted_images = np.array(shifted_images)
true_eye_positions = np.array(true_eye_positions)

# -----------------------
# Predict eye positions
# -----------------------
pred_eye_positions = model.predict(shifted_images)

# -----------------------
# Visualize
# -----------------------
for i in range(len(shifted_images)):
    img = shifted_images[i]
    img_disp = img[...,0] if img.shape[-1] > 1 else img

    plt.imshow(img_disp, cmap='gray')

    # True eye
    true_x, true_y = true_eye_positions[i]
    plt.scatter(true_x, true_y, c='red', s=100, marker='o', edgecolors='black', label='True Eye' if i==0 else "")

    # Predicted eye
    pred_x, pred_y = pred_eye_positions[i]
    plt.scatter(pred_x, pred_y, c='green', s=100, marker='x', label='Predicted Eye' if i==0 else "")

    plt.title(f'Frame {i}')
    plt.axis('off')
    if i==0:
        plt.legend()
    plt.show()
