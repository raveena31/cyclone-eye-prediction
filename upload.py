import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# -----------------------
# Load trained model
# -----------------------
model_path = 'cyclone_eye_detector.h5'
model = load_model(model_path, compile=False)
print("✅ Model loaded.")

# -----------------------
# Ask user for image path
# -----------------------
image_path = input("Enter path of the image: ").strip()

# -----------------------
# Load and preprocess image
# -----------------------
img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)  # read as grayscale or color

# Resize to 201x201 (model input size)
img_resized = cv2.resize(img, (201, 201))

# Convert to float and normalize
img_resized = img_resized.astype('float32') / 255.0

# Ensure 4 channels
if img_resized.ndim == 2:
    # grayscale -> duplicate 4 times
    img_resized = np.stack([img_resized]*4, axis=-1)
elif img_resized.shape[2] == 3:
    # RGB -> add 1 dummy channel
    dummy = np.zeros((201, 201, 1), dtype=np.float32)
    img_resized = np.concatenate([img_resized, dummy], axis=-1)

# Add batch dimension
input_img = np.expand_dims(img_resized, axis=0)

# -----------------------
# Predict cyclone eye
# -----------------------
pred_x, pred_y = model.predict(input_img)[0]
print(f"Predicted cyclone eye at pixel coordinates: x={pred_x:.2f}, y={pred_y:.2f}")

# -----------------------
# Display result
# -----------------------
plt.imshow(img_resized[...,0], cmap='gray')  # show first channel
plt.scatter(pred_x, pred_y, c='green', s=150, marker='x', label='Predicted Eye')
plt.title('Predicted Cyclone Eye')
plt.axis('off')
plt.legend()
plt.show()
