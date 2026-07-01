import numpy as np
import h5py
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

# -----------------------
# Paths
# -----------------------
model_path = 'cyclone_eye_detector.h5'
data_path = 'TCIR-ALL_2017.h5'

# -----------------------
# Load trained model
# -----------------------
model = load_model(model_path, compile=False)
print("✅ Model loaded.")

# -----------------------
# Load dataset
# -----------------------
with h5py.File(data_path, 'r') as hf:
    images = hf['matrix'][:]

images = images.astype('float32') / 255.0
if len(images.shape) == 3:
    images = np.expand_dims(images, axis=-1)

img_size = images.shape[1]
center_pixel = img_size // 2
max_shift = 20

# -----------------------
# Apply same random shifts as training
# -----------------------
shifted_images = []
true_eye_positions = []

for img in images:
    dx = np.random.randint(-max_shift, max_shift+1)
    dy = np.random.randint(-max_shift, max_shift+1)
    shifted_img = np.roll(img, shift=dx, axis=1)
    shifted_img = np.roll(shifted_img, shift=dy, axis=0)
    shifted_images.append(shifted_img)
    true_eye_positions.append([center_pixel + dx, center_pixel + dy])

shifted_images = np.array(shifted_images)
true_eye_positions = np.array(true_eye_positions)

# -----------------------
# Split to test set
# -----------------------
_, X_test, _, y_test = train_test_split(
    shifted_images, true_eye_positions, test_size=0.2, random_state=42
)

# -----------------------
# Predict on test set
# -----------------------
y_pred = model.predict(X_test)

# -----------------------
# Regression metrics
# -----------------------
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
max_error = np.max(np.linalg.norm(y_test - y_pred, axis=1))

print("\n✅ Regression Metrics:")
print(f"MSE: {mse:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"MAE: {mae:.2f}")
print(f"R² score: {r2:.3f}")
print(f"Max error (pixels): {max_error:.2f}")

# -----------------------
# Detection metrics (within tolerance)
# -----------------------
tolerances = [1, 5, 10, 20]  # pixels
distances = np.linalg.norm(y_test - y_pred, axis=1)

print("\n✅ Detection Metrics:")
for tol in tolerances:
    correct = distances <= tol
    accuracy = np.mean(correct)
    print(f"Accuracy within ±{tol} pixels: {accuracy*100:.2f}%")

    # Optional precision/recall/F1 using tolerance
    y_true_binary = correct.astype(int)
    y_pred_binary = np.ones_like(y_true_binary)
    precision = precision_score(y_true_binary, y_pred_binary)
    recall = recall_score(y_true_binary, y_pred_binary)
    f1 = f1_score(y_true_binary, y_pred_binary)
    print(f"  Precision: {precision:.2f}, Recall: {recall:.2f}, F1-score: {f1:.2f}")

# -----------------------
# Error distribution plot
# -----------------------
plt.figure(figsize=(8,5))
plt.hist(distances, bins=50, color='skyblue', edgecolor='black')
plt.title('Distribution of Euclidean prediction errors (pixels)')
plt.xlabel('Distance error (pixels)')
plt.ylabel('Number of predictions')
plt.show()
