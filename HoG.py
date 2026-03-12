import cv2
import numpy as np
import matplotlib.pyplot as plt

def preprocess_image(image_path, size=(64,128)):
    img_raw = cv2.imread(image_path)
    if img_raw is None:
        raise FileNotFoundError(f"Image at {image_path} not found.")
    img_gray = cv2.cvtColor(img_raw, cv2.COLOR_BGR2GRAY)
    # Note: cv2.resize uses (width, height)
    img = cv2.resize(img_gray, size)
    return img

def compute_gradients_scratch(img):
    Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)
    Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=np.float64)
    
    h, w = img.shape

    gx = np.zeros((h, w))
    gy = np.zeros((h, w))
    
    padded = np.pad(img, 1, mode='edge')

    for i in range(h):
        for j in range(w):
            window = padded[i:i+3, j:j+3]
            gx[i, j] = np.sum(window * Kx)
            gy[i, j] = np.sum(window * Ky)
            
    return gx, gy

def get_mag_and_angle(gx, gy):
    magnitude = np.sqrt(gx**2 + gy**2)
    angle = np.arctan2(gy, gx) * (180 / np.pi)
    angle[angle < 0] += 180
    return magnitude, angle

def build_histograms(magnitude, angle, cell_size=8, bins=9):
    h, w = magnitude.shape
    cells_y, cells_x = h // cell_size, w // cell_size
    bin_size = 180 // bins
    histograms = np.zeros((cells_y, cells_x, bins))

    for i in range(cells_y):
        for j in range(cells_x):
            m_cell = magnitude[i*cell_size:(i+1)*cell_size, j*cell_size:(j+1)*cell_size]
            a_cell = angle[i*cell_size:(i+1)*cell_size, j*cell_size:(j+1)*cell_size]

            for y in range(cell_size):
                for x in range(cell_size):
                    mag = m_cell[y, x]
                    ang = a_cell[y, x]

                    bin_idx = int(ang // bin_size) % bins
                    next_bin = (bin_idx + 1) % bins
                    ratio = (ang % bin_size) / bin_size

                    histograms[i, j, bin_idx] += mag * (1 - ratio)
                    histograms[i, j, next_bin] += mag * ratio
    return histograms

def normalize_and_flatten(histograms):
    cells_y, cells_x, bins = histograms.shape
    feature_vector = []

    for i in range(cells_y - 1):
        for j in range(cells_x - 1):
            block = histograms[i:i+2, j:j+2, :].flatten()
            K = np.linalg.norm(block) + 1e-6
            feature_vector.extend(block / K)
            
    return np.array(feature_vector)

def create_hog_visualization(img, histograms, cell_size=8, bins=9):
    cells_y, cells_x, _ = histograms.shape
    bin_size = 180 // bins
    hog_viz = np.zeros_like(img, dtype=np.float32)

    max_mag = histograms.max() + 1e-6

    for i in range(cells_y):
        for j in range(cells_x):
            cell_hist = histograms[i, j]
            for b in range(bins):
                angle_rad = np.deg2rad(b * bin_size + bin_size / 2)
                
                length = cell_size / 2

                if cell_hist[b] > 0.05 * max_mag:
                    cx = j * cell_size + cell_size // 2
                    cy = i * cell_size + cell_size // 2

                    dx = length * np.cos(angle_rad)
                    dy = length * np.sin(angle_rad)

                    cv2.line(hog_viz,
                             (int(cx - dx), int(cy - dy)),
                             (int(cx + dx), int(cy + dy)),
                             255, 1)
    return hog_viz

# --- Main Execution ---

# 1. Load and Preprocess
img = preprocess_image("haland.png", size=(64, 128))

# 2. Extract Gradients (From Scratch)
gx, gy = compute_gradients_scratch(img)
magnitude, angle = get_mag_and_angle(gx, gy)

# 3. Create Histograms
hists = build_histograms(magnitude, angle)

# 4. Normalize and get Feature Vector
features = normalize_and_flatten(hists)
print(f"Feature Vector Length: {len(features)}")
print(f"random Features: {features[1000:1100]}")

# 5. Visualization (Style from Code 2)
hog_image = create_hog_visualization(img, hists)

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Original (1:2 Aspect)")
plt.imshow(img, cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title("HoG Visualization")
plt.imshow(hog_image, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()