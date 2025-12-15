"""
Plate Number Recognition - SVM vs CNN Comparison
=================================================
Fitur:
- SVM + HOG (model lama)
- CNN dengan TensorFlow/Keras (model baru)
- Perbandingan visual hasil
- Debug mode untuk analisis
"""

import cv2
import numpy as np
import os
import time
from sklearn.svm import LinearSVC
from skimage.feature import hog
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from collections import defaultdict

# ============================================
# KONFIGURASI
# ============================================
DEBUG = True
DEBUG_SAVE_IMAGES = True
USE_CNN = True
USE_SVM = True

DATASET_PATH = r"D:\Semester 5\datasetPCVK\DatasetCharacter\DatasetCharacter"
CNN_MODEL_PATH = "cnn_model.keras"
LIMIT_PER_CLASS = 10000

# ============================================
# TensorFlow/Keras imports
# ============================================
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    from tensorflow.keras.utils import to_categorical
    TF_AVAILABLE = True
    # Suppress TF warnings
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    tf.get_logger().setLevel('ERROR')
except ImportError:
    TF_AVAILABLE = False
    print("[!] TensorFlow tidak terinstall. CNN tidak tersedia.")
    print("    Install dengan: pip install tensorflow")
    USE_CNN = False

# ============================================
# HELPER FUNCTIONS
# ============================================
def debug_print(*args, **kwargs):
    """Print hanya jika DEBUG=True"""
    if DEBUG:
        print(*args, **kwargs)

def delete_debug_files(directory):
    """Hapus semua file debug_* di direktori"""
    for file in os.listdir(directory):
        if file.startswith("debug_"):
            try:
                os.remove(os.path.join(directory, file))
                debug_print(f"Deleted: {file}")
            except Exception as e:
                debug_print(f"Failed to delete {file}: {e}")

def calculate_mse(img1, img2):
    """Hitung Mean Squared Error antara dua gambar"""
    # Pastikan ukuran sama
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    
    # Konversi ke float
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    
    mse = np.mean((img1 - img2) ** 2)
    return mse

def calculate_psnr(img1, img2):
    """Hitung Peak Signal-to-Noise Ratio antara dua gambar"""
    mse = calculate_mse(img1, img2)
    if mse == 0:
        return float('inf')  # Gambar identik
    
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

def analyze_preprocessing_quality(plate_img):
    """Analisis kualitas gambar sebelum dan sesudah preprocessing"""
    results = {}
    
    # Original (grayscale)
    if len(plate_img.shape) == 3:
        original_gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    else:
        original_gray = plate_img.copy()
    
    # Step 1: CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    after_clahe = clahe.apply(original_gray)
    
    # Step 2: Bilateral Filter
    after_bilateral = cv2.bilateralFilter(after_clahe, 11, 75, 75)
    
    # Step 3: Thresholding
    median_brightness = np.median(original_gray)
    is_dark_plate = median_brightness < 128
    
    if is_dark_plate:
        _, after_thresh = cv2.threshold(after_bilateral, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        _, after_thresh = cv2.threshold(after_bilateral, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Calculate metrics
    results['original'] = original_gray
    results['after_clahe'] = after_clahe
    results['after_bilateral'] = after_bilateral
    results['after_thresh'] = after_thresh
    
    # PSNR dan MSE untuk setiap tahap
    results['psnr_clahe'] = calculate_psnr(original_gray, after_clahe)
    results['mse_clahe'] = calculate_mse(original_gray, after_clahe)
    
    results['psnr_bilateral'] = calculate_psnr(after_clahe, after_bilateral)
    results['mse_bilateral'] = calculate_mse(after_clahe, after_bilateral)
    
    results['psnr_thresh'] = calculate_psnr(after_bilateral, after_thresh)
    results['mse_thresh'] = calculate_mse(after_bilateral, after_thresh)
    
    # Overall (original vs final before thresh)
    results['psnr_overall'] = calculate_psnr(original_gray, after_bilateral)
    results['mse_overall'] = calculate_mse(original_gray, after_bilateral)
    
    return results

# ============================================
# IMAGE PROCESSING FUNCTIONS
# ============================================
def split_characters(plate_img, debug_prefix="debug"):
    """Segmentasi karakter dari gambar plat"""
    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    
    median_brightness = np.median(gray)
    is_dark_plate = median_brightness < 128
    
    debug_print(f"Median brightness: {median_brightness:.1f}, Dark plate: {is_dark_plate}")
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)
    gray = cv2.bilateralFilter(gray, 11, 75, 75)
    
    if is_dark_plate:
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_open, iterations=1)
    
    if is_dark_plate:
        kernel_erode = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1))
        thresh = cv2.erode(thresh, kernel_erode, iterations=2)
    else:
        kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_close, iterations=1)
    
    # Hapus border
    h, w = thresh.shape
    border_h = int(h * 0.05)
    border_w = int(w * 0.03)
    thresh[:border_h, :] = 0
    thresh[-border_h:, :] = 0
    thresh[:, :border_w] = 0
    thresh[:, -border_w:] = 0
    
    if DEBUG_SAVE_IMAGES:
        cv2.imwrite(f"{debug_prefix}_plate.jpg", plate_img)
        cv2.imwrite(f"{debug_prefix}_thresh.jpg", thresh)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    chars = []
    img_h, img_w = plate_img.shape[:2]
    img_area = img_h * img_w
    
    # Filter contour besar
    valid_contours = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        area = w * h
        if area < img_area * 0.3 and w < img_w * 0.5:
            valid_contours.append(c)
    
    # Hitung height rata-rata
    heights = []
    y_positions = []
    for c in valid_contours:
        _, y, _, h = cv2.boundingRect(c)
        if h > img_h * 0.25:
            heights.append(h)
            y_positions.append(y)
    
    if heights:
        avg_height = np.mean(heights)
        avg_y = np.mean(y_positions)
        min_height = avg_height * 0.6
        max_height = avg_height * 1.4
        y_tolerance = avg_height * 0.8
        min_y = avg_y - y_tolerance
        max_y = avg_y + y_tolerance
    else:
        min_height = img_h * 0.3
        max_height = img_h * 0.95
        min_y = 0
        max_y = img_h
    
    min_area = img_area * 0.01
    
    for c in valid_contours:
        x, y, w, h = cv2.boundingRect(c)
        aspect_ratio = w / float(h)
        area = w * h
        
        if min_height < h < max_height and 0.15 < aspect_ratio < 0.95 and area > min_area and min_y < y < max_y:
            char_img = thresh[y:y+h, x:x+w]
            
            # Trim whitespace
            row_sums = np.sum(char_img, axis=1)
            non_zero_rows = np.where(row_sums > 0)[0]
            if len(non_zero_rows) > 0:
                y_min = non_zero_rows[0]
                y_max = non_zero_rows[-1] + 1
                char_img = char_img[y_min:y_max, :]
            
            # Padding untuk karakter tipis
            char_h, char_w = char_img.shape
            if char_w < char_h * 0.4:
                target_w = int(char_h * 0.5)
                pad_left = (target_w - char_w) // 2
                pad_right = target_w - char_w - pad_left
                char_img = cv2.copyMakeBorder(char_img, 0, 0, pad_left, pad_right, 
                                             cv2.BORDER_CONSTANT, value=0)
            
            char_img = cv2.resize(char_img, (32, 32))
            chars.append((x, char_img, aspect_ratio))
    
    chars = sorted(chars, key=lambda x: x[0])
    return [(c[1], c[2]) for c in chars]

# ============================================
# SVM MODEL
# ============================================
class SVMModel:
    def __init__(self):
        self.model = LinearSVC(max_iter=10000)
        self.trained = False
    
    def load_data(self, dataset_path):
        """Load dan extract HOG features"""
        X, y = [], []
        
        if not os.path.exists(dataset_path):
            print(f"[ERROR] Dataset tidak ditemukan: {dataset_path}")
            return np.array([]), np.array([])
        
        labels = sorted(os.listdir(dataset_path))
        for label in labels:
            folder = os.path.join(dataset_path, label)
            if not os.path.isdir(folder):
                continue
            
            debug_print(f"Loading label: {label}", end=" ")
            count = 0
            for img_name in sorted(os.listdir(folder)):
                if count >= LIMIT_PER_CLASS:
                    break
                
                img_path = os.path.join(folder, img_name)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                
                try:
                    img = cv2.resize(img, (32, 32))
                    fd = hog(img, orientations=9, pixels_per_cell=(8,8),
                             cells_per_block=(2,2), block_norm='L2-Hys')
                    X.append(fd)
                    y.append(label)
                    count += 1
                except:
                    continue
            
            debug_print(f"({count})")
        
        print(f"[OK] SVM Dataset loaded: {len(X)} samples")
        return np.array(X), np.array(y)
    
    def train(self, X, y):
        """Train SVM model"""
        print("[...] Training SVM...")
        start_time = time.time()
        self.model.fit(X, y)
        self.trained = True
        elapsed = time.time() - start_time
        print(f"[OK] SVM trained in {elapsed:.2f} seconds")
        return elapsed
    
    def predict(self, char_img):
        """Predict single character"""
        if not self.trained:
            return "?"
        
        fd = hog(char_img, orientations=9, pixels_per_cell=(8,8),
                 cells_per_block=(2,2), block_norm='L2-Hys')
        pred = self.model.predict([fd])[0]
        
        # Get confidence scores
        scores = self.model.decision_function([fd])[0]
        top_indices = np.argsort(scores)[-3:][::-1]
        top_preds = [self.model.classes_[idx] for idx in top_indices]
        
        return pred, top_preds

# ============================================
# CNN MODEL
# ============================================
class CNNModel:
    def __init__(self):
        self.model = None
        self.label_map = {}
        self.reverse_map = {}
        self.trained = False
    
    def build_model(self, num_classes):
        """Build CNN architecture"""
        model = keras.Sequential([
            # Input
            layers.Input(shape=(32, 32, 1)),
            
            # Conv Block 1
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Conv Block 2
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Conv Block 3
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Dense layers
            layers.Flatten(),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def load_data(self, dataset_path):
        """Load data untuk CNN"""
        X, y = [], []
        labels = set()
        
        if not os.path.exists(dataset_path):
            print(f"[ERROR] Dataset tidak ditemukan: {dataset_path}")
            return None, None, None
        
        folder_labels = sorted(os.listdir(dataset_path))
        for label in folder_labels:
            folder = os.path.join(dataset_path, label)
            if not os.path.isdir(folder):
                continue
            
            labels.add(label)
            debug_print(f"Loading label: {label}", end=" ")
            count = 0
            for img_name in sorted(os.listdir(folder)):
                if count >= LIMIT_PER_CLASS:
                    break
                
                img_path = os.path.join(folder, img_name)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                
                try:
                    img = cv2.resize(img, (32, 32))
                    X.append(img)
                    y.append(label)
                    count += 1
                except:
                    continue
            
            debug_print(f"({count})")
        
        # Create label mapping
        sorted_labels = sorted(labels)
        self.label_map = {label: i for i, label in enumerate(sorted_labels)}
        self.reverse_map = {i: label for label, i in self.label_map.items()}
        
        # Convert to numpy
        X = np.array(X).reshape(-1, 32, 32, 1) / 255.0
        y_encoded = np.array([self.label_map[label] for label in y])
        y_onehot = to_categorical(y_encoded, num_classes=len(sorted_labels))
        
        print(f"[OK] CNN Dataset loaded: {len(X)} samples, {len(sorted_labels)} classes")
        return X, y_onehot, len(sorted_labels)
    
    def train(self, X, y, epochs=15):
        """Train CNN model"""
        if self.model is None:
            num_classes = y.shape[1]
            self.model = self.build_model(num_classes)
        
        print(f"[...] Training CNN ({epochs} epochs)...")
        
        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=2)
        ]
        
        start_time = time.time()
        history = self.model.fit(
            X, y,
            epochs=epochs,
            batch_size=64,
            validation_split=0.1,
            callbacks=callbacks,
            verbose=1 if DEBUG else 0
        )
        elapsed = time.time() - start_time
        
        self.trained = True
        final_acc = history.history['val_accuracy'][-1] * 100
        print(f"[OK] CNN trained in {elapsed:.2f} seconds (Val Accuracy: {final_acc:.1f}%)")
        
        # Save model
        self.model.save(CNN_MODEL_PATH)
        print(f"[SAVED] Model saved to {CNN_MODEL_PATH}")
        
        return elapsed, history
    
    def load_model(self):
        """Load pre-trained model"""
        if os.path.exists(CNN_MODEL_PATH):
            self.model = keras.models.load_model(CNN_MODEL_PATH)
            self.trained = True
            print(f"[OK] CNN model loaded from {CNN_MODEL_PATH}")
            return True
        return False
    
    def predict(self, char_img):
        """Predict single character"""
        if not self.trained or self.model is None:
            return "?", []
        
        # Preprocess
        img = char_img.reshape(1, 32, 32, 1) / 255.0
        
        # Predict
        probs = self.model.predict(img, verbose=0)[0]
        top_indices = np.argsort(probs)[-3:][::-1]
        
        pred = self.reverse_map[top_indices[0]]
        top_preds = [self.reverse_map[i] for i in top_indices]
        confidence = probs[top_indices[0]] * 100
        
        return pred, top_preds, confidence

# ============================================
# PLATE RECOGNITION
# ============================================
def correct_plate_context(plate_str):
    """Post-processing koreksi konteks plat Indonesia"""
    if len(plate_str) < 4:
        return plate_str
    
    corrected = list(plate_str)
    digit_to_letter = {'0': 'Q', '8': 'B', '6': 'G', '5': 'S'}
    letter_to_digit = {'O': '0', 'I': '1', 'Z': '2', 'S': '5'}
    
    # Karakter pertama harus huruf
    if corrected[0].isdigit() and corrected[0] in digit_to_letter:
        corrected[0] = digit_to_letter[corrected[0]]
    
    # Cari indeks angka pertama
    first_digit_idx = -1
    for i in range(1, len(corrected)):
        if corrected[i].isdigit():
            first_digit_idx = i
            break
    
    if first_digit_idx == -1:
        return ''.join(corrected)
    
    # Cari huruf akhir
    last_trailing_letter_idx = -1
    for i in range(len(corrected) - 1, first_digit_idx, -1):
        if corrected[i].isalpha() and corrected[i] not in letter_to_digit:
            last_trailing_letter_idx = i
    
    if last_trailing_letter_idx > 0:
        end_of_numbers = first_digit_idx
        for i in range(first_digit_idx, last_trailing_letter_idx):
            if corrected[i].isdigit() and corrected[i] not in digit_to_letter:
                end_of_numbers = i
            elif corrected[i].isdigit() and i + 1 < len(corrected) and corrected[i + 1].isdigit():
                end_of_numbers = i
        
        for i in range(end_of_numbers + 1, len(corrected)):
            if corrected[i].isdigit() and corrected[i] in digit_to_letter:
                corrected[i] = digit_to_letter[corrected[i]]
    
    return ''.join(corrected)

def recognize_plate_with_timing(plate_path, svm_model=None, cnn_model=None, debug_prefix="debug"):
    """Recognize plate menggunakan SVM dan/atau CNN dengan timing terpisah"""
    results = {'predictions': {}, 'svm_time': 0, 'cnn_time': 0}
    
    # Load image
    plate = cv2.imread(plate_path)
    if plate is None:
        return {'predictions': {"error": "Gambar tidak ditemukan"}, 'svm_time': 0, 'cnn_time': 0}
    
    filename = os.path.basename(plate_path).split('.')[0]
    debug_print(f"\n{'='*60}")
    debug_print(f"Processing: {filename}")
    debug_print(f"{'='*60}")
    
    # Segmentasi karakter
    chars = split_characters(plate, debug_prefix=f"debug_{filename}")
    if not chars:
        return {'predictions': {"error": "Tidak ada karakter terdeteksi"}, 'svm_time': 0, 'cnn_time': 0}
    
    debug_print(f"Found {len(chars)} characters")
    
    # SVM Recognition with timing
    if USE_SVM and svm_model and svm_model.trained:
        svm_start = time.time()
        hasil_svm = ""
        for i, (ch, ar) in enumerate(chars):
            if DEBUG_SAVE_IMAGES:
                cv2.imwrite(f"debug_{filename}_char_{i}.jpg", ch)
            
            pred, top_preds = svm_model.predict(ch)
            
            # Koreksi aspect ratio
            if pred == '0' and ar < 0.35:
                pred = '1'
            elif pred == 'U' and ar < 0.35:
                pred = '1'
            
            hasil_svm += pred
            debug_print(f"  SVM Char {i}: {pred} (top: {top_preds[:2]})")
        
        hasil_svm = correct_plate_context(hasil_svm)
        results['predictions']['svm'] = hasil_svm
        results['svm_time'] = time.time() - svm_start
    
    # CNN Recognition with timing
    if USE_CNN and cnn_model and cnn_model.trained:
        cnn_start = time.time()
        hasil_cnn = ""
        for i, (ch, ar) in enumerate(chars):
            pred, top_preds, conf = cnn_model.predict(ch)
            
            # Koreksi aspect ratio
            if pred == '0' and ar < 0.35:
                pred = '1'
            elif pred == 'U' and ar < 0.35:
                pred = '1'
            
            hasil_cnn += pred
            debug_print(f"  CNN Char {i}: {pred} ({conf:.1f}%) (top: {top_preds[:2]})")
        
        hasil_cnn = correct_plate_context(hasil_cnn)
        results['predictions']['cnn'] = hasil_cnn
        results['cnn_time'] = time.time() - cnn_start
    
    return results

def recognize_plate(plate_path, svm_model=None, cnn_model=None, debug_prefix="debug"):
    """Wrapper untuk compatibility - returns only predictions"""
    result = recognize_plate_with_timing(plate_path, svm_model, cnn_model, debug_prefix)
    return result['predictions']

# ============================================
# VISUALIZATION
# ============================================
def visualize_comparison(image_results, output_path="comparison_results.png"):
    """Buat visualisasi perbandingan SVM vs CNN"""
    
    n_images = len(image_results)
    fig, axes = plt.subplots(n_images, 1, figsize=(12, 2 * n_images))
    
    if n_images == 1:
        axes = [axes]
    
    for i, (filename, expected, results) in enumerate(image_results):
        ax = axes[i]
        
        svm_result = results.get('svm', 'N/A')
        cnn_result = results.get('cnn', 'N/A')
        
        # Check accuracy
        svm_correct = svm_result.replace(' ', '') == expected.replace(' ', '')
        cnn_correct = cnn_result.replace(' ', '') == expected.replace(' ', '')
        
        # Colors
        svm_color = 'green' if svm_correct else 'red'
        cnn_color = 'green' if cnn_correct else 'red'
        
        # Text
        text = f"{filename}\n"
        text += f"Expected: {expected}\n"
        text += f"SVM: {svm_result} {'[OK]' if svm_correct else '[X]'}\n"
        text += f"CNN: {cnn_result} {'[OK]' if cnn_correct else '[X]'}"
        
        ax.text(0.5, 0.5, text, ha='center', va='center', fontsize=11,
                family='monospace', transform=ax.transAxes)
        
        # Background color based on accuracy
        if svm_correct and cnn_correct:
            ax.set_facecolor('#d4edda')  # Green
        elif svm_correct or cnn_correct:
            ax.set_facecolor('#fff3cd')  # Yellow
        else:
            ax.set_facecolor('#f8d7da')  # Red
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n[CHART] Comparison saved to: {output_path}")

def print_summary(image_results):
    """Print summary hasil"""
    print("\n" + "="*70)
    print("SUMMARY HASIL")
    print("="*70)
    print(f"{'File':<20} {'Expected':<12} {'SVM':<12} {'CNN':<12} {'Winner'}")
    print("-"*70)
    
    svm_correct = 0
    cnn_correct = 0
    
    for filename, expected, results in image_results:
        svm_result = results.get('svm', 'N/A').replace(' ', '')
        cnn_result = results.get('cnn', 'N/A').replace(' ', '')
        expected_clean = expected.replace(' ', '')
        
        svm_ok = svm_result == expected_clean
        cnn_ok = cnn_result == expected_clean
        
        if svm_ok:
            svm_correct += 1
        if cnn_ok:
            cnn_correct += 1
        
        # Determine winner
        if svm_ok and cnn_ok:
            winner = "BOTH OK"
        elif cnn_ok:
            winner = "CNN OK"
        elif svm_ok:
            winner = "SVM OK"
        else:
            winner = "NONE"
        
        svm_display = results.get('svm', 'N/A')[:10]
        cnn_display = results.get('cnn', 'N/A')[:10]
        
        print(f"{filename:<20} {expected:<12} {svm_display:<12} {cnn_display:<12} {winner}")
    
    print("-"*70)
    total = len(image_results)
    print(f"SVM Accuracy: {svm_correct}/{total} ({svm_correct/total*100:.1f}%)")
    print(f"CNN Accuracy: {cnn_correct}/{total} ({cnn_correct/total*100:.1f}%)")
    print("="*70)

def visualize_training_history(history, output_path="training_history.png"):
    """Visualisasi training history CNN (loss & accuracy)"""
    if history is None:
        print("[SKIP] No training history available")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot Accuracy
    ax1 = axes[0]
    ax1.plot(history.history['accuracy'], 'b-', label='Training Accuracy', linewidth=2)
    ax1.plot(history.history['val_accuracy'], 'r--', label='Validation Accuracy', linewidth=2)
    ax1.set_title('Model Accuracy During Training', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.legend(loc='lower right', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 1.05])
    
    # Plot Loss
    ax2 = axes[1]
    ax2.plot(history.history['loss'], 'b-', label='Training Loss', linewidth=2)
    ax2.plot(history.history['val_loss'], 'r--', label='Validation Loss', linewidth=2)
    ax2.set_title('Model Loss During Training', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Loss', fontsize=12)
    ax2.legend(loc='upper right', fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"[CHART] Training history saved to: {output_path}")

def visualize_accuracy_comparison(svm_acc, cnn_acc, output_path="accuracy_comparison.png"):
    """Bar chart perbandingan akurasi SVM vs CNN"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    models = ['SVM + HOG', 'CNN']
    accuracies = [svm_acc, cnn_acc]
    colors = ['#3498db', '#e74c3c']
    
    bars = ax.bar(models, accuracies, color=colors, width=0.5, edgecolor='black', linewidth=1.5)
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax.annotate(f'{acc:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 5),
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=16, fontweight='bold')
    
    ax.set_ylabel('Accuracy (%)', fontsize=14)
    ax.set_title('Perbandingan Akurasi: SVM vs CNN', fontsize=16, fontweight='bold')
    ax.set_ylim([0, 110])
    ax.grid(axis='y', alpha=0.3)
    
    # Add color legend
    ax.axhline(y=100, color='green', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"[CHART] Accuracy comparison saved to: {output_path}")

def visualize_time_comparison(svm_train_time, cnn_train_time, svm_inf_times, cnn_inf_times, 
                               output_path="time_comparison.png", cnn_pretrained=False):
    """Visualisasi perbandingan waktu training dan inference"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Training Time
    ax1 = axes[0]
    models = ['SVM + HOG', 'CNN']
    train_times = [svm_train_time, cnn_train_time]
    colors = ['#3498db', '#e74c3c']
    
    bars1 = ax1.bar(models, train_times, color=colors, width=0.5, edgecolor='black', linewidth=1.5)
    for i, (bar, t) in enumerate(zip(bars1, train_times)):
        height = bar.get_height()
        # Special label for pre-trained CNN
        if i == 1 and cnn_pretrained:
            label = 'Pre-trained\n(loaded from file)'
        else:
            label = f'{t:.2f}s'
        ax1.annotate(label,
                    xy=(bar.get_x() + bar.get_width() / 2, max(height, 0.5)),
                    xytext=(0, 5),
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=12, fontweight='bold')
    ax1.set_ylabel('Time (seconds)', fontsize=12)
    ax1.set_title('Training Time Comparison', fontsize=14, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    if cnn_pretrained:
        ax1.text(0.02, 0.98, '* CNN model loaded from pre-trained file', 
                 transform=ax1.transAxes, fontsize=9, verticalalignment='top',
                 style='italic', color='gray')
    
    # Inference Time per Image
    ax2 = axes[1]
    if svm_inf_times and cnn_inf_times:
        avg_svm_inf = np.mean(svm_inf_times) * 1000  # Convert to ms
        avg_cnn_inf = np.mean(cnn_inf_times) * 1000
        inf_times = [avg_svm_inf, avg_cnn_inf]
        
        bars2 = ax2.bar(models, inf_times, color=colors, width=0.5, edgecolor='black', linewidth=1.5)
        for bar, t in zip(bars2, inf_times):
            height = bar.get_height()
            ax2.annotate(f'{t:.1f}ms',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 5),
                        textcoords="offset points",
                        ha='center', va='bottom',
                        fontsize=14, fontweight='bold')
        ax2.set_ylabel('Time (milliseconds)', fontsize=12)
        ax2.set_title('Average Inference Time per Image', fontsize=14, fontweight='bold')
        ax2.grid(axis='y', alpha=0.3)
    else:
        ax2.text(0.5, 0.5, 'No inference time data', ha='center', va='center', fontsize=14)
        ax2.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"[CHART] Time comparison saved to: {output_path}")

def visualize_segmentation_pipeline(plate_path, output_path="segmentation_pipeline.png"):
    """Visualisasi step-by-step proses segmentasi karakter"""
    plate_img = cv2.imread(plate_path)
    if plate_img is None:
        print(f"[ERROR] Cannot load image: {plate_path}")
        return
    
    # Convert ke RGB untuk matplotlib
    plate_rgb = cv2.cvtColor(plate_img, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    
    # Apply preprocessing steps
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray_clahe = clahe.apply(gray)
    gray_bilateral = cv2.bilateralFilter(gray_clahe, 11, 75, 75)
    
    median_brightness = np.median(gray)
    is_dark_plate = median_brightness < 128
    
    if is_dark_plate:
        _, thresh = cv2.threshold(gray_bilateral, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        _, thresh = cv2.threshold(gray_bilateral, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Morphological operations
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    thresh_morph = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_open, iterations=1)
    
    # Find contours and draw
    contours, _ = cv2.findContours(thresh_morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_img = plate_rgb.copy()
    cv2.drawContours(contour_img, contours, -1, (0, 255, 0), 2)
    
    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    
    steps = [
        (plate_rgb, "1. Original Image", None),
        (gray, "2. Grayscale", 'gray'),
        (gray_clahe, "3. CLAHE Enhancement", 'gray'),
        (gray_bilateral, "4. Bilateral Filter", 'gray'),
        (thresh_morph, "5. Thresholding + Morphology", 'gray'),
        (contour_img, "6. Contour Detection", None),
    ]
    
    for idx, (img, title, cmap) in enumerate(steps):
        ax = axes[idx // 3, idx % 3]
        ax.imshow(img, cmap=cmap)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.axis('off')
    
    plt.suptitle('Character Segmentation Pipeline', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"[CHART] Segmentation pipeline saved to: {output_path}")

def visualize_psnr_mse(all_psnr_data, all_mse_data, output_path="viz_psnr_mse.png"):
    """Visualisasi PSNR dan MSE untuk evaluasi preprocessing"""
    if not all_psnr_data or not all_mse_data:
        print("[SKIP] No PSNR/MSE data available")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Calculate averages
    stages = ['CLAHE', 'Bilateral Filter', 'Overall']
    
    avg_psnr = [
        np.mean([d['psnr_clahe'] for d in all_psnr_data if d['psnr_clahe'] != float('inf')]),
        np.mean([d['psnr_bilateral'] for d in all_psnr_data if d['psnr_bilateral'] != float('inf')]),
        np.mean([d['psnr_overall'] for d in all_psnr_data if d['psnr_overall'] != float('inf')]),
    ]
    
    avg_mse = [
        np.mean([d['mse_clahe'] for d in all_mse_data]),
        np.mean([d['mse_bilateral'] for d in all_mse_data]),
        np.mean([d['mse_overall'] for d in all_mse_data]),
    ]
    
    colors = ['#3498db', '#2ecc71', '#9b59b6']
    
    # PSNR Chart
    ax1 = axes[0]
    bars1 = ax1.bar(stages, avg_psnr, color=colors, width=0.6, edgecolor='black', linewidth=1.5)
    for bar, val in zip(bars1, avg_psnr):
        height = bar.get_height()
        ax1.annotate(f'{val:.2f} dB',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 5),
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=12, fontweight='bold')
    ax1.set_ylabel('PSNR (dB)', fontsize=12)
    ax1.set_title('Peak Signal-to-Noise Ratio (PSNR)\nSebelum vs Sesudah Preprocessing', fontsize=14, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_ylim([0, max(avg_psnr) * 1.2])
    
    # MSE Chart
    ax2 = axes[1]
    bars2 = ax2.bar(stages, avg_mse, color=colors, width=0.6, edgecolor='black', linewidth=1.5)
    for bar, val in zip(bars2, avg_mse):
        height = bar.get_height()
        ax2.annotate(f'{val:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 5),
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=12, fontweight='bold')
    ax2.set_ylabel('MSE', fontsize=12)
    ax2.set_title('Mean Squared Error (MSE)\nSebelum vs Sesudah Preprocessing', fontsize=14, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"[CHART] PSNR/MSE comparison saved to: {output_path}")

def visualize_character_accuracy_heatmap(char_results_svm, char_results_cnn, 
                                          output_path="character_accuracy_heatmap.png"):
    """Heatmap akurasi per karakter untuk SVM dan CNN"""
    if not char_results_svm and not char_results_cnn:
        print("[SKIP] No character results for heatmap")
        return
    
    # Collect all unique characters
    all_chars = sorted(set(list(char_results_svm.keys()) + list(char_results_cnn.keys())))
    
    if not all_chars:
        print("[SKIP] No characters to visualize")
        return
    
    # Calculate accuracy per character
    svm_accs = []
    cnn_accs = []
    
    for char in all_chars:
        # SVM accuracy for this char
        if char in char_results_svm and char_results_svm[char]['total'] > 0:
            svm_acc = char_results_svm[char]['correct'] / char_results_svm[char]['total'] * 100
        else:
            svm_acc = 0
        svm_accs.append(svm_acc)
        
        # CNN accuracy for this char
        if char in char_results_cnn and char_results_cnn[char]['total'] > 0:
            cnn_acc = char_results_cnn[char]['correct'] / char_results_cnn[char]['total'] * 100
        else:
            cnn_acc = 0
        cnn_accs.append(cnn_acc)
    
    # Create heatmap data
    data = np.array([svm_accs, cnn_accs])
    
    fig, ax = plt.subplots(figsize=(max(12, len(all_chars) * 0.5), 4))
    
    sns.heatmap(data, annot=True, fmt='.0f', cmap='RdYlGn', 
                xticklabels=all_chars, yticklabels=['SVM', 'CNN'],
                cbar_kws={'label': 'Accuracy (%)'}, ax=ax,
                vmin=0, vmax=100, linewidths=0.5)
    
    ax.set_title('Per-Character Recognition Accuracy', fontsize=14, fontweight='bold')
    ax.set_xlabel('Character', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"[CHART] Character accuracy heatmap saved to: {output_path}")

def visualize_sample_predictions(test_images, svm_model, cnn_model, 
                                  output_path="sample_predictions.png", max_samples=6):
    """Grid visualisasi sample prediksi dengan gambar plate"""
    n_samples = min(len(test_images), max_samples)
    if n_samples == 0:
        print("[SKIP] No samples to visualize")
        return
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, img_path in enumerate(test_images[:n_samples]):
        ax = axes[idx]
        
        # Load and display image
        plate = cv2.imread(img_path)
        if plate is None:
            continue
        plate_rgb = cv2.cvtColor(plate, cv2.COLOR_BGR2RGB)
        
        filename = os.path.basename(img_path)
        # Extract plate number from filename (format: COLOR_PLATNUMBER.jpg)
        name_without_ext = filename.split('.')[0]
        if '_' in name_without_ext:
            expected = name_without_ext.split('_', 1)[1]  # Get part after first underscore
        else:
            expected = name_without_ext  # Fallback if no underscore
        
        # Get predictions
        results = recognize_plate(img_path, svm_model, cnn_model, debug_prefix=f"debug_sample_{idx}")
        svm_result = results.get('svm', 'N/A')
        cnn_result = results.get('cnn', 'N/A')
        
        # Check correctness
        svm_ok = svm_result.replace(' ', '') == expected.replace(' ', '')
        cnn_ok = cnn_result.replace(' ', '') == expected.replace(' ', '')
        
        ax.imshow(plate_rgb)
        
        # Create label text
        title = f"Expected: {expected}\n"
        title += f"SVM: {svm_result} "
        title += "✓" if svm_ok else "✗"
        title += f"\nCNN: {cnn_result} "
        title += "✓" if cnn_ok else "✗"
        
        # Color background based on results
        if svm_ok and cnn_ok:
            ax.set_facecolor('#d4edda')
        elif svm_ok or cnn_ok:
            ax.set_facecolor('#fff3cd')
        else:
            ax.set_facecolor('#f8d7da')
        
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.axis('off')
    
    # Hide unused axes
    for idx in range(n_samples, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle('Sample Predictions: SVM vs CNN', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"[CHART] Sample predictions saved to: {output_path}")

def visualize_confusion_matrix_summary(all_predictions_svm, all_predictions_cnn, 
                                        all_ground_truths, output_path="confusion_summary.png"):
    """Visualisasi confusion matrix summary untuk prediksi plat keseluruhan"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # SVM Results
    ax1 = axes[0]
    svm_correct = sum(1 for pred, gt in zip(all_predictions_svm, all_ground_truths) 
                      if pred.replace(' ', '') == gt.replace(' ', ''))
    svm_incorrect = len(all_predictions_svm) - svm_correct
    
    wedges1, texts1, autotexts1 = ax1.pie([svm_correct, svm_incorrect], 
                                           labels=['Correct', 'Incorrect'],
                                           colors=['#2ecc71', '#e74c3c'],
                                           autopct='%1.1f%%',
                                           startangle=90,
                                           explode=(0.05, 0),
                                           textprops={'fontsize': 12})
    ax1.set_title(f'SVM Results\n({svm_correct}/{len(all_predictions_svm)} correct)', 
                  fontsize=14, fontweight='bold')
    
    # CNN Results
    ax2 = axes[1]
    cnn_correct = sum(1 for pred, gt in zip(all_predictions_cnn, all_ground_truths) 
                      if pred.replace(' ', '') == gt.replace(' ', ''))
    cnn_incorrect = len(all_predictions_cnn) - cnn_correct
    
    wedges2, texts2, autotexts2 = ax2.pie([cnn_correct, cnn_incorrect], 
                                           labels=['Correct', 'Incorrect'],
                                           colors=['#2ecc71', '#e74c3c'],
                                           autopct='%1.1f%%',
                                           startangle=90,
                                           explode=(0.05, 0),
                                           textprops={'fontsize': 12})
    ax2.set_title(f'CNN Results\n({cnn_correct}/{len(all_predictions_cnn)} correct)', 
                  fontsize=14, fontweight='bold')
    
    plt.suptitle('Recognition Results Overview', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"[CHART] Confusion summary saved to: {output_path}")

def generate_all_visualizations(image_results, svm_model, cnn_model, test_images,
                                 svm_train_time, cnn_train_time, cnn_history,
                                 svm_inf_times, cnn_inf_times,
                                 char_results_svm, char_results_cnn, cnn_pretrained=False,
                                 psnr_mse_data=None):
    """Generate semua visualisasi untuk laporan"""
    print("\n" + "="*60)
    print("GENERATING VISUALIZATIONS FOR REPORT")
    print("="*60)
    
    # 1. Training History (CNN only)
    if cnn_history is not None:
        visualize_training_history(cnn_history, "viz_training_history.png")
    
    # 2. Accuracy Comparison
    svm_correct = sum(1 for _, exp, res in image_results 
                      if res.get('svm', '').replace(' ', '') == exp.replace(' ', ''))
    cnn_correct = sum(1 for _, exp, res in image_results 
                      if res.get('cnn', '').replace(' ', '') == exp.replace(' ', ''))
    total = len(image_results)
    
    if total > 0:
        svm_acc = svm_correct / total * 100
        cnn_acc = cnn_correct / total * 100
        visualize_accuracy_comparison(svm_acc, cnn_acc, "viz_accuracy_comparison.png")
        
        # 3. Time Comparison
        visualize_time_comparison(svm_train_time, cnn_train_time, 
                                   svm_inf_times, cnn_inf_times, "viz_time_comparison.png",
                                   cnn_pretrained=cnn_pretrained)
        
        # 4. Confusion Summary (Pie Charts)
        all_predictions_svm = [res.get('svm', '') for _, _, res in image_results]
        all_predictions_cnn = [res.get('cnn', '') for _, _, res in image_results]
        all_ground_truths = [exp for _, exp, _ in image_results]
        visualize_confusion_matrix_summary(all_predictions_svm, all_predictions_cnn,
                                            all_ground_truths, "viz_confusion_summary.png")
    
    # 5. Segmentation Pipeline (use first test image)
    if test_images:
        visualize_segmentation_pipeline(test_images[0], "viz_segmentation_pipeline.png")
    
    # 6. Character Accuracy Heatmap
    visualize_character_accuracy_heatmap(char_results_svm, char_results_cnn,
                                          "viz_character_heatmap.png")
    
    # 7. Sample Predictions Grid
    if test_images:
        visualize_sample_predictions(test_images, svm_model, cnn_model,
                                      "viz_sample_predictions.png", max_samples=6)
    
    # 8. PSNR/MSE Visualization
    if psnr_mse_data:
        visualize_psnr_mse(psnr_mse_data, psnr_mse_data, "viz_psnr_mse.png")
    
    print("\n[DONE] All visualizations generated!")
    print("Files created:")
    print("  - viz_training_history.png (CNN training progress)")
    print("  - viz_accuracy_comparison.png (SVM vs CNN accuracy)")
    print("  - viz_time_comparison.png (Training & inference time)")
    print("  - viz_confusion_summary.png (Pie chart results)")
    print("  - viz_segmentation_pipeline.png (Preprocessing steps)")
    print("  - viz_character_heatmap.png (Per-character accuracy)")
    print("  - viz_sample_predictions.png (Sample results with images)")
    print("  - viz_psnr_mse.png (PSNR/MSE preprocessing quality)")

# ============================================
# MAIN
# ============================================
if __name__ == "__main__":
    print("="*60)
    print("PLATE NUMBER RECOGNITION - SVM vs CNN")
    print("="*60)
    print(f"DEBUG: {DEBUG}")
    print(f"USE_SVM: {USE_SVM}")
    print(f"USE_CNN: {USE_CNN}")
    print(f"TensorFlow Available: {TF_AVAILABLE}")
    print("="*60)
    
    current_dir = os.getcwd()
    
    # Hapus debug files
    delete_debug_files(current_dir)
    
    # Initialize models
    svm_model = SVMModel() if USE_SVM else None
    cnn_model = CNNModel() if USE_CNN and TF_AVAILABLE else None
    
    # Storage for visualization data
    cnn_history = None
    svm_inf_times = []
    cnn_inf_times = []
    char_results_svm = defaultdict(lambda: {'correct': 0, 'total': 0})
    char_results_cnn = defaultdict(lambda: {'correct': 0, 'total': 0})
    
    # Train SVM
    svm_time = 0
    if USE_SVM:
        X_svm, y_svm = svm_model.load_data(DATASET_PATH)
        if len(X_svm) > 0:
            svm_time = svm_model.train(X_svm, y_svm)
    
    # Train CNN
    cnn_time = 0
    if USE_CNN and TF_AVAILABLE:
        # Try to load existing model first
        if not cnn_model.load_model():
            X_cnn, y_cnn, num_classes = cnn_model.load_data(DATASET_PATH)
            if X_cnn is not None:
                cnn_time, cnn_history = cnn_model.train(X_cnn, y_cnn, epochs=15)
        
        # Load label mapping if model was loaded
        if cnn_model.trained and not cnn_model.label_map:
            # Rebuild label map from dataset
            labels = sorted(os.listdir(DATASET_PATH))
            labels = [l for l in labels if os.path.isdir(os.path.join(DATASET_PATH, l))]
            cnn_model.label_map = {label: i for i, label in enumerate(labels)}
            cnn_model.reverse_map = {i: label for label, i in cnn_model.label_map.items()}
    
    # Find test images
    test_images = []
    for file in os.listdir(current_dir):
        if file.lower().endswith(('.jpg', '.jpeg')) and not file.startswith('debug_'):
            test_images.append(os.path.join(current_dir, file))
    
    if not test_images:
        print(f"\n[ERROR] Tidak ada gambar test di: {current_dir}")
        exit()
    
    print(f"\n[INFO] Found {len(test_images)} test images")
    
    # Track if CNN was loaded from pretrained file
    cnn_pretrained = USE_CNN and TF_AVAILABLE and cnn_model.trained and cnn_time == 0
    
    # Storage for PSNR/MSE data
    psnr_mse_data = []
    
    # Process images with timing
    image_results = []
    for img_path in sorted(test_images):
        filename = os.path.basename(img_path)
        # Extract plate number from filename (format: COLOR_PLATNUMBER.jpg)
        name_without_ext = filename.split('.')[0]
        if '_' in name_without_ext:
            expected = name_without_ext.split('_', 1)[1]
        else:
            expected = name_without_ext
        
        # Analyze preprocessing quality (PSNR/MSE)
        plate_img = cv2.imread(img_path)
        if plate_img is not None:
            quality_results = analyze_preprocessing_quality(plate_img)
            psnr_mse_data.append(quality_results)
            debug_print(f"[PSNR/MSE] {filename}: PSNR={quality_results['psnr_overall']:.2f}dB, MSE={quality_results['mse_overall']:.2f}")
        
        # Measure inference time separately for each model
        result_with_timing = recognize_plate_with_timing(img_path, svm_model, cnn_model)
        results = result_with_timing['predictions']
        
        # Collect individual timing
        if result_with_timing['svm_time'] > 0:
            svm_inf_times.append(result_with_timing['svm_time'])
        if result_with_timing['cnn_time'] > 0:
            cnn_inf_times.append(result_with_timing['cnn_time'])
        
        if 'error' not in results:
            image_results.append((filename, expected, results))
            
            # Track character-level accuracy
            svm_result = results.get('svm', '')
            cnn_result = results.get('cnn', '')
            expected_clean = expected.replace(' ', '')
            
            # Per-character tracking
            for i, char in enumerate(expected_clean):
                # SVM character accuracy
                if i < len(svm_result):
                    char_results_svm[char]['total'] += 1
                    if svm_result[i] == char:
                        char_results_svm[char]['correct'] += 1
                
                # CNN character accuracy
                if i < len(cnn_result):
                    char_results_cnn[char]['total'] += 1
                    if cnn_result[i] == char:
                        char_results_cnn[char]['correct'] += 1
            
            # Print results
            print(f"\n>>> {filename}")
            if 'svm' in results:
                print(f"    SVM: {results['svm']}")
            if 'cnn' in results:
                print(f"    CNN: {results['cnn']}")
    
    # Print summary
    if image_results:
        print_summary(image_results)
        visualize_comparison(image_results)
        
        # Generate all visualizations for report
        generate_all_visualizations(
            image_results=image_results,
            svm_model=svm_model,
            cnn_model=cnn_model,
            test_images=test_images,
            svm_train_time=svm_time,
            cnn_train_time=cnn_time,
            cnn_history=cnn_history,
            svm_inf_times=svm_inf_times,
            cnn_inf_times=cnn_inf_times,
            char_results_svm=dict(char_results_svm),
            char_results_cnn=dict(char_results_cnn),
            cnn_pretrained=cnn_pretrained,
            psnr_mse_data=psnr_mse_data
        )
    
    # Training time comparison
    print(f"\n[TIME] Training Time:")
    print(f"   SVM: {svm_time:.2f} seconds")
    print(f"   CNN: {cnn_time:.2f} seconds")

