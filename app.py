import os
import cv2
import numpy as np
from flask import Flask, render_template, request, url_for
from sklearn.cluster import KMeans
from skimage.measure import regionprops, label
from scipy import ndimage
from PIL import Image
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

ALLOWED = {'jpg', 'jpeg', 'png', 'bmp'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED

def preprocess(img_rgb):
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return gray, binary

def morphology(binary):
    filled = ndimage.binary_fill_holes(binary > 0).astype(np.uint8) * 255
    num, labels, stats, _ = cv2.connectedComponentsWithStats(filled)
    cleaned = np.zeros_like(filled)
    for i in range(1, num):
        if stats[i, cv2.CC_STAT_AREA] >= 5000:
            cleaned[labels == i] = 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    return cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)

def kmeans_segmentation(img_rgb):
    h, w, _ = img_rgb.shape
    pixels = img_rgb.reshape(-1, 3).astype(np.float32)
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    labels = kmeans.fit_predict(pixels)
    centers = kmeans.cluster_centers_.astype(int)
    obj_cluster = np.argmax(centers.sum(axis=1))
    mask = (labels == obj_cluster).reshape(h, w).astype(np.uint8) * 255
    return mask

def extract_features(mask):
    labeled = label(mask > 0)
    regions = regionprops(labeled)
    if len(regions) == 0: return None
    region = max(regions, key=lambda r: r.area)
    
    area = region.area
    perimeter = region.perimeter
    metric = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0
    eccentricity = float(region.eccentricity) if region.eccentricity else 0

    return {
        "Area": int(area),
        "Perimeter": round(perimeter, 2),
        "Metric": round(metric, 5),
        "Eccentricity": round(eccentricity, 5)
    }

def analyze_color_texture(img_rgb, mask):
    egg_pixels_bgr = cv2.bitwise_and(img_rgb, img_rgb, mask=mask)
    egg_pixels = egg_pixels_bgr[mask > 0] 
    
    if len(egg_pixels) == 0:
        return {"Warna_Dominan_RGB": "-", "Jenis_Telur": "Unknown", "Kualitas_Warna": "Unknown", "Tingkat_Corak": 0.0}

    avg_bgr = np.mean(egg_pixels, axis=0)
    avg_rgb = (int(avg_bgr[2]), int(avg_bgr[1]), int(avg_bgr[0])) 
    
    avg_pixel_img = np.uint8([[[avg_bgr[0], avg_bgr[1], avg_bgr[2]]]])
    avg_hsv = cv2.cvtColor(avg_pixel_img, cv2.COLOR_BGR2HSV)[0][0]
    hue, sat, val = avg_hsv

    r, g, b = avg_rgb

    gray_img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    gray_pixels = gray_img[mask > 0]
    mean_gray = np.mean(gray_pixels)
    
    dark_spots = np.sum(gray_pixels < (mean_gray * 0.65))
    total_pixels = len(gray_pixels)
    corak_score = round((dark_spots / total_pixels) * 100, 2)

    if corak_score > 8.0:
        jenis = "Telur Puyuh (Bercorak)"
    elif r > 140 and g > 90 and b < 110 and r > b:
        jenis = "Telur Ayam (Coklat)"
    elif val > 120 and sat < 100:
        jenis = "Telur Bebek (Putih/Kehijauan)"
    else:
        jenis = "Telur Ayam / Bebek"

    if val > 220 and sat < 50:
        kualitas = "Pucet / Kurang Sehat"
    else:
        kualitas = "Normal / Sehat"

    return {
        "Warna_Dominan_RGB": f"R:{r} G:{g} B:{b}",
        "Jenis_Telur": jenis,
        "Tingkat_Corak": f"{corak_score}% area",
        "Kualitas_Warna": kualitas
    }

def mse_rmse_psnr(original, segmented):
    orig = original.astype(np.float64) / 255.0
    seg = segmented.astype(np.float64) / 255.0
    mse = float(np.mean((orig - seg) ** 2))
    rmse = float(np.sqrt(mse))
    psnr = 20 * np.log10(1.0 / np.sqrt(mse)) if mse > 0 else 0.0
    return round(mse, 8), round(rmse, 6), round(psnr, 4)

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    if request.method == 'POST':
        file = request.files.get('image')
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            img_pil = Image.open(filepath).convert('RGB')
            img_rgb = np.array(img_pil)
            h, w = img_rgb.shape[:2]

            if max(h, w) > 500:
                scale = 500 / max(h, w)
                img_rgb = cv2.resize(img_rgb, (int(w * scale), int(h * scale)))

            gray, binary = preprocess(img_rgb)
            morph = morphology(binary)
            km_mask = kmeans_segmentation(img_rgb)
            km_morph = morphology(km_mask)
            feats = extract_features(km_morph)

            if feats is None:
                result = {"error": "Objek tidak terdeteksi. Coba gambar lain."}
            else:

                area = feats["Area"]
                label_ukuran = "BESAR" if area > 45000 else "KECIL"
                

                color_info = analyze_color_texture(img_rgb, km_morph)
                
                mse, rmse, psnr = mse_rmse_psnr(gray, km_morph)

                vis_path = os.path.join(app.config['UPLOAD_FOLDER'], 'vis_' + filename)
                fig = np.hstack([
                    cv2.cvtColor(cv2.resize(img_rgb, (200, 200)), cv2.COLOR_RGB2BGR),
                    cv2.cvtColor(cv2.resize(gray, (200, 200)), cv2.COLOR_GRAY2BGR),
                    cv2.cvtColor(cv2.resize(binary, (200, 200)), cv2.COLOR_GRAY2BGR),
                    cv2.cvtColor(cv2.resize(km_morph, (200, 200)), cv2.COLOR_GRAY2BGR),
                ])
                cv2.imwrite(vis_path, fig)

                result = {
                    "label_ukuran": label_ukuran,
                    "fitur_bentuk": feats,
                    "fitur_warna": color_info,
                    "mse": mse,
                    "rmse": rmse,
                    "psnr": psnr,
                    "img_vis": url_for('static', filename=f'uploads/vis_{filename}'),
                }

    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)