import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from matplotlib.gridspec import GridSpec
import plotly.graph_objects as go
from sklearn.cluster import KMeans

# ===================== Image Loading and Preprocessing =====================
def load_images_from_folder(folder):
    """Load and preprocess images from a folder, removing duplicates."""
    images = []
    stored_images = []
    if not os.path.exists(folder):
        raise FileNotFoundError(f"Folder {folder} does not exist")
    for filename in sorted(os.listdir(folder)):
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path)
        if img is None:
            continue
        # Preprocess each image as it's loaded
        img = preprocess_image(img)
        img_resized = cv2.resize(img, (512, 512))
        is_duplicate = any(
            ssim(cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY),
                 cv2.cvtColor(stored_img, cv2.COLOR_BGR2GRAY)) > 0.95
            for stored_img in stored_images
        )
        if not is_duplicate:
            stored_images.append(img_resized)
            images.append(img_resized)
    if not images:
        raise ValueError("No valid images found in the folder")
    return images

def preprocess_image(image):
    """Enhance image quality before processing."""
    # CLAHE for contrast enhancement
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    # Denoising
    image = cv2.GaussianBlur(image, (3, 3), 0)
    # White balance correction
    avg_b = np.mean(image[:, :, 0])
    avg_g = np.mean(image[:, :, 1])
    avg_r = np.mean(image[:, :, 2])
    image[:, :, 0] = np.minimum(image[:, :, 0] * (avg_g / avg_b), 255)
    image[:, :, 2] = np.minimum(image[:, :, 2] * (avg_g / avg_r), 255)
    return image

# ===================== Image Stitching =====================
def stitch_images(images):
    """Stitch images using OpenCV and crop black edges."""
    if len(images) < 2:
        return images[0]
    stitcher = cv2.Stitcher_create()
    status, stitched_image = stitcher.stitch(images)
    if status == cv2.Stitcher_OK:
        gray = cv2.cvtColor(stitched_image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            cnt = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(cnt)
            stitched_image = stitched_image[y:y + h, x:x + w]
        return stitched_image
    else:
        return images[0]

# ===================== VARI Calculation and Classification =====================
def calculate_vari(image):
    """
    Calculate Visible Atmospherically Resistant Index (VARI).
    Args:
        image (numpy.ndarray): Input RGB image
    Returns:
        numpy.ndarray: VARI index values
    """
    image = image.astype(np.float32)
    green = image[:, :, 1]
    red = image[:, :, 0]
    blue = image[:, :, 2]
    vari = np.divide(
        green - red,
        green + red - blue,
        out=np.zeros_like(red, dtype=float),
        where=(green + red - blue) != 0
    )
    return vari

def classify_vegetation_vari(vari):
    """
    Classify vegetation based on VARI values.
    Args:
        vari (numpy.ndarray): VARI index values
    Returns:
        numpy.ndarray: Classified vegetation map
    """
    healthy_vegetation = vari > 0.3
    moderate_vegetation = (vari > 0.1) & (vari <= 0.3)
    sparse_vegetation = (vari > 0) & (vari <= 0.1)
    no_vegetation = vari <= 0
    vegetation_map = np.zeros((*vari.shape, 3), dtype=np.uint8)
    vegetation_map[healthy_vegetation] = [0, 255, 0]  # Bright Green
    vegetation_map[moderate_vegetation] = [128, 255, 0]  # Lime Green
    vegetation_map[sparse_vegetation] = [255, 255, 0]  # Yellow
    vegetation_map[no_vegetation] = [255, 0, 0]  # Red
    return vegetation_map

def analyze_vegetation_percentage_vari(vari):
    total_pixels = vari.size
    percentages = {
        'Healthy Vegetation': np.sum(vari > 0.3) / total_pixels * 100,
        'Moderate Vegetation': np.sum((vari > 0.1) & (vari <= 0.3)) / total_pixels * 100,
        'Sparse Vegetation': np.sum((vari > 0) & (vari <= 0.1)) / total_pixels * 100,
        'No Vegetation/Stressed': np.sum(vari <= 0) / total_pixels * 100
    }
    return percentages

# ===================== Visualization =====================
def visualize_analysis(original_image, vari, vari_map, vari_percentages):
    """Create comprehensive visualizations for VARI."""
    fig, axes = plt.subplots(2, 2, figsize=(20, 12))
    fig.suptitle("Vegetation Analysis Using VARI", fontsize=20)

    # Original Image
    axes[0, 0].imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title("Original Stitched Image")
    axes[0, 0].axis('off')

    # VARI Heatmap
    vari_display = cv2.normalize(vari, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    axes[0, 1].imshow(vari_display, cmap='viridis')
    axes[0, 1].set_title("VARI Heatmap")
    axes[0, 1].axis('off')

    # VARI Classification Map
    axes[1, 0].imshow(vari_map)
    axes[1, 0].set_title("VARI Classification Map")
    axes[1, 0].axis('off')

    # Pie Chart
    axes[1, 1].pie(
        list(vari_percentages.values()),
        labels=list(vari_percentages.keys()),
        autopct='%1.1f%%',
        colors=['green', 'lime', 'yellow', 'red']
    )
    axes[1, 1].set_title("Vegetation Coverage")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

# ===================== Export Results =====================
def export_results(vari, vari_map, percentages, output_dir='output'):
    """Export analysis results to files."""
    os.makedirs(output_dir, exist_ok=True)

    # Save VARI data
    np.save(os.path.join(output_dir, 'vari_values.npy'), vari)

    # Save VARI vegetation map
    cv2.imwrite(os.path.join(output_dir, 'vegetation_map.png'), vari_map)

    # Save statistics to CSV
    import pandas as pd
    stats_df = pd.DataFrame.from_dict(percentages, orient='index', columns=['Percentage'])
    stats_df.loc['Total Vegetation'] = stats_df.iloc[:3].sum()
    stats_df.loc['Total Stressed/Drought'] = stats_df.iloc[3]
    stats_df.to_csv(os.path.join(output_dir, 'vegetation_stats.csv'))
    print(f"Results exported to {output_dir} directory")

# ===================== Main Workflow =====================
def process_and_analyze(folder_path, output_dir='output'):
    try:
        print("Loading and preprocessing images...")
        images = load_images_from_folder(folder_path)
        print(f"Loaded {len(images)} unique images")
        print("Stitching images...")
        stitched_image = stitch_images(images)
        print("Stitching completed successfully")

        print("Calculating VARI...")
        vari = calculate_vari(stitched_image)

        print("Classifying vegetation using VARI...")
        vari_map = classify_vegetation_vari(vari)
        vari_percentages = analyze_vegetation_percentage_vari(vari)

        print("Generating visualizations...")
        visualize_analysis(stitched_image, vari, vari_map, vari_percentages)

        print("Exporting results...")
        export_results(vari, vari_map, vari_percentages, output_dir)

        print("\nVegetation Coverage Analysis:")
        for category, percentage in vari_percentages.items():
            print(f"{category}: {percentage:.2f}%")

        total_veg = vari_percentages['Healthy Vegetation'] + vari_percentages['Moderate Vegetation'] + vari_percentages['Sparse Vegetation']
        drought = vari_percentages['No Vegetation/Stressed']
        print(f"\nTotal Vegetation: {total_veg:.2f}%")
        print(f"Total Stressed/Drought: {drought:.2f}%")

        return stitched_image, vari, vari_map, vari_percentages

    except Exception as e:
        print(f"Error: {str(e)}")
        return None, None, None, None

if __name__ == "__main__":
    dataset_path = "/content/drive/MyDrive/DataSet/Dataset/Field (3)"
    process_and_analyze(dataset_path)