# Korean Letter Crop with PaddleOCR and OpenCV

## 📖 Overview
This project extracts individual Korean characters from images using **PaddleOCR** and **OpenCV**. It processes images by detecting bounding boxes, cropping characters, and refining them for further use, such as Optical Character Recognition (OCR) or dataset preparation.

### Key Features:
- **Bounding Box Detection**: Detects text regions using PaddleOCR.
- **Character Cropping**: Crops individual characters from detected bounding boxes.
- **Projection-Based Segmentation**: Separates characters within a bounding box using vertical projection.
- **Size Filtering**: Filters out characters that are too small to ensure quality.
- **Final Character Extraction**: Extracts and saves individual Korean characters with high confidence.

---

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- CUDA-enabled GPU (for PaddleOCR GPU version)

### Install Dependencies
1. Clone the repository:
```bash
git clone https://github.com/your-repo/korean-letter-crop.git
cd korean-letter-crop
```
2. Install required Python packages:
```bash
conda create -n ENVNAME --python=3.12.9
conda activate ENVNAME
pip install -r requirements.txt
```

---

## 🚀 Usage
1. Run the Program
To process an input image and extract Korean characters:

2. Arguments
-i: Path to the input image (default: ./inputs/image.png)  
-o: Path to the output directory for extracted characters (default: ./final/)

3. Output
Cropped Images: Saved in the ./cropped/ directory.  
Projection Images: Saved in the ./projection/ directory.  
Filtered Characters: Saved in the ./best_size/ directory.  
Final Characters: Saved in the ./final/ directory with filenames corresponding to the recognized characters.  

---

## 📂 Project Structure
```text
korean-letter-crop/
├── inputs/                # Input images
├── cropped/               # Cropped bounding box images
├── projection/            # Projection-based segmented images
├── best_size/             # Filtered images based on size
├── final/                 # Final extracted characters
├── [run.py]              # Main script
├── [requirements.txt]     # Python dependencies
└── [Readme.md]           # Project documentation
```

---

## ⚙️ How It Works

1. Bounding Box Detection:

Uses PaddleOCR to detect text regions and generate bounding boxes.

2. Cropping:

Crops the detected bounding boxes and saves them as individual images.

3. Projection-Based Segmentation:

Uses vertical projection to separate characters within a bounding box.

4. Size Filtering:

Filters out characters that are too small to ensure quality.

5. Final Character Extraction:

Uses PaddleOCR to recognize and save individual Korean characters with high confidence.

---

## 🧪 Testing

To test the program with sample images:

1. Place your test images in the ./inputs/ directory.
2. Run the program
```bash
python run.py -i INPUT_IMAGE_PATH -o OUTPUT_DIR_PATH
```
