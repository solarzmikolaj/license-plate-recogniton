# 🚗 License Plate Recognition

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green?style=for-the-badge&logo=opencv&logoColor=white)
![Status](https://img.shields.io/badge/Status-Active-success?style=for-the-badge)

A powerful and efficient tool for **Automatic License Plate Recognition (ALPR)** developed in Python. This project utilizes YOLO for license plate detection and Florence-2 model for OCR to extract text from vehicle license plates through a user-friendly Gradio web interface.

## 🌟 Features

* 📷 **Advanced Detection:** YOLO-based license plate detection with high accuracy
* 🔍 **OCR Integration:** Florence-2 model for optical character recognition
* 🚀 **Web Interface:** Easy-to-use Gradio web application
* 💻 **GPU Support:** Optimized for GPU acceleration (CUDA)
* 🐳 **Docker Ready:** Containerized deployment support

## 🛠️ Technologies Used

* **Python 3** - Core programming language
* **PyTorch** - Deep learning framework
* **YOLO (Ultralytics)** - License plate detection
* **Florence-2** - OCR model for text extraction
* **Gradio** - Web interface framework
* **OpenCV** - Image processing
* **NumPy** - Matrix operations

## 🚀 Getting Started

Follow these steps to set up the project locally.

### Prerequisites

Make sure you have Python 3.8+ installed on your system. You can download it [here](https://www.python.org/downloads/).

**Note:** For optimal performance, it's recommended to have:
- CUDA-compatible GPU (for faster inference)
- At least 4GB of RAM
- 2GB+ free disk space for models

### Installation

1. **Clone the repository:**

```bash
git clone https://github.com/solarzmikolaj/license-plate-recogniton.git
cd license-plate-recogniton
```

2. **Create a virtual environment (optional but recommended):**

```bash
python -m venv venv
```

Activate the virtual environment:
- **On Windows:**
  ```bash
  venv\Scripts\activate
  ```
- **On Linux/Mac:**
  ```bash
  source venv/bin/activate
  ```

3. **Install dependencies:**

```bash
pip install -r requirements.txt
```

**Note:** If you encounter issues with PyTorch installation, you may need to install it separately based on your system:
- For CUDA support: Visit [PyTorch official website](https://pytorch.org/get-started/locally/)
- For CPU-only: The requirements.txt should work, but may need adjustments

## 🎮 Usage

### Running the Web Application

To start the Gradio web interface, execute:

```bash
python app.py
```

The application will start a local web server (typically at `http://127.0.0.1:7860`). Open this URL in your web browser to access the interface.

### Using the Interface

1. Upload an image containing a vehicle with a license plate
2. Click "Uruchom analizę" (Run analysis)
3. View the detected license plate(s) and recognized text
4. Check the confidence scores and multiple detection results in the table

### Docker Deployment

If you prefer using Docker:

```bash
docker build -t license-plate-recognition .
docker run -p 7860:7860 license-plate-recognition
```

## 📸 Screenshots

| Original Image | Detected Plate | Recognized Text |
|:--------------:|:--------------:|:---------------:|
| <img src="https://via.placeholder.com/300x200?text=Car+Image" width="300"> | <img src="https://via.placeholder.com/300x100?text=Plate" width="300"> | **WA 12345** |

## 📂 Project Structure

```
license-plate-recogniton/
├── app.py                    # Main Gradio application
├── model/
│   ├── engine.py            # YOLO + Florence-2 engine
│   └── license_plate_detector.pt  # Trained YOLO weights
├── requirements.txt         # Python dependencies
├── Dockerfile               # Docker configuration
└── README.md               # Project documentation
```

## 🤝 Contributing

Contributions are welcome! If you have suggestions for improvements or bug fixes, please feel free to:

1. Fork the repository
2. Create a new branch (`git checkout -b feature-branch`)
3. Commit your changes (`git commit -m 'Add some feature'`)
4. Push to the branch (`git push origin feature-branch`)
5. Open a Pull Request

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

<p align="center">Created by <a href="https://github.com/solarzmikolaj">Mikołaj Solarz</a>, <a href="https://github.com/cpetryka">Cezary Petryka</a> and <a href="https://github.com/KubaKarwow">Kuba Karwowski</a></p>
