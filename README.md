# Breast Cancer Ultrasound Segmentation using U-Net

This repository implements a U-Net-based deep learning model to segment breast cancer regions from ultrasound images. The dataset used is the **BUSI (Breast Ultrasound Images Dataset)** which includes benign, malignant, and normal cases with ground truth masks.

## 🧠 Project Overview
- Segment breast lesions in ultrasound images using U-Net.
- Input: Grayscale ultrasound images (256×256)
- Output: Binary mask highlighting tumor region.

## 📁 Dataset
Dataset: [Kaggle Link](https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset)
## Organize Dataset like:
Dataset_BUSI_with_GT/
├── benign/
│ ├── image1.png
│ ├── image1_mask.png
│ └── ...
├── malignant/
├── normal/

## ⚙️ Setup Instructions
```bash
git clone https://github.com/your-username/breast-cancer-unet.git
cd breast-cancer-unet
pip install -r requirements.txt
```

## 🚀 Run the Model
```bash
python unet_train.py
```

## 📦 Output
- Trained model: `unet_model.h5`

## 🤝 Contribution
Pull requests and suggestions are welcome.

## 📄 License
This project is open-source under the [MIT License](LICENSE).
