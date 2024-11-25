# One Piece Card Model

Welcome to the **One Piece Card Model** repository! This project is a machine learning prototype for identifying trading card images, classifying them, and visualizing historical pricing trends. It is built using PyTorch and a fine-tuned ResNet model.

## Features
- **Card Classification**: Identifies and classifies One Piece trading cards from an input image.
- **Pre-trained Model**: Includes a trained ResNet model (`best_model.pth`) and supporting files (`state.db`).

---

## Repository Contents
- `One-Piece-Model/`: Contains the prototype code and supporting files.
  - `one-piece-cards/`: Dataset of card images used for the model.
  - `best_model.pth`: Trained model weights.
  - `state.db`: Supporting database for price data. ~ Note: unused at the moment!
  - `file.py`: Python file containing the prototype code.

## Prerequisites
- Python 3.8+
- Libraries:
  - `torch`
  - `torchvision`
  - `pandas`
  - `Pillow`
  - `Tkinter`

Install dependencies via pip:
```bash
pip install torch torchvision pandas Pillow tkinter
```

## Usage
- Clone the Repository
```bash
git clone https://github.com/jaredthecarrot/One-Piece-Model.git
cd One-Piece-Model
```
- Ensure a webcam is connected and functional
- Run the `file.py` script:
```bash
python file.py
```
