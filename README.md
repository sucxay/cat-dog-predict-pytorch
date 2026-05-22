# Cat vs Dog Image Classification using PyTorch

A deep learning project for binary image classification using a Convolutional Neural Network (CNN) built with PyTorch. The model classifies images as either cats or dogs using grayscale image preprocessing and a custom CNN architecture.

## Project Overview

This project covers the complete workflow for an image classification task:

- Downloading dataset from Kaggle
- Preprocessing image data
- Building a CNN model in PyTorch
- Training and evaluating the model
- Predicting custom images
- Exporting and uploading the project to GitHub

The implementation was developed and tested in Google Colab.

---

## Dataset

Dataset used:

- Dog and Cat Classification Dataset from Kaggle

Dataset structure:

```text
PetImages/
│
├── Cat/
├── Dog/
```

---

## Technologies Used

- Python
- PyTorch
- Torchvision
- Google Colab
- Kaggle API
- Matplotlib
- PIL

---

## Image Preprocessing

The following preprocessing steps were applied:

- Converted images to grayscale
- Resized images to `128x128`
- Converted images to tensors

```python
transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((128,128)),
    transforms.ToTensor()
])
```

---

## Model Architecture

Custom CNN architecture built using PyTorch:

- Convolutional Layers
- ReLU Activation
- Max Pooling
- Fully Connected Layers
- Softmax Classification

Loss Function:

```python
nn.CrossEntropyLoss()
```

Optimizer:

```python
optim.Adam(model.parameters(), lr=0.001)
```

---

## Train/Test Split

- Training Data: 80%
- Testing Data: 20%

```python
random_split(dataset, [train_size, test_size])
```

---

## Training

The model was trained using:

```python
DataLoader(batch_size=32, shuffle=True)
```

The training pipeline includes:

- Forward propagation
- Loss calculation
- Backpropagation
- Parameter optimization

---

## Evaluation

Model evaluation was performed using test accuracy:

```python
correct += (predicted == labels).sum().item()
```

---

## Prediction on Custom Images

Users can upload custom images and test predictions directly in Google Colab.

Supported workflow:

1. Upload image
2. Preprocess image
3. Run inference
4. Display prediction

---

## Project Structure

```text
Cat-VS-Dog/
│
├── Cat_VS_Dog.ipynb
├── .gitignore
├── README.md
└── requirements.txt
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/sucxay/cat-dog-predict-pytorch.git
```

Install dependencies:

```bash
pip install torch torchvision matplotlib pillow kaggle
```

---

## Running the Project

Open the notebook in Google Colab or Jupyter Notebook and run all cells sequentially.

---

## Future Improvements

- Use RGB images instead of grayscale
- Add data augmentation
- Improve CNN architecture
- Use pretrained models like ResNet or EfficientNet
- Deploy model using Flask or Streamlit

---

## Results

The model successfully performs binary image classification between cats and dogs using a custom CNN implemented entirely in PyTorch.

---

## Author

Suchay Joshi

GitHub Repository:

https://github.com/sucxay/cat-dog-predict-pytorch
