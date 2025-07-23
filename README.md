# 🧠 Cat vs. Non-Cat Classifier – Deep Neural Network

This project implements a deep feedforward neural network **from scratch using NumPy** to classify images as either **cat** or **non-cat**, based on pixel-level RGB data. The dataset used is the [Cat vs. Non-Cat Dataset from Kaggle](https://www.kaggle.com/datasets/sagar2522/cat-vs-non-cat), composed of small, labeled images for binary classification.

> 📘 This implementation is inspired by the principles and exercises taught in [Andrew Ng’s Coursera Course: Neural Networks and Deep Learning](https://www.coursera.org/learn/neural-networks-deep-learning/), with adaptations and extensions.

---

## 📁 Dataset

The dataset consists of RGB images with dimensions 64×64×3, labeled as:

- `1` if the image contains a **cat**
- `0` otherwise

Split:
- **Training set**: 209 images
- **Test set**: 50 images

Before training, the images are flattened and normalized.

---

## 🧪 What’s Implemented

### ✅ Core Components

- **Parameter Initialization**, including He initialization for ReLU
- **Forward Propagation**:
  - Linear + ReLU for hidden layers
  - Linear + Sigmoid for output layer
- **Cost Function**: Binary cross-entropy with numerical stability
- **Backward Propagation**
- **Gradient Descent Update**
- **Prediction and Accuracy Evaluation**

### 🧩 Training Details

- Cost is printed every 100 iterations to monitor convergence
- Network is trained using batch gradient descent
- The model uses a fully connected architecture

---

## 📊 Results

- The cost function shows a steady decrease across iterations
- With a learning rate of **0.01** and architecture of `[12288, 128, 64, 32, 1]`, the model reaches:
  - ✅ **Training Accuracy**: ~99%
  - ✅ **Test Accuracy**: ~72%–85% (varies slightly based on seed and config)

This result confirms that the network is learning to distinguish cats from non-cats, although generalization could improve with more data or regularization.

---

## 🚀 Future Work

- 🔄 Implement **mini-batch gradient descent** to speed up training and improve stability
- 📉 Add **L2 regularization** and **dropout** to reduce overfitting
- 🧠 Convert to **CNN** for better spatial feature learning
- 🔁 Expand to **multiclass classification**
- ⚙️ Reimplement using **TensorFlow** or **PyTorch**

---


## 💡 Conclusion

This project demonstrates the full deep learning pipeline—from scratch—without using any high-level libraries. By working through initialization, propagation, and optimization manually, it reinforces foundational concepts and provides a solid base for transitioning to more advanced architectures and frameworks.
