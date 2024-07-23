<div style="text-align: center;">
    <img src="banner/banner.jpg" style="width:950px;height:450px;">
</div>


# Softmax Function Implementation in Neural Network

## Overview

This project demonstrates the implementation of the softmax function within a neural network using TensorFlow. It includes steps for data generation, model building, training, and evaluation, showcasing the process of constructing and assessing a multi-class classification model on synthetic data.

## Features

- **Data Generation**: Uses the `make_blobs` function from `sklearn.datasets` to create synthetic data for training.
- **Model Building**: Constructs a sequential neural network with multiple dense layers.
- **Training**: Trains the model on the generated data.
- **Evaluation**: Evaluates the model's performance and displays predictions.

## Requirements

- Python 3.x
- NumPy
- Matplotlib
- TensorFlow
- Scikit-learn

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/HoomKh/softmax-neural-network.git
   ```
2. Navigate to the project directory:
   ```bash
   cd softmax-neural-network
   ```
3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Import Libraries**:
   ```python
   import numpy as np
   import matplotlib.pyplot as plt
   import tensorflow as tf
   from tensorflow.keras.models import Sequential
   from tensorflow.keras.layers import Dense
   from sklearn.datasets import make_blobs
   ```

2. **Define Softmax Function**:
   ```python
   def my_softmax(z):
       ez = np.exp(z)
       sm = ez / np.sum(ez)
       return sm
   ```

3. **Generate Data**:
   ```python
   centers = [[-5, 2], [-2, -2], [1, 2], [5, -2]]
   X_train, y_train = make_blobs(n_samples=2000, centers=centers, cluster_std=1.0, random_state=30)
   ```

4. **Build and Train the Model**:
   ```python
   model = Sequential([
       Dense(25, activation='relu'),
       Dense(15, activation='relu'),
       Dense(4, activation='softmax')
   ])
   model.compile(
       loss=tf.keras.losses.SparseCategoricalCrossentropy(), 
       optimizer=tf.keras.optimizers.Adam(0.01)
   )
   model.fit(X_train, y_train, epochs=10)
   ```

5. **Evaluate the Model**:
   ```python
   predictions = model.predict(X_train)
   print(predictions[:2])
   ```

## Results

The trained model outputs the softmax probabilities for the given inputs. You can observe the predictions and analyze the model's performance on the synthetic dataset.

## Contributing

Feel free to fork the repository and submit pull requests. For major changes, please open an issue first to discuss what you would like to change.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---
