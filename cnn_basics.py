"""
Basic CNN Example - Introduction to Convolutional Neural Networks

This script demonstrates a simple CNN for image classification using the MNIST dataset.
It's designed to help you understand the basic architecture and workflow of CNNs.

Key Concepts:
- Convolutional layers: Extract spatial features from images
- Pooling layers: Reduce spatial dimensions
- Dense layers: Make final predictions
- Training loop: How neural networks learn from data
"""

import tensorflow as tf
import keras
from keras import layers
import matplotlib.pyplot as plt
import numpy as np

def create_basic_cnn():
    """
    Create a simple CNN model for MNIST digit classification.
    
    Architecture:
    - Conv2D: 32 filters, 3x3 kernel -> Extract low-level features
    - MaxPooling: 2x2 -> Reduce dimensions
    - Conv2D: 64 filters, 3x3 kernel -> Extract higher-level features
    - MaxPooling: 2x2 -> Reduce dimensions further
    - Flatten: Convert 2D features to 1D
    - Dense: 64 units -> Learn complex patterns
    - Dense: 10 units -> Output probabilities for 10 digits
    """
    model = keras.Sequential([
        # First convolutional block
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),
        
        # Second convolutional block
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        # Flatten and dense layers
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(10, activation='softmax')
    ])
    
    return model

def generate_synthetic_mnist_data():
    """
    Generate synthetic data similar to MNIST for demonstration purposes.
    This is used when the actual MNIST dataset cannot be downloaded.
    
    Creates simple digit-like patterns for educational purposes.
    """
    print("Generating synthetic digit-like data for demonstration...")
    np.random.seed(42)
    
    # Generate training data
    x_train = np.random.rand(6000, 28, 28, 1).astype('float32')
    y_train = np.random.randint(0, 10, 6000)
    
    # Add some simple patterns based on labels
    for i in range(len(x_train)):
        label = y_train[i]
        # Create a simple pattern based on the digit
        x_train[i] = x_train[i] * 0.3  # Base noise
        # Add a characteristic pattern for each digit
        start_row = 5 + label
        start_col = 5 + label
        x_train[i, start_row:start_row+10, start_col:start_col+10, 0] += 0.5
        
    # Generate test data
    x_test = np.random.rand(1000, 28, 28, 1).astype('float32')
    y_test = np.random.randint(0, 10, 1000)
    
    for i in range(len(x_test)):
        label = y_test[i]
        x_test[i] = x_test[i] * 0.3
        start_row = 5 + label
        start_col = 5 + label
        x_test[i, start_row:start_row+10, start_col:start_col+10, 0] += 0.5
    
    print(f"Training data shape: {x_train.shape}")
    print(f"Test data shape: {x_test.shape}")
    
    return (x_train, y_train), (x_test, y_test)

def load_and_preprocess_data():
    """Load MNIST dataset and preprocess it for CNN training."""
    try:
        print("Attempting to load MNIST dataset...")
        (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
        
        # Normalize pixel values to [0, 1]
        x_train = x_train.astype('float32') / 255.0
        x_test = x_test.astype('float32') / 255.0
        
        # Reshape to add channel dimension (required for CNN)
        x_train = np.expand_dims(x_train, -1)
        x_test = np.expand_dims(x_test, -1)
        
        print(f"Training data shape: {x_train.shape}")
        print(f"Test data shape: {x_test.shape}")
        
        return (x_train, y_train), (x_test, y_test)
    except Exception as e:
        print(f"Could not download MNIST: {e}")
        print("Falling back to synthetic data for demonstration...")
        return generate_synthetic_mnist_data()

def train_model(model, x_train, y_train, x_test, y_test, epochs=5):
    """Train the CNN model and return training history."""
    print("\nCompiling model...")
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("\nTraining model...")
    print("=" * 60)
    history = model.fit(
        x_train, y_train,
        batch_size=128,
        epochs=epochs,
        validation_split=0.1,
        verbose=1
    )
    
    print("\nEvaluating on test set...")
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test accuracy: {test_accuracy:.4f}")
    
    return history

def visualize_predictions(model, x_test, y_test, num_examples=5):
    """Visualize some predictions made by the model."""
    predictions = model.predict(x_test[:num_examples], verbose=0)
    
    fig, axes = plt.subplots(1, num_examples, figsize=(15, 3))
    for i in range(num_examples):
        axes[i].imshow(x_test[i].squeeze(), cmap='gray')
        predicted_label = np.argmax(predictions[i])
        true_label = y_test[i]
        color = 'green' if predicted_label == true_label else 'red'
        axes[i].set_title(f'Pred: {predicted_label}\nTrue: {true_label}', color=color)
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig('cnn_basics_predictions.png', dpi=150, bbox_inches='tight')
    print("\nPrediction visualization saved to 'cnn_basics_predictions.png'")
    plt.close()

def visualize_training_history(history):
    """Plot training and validation accuracy/loss over epochs."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Accuracy plot
    ax1.plot(history.history['accuracy'], label='Training')
    ax1.plot(history.history['val_accuracy'], label='Validation')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True)
    
    # Loss plot
    ax2.plot(history.history['loss'], label='Training')
    ax2.plot(history.history['val_loss'], label='Validation')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('cnn_basics_training.png', dpi=150, bbox_inches='tight')
    print("Training history saved to 'cnn_basics_training.png'")
    plt.close()

def main():
    """Main function to run the basic CNN example."""
    print("=" * 60)
    print("BASIC CNN EXAMPLE - MNIST Digit Classification")
    print("=" * 60)
    
    # Load data
    (x_train, y_train), (x_test, y_test) = load_and_preprocess_data()
    
    # Create model
    print("\nCreating CNN model...")
    model = create_basic_cnn()
    model.summary()
    
    # Train model
    history = train_model(model, x_train, y_train, x_test, y_test, epochs=5)
    
    # Visualize results
    visualize_training_history(history)
    visualize_predictions(model, x_test, y_test)
    
    print("\n" + "=" * 60)
    print("Basic CNN example completed successfully!")
    print("Check the generated PNG files to see the results.")
    print("=" * 60)

if __name__ == "__main__":
    main()
