"""
CNN for Climate Modeling - Solar Irradiance Prediction in Colombia

This script demonstrates how CNNs can be applied to climate modeling for solar energy predictions.
It uses synthetic climate data that mimics real-world patterns for educational purposes.

Application Context:
- Solar energy prediction is crucial for grid management and energy planning
- Colombia has varying solar irradiance patterns due to its geography
- CNNs can learn spatial patterns from satellite imagery and climate maps
- This example uses simulated data representing temperature, cloud cover, and humidity maps

Key Learning Points:
1. Multi-channel input (like RGB images, but climate variables)
2. Spatial feature extraction from weather maps
3. Regression output (predicting continuous values, not classes)
4. Time-series aspects of climate data
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def generate_synthetic_climate_data(num_samples=1000, spatial_size=32):
    """
    Generate synthetic climate data for Colombia.
    
    This simulates:
    - Channel 0: Temperature map (°C)
    - Channel 1: Cloud cover (%)
    - Channel 2: Relative humidity (%)
    
    Target: Daily average solar irradiance (kWh/m²/day)
    
    In real applications, this data would come from:
    - Satellite imagery (e.g., GOES-16)
    - Numerical weather prediction models
    - Ground station measurements
    """
    np.random.seed(42)
    
    # Generate climate maps (spatial_size x spatial_size x 3 channels)
    climate_data = np.zeros((num_samples, spatial_size, spatial_size, 3))
    solar_irradiance = np.zeros(num_samples)
    
    for i in range(num_samples):
        # Temperature: 20-35°C with spatial variation
        base_temp = np.random.uniform(22, 32)
        temp_map = base_temp + np.random.randn(spatial_size, spatial_size) * 2
        climate_data[i, :, :, 0] = temp_map
        
        # Cloud cover: 0-100% with clusters
        cloud_center_x = np.random.randint(0, spatial_size)
        cloud_center_y = np.random.randint(0, spatial_size)
        x, y = np.meshgrid(range(spatial_size), range(spatial_size))
        cloud_map = 100 * np.exp(-((x - cloud_center_x)**2 + (y - cloud_center_y)**2) / (2 * 100))
        cloud_map += np.random.uniform(0, 30, (spatial_size, spatial_size))
        cloud_map = np.clip(cloud_map, 0, 100)
        climate_data[i, :, :, 1] = cloud_map
        
        # Humidity: 50-90% correlated with clouds
        humidity_map = 60 + cloud_map * 0.3 + np.random.randn(spatial_size, spatial_size) * 5
        humidity_map = np.clip(humidity_map, 50, 95)
        climate_data[i, :, :, 2] = humidity_map
        
        # Solar irradiance: inversely related to cloud cover, positively to temperature
        avg_cloud = np.mean(cloud_map)
        avg_temp = np.mean(temp_map)
        # Colombia receives 4-6 kWh/m²/day on average
        irradiance = 6.5 - (avg_cloud / 100) * 3.0 + (avg_temp - 27) * 0.1
        irradiance += np.random.randn() * 0.3  # Add noise
        irradiance = np.clip(irradiance, 2.0, 7.0)
        solar_irradiance[i] = irradiance
    
    # Normalize the input features
    climate_data[:, :, :, 0] = (climate_data[:, :, :, 0] - 27) / 10  # Temperature
    climate_data[:, :, :, 1] = climate_data[:, :, :, 1] / 100  # Cloud cover
    climate_data[:, :, :, 2] = (climate_data[:, :, :, 2] - 70) / 20  # Humidity
    
    return climate_data, solar_irradiance

def create_climate_cnn(input_shape=(32, 32, 3)):
    """
    Create a CNN for predicting solar irradiance from climate maps.
    
    Architecture designed for spatial climate pattern recognition:
    - Multiple convolutional layers to capture weather patterns at different scales
    - Global average pooling to aggregate spatial information
    - Dense layers for final regression
    """
    model = keras.Sequential([
        # First conv block: Local feature extraction
        layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        
        # Second conv block: Mid-level pattern recognition
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        
        # Third conv block: High-level feature extraction
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.GlobalAveragePooling2D(),
        
        # Regression head
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(1)  # Single output: solar irradiance
    ])
    
    return model

def train_climate_model(model, x_train, y_train, x_val, y_val, epochs=50):
    """Train the climate CNN model."""
    print("\nCompiling model...")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    
    # Add callbacks for better training
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            verbose=1
        )
    ]
    
    print("\nTraining climate model...")
    print("=" * 60)
    history = model.fit(
        x_train, y_train,
        batch_size=32,
        epochs=epochs,
        validation_data=(x_val, y_val),
        callbacks=callbacks,
        verbose=1
    )
    
    return history

def visualize_climate_predictions(model, x_test, y_test, num_examples=4):
    """Visualize climate maps and corresponding predictions."""
    predictions = model.predict(x_test[:num_examples], verbose=0)
    
    fig, axes = plt.subplots(num_examples, 4, figsize=(16, 4*num_examples))
    
    for i in range(num_examples):
        # Temperature map
        axes[i, 0].imshow(x_test[i, :, :, 0], cmap='RdYlBu_r')
        axes[i, 0].set_title('Temperature Map')
        axes[i, 0].axis('off')
        
        # Cloud cover map
        axes[i, 1].imshow(x_test[i, :, :, 1], cmap='gray_r')
        axes[i, 1].set_title('Cloud Cover Map')
        axes[i, 1].axis('off')
        
        # Humidity map
        axes[i, 2].imshow(x_test[i, :, :, 2], cmap='Blues')
        axes[i, 2].set_title('Humidity Map')
        axes[i, 2].axis('off')
        
        # Prediction vs actual
        axes[i, 3].bar(['Predicted', 'Actual'], 
                       [predictions[i][0], y_test[i]],
                       color=['blue', 'green'])
        axes[i, 3].set_ylabel('Solar Irradiance (kWh/m²/day)')
        axes[i, 3].set_ylim(0, 8)
        axes[i, 3].set_title(f'Error: {abs(predictions[i][0] - y_test[i]):.2f}')
        axes[i, 3].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('climate_cnn_predictions.png', dpi=150, bbox_inches='tight')
    print("\nClimate prediction visualization saved to 'climate_cnn_predictions.png'")
    plt.close()

def visualize_training_history(history):
    """Plot training history for the climate model."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss plot
    ax1.plot(history.history['loss'], label='Training Loss', linewidth=2)
    ax1.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
    ax1.set_title('Model Loss (MSE)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Mean Squared Error', fontsize=12)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # MAE plot
    ax2.plot(history.history['mae'], label='Training MAE', linewidth=2)
    ax2.plot(history.history['val_mae'], label='Validation MAE', linewidth=2)
    ax2.set_title('Model MAE', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Mean Absolute Error (kWh/m²/day)', fontsize=12)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('climate_cnn_training.png', dpi=150, bbox_inches='tight')
    print("Training history saved to 'climate_cnn_training.png'")
    plt.close()

def evaluate_model_performance(model, x_test, y_test):
    """Evaluate and print detailed model performance metrics."""
    predictions = model.predict(x_test, verbose=0).flatten()
    
    mse = np.mean((predictions - y_test) ** 2)
    mae = np.mean(np.abs(predictions - y_test))
    rmse = np.sqrt(mse)
    
    # R² score
    ss_res = np.sum((y_test - predictions) ** 2)
    ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    
    print("\n" + "=" * 60)
    print("MODEL PERFORMANCE METRICS")
    print("=" * 60)
    print(f"Mean Absolute Error (MAE):  {mae:.4f} kWh/m²/day")
    print(f"Root Mean Squared Error:    {rmse:.4f} kWh/m²/day")
    print(f"R² Score:                   {r2:.4f}")
    print("=" * 60)
    
    # Create scatter plot
    plt.figure(figsize=(8, 8))
    plt.scatter(y_test, predictions, alpha=0.5, s=30)
    plt.plot([y_test.min(), y_test.max()], 
             [y_test.min(), y_test.max()], 
             'r--', linewidth=2, label='Perfect Prediction')
    plt.xlabel('Actual Solar Irradiance (kWh/m²/day)', fontsize=12)
    plt.ylabel('Predicted Solar Irradiance (kWh/m²/day)', fontsize=12)
    plt.title('Predicted vs Actual Solar Irradiance', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('climate_cnn_scatter.png', dpi=150, bbox_inches='tight')
    print("\nScatter plot saved to 'climate_cnn_scatter.png'")
    plt.close()

def main():
    """Main function to run the climate CNN example."""
    print("=" * 60)
    print("CNN FOR CLIMATE MODELING - Solar Irradiance Prediction")
    print("Application: Solar Energy Forecasting in Colombia")
    print("=" * 60)
    
    # Generate synthetic climate data
    print("\nGenerating synthetic climate data...")
    X, y = generate_synthetic_climate_data(num_samples=2000, spatial_size=32)
    print(f"Generated {len(X)} samples")
    print(f"Input shape: {X.shape}")
    print(f"Target range: {y.min():.2f} - {y.max():.2f} kWh/m²/day")
    
    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    print(f"\nTraining set: {len(X_train)} samples")
    print(f"Validation set: {len(X_val)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # Create and train model
    print("\nCreating climate CNN model...")
    model = create_climate_cnn(input_shape=X.shape[1:])
    model.summary()
    
    # Train the model
    history = train_climate_model(model, X_train, y_train, X_val, y_val, epochs=50)
    
    # Evaluate and visualize
    print("\nEvaluating model on test set...")
    evaluate_model_performance(model, X_test, y_test)
    
    visualize_training_history(history)
    visualize_climate_predictions(model, X_test, y_test)
    
    print("\n" + "=" * 60)
    print("Climate CNN example completed successfully!")
    print("Check the generated PNG files to see the results.")
    print("\nNext Steps:")
    print("- Try modifying the CNN architecture")
    print("- Experiment with different synthetic data patterns")
    print("- Learn about real climate data sources (ERA5, GOES, etc.)")
    print("- Explore transfer learning with pre-trained models")
    print("=" * 60)

if __name__ == "__main__":
    main()
