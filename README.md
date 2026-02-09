# Emergente - CNN for Climate Modeling & Solar Energy Prediction

Welcome to your internship starter repository! This project demonstrates how **Convolutional Neural Networks (CNNs)** can be applied to **climate modeling** for **solar energy predictions in Colombia**.

## 🎯 Project Overview

This repository contains educational examples designed to help you understand:

1. **Basic CNN Concepts**: How CNNs work with practical examples
2. **Climate Modeling Application**: Using CNNs for solar irradiance prediction
3. **Computer Vision in Climate Science**: Processing spatial climate data (temperature, cloud cover, humidity maps)

## 📚 What You'll Learn

### Convolutional Neural Networks (CNNs)
- **Convolutional Layers**: Extract spatial features from images/maps
- **Pooling Layers**: Reduce spatial dimensions while preserving important features
- **Dense Layers**: Make final predictions based on extracted features
- **Training Process**: How neural networks learn from data

### Climate Modeling for Solar Energy
- **Multi-channel Input**: Processing multiple climate variables simultaneously
- **Spatial Pattern Recognition**: Learning weather patterns from map data
- **Regression Tasks**: Predicting continuous values (solar irradiance)
- **Real-world Applications**: Grid management and energy planning in Colombia

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Installation

1. Clone this repository (if you haven't already):
```bash
git clone https://github.com/LiamKamphuis/Emergente.git
cd Emergente
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

This will install:
- TensorFlow (deep learning framework)
- NumPy (numerical computing)
- Matplotlib (visualization)
- Pandas (data manipulation)
- scikit-learn (machine learning utilities)

## 🎓 Example Scripts

### 1. Basic CNN Example (`cnn_basics.py`)

Start here to understand CNN fundamentals!

**What it does:**
- Trains a CNN to recognize handwritten digits (MNIST dataset)
- Shows how convolutional layers extract features from images
- Visualizes training progress and predictions

**Run it:**
```bash
python cnn_basics.py
```

**Expected output:**
- Training progress with accuracy metrics
- `cnn_basics_training.png` - Training and validation curves
- `cnn_basics_predictions.png` - Sample predictions on test images

**Key concepts demonstrated:**
- CNN architecture (Conv2D, MaxPooling, Dense layers)
- Image preprocessing (normalization)
- Training loop with validation
- Model evaluation

### 2. Climate Modeling Example (`cnn_climate_modeling.py`)

This is where CNNs meet climate science!

**What it does:**
- Predicts solar irradiance from synthetic climate maps
- Uses 3 input channels: temperature, cloud cover, and humidity
- Demonstrates how CNNs can learn spatial weather patterns
- Relevant to solar energy forecasting in Colombia

**Run it:**
```bash
python cnn_climate_modeling.py
```

**Expected output:**
- Model training with MAE (Mean Absolute Error) metrics
- `climate_cnn_training.png` - Training history
- `climate_cnn_predictions.png` - Climate maps and predictions
- `climate_cnn_scatter.png` - Predicted vs actual irradiance

**Key concepts demonstrated:**
- Multi-channel input processing (like RGB, but climate variables)
- Regression vs classification
- Synthetic data generation for learning
- Performance metrics for continuous predictions

## 🌍 Context: Solar Energy in Colombia

Colombia has significant solar energy potential due to its location near the equator. Key considerations:

- **Geographic Diversity**: Coastal regions, mountains, and plains have different solar patterns
- **Cloud Cover Variability**: Tropical climate means cloud patterns significantly affect solar irradiance
- **Daily Irradiance**: Colombia typically receives 4-6 kWh/m²/day on average
- **Applications**: Solar farms, grid planning, energy storage optimization

## 🔍 Understanding the Code

### CNN Architecture Explained

```python
# Example from climate model
Conv2D(32, (3,3))      # 32 filters scanning 3x3 regions
    ↓
MaxPooling2D(2,2)      # Reduce spatial size by 2x
    ↓
Conv2D(64, (3,3))      # 64 filters for complex patterns
    ↓
MaxPooling2D(2,2)      # Further reduction
    ↓
GlobalAveragePooling   # Aggregate spatial information
    ↓
Dense(128)             # Learn relationships
    ↓
Dense(1)               # Output: solar irradiance
```

### Why CNNs for Climate Data?

1. **Spatial Patterns**: Weather patterns have spatial structure (e.g., cloud formations)
2. **Local Features**: CNNs detect local patterns (storms, clear zones)
3. **Translation Invariance**: Same weather pattern at different locations
4. **Hierarchical Learning**: Low-level features → high-level patterns

## 📊 Interpreting Results

### Training Curves
- **Loss decreasing**: Model is learning
- **Validation similar to training**: Good generalization
- **Validation worse than training**: May need regularization (dropout, etc.)

### Performance Metrics
- **MAE (Mean Absolute Error)**: Average prediction error in kWh/m²/day
- **RMSE**: Penalizes larger errors more
- **R² Score**: Proportion of variance explained (closer to 1 is better)

## 🎯 Next Steps for Your Internship

1. **Experiment with the Code**:
   - Modify CNN architectures (add/remove layers)
   - Change hyperparameters (learning rate, batch size)
   - Try different activation functions

2. **Real Data Integration**:
   - Learn about ERA5 reanalysis data (climate data)
   - Explore GOES satellite imagery
   - Study Colombian meteorological data sources

3. **Advanced Topics**:
   - Recurrent Neural Networks (RNNs) for time-series
   - Attention mechanisms for spatial weighting
   - Transfer learning from pre-trained models
   - Ensemble methods

4. **Practical Applications**:
   - Day-ahead solar forecasting
   - Grid balancing optimization
   - Solar farm site selection
   - Energy storage planning

## 📖 Additional Resources

### Deep Learning & CNNs
- [Deep Learning Specialization by Andrew Ng](https://www.coursera.org/specializations/deep-learning)
- [CS231n: Convolutional Neural Networks for Visual Recognition](http://cs231n.stanford.edu/)
- [TensorFlow Tutorials](https://www.tensorflow.org/tutorials)

### Climate Modeling & Solar Energy
- [ERA5 Climate Reanalysis Data](https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-single-levels)
- [NOAA Solar Calculator](https://www.esrl.noaa.gov/gmd/grad/solcalc/)
- [IDEAM - Colombian Institute of Hydrology, Meteorology and Environmental Studies](http://www.ideam.gov.co/)

### Computer Vision in Climate Science
- [Climate Change AI](https://www.climatechange.ai/)
- [AI for Earth](https://www.microsoft.com/en-us/ai/ai-for-earth)

## 🤝 Contributing

Feel free to experiment and add your own examples! This is your learning environment.

## 📝 License

This is an educational project for internship learning purposes.

## 💡 Tips for Learning

1. **Run the code first**: See it work before diving deep
2. **Modify one thing at a time**: Understand cause and effect
3. **Visualize everything**: Use plots to understand what's happening
4. **Start simple**: Master basics before advanced concepts
5. **Ask questions**: Research what you don't understand

## 🔧 Troubleshooting

### Common Issues

**Issue**: `ImportError: No module named 'tensorflow'`
**Solution**: Run `pip install -r requirements.txt`

**Issue**: Training is very slow
**Solution**: CNNs can be slow on CPU. Consider using Google Colab for free GPU access

**Issue**: Out of memory errors
**Solution**: Reduce batch size or model size (fewer filters/layers)

## 📞 Getting Help

- Check TensorFlow documentation: https://www.tensorflow.org/
- Python community: https://stackoverflow.com/
- Discuss with your internship supervisor

---

**Happy Learning! 🚀☀️**

Welcome to the exciting intersection of deep learning and climate science!