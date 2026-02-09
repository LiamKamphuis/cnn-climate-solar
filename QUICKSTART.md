# Quick Start Guide 🚀

Welcome to your first day! Here's how to get started in 5 minutes:

## Step 1: Install Dependencies (2 minutes)

```bash
pip install -r requirements.txt
```

This installs TensorFlow, NumPy, Matplotlib, and other necessary libraries.

## Step 2: Run Basic CNN Example (3 minutes)

```bash
python cnn_basics.py
```

**What you'll see:**
- Training progress showing how the CNN learns
- Two PNG images with visualizations:
  - `cnn_basics_training.png` - Training curves
  - `cnn_basics_predictions.png` - Sample predictions

**What you'll learn:**
- How CNNs process images layer by layer
- How training reduces loss and improves accuracy
- How to evaluate model predictions

## Step 3: Run Climate Modeling Example (5 minutes)

```bash
python cnn_climate_modeling.py
```

**What you'll see:**
- Model training on synthetic climate data
- Three PNG images:
  - `climate_cnn_training.png` - Loss and MAE over epochs
  - `climate_cnn_predictions.png` - Climate maps and predictions
  - `climate_cnn_scatter.png` - Predicted vs actual irradiance

**What you'll learn:**
- How CNNs can process multi-channel spatial data (temperature, clouds, humidity)
- How to predict continuous values (regression) vs classes (classification)
- How weather patterns affect solar irradiance in Colombia

## What Each File Does

| File | Purpose |
|------|---------|
| `cnn_basics.py` | Learn CNN fundamentals with simple digit classification |
| `cnn_climate_modeling.py` | Apply CNNs to solar energy prediction |
| `requirements.txt` | List of Python packages needed |
| `README.md` | Detailed documentation and learning resources |

## Next Steps

1. ✅ Run both examples to see them work
2. 📖 Read the code comments to understand what each part does
3. 🔧 Try modifying the code:
   - Change the number of epochs
   - Add/remove layers
   - Adjust learning rates
4. 📚 Read the full README.md for deeper understanding
5. 💬 Discuss with your supervisor about real climate data sources

## Troubleshooting

**Problem:** `ModuleNotFoundError`  
**Solution:** Run `pip install -r requirements.txt`

**Problem:** "Training is slow"  
**Solution:** Normal on CPU. Each example takes 2-5 minutes.

**Problem:** "Out of memory"  
**Solution:** Reduce batch size in the code (e.g., change `batch_size=128` to `batch_size=32`)

## Key Concepts for Your Internship

### Convolutional Neural Networks (CNNs)
- Specialized for spatial data (images, maps, grids)
- Extract features at multiple scales
- Translation invariant (same pattern anywhere in image)

### Climate Modeling for Solar Energy
- **Input:** Spatial maps of weather variables (temperature, clouds, humidity)
- **Output:** Solar irradiance prediction (kWh/m²/day)
- **Application:** Grid planning, energy storage, solar farm optimization

### Colombia Context
- Located near equator → high solar potential
- Variable cloud cover → needs accurate forecasting
- Growing renewable energy sector → practical applications

## Questions to Explore

1. Why do we use multiple convolutional layers?
2. How does cloud cover affect solar predictions?
3. What's the difference between classification and regression in CNNs?
4. How would real satellite data differ from our synthetic data?

---

**Need Help?** Check README.md or ask your supervisor!

**Ready to Code?** Start with `python cnn_basics.py` 🎯
