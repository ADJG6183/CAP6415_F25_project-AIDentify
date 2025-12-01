# AIDentify User Interface Guide

AIDentify provides **two different user interfaces** to make AI-generated image detection accessible to everyone:

1. **Web UI (Gradio)** - Modern, browser-based interface
2. **Desktop UI (Tkinter)** - Native desktop application

## 🌐 Web UI (Recommended)

### Features
- ✨ Modern, intuitive interface
- 📊 Visual detection results with charts
- 📁 Batch processing support
- 🌍 Access from any device with a browser
- 📱 Mobile-friendly

### Quick Start

```bash
# Install dependencies (if not already done)
pip install -r requirements.txt

# Run the web interface
python app.py
```

The web interface will automatically open in your browser at `http://localhost:7860`

### Web UI Screenshots & Usage

#### Single Image Detection Tab

1. **Upload an Image**: Click the image upload area or drag & drop
2. **Configure Settings**:
   - **Detection Method**: Choose how to combine detectors
     - `weighted_average`: Best for general use (default)
     - `ml_model`: Uses trained ML model (requires training first)
     - `voting`: Majority vote among detectors
   - **Threshold**: Adjust sensitivity (0.5 is default)
   - **Use CNN**: Enable for best accuracy (slower)
3. **Click "Analyze Image"**
4. **View Results**:
   - Verdict (AI-Generated or Real)
   - Probability and confidence scores
   - Visualization charts
   - Individual detector breakdowns

#### Batch Processing Tab

Perfect for analyzing multiple images at once:

1. **Upload Multiple Images**: Select multiple files at once
2. **Configure Detection Settings**: Same as single image
3. **Click "Analyze All Images"**
4. **View Results Table**: Shows verdict, probability, and confidence for each image

#### About Tab

- Learn about detection methods
- Understand the science behind AIDentify
- See performance metrics
- Get training instructions

### Advanced Options

#### Share Your Interface

To create a public link (accessible from anywhere):

```python
# Edit app.py, change this line:
demo.launch(
    share=True,  # Creates a public URL
    server_name="0.0.0.0",
    server_port=7860
)
```

#### Custom Port

```bash
# Run on a different port
python app.py  # Default: 7860
```

Edit `server_port` in `app.py` to change the default port.

---

## 🖥️ Desktop UI (Alternative)

### Features
- 💻 Native desktop application
- 🚀 No browser required
- 🎨 Traditional desktop interface
- 📦 Standalone application

### Quick Start

```bash
# Install dependencies (if not already done)
pip install -r requirements.txt

# Run the desktop application
python app_desktop.py
```

A desktop window will open with the AIDentify interface.

### Desktop UI Usage

#### Main Interface

The desktop UI has two main panels:

**Left Panel - Image Display**
- 📁 Load Image: Select an image file
- 🔍 Detect: Run detection on loaded image
- 🗑️ Clear: Remove current image

**Right Panel - Settings & Results**

**Settings:**
- Detection Method dropdown
- Threshold slider (0.0 - 1.0)
- Use CNN checkbox

**Results:**
- Large verdict display (AI-Generated or Real)
- Detailed breakdown of:
  - Probability
  - Confidence
  - Individual detector scores
  - Method used

#### Workflow

1. Click **"Load Image"** and select an image file
2. Adjust settings if desired
3. Click **"Detect"** to analyze
4. View results in the right panel
5. Load another image or clear to start over

### Tips for Desktop UI

- The interface runs detection in a background thread, so it won't freeze
- A progress bar shows when analysis is running
- Status bar at bottom shows current operation
- Results are scrollable if long

---

## 🔄 Comparison: Web UI vs Desktop UI

| Feature | Web UI | Desktop UI |
|---------|--------|------------|
| Interface | Modern, browser-based | Traditional desktop |
| Batch Processing | ✅ Yes | ❌ No |
| Visualizations | ✅ Charts & graphs | ⚠️ Text only |
| Mobile Access | ✅ Yes | ❌ No |
| Installation | Requires Gradio | Built-in (Tkinter) |
| Resource Usage | Slightly higher | Lower |
| Best For | General use, multiple images | Quick single image checks |

**Recommendation**: Use the **Web UI** for most cases. It's more feature-rich and easier to use.

---

## 🎯 Usage Examples

### Example 1: Quick Check (Web UI)

```bash
# Start web interface
python app.py

# In browser:
# 1. Upload image
# 2. Click "Analyze Image"
# 3. See immediate results
```

### Example 2: Batch Analysis (Web UI)

```bash
python app.py

# In browser - Batch Detection tab:
# 1. Upload multiple images
# 2. Configure settings
# 3. Click "Analyze All Images"
# 4. Get a results table
```

### Example 3: High-Accuracy Detection (Desktop UI)

```bash
python app_desktop.py

# In application:
# 1. Load image
# 2. Enable "Use CNN Detector"
# 3. Click Detect
# 4. Wait for results (may take longer)
```

---

## ⚙️ Configuration Tips

### For Best Accuracy

1. **Train Models First**:
   ```bash
   python src/train.py --data_dir data --model_type both --epochs 50
   ```

2. **Use These Settings**:
   - Detection Method: `weighted_average` or `ml_model`
   - Enable CNN: ✅ Yes
   - Threshold: 0.5 (adjust based on your needs)

### For Fastest Speed

1. **Use These Settings**:
   - Detection Method: `weighted_average`
   - Enable CNN: ❌ No
   - Threshold: 0.5

This gives ~70% accuracy but runs very fast.

### For Balanced Performance

1. **Use These Settings**:
   - Detection Method: `ml_model`
   - Enable CNN: ❌ No (or ✅ Yes if you have trained model)
   - Threshold: 0.5

This gives 75-85% accuracy with good speed.

---

## 🐛 Troubleshooting

### Web UI Won't Start

**Problem**: `ModuleNotFoundError: No module named 'gradio'`

**Solution**:
```bash
pip install gradio>=4.0.0
```

**Problem**: Port already in use

**Solution**: Edit `app.py` and change `server_port=7860` to another port like `7861`

### Desktop UI Issues

**Problem**: Window doesn't display correctly

**Solution**: Tkinter issues vary by OS. Try:
```bash
# macOS
brew install python-tk

# Ubuntu/Debian
sudo apt-get install python3-tk

# Windows
# Tkinter usually included, try reinstalling Python
```

**Problem**: Detection is slow

**Solution**:
- Disable CNN detector (uncheck the checkbox)
- Close other applications
- Consider using Web UI instead

### Both UIs

**Problem**: "No trained models found" warning

**Solution**: Either:
1. Train models: `python src/train.py --data_dir data --model_type both`
2. Use heuristic mode (works without training, ~70% accuracy)

**Problem**: Out of memory errors

**Solution**:
- Disable CNN detector
- Reduce image size before uploading
- Close other applications

---

## 📚 Advanced Usage

### Running Web UI on a Server

To make the web UI accessible from other computers on your network:

```python
# In app.py
demo.launch(
    share=False,
    server_name="0.0.0.0",  # Listen on all interfaces
    server_port=7860,
    show_error=True
)
```

Then access from other devices using: `http://YOUR_SERVER_IP:7860`

### Customizing the Web UI

Edit `app.py` to:
- Change theme: `gr.themes.Soft()` → `gr.themes.Base()` or `gr.themes.Glass()`
- Modify default settings
- Add custom tabs
- Change colors and layout

### Creating a Standalone Desktop App

To create an executable:

```bash
# Install PyInstaller
pip install pyinstaller

# Create standalone executable
pyinstaller --onefile --windowed app_desktop.py
```

The executable will be in the `dist/` folder.

---

## 💡 Tips & Best Practices

### General Tips

1. **Use High-Quality Images**: Better input = better results
2. **Test Multiple Images**: One result isn't definitive
3. **Adjust Threshold**: Lower for more detections, higher for fewer false positives
4. **Train on Your Data**: For best results, train on images similar to what you'll analyze

### For Web UI

1. **Batch Processing**: Use batch mode for analyzing entire folders
2. **Bookmark**: Add the local URL to bookmarks for easy access
3. **Mobile**: Works great on tablets and phones
4. **Share Carefully**: Only enable `share=True` if you need public access

### For Desktop UI

1. **Keyboard**: Use Tab to navigate between controls
2. **Multiple Windows**: Can run multiple instances for parallel analysis
3. **Logs**: Check terminal output for detailed information

---

## 🎓 Learning Resources

- **README.md**: Overview and quick start
- **IMPLEMENTATION_GUIDE.md**: Technical details and principles
- **notebooks/demo_detection.ipynb**: Interactive Python examples
- **example_usage.py**: Programmatic usage examples

---

## 🆘 Getting Help

If you encounter issues:

1. Check this guide's Troubleshooting section
2. Review README.md and IMPLEMENTATION_GUIDE.md
3. Check terminal/console output for error messages
4. Open an issue on GitHub with:
   - What you were trying to do
   - Error message (if any)
   - Your OS and Python version

---

## 🚀 Quick Command Reference

```bash
# Web UI (recommended)
python app.py

# Desktop UI
python app_desktop.py

# Command-line detection
python detect.py --image photo.jpg

# Train models
python src/train.py --data_dir data --model_type both

# Evaluate models
python src/evaluate.py --data_dir data/test

# Run examples
python example_usage.py
```

---

**Choose your interface and start detecting AI-generated images today!** 🎯
