# Raspberry Pi Deployment Guide

## Hardware Requirements
- Raspberry Pi 4 (recommended: 4GB or 8GB RAM model)
- Raspbian OS (64-bit recommended)
- MicroSD card (32GB+ recommended)

## Installation Steps

1. **Set up Raspberry Pi**:
   ```bash
   # Update system
   sudo apt-get update && sudo apt-get upgrade
   
   # Install Python dependencies
   sudo apt-get install python3-pip python3-venv
   ```

2. **Create Virtual Environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements_rpi.txt
   ```

3. **Convert Model to TF Lite**:
   - Run the conversion script on your development machine
   - Transfer the `.tflite` model to Raspberry Pi

4. **Run Inference**:
   ```bash
   python3 inference.py
   ```

## Performance Notes
- First inference might be slower due to model loading
- Monitor system resources using `htop` or included utilities
- Consider using USB SSD instead of SD card for better I/O performance

## Troubleshooting
- If out of memory, reduce batch size
- Use `top` or `free -h` to monitor memory usage
- Check CPU temperature with `vcgencmd measure_temp`
