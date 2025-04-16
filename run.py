"""
Launch script for ID Document Verification system

This script checks for required dependencies and launches the Streamlit application.
"""

import subprocess
import sys
import os

def check_python_version():
    """Check if Python version is compatible"""
    major, minor = sys.version_info[:2]
    if major == 3 and (7 <= minor <= 10 or minor >= 13):
        return True
    print(f"Warning: This application is designed for Python 3.7-3.10 or 3.13+, but you're using Python {major}.{minor}")
    print("You may encounter compatibility issues with some dependencies.")
    return False

def check_requirements():
    """Check if all required packages are installed"""
    try:
        import streamlit
        import cv2
        import numpy
        import google.generativeai
        import PIL
        return True
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Installing required packages...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
            return True
        except subprocess.CalledProcessError as error:
            print(f"Failed to install dependencies: {str(error)}")
            print("Please try installing them manually with: pip install -r requirements.txt")
            return False

def launch_app():
    """Launch the Streamlit application"""
    is_compatible = check_python_version()
    if not is_compatible:
        user_input = input("Continue anyway? (y/n): ")
        if user_input.lower() != 'y':
            print("Exiting application.")
            return
            
    if check_requirements():
        print("Starting ID Document Verification System...")
        subprocess.run(["streamlit", "run", "app.py", "--server.headless", "true"])
    else:
        print("Failed to start application due to missing dependencies.")

if __name__ == "__main__":
    launch_app() 