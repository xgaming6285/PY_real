# ID Document Verification System

A Python application that uses Gemini 2.5 Pro AI to extract data from ID documents and verify a person's identity.

## Features

1. **Automatic ID Document Detection**: Opens camera feed and automatically detects the ID document borders from the live video.
2. **Data Extraction**: Extracts personal information from the ID document using Gemini 2.5 Pro.
3. **Identity Verification**: Verifies the person's identity by comparing their face with the photo on the ID document.
4. **User-Friendly Interface**: Simple, intuitive UI built with Streamlit.

## Requirements

- Python 3.7-3.10 or Python 3.13+
- Webcam
- Internet connection (for Gemini API)

## Installation

### Automatic Installation

1. Clone this repository:
   ```
   git clone <repository-url>
   cd <repository-directory>
   ```

2. Run the application:
   - **Windows**: Double-click on `run_app.bat`
   - **macOS/Linux**: Run `python run.py` from your terminal

The script will automatically check for and install required dependencies.

### Manual Installation

1. Clone this repository:
   ```
   git clone <repository-url>
   cd <repository-directory>
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

3. Run the Streamlit app:
   ```
   streamlit run app.py
   ```

   **Note for Python 3.13 users:** The app has been updated to work with Python 3.13. If you encounter any issues, please ensure you have the latest pip version by running `python -m pip install --upgrade pip` before installing the dependencies.

## Usage

1. The application will open in your default web browser.
2. Follow the on-screen instructions:
   - Position your ID document in front of the camera
   - Wait for auto-detection (green border)
   - Review the extracted information
   - Proceed to face verification
   - View the final verification result

## How It Works

### ID Document Detection

The application uses OpenCV to detect the edges of ID documents in real-time. When a rectangular document is detected for several consecutive frames (to ensure stability), it is highlighted with a green border, and you can capture it.

### Data Extraction

The captured ID image is sent to Google's Gemini 2.5 Pro AI model, which extracts structured information like:
- Document type
- ID number
- Full name
- Date of birth
- Gender
- And other fields present on the document

### Face Verification

The application then activates the camera again for face verification. It captures your face and sends both the ID document and face image to Gemini AI for comparison. The AI returns a confidence score and remarks about the match.

## Troubleshooting

### Camera Issues

- Ensure your webcam is properly connected and not in use by another application
- Provide camera permissions to your browser when prompted
- If using a laptop with built-in webcam, ensure it's not disabled in your system settings

### ID Detection Problems

- Ensure good lighting conditions
- Position the ID on a contrasting background
- Keep the ID flat and avoid glare
- Try moving closer or farther from the camera

### Gemini API Issues

- Check your internet connection
- Verify the API key is valid
- If encountering rate limits, try again later

## Technical Details

- **Computer Vision**: OpenCV for camera access and document detection
- **AI/ML**: Gemini 2.5 Pro for document information extraction and face verification
- **Frontend**: Streamlit for the user interface
- **Language**: Python 3.7+

## Notes

- The API key provided in the code is for demonstration purposes only. In a production environment, you should use a secure method to store and retrieve API keys.
- This application is for educational purposes and should not be used for official identity verification without proper authorization. 