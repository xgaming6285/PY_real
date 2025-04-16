import streamlit as st
import cv2
import numpy as np
import google.generativeai as genai
from PIL import Image
import io
import time
import json
import os
from document_detection import detect_document, extract_document
from json_utils import extract_json_from_text, format_json_for_display, validate_id_data

# Configure the Google Generative AI API
API_KEY = "AIzaSyAcD4nCukDuiL3PLRwjwC3Qu_K0atpC20Y"
genai.configure(api_key=API_KEY)

# Initialize Gemini model
model = genai.GenerativeModel('gemini-1.5-pro-vision')

# Page config
st.set_page_config(
    page_title="ID Document Scanner", 
    page_icon="🆔",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load custom CSS
def local_css(file_name):
    with open(file_name, "r") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Apply CSS if the file exists
if os.path.exists("style.css"):
    local_css("style.css")

def process_id_document(image):
    """Process ID document using Gemini AI"""
    try:
        # Convert numpy array to PIL Image
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        # Create a byte stream
        byte_stream = io.BytesIO()
        pil_image.save(byte_stream, format='JPEG')
        byte_stream.seek(0)
        
        # Generate prompt for Gemini
        prompt = """
        Extract all information from this ID document image.
        Return the data in the following JSON format:
        {
            "document_type": "ID Card/Passport/Driver's License",
            "id_number": "",
            "full_name": "",
            "date_of_birth": "",
            "date_of_issue": "",
            "date_of_expiry": "",
            "nationality": "",
            "gender": "",
            "additional_info": {}
        }
        If any field is not visible or cannot be determined, leave it as an empty string.
        For additional_info, include any other fields present on the document.
        """
        
        # Call Gemini model
        response = model.generate_content([prompt, {"mime_type": "image/jpeg", "data": byte_stream.getvalue()}])
        
        # Extract JSON from response text
        _, json_str = extract_json_from_text(response.text)
        
        # Validate the extracted data
        is_valid, validation_message = validate_id_data(json_str)
        
        if not is_valid:
            return False, f"Invalid ID data: {validation_message}"
        
        # Format the JSON for display
        formatted_json = format_json_for_display(json_str)
        
        return True, formatted_json
    
    except Exception as e:
        return False, str(e)

def verify_face(id_image, face_image):
    """Verify if the face matches the ID document"""
    try:
        # Convert numpy arrays to PIL Images
        id_pil = Image.fromarray(cv2.cvtColor(id_image, cv2.COLOR_BGR2RGB))
        face_pil = Image.fromarray(cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB))
        
        # Create byte streams
        id_stream = io.BytesIO()
        face_stream = io.BytesIO()
        
        id_pil.save(id_stream, format='JPEG')
        face_pil.save(face_stream, format='JPEG')
        
        id_stream.seek(0)
        face_stream.seek(0)
        
        # Generate prompt for Gemini
        prompt = """
        Compare the person in these two images. The first image is from an ID document, and the second image is a live photo of a person.
        Verify if they are the same person and provide a confidence score between 0 and 100.
        Return your analysis in the following JSON format:
        {
            "same_person": true/false,
            "confidence_score": 85,
            "remarks": "Brief explanation of your decision"
        }
        """
        
        # Call Gemini model
        response = model.generate_content([
            prompt, 
            {"mime_type": "image/jpeg", "data": id_stream.getvalue()},
            {"mime_type": "image/jpeg", "data": face_stream.getvalue()}
        ])
        
        # Extract JSON from response text
        _, json_str = extract_json_from_text(response.text)
        
        # Format the JSON for display
        formatted_json = format_json_for_display(json_str)
        
        return True, formatted_json
    
    except Exception as e:
        return False, str(e)

def capture_id_document():
    """Capture ID document using advanced document detection"""
    st.title("ID Document Scanner")
    st.write("Please position your ID document in front of the camera.")
    
    # Create a container for the webcam feed for better styling
    webcam_container = st.container()
    with webcam_container:
        # Placeholders for video feed and buttons
        video_placeholder = st.empty()
        button_col1, button_col2, button_col3 = st.columns([1, 1, 1])
        capture_button = button_col2.button("📸 Capture", key="main_capture")
    
    # Initialize session state variables
    if "id_image" not in st.session_state:
        st.session_state.id_image = None
    if "document_detected" not in st.session_state:
        st.session_state.document_detected = False
    if "detection_counter" not in st.session_state:
        st.session_state.detection_counter = 0
    
    # Start video capture
    cap = cv2.VideoCapture(0)
    
    # Check if the webcam is opened correctly
    if not cap.isOpened():
        st.error("Could not open webcam. Please check your camera connection.")
        return None
    
    # Process frames until capture button is pressed
    while not capture_button:
        success, frame = cap.read()
        if not success:
            st.error("Failed to read from webcam")
            break
        
        # Apply document detection
        processed_frame, document_detected, document_coords = detect_document(frame)
        
        # Update the detection counter
        if document_detected:
            st.session_state.detection_counter += 1
        else:
            st.session_state.detection_counter = 0
        
        # If document is detected for several consecutive frames, indicate it's stable
        if st.session_state.detection_counter >= 10:
            st.session_state.document_detected = True
            
            # Add a "Ready to capture" text
            cv2.putText(processed_frame, "READY TO CAPTURE - Press the button", (10, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # When detected, extract and store the document
            extracted_doc = extract_document(frame, document_coords)
            if extracted_doc is not None:
                st.session_state.id_image = extracted_doc
        else:
            st.session_state.document_detected = False
        
        # Display the frame
        video_placeholder.image(cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB), 
                               channels="RGB", 
                               use_column_width=True,
                               caption="Live Camera Feed")
        
        # Re-check the button state
        if button_col2.button("📸 Capture", key="capture_id"):
            if st.session_state.document_detected and st.session_state.id_image is not None:
                break
            else:
                st.warning("Please align the ID document properly before capturing")
        
        # Small pause to reduce CPU usage
        time.sleep(0.1)
    
    # Release the capture
    cap.release()
    
    # Return the extracted document if available
    if st.session_state.document_detected and st.session_state.id_image is not None:
        return st.session_state.id_image
    else:
        # If capture button was pressed but no document was detected, return the raw frame
        return frame if success else None

def capture_face():
    """Capture face for verification"""
    st.title("Face Verification")
    st.write("Please position your face in front of the camera.")
    
    # Create a container for the webcam feed
    webcam_container = st.container()
    with webcam_container:
        # Placeholders for video feed and buttons
        video_placeholder = st.empty()
        button_col1, button_col2, button_col3 = st.columns([1, 1, 1])
        capture_button = button_col2.button("📸 Capture", key="main_face_capture")
    
    # Initialize face image in session state
    if "face_image" not in st.session_state:
        st.session_state.face_image = None
    
    # Start video capture
    cap = cv2.VideoCapture(0)
    
    # Check if the webcam is opened correctly
    if not cap.isOpened():
        st.error("Could not open webcam. Please check your camera connection.")
        return None
    
    # Process frames until capture button is pressed
    while not capture_button:
        success, frame = cap.read()
        if not success:
            st.error("Failed to read from webcam")
            break
        
        # Draw face guide oval
        height, width = frame.shape[:2]
        center = (int(width/2), int(height/2))
        axes = (int(width/4), int(height/3))
        cv2.ellipse(frame, center, axes, 0, 0, 360, (0, 255, 0), 2)
        
        # Add text guide
        cv2.putText(frame, "POSITION FACE WITHIN OVAL", (int(width*0.2), int(height*0.15)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Display the frame
        video_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), 
                               channels="RGB", 
                               use_column_width=True,
                               caption="Live Camera Feed")
        
        # Re-check the button state
        if button_col2.button("📸 Capture", key="capture_face"):
            st.session_state.face_image = frame.copy()
            break
        
        # Small pause to reduce CPU usage
        time.sleep(0.1)
    
    # Release the capture
    cap.release()
    
    # Process captured image if available
    if st.session_state.face_image is not None:
        return st.session_state.face_image
    else:
        return None

def render_success_message(message):
    """Render a styled success message"""
    st.markdown(f"""
    <div class="success-message">
        ✅ {message}
    </div>
    """, unsafe_allow_html=True)

def render_error_message(message):
    """Render a styled error message"""
    st.markdown(f"""
    <div class="error-message">
        ❌ {message}
    </div>
    """, unsafe_allow_html=True)

# Main app flow
def main():
    # Set up session state
    if "page" not in st.session_state:
        st.session_state.page = "id_scan"
    if "id_data" not in st.session_state:
        st.session_state.id_data = None
    if "id_image" not in st.session_state:
        st.session_state.id_image = None
    if "verification_result" not in st.session_state:
        st.session_state.verification_result = None
    
    # Page title and sidebar
    st.sidebar.title("ID Verification System")
    st.sidebar.markdown("""
    This application uses Gemini 2.5 Pro AI to extract information from ID documents and verify identity.
    
    ### Features
    - Automatic ID document detection
    - Data extraction using AI
    - Face verification
    """)
    
    # Display progress status in sidebar
    st.sidebar.subheader("Progress")
    if st.session_state.page == "id_scan":
        st.sidebar.markdown("✅ **Step 1: ID Scanning** (Current)")
        st.sidebar.markdown("⬜ Step 2: Data Review")
        st.sidebar.markdown("⬜ Step 3: Face Verification")
        st.sidebar.markdown("⬜ Step 4: Results")
    elif st.session_state.page == "show_data":
        st.sidebar.markdown("✅ Step 1: ID Scanning")
        st.sidebar.markdown("✅ **Step 2: Data Review** (Current)")
        st.sidebar.markdown("⬜ Step 3: Face Verification")
        st.sidebar.markdown("⬜ Step 4: Results")
    elif st.session_state.page == "face_verify":
        st.sidebar.markdown("✅ Step 1: ID Scanning")
        st.sidebar.markdown("✅ Step 2: Data Review")
        st.sidebar.markdown("✅ **Step 3: Face Verification** (Current)")
        st.sidebar.markdown("⬜ Step 4: Results")
    elif st.session_state.page == "result":
        st.sidebar.markdown("✅ Step 1: ID Scanning")
        st.sidebar.markdown("✅ Step 2: Data Review")
        st.sidebar.markdown("✅ Step 3: Face Verification")
        st.sidebar.markdown("✅ **Step 4: Results** (Current)")
    
    # ID Document Scanning Page
    if st.session_state.page == "id_scan":
        st.header("ID Document Scanner")
        st.write("Step 1: Scan your ID document")
        
        id_image = capture_id_document()
        
        if id_image is not None:
            # Create a container for the captured image
            image_container = st.container()
            with image_container:
                st.image(cv2.cvtColor(id_image, cv2.COLOR_BGR2RGB), 
                        caption="Captured ID Document",
                        use_column_width=True)
            
            with st.spinner("Processing ID document with Gemini AI..."):
                success, data = process_id_document(id_image)
            
            if success:
                render_success_message("ID document processed successfully!")
                st.session_state.id_data = data
                st.session_state.id_image = id_image
                st.session_state.page = "show_data"
                st.experimental_rerun()
            else:
                render_error_message(f"Failed to process ID document: {data}")
    
    # Display Extracted Data Page
    elif st.session_state.page == "show_data":
        st.header("Extracted ID Information")
        st.write("Step 2: Verify extracted information")
        
        # Create two columns for layout
        col1, col2 = st.columns([1, 1])
        
        # Display the ID image in the first column
        with col1:
            st.subheader("ID Document Image")
            st.image(cv2.cvtColor(st.session_state.id_image, cv2.COLOR_BGR2RGB), 
                    use_column_width=True,
                    caption="Captured ID Document")
        
        # Display the extracted data in the second column
        with col2:
            st.subheader("Extracted Information")
            st.json(st.session_state.id_data)
        
        # Navigation buttons
        st.write("")  # Add some space
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            if st.button("⬅️ Rescan ID"):
                st.session_state.page = "id_scan"
                st.experimental_rerun()
        
        with col3:
            if st.button("Continue to Face Verification ➡️"):
                st.session_state.page = "face_verify"
                st.experimental_rerun()
    
    # Face Verification Page
    elif st.session_state.page == "face_verify":
        st.header("Face Verification")
        st.write("Step 3: Verify your identity")
        
        face_image = capture_face()
        
        if face_image is not None:
            # Create a container for the captured face
            image_container = st.container()
            with image_container:
                st.image(cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB), 
                        caption="Captured Face",
                        use_column_width=True)
            
            with st.spinner("Verifying identity with Gemini AI..."):
                success, result = verify_face(st.session_state.id_image, face_image)
            
            if success:
                render_success_message("Identity verification completed!")
                st.session_state.verification_result = result
                st.session_state.page = "result"
                st.experimental_rerun()
            else:
                render_error_message(f"Failed to verify identity: {result}")
                
            col1, col2, col3 = st.columns([1, 1, 1])
            with col1:
                if st.button("⬅️ Back to ID Data"):
                    st.session_state.page = "show_data"
                    st.experimental_rerun()
    
    # Final Result Page
    elif st.session_state.page == "result":
        st.header("Verification Complete")
        st.write("Step 4: Review verification result")
        
        # Create card-like containers
        id_card = st.container()
        with id_card:
            st.markdown("### 📋 ID Information")
            
            # Create two columns for layout
            col1, col2 = st.columns([1, 2])
            
            # Display the ID image in the first column
            with col1:
                st.image(cv2.cvtColor(st.session_state.id_image, cv2.COLOR_BGR2RGB), 
                        use_column_width=True,
                        caption="ID Document")
            
            # Display the extracted data in the second column
            with col2:
                st.json(st.session_state.id_data)
        
        # Add some space
        st.write("")
        
        # Verification result card
        verify_card = st.container()
        with verify_card:
            st.markdown("### 🔍 Verification Result")
            st.json(st.session_state.verification_result)
        
        # Navigation button for starting over
        st.write("")  # Add some space
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col2:
            if st.button("🔄 Start Over"):
                # Reset session state
                st.session_state.page = "id_scan"
                st.session_state.id_data = None
                st.session_state.id_image = None
                st.session_state.verification_result = None
                st.experimental_rerun()

if __name__ == "__main__":
    main() 