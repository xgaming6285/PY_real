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

# Initialize database manager
db_manager = DatabaseManager()

# Configure the Google Generative AI API
API_KEY = "AIzaSyAcD4nCukDuiL3PLRwjwC3Qu_K0atpC20Y"
genai.configure(api_key=API_KEY)

# Initialize Gemini model
model = genai.GenerativeModel('gemini-2.5-pro-exp-03-25')

# Initialize face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

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
        
        # Call Gemini model with image
        response = model.generate_content([
            prompt, 
            {"mime_type": "image/jpeg", "data": byte_stream.getvalue()}
        ])
        
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
            "confidence_score": ,
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
    """Capture ID document using Streamlit's camera input with real-time detection"""
    st.title("ID Document Scanner")
    st.write("Please position your ID document in front of the camera.")
    
    # Initialize session state variables
    if "id_image" not in st.session_state:
        st.session_state.id_image = None
    if "document_detected" not in st.session_state:
        st.session_state.document_detected = False
    if "detection_attempts" not in st.session_state:
        st.session_state.detection_attempts = 0
    if "auto_capture" not in st.session_state:
        st.session_state.auto_capture = False
        
    # Create a container for instructions
    instruction_container = st.container()
    with instruction_container:
        st.info("📱 Position your ID document within the camera frame.")
    
    # Add option for auto-capture
    st.session_state.auto_capture = st.checkbox("Enable automatic capture when document is detected", 
                                                value=st.session_state.auto_capture)
    
    # Use Streamlit's camera input
    camera_col1, camera_col2 = st.columns([3, 1])
    with camera_col1:
        camera_image = st.camera_input("ID Document Camera", key="id_document_camera")
    
    with camera_col2:
        st.write("Detection Status:")
        status_placeholder = st.empty()
        if st.session_state.document_detected:
            status_placeholder.success("Document Detected")
        else:
            status_placeholder.warning("Waiting for Document")
        
        # Manual capture button
        if not st.session_state.auto_capture:
            st.button("Capture Document", key="manual_capture_btn")
    
    if camera_image is not None:
        # Convert the image to a format we can process
        bytes_data = camera_image.getvalue()
        image = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
        
        # Apply document detection
        processed_frame, document_detected, document_coords = detect_document(image)
        
        # Show the processed frame with detection overlays
        st.image(cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB), 
                use_container_width=True,
                caption="Detection Result")
                
        # Update detection status
        if document_detected:
            status_placeholder.success("Document Detected")
            
            # Auto-capture if enabled
            if st.session_state.auto_capture:
                # Extract the document
                extracted_doc = extract_document(image, document_coords)
                if extracted_doc is not None:
                    st.session_state.id_image = extracted_doc
                    st.session_state.document_detected = True
                    st.success("✅ Document detected and captured automatically!")
                    return extracted_doc
        else:
            status_placeholder.warning("No Document Detected")
        
        # Manually capture current frame
        if st.button("Use This Frame", key="use_current_frame") or st.session_state.get("manual_capture_btn", False):
            if document_detected:
                # Extract the document
                extracted_doc = extract_document(image, document_coords)
                if extracted_doc is not None:
                    st.session_state.id_image = extracted_doc
                    st.session_state.document_detected = True
                    st.success("✅ Document captured successfully!")
                    return extracted_doc
                else:
                    st.warning("Document detected but could not be extracted properly. Please try again.")
                    return image  # Return the original image as fallback
            else:
                # Increment detection attempts
                st.session_state.detection_attempts += 1
                
                # Provide more helpful guidance after failed attempts
                if st.session_state.detection_attempts > 1:
                    st.warning("No document detected. Tips: Ensure good lighting, hold the ID still, and make sure all four corners are visible.")
                else:
                    st.warning("No document detected in the image. Please try again with better positioning.")
                
                # After multiple failed attempts, allow using the original image anyway
                if st.session_state.detection_attempts >= 3:
                    if st.button("Use this image anyway", key="force_use_image"):
                        st.session_state.id_image = image
                        st.session_state.document_detected = True
                        return image
                
                return image  # Return the original image
    
    return None

def capture_face():
    """Capture face for verification using Streamlit's camera input with real-time face detection"""
    st.title("Face Verification")
    st.write("Please position your face in front of the camera.")
    
    # Initialize face image in session state
    if "face_image" not in st.session_state:
        st.session_state.face_image = None
    if "face_detected" not in st.session_state:
        st.session_state.face_detected = False
    if "auto_capture_face" not in st.session_state:
        st.session_state.auto_capture_face = False
    
    # Create a container for instructions
    instruction_container = st.container()
    with instruction_container:
        st.info("📱 Position your face centered in the camera frame.")
    
    # Add option for auto-capture
    st.session_state.auto_capture_face = st.checkbox("Enable automatic capture when face is detected", 
                                                    value=st.session_state.auto_capture_face)
    
    # Use Streamlit's camera input
    camera_col1, camera_col2 = st.columns([3, 1])
    with camera_col1:
        camera_image = st.camera_input("Face Verification Camera", key="face_verification_camera")
    
    with camera_col2:
        st.write("Detection Status:")
        status_placeholder = st.empty()
        if st.session_state.face_detected:
            status_placeholder.success("Face Detected")
        else:
            status_placeholder.warning("Waiting for Face")
        
        # Manual capture button
        if not st.session_state.auto_capture_face:
            st.button("Capture Face", key="manual_capture_face_btn")
    
    if camera_image is not None:
        # Convert the image to a format we can process
        bytes_data = camera_image.getvalue()
        image = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
        
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        # Draw rectangles around faces
        processed_frame = image.copy()
        for (x, y, w, h) in faces:
            cv2.rectangle(processed_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Show the processed frame with face detection
        st.image(cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB), 
                use_container_width=True,
                caption="Face Detection Result")
        
        # Update detection status
        if len(faces) > 0:
            status_placeholder.success("Face Detected")
            st.session_state.face_detected = True
            
            # Auto-capture if enabled
            if st.session_state.auto_capture_face:
                st.session_state.face_image = image
                st.success("✅ Face detected and captured automatically!")
                return image
        else:
            status_placeholder.warning("No Face Detected")
            st.session_state.face_detected = False
        
        # Manually capture current frame
        if st.button("Use This Frame", key="use_current_face_frame") or st.session_state.get("manual_capture_face_btn", False):
            if len(faces) > 0:
                st.session_state.face_image = image
                st.success("✅ Face captured successfully!")
                return image
            else:
                st.warning("No face detected in the image. Please try again with better positioning.")
                return image
    
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
    if "verification_id" not in st.session_state:
        st.session_state.verification_id = None
    
    # Page title and sidebar
    st.sidebar.title("ID Verification System")
    st.sidebar.markdown("""
    This application uses Gemini 2.5 Pro AI to extract information from ID documents and verify identity.
    
    ### Features
    - Automatic ID document detection
    - Data extraction using AI
    - Face verification
    - Secure storage in MongoDB and S3
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
                        use_container_width=True)
            
            with st.spinner("Processing ID document with Gemini AI..."):
                success, data = process_id_document(id_image)
            
            if success:
                render_success_message("ID document processed successfully!")
                st.session_state.id_data = data
                st.session_state.id_image = id_image
                st.session_state.page = "show_data"
                st.rerun()
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
                    use_container_width=True,
                    caption="ID Document")
        
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
                st.rerun()
        
        with col3:
            if st.button("Continue to Face Verification ➡️"):
                st.session_state.page = "face_verify"
                st.rerun()
    
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
                        use_container_width=True)
            
            with st.spinner("Verifying identity with Gemini AI..."):
                success, result = verify_face(st.session_state.id_image, face_image)
            
            if success:
                render_success_message("Identity verification completed!")
                st.session_state.verification_result = result
                st.session_state.face_image = face_image
                st.session_state.page = "result"
                st.rerun()
            else:
                render_error_message(f"Failed to verify identity: {result}")
                
            col1, col2, col3 = st.columns([1, 1, 1])
            with col1:
                if st.button("⬅️ Back to ID Data"):
                    st.session_state.page = "show_data"
                    st.rerun()
    
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
                        use_container_width=True,
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
        
        # Save verification data to database
        if st.session_state.verification_id is None:
            with st.spinner("Saving verification data to database..."):
                doc_id = db_manager.store_verification_data(
                    st.session_state.id_data,
                    st.session_state.verification_result,
                    st.session_state.id_image,
                    st.session_state.face_image
                )
                if doc_id:
                    st.session_state.verification_id = doc_id
                    render_success_message(f"Data saved to database. Verification ID: {doc_id}")
                else:
                    render_error_message("Failed to save data to database.")
        else:
            st.success(f"Data saved to database. Verification ID: {st.session_state.verification_id}")
        
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
                st.session_state.verification_id = None
                st.rerun()

if __name__ == "__main__":
    main() 
