import os
import io
import json
from datetime import datetime
from pymongo import MongoClient
import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv
import streamlit as st

# Load environment variables
load_dotenv()

# Load credentials from Streamlit secrets or env variables
try:
    # First try to get from Streamlit secrets
    MONGO_URI = st.secrets["MONGO_URI"]
    AWS_ACCESS_KEY = st.secrets["AWS_ACCESS_KEY"]
    AWS_SECRET_KEY = st.secrets["AWS_SECRET_KEY"]
    S3_BUCKET_NAME = st.secrets["S3_BUCKET_NAME"]
    print("Using credentials from Streamlit secrets")
except (KeyError, AttributeError):
    # Fall back to environment variables if not in Streamlit secrets
    MONGO_URI = os.getenv('MONGO_URI')
    AWS_ACCESS_KEY = os.getenv('AWS_ACCESS_KEY')
    AWS_SECRET_KEY = os.getenv('AWS_SECRET_KEY')
    S3_BUCKET_NAME = os.getenv('S3_BUCKET_NAME')
    print("Using credentials from environment variables")

class DatabaseManager:
    def __init__(self):
        # Initialize MongoDB connection
        try:
            self.client = MongoClient(MONGO_URI)
            self.db = self.client['verification_data']
            self.verifications = self.db['verifications']
            print("MongoDB connection successful")
        except Exception as e:
            print(f"MongoDB connection error: {e}")
            self.client = None
            self.db = None
            self.verifications = None
        
        # Initialize S3 connection
        try:
            self.s3_client = boto3.client(
                's3',
                aws_access_key_id=AWS_ACCESS_KEY,
                aws_secret_access_key=AWS_SECRET_KEY
            )
            print("S3 connection successful")
        except Exception as e:
            print(f"S3 connection error: {e}")
            self.s3_client = None
    
    def store_verification_data(self, id_data, verification_result, id_image=None, face_image=None):
        """
        Store verification data in MongoDB and images in S3
        
        Args:
            id_data (dict): Extracted ID document data
            verification_result (dict): Face verification result
            id_image (numpy.ndarray): ID document image
            face_image (numpy.ndarray): Face image
            
        Returns:
            str: Document ID if successful, None otherwise
        """
        try:
            # Create record with timestamp
            record = {
                "id_data": json.loads(id_data) if isinstance(id_data, str) else id_data,
                "verification_result": json.loads(verification_result) if isinstance(verification_result, str) else verification_result,
                "timestamp": datetime.utcnow(),
                "id_image_url": None,
                "face_image_url": None
            }
            
            # Insert record to get document ID
            result = self.verifications.insert_one(record)
            doc_id = str(result.inserted_id)
            
            # Update URLs if images are provided
            if id_image is not None and self.s3_client:
                id_image_url = self._upload_image_to_s3(id_image, f"id_{doc_id}.jpg")
                if id_image_url:
                    self.verifications.update_one(
                        {"_id": result.inserted_id},
                        {"$set": {"id_image_url": id_image_url}}
                    )
            
            if face_image is not None and self.s3_client:
                face_image_url = self._upload_image_to_s3(face_image, f"face_{doc_id}.jpg")
                if face_image_url:
                    self.verifications.update_one(
                        {"_id": result.inserted_id},
                        {"$set": {"face_image_url": face_image_url}}
                    )
            
            return doc_id
            
        except Exception as e:
            print(f"Error storing verification data: {e}")
            return None
    
    def _upload_image_to_s3(self, image, key):
        """
        Upload image to S3 bucket
        
        Args:
            image (numpy.ndarray): Image to upload
            key (str): S3 object key
            
        Returns:
            str: S3 URL if successful, None otherwise
        """
        import cv2
        from PIL import Image as PILImage
        import numpy as np
        
        try:
            # Convert numpy array to PIL Image
            if isinstance(image, np.ndarray):
                pil_image = PILImage.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            else:
                return None
            
            # Create byte stream
            img_byte_arr = io.BytesIO()
            pil_image.save(img_byte_arr, format='JPEG')
            img_byte_arr.seek(0)
            
            # Upload to S3
            self.s3_client.upload_fileobj(
                img_byte_arr,
                S3_BUCKET_NAME,
                key,
                ExtraArgs={'ContentType': 'image/jpeg'}
            )
            
            # Generate URL
            url = f"https://{S3_BUCKET_NAME}.s3.amazonaws.com/{key}"
            return url
            
        except Exception as e:
            print(f"Error uploading image to S3: {e}")
            return None
    
    def get_verification_by_id(self, doc_id):
        """
        Retrieve verification data by document ID
        
        Args:
            doc_id (str): Document ID
            
        Returns:
            dict: Verification data if found, None otherwise
        """
        from bson.objectid import ObjectId
        
        try:
            result = self.verifications.find_one({"_id": ObjectId(doc_id)})
            return result
        except Exception as e:
            print(f"Error retrieving verification data: {e}")
            return None
    
    def close(self):
        """Close database connections"""
        if self.client:
            self.client.close() 
