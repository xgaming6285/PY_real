import json
import re

def extract_json_from_text(text):
    """
    Extract JSON object from text that may contain markdown or other formatting
    
    Args:
        text: Text containing JSON (potentially with markdown formatting)
        
    Returns:
        parsed_json: Parsed JSON object or None if parsing failed
        json_str: The extracted JSON string
    """
    try:
        # Try to directly parse the text as JSON first
        try:
            parsed_json = json.loads(text)
            return parsed_json, text
        except json.JSONDecodeError:
            # Continue with extraction methods
            pass
        
        # Check for JSON inside markdown code blocks
        if "```json" in text:
            # Extract content between ```json and ``` markers
            start = text.find("```json") + 7
            end = text.find("```", start)
            json_str = text[start:end].strip()
        elif "```" in text:
            # Extract content between ``` markers (generic code block)
            start = text.find("```") + 3
            end = text.find("```", start)
            json_str = text[start:end].strip()
        else:
            # Try to find JSON object using regex
            json_pattern = r'\{(?:[^{}]|(?:\{(?:[^{}]|(?:\{(?:[^{}])*\}))*\}))*\}'
            match = re.search(json_pattern, text)
            if match:
                json_str = match.group(0)
            else:
                json_str = text
        
        # Parse the extracted JSON string
        parsed_json = json.loads(json_str)
        return parsed_json, json_str
    
    except Exception as e:
        # Return raw text if JSON parsing fails
        return None, text

def format_json_for_display(json_str, add_color=True):
    """
    Format JSON string for better display in UI
    
    Args:
        json_str: JSON string to format
        add_color: Add syntax highlighting
        
    Returns:
        formatted_json: Formatted JSON string
    """
    try:
        # Parse the JSON
        parsed_json = json.loads(json_str)
        
        # Pretty print with indentation
        formatted_json = json.dumps(parsed_json, indent=2)
        
        return formatted_json
    
    except Exception:
        # Return original string if formatting fails
        return json_str

def validate_id_data(json_str):
    """
    Validate that ID data contains required fields
    
    Args:
        json_str: JSON string with ID data
        
    Returns:
        is_valid: Boolean indicating if data is valid
        message: Message about validation result
    """
    required_fields = ["document_type", "id_number", "full_name"]
    
    try:
        # Parse the JSON
        data, _ = extract_json_from_text(json_str)
        
        if data is None:
            return False, "Could not parse JSON data"
        
        # Check required fields
        missing_fields = [field for field in required_fields if not data.get(field)]
        
        if missing_fields:
            return False, f"Missing required fields: {', '.join(missing_fields)}"
        
        return True, "ID data is valid"
    
    except Exception as e:
        return False, f"Error validating ID data: {str(e)}" 