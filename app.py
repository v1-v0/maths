import streamlit as st
import ollama
from PIL import Image
import numpy as np
import cv2
import re
import io
import time

# Configuration parameters
CONFIG = {
    "primary_model": "llama3.2-vision",
    "fallback_models": ["llama3-vision", "llama2-vision"],
    "temperature": 0.1,
    "max_tokens": 1500,
    "ollama_base_url": "http://localhost:11434",
    "image_types": ['png', 'jpg', 'jpeg'],
    "max_file_size_mb": 5,
    "preprocessing_enabled": True
}

# Extraction prompts
EXTRACTION_PROMPTS = {
    "standard": r"""
Extract all text and mathematical expressions from this image.

For text: Extract exactly as written.
For math: Convert to LaTeX using \( \) for inline math.

Important:
- Include the COMPLETE equation, capturing all terms and operations
- Maintain the exact structure and sequence of the content
- Do not add any content not present in the image
- Do not simplify or modify equations
""",
    
    "example_based": r"""
Extract text and convert math to LaTeX from this image.

Examples of expected output:
Input: "The expression cos(mx + nx) is equal to"
Output: "The expression \(\cos(mx + nx)\) is equal to"

Input: "2 cos mx cos nx"
Output: "2 \(\cos mx \cos nx\)"

Now extract from the provided image, ensuring ALL mathematical expressions are completely captured.
""",
    
    "detailed": r"""
Extract ALL text and mathematical expressions from this image with these specific requirements:

1. Extract ALL visible text exactly as written (questions, options, etc.)
2. For mathematical expressions:
   - Use proper LaTeX notation: \(\cos\) instead of cos, \(\sin\) instead of sin
   - Capture complete expressions including ALL terms and operations
   - Preserve the exact structure (fractions, exponents, etc.)

3. Format requirements:
   - Use \( and \) for inline math
   - Maintain the original layout and sequence
   - Include ALL parts of equations, especially + and - terms

4. DO NOT:
   - Add explanations or text not in the image
   - Simplify or modify equations
   - Skip any part of the content
   - Generate content not present in the image

This is for a math education application where complete and accurate extraction is critical.
"""
}

# Page configuration
st.set_page_config(
    page_title="Enhanced Math Equation Extractor",
    page_icon="🦙",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Helper functions
def preprocess_image(image):
    """Preprocess image to improve OCR quality"""
    if not CONFIG["preprocessing_enabled"]:
        return image, None
    
    # Convert to numpy array if needed
    if not isinstance(image, np.ndarray):
        image_np = np.array(image)
    else:
        image_np = image
    
    # Create a copy of the original image for processing
    processed_np = image_np.copy()
    
    # Convert to grayscale if it's a color image
    if len(processed_np.shape) == 3 and processed_np.shape[2] == 3:
        gray = cv2.cvtColor(processed_np, cv2.COLOR_RGB2GRAY)
    else:
        gray = processed_np
    
    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 11, 2
    )
    
    # Denoise
    denoised = cv2.fastNlMeansDenoising(thresh, None, 10, 7, 21)
    
    # Convert back to PIL Image
    processed_image = Image.fromarray(denoised)
    
    # Also return the intermediate processing steps for debugging
    processing_steps = {
        "grayscale": Image.fromarray(gray),
        "threshold": Image.fromarray(thresh),
        "denoised": processed_image
    }
    
    return processed_image, processing_steps

def check_image_quality(image):
    """Check if image meets quality standards for OCR"""
    # Convert to numpy array if needed
    if not isinstance(image, np.ndarray):
        image = np.array(image)
    
    # Check resolution
    height, width = image.shape[:2]
    if width < 300 or height < 300:
        return False, "Image resolution too low for reliable extraction"
    
    # Check contrast
    if len(image.shape) == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    
    min_val, max_val, _, _ = cv2.minMaxLoc(gray)
    contrast = max_val - min_val
    
    if contrast < 30:
        return False, "Image contrast too low for reliable extraction"
    
    return True, "Image quality acceptable"

def validate_extraction(extracted_text, expected_patterns=None):
    """Validate the extracted content meets quality standards"""
    issues = []
    
    # Check for common indicators of incomplete extraction
    if "..." in extracted_text or "[...]" in extracted_text:
        issues.append("Incomplete extraction detected")
    
    # Check for balanced delimiters
    if extracted_text.count("\\(") != extracted_text.count("\\)"):
        issues.append("Unbalanced LaTeX delimiters")
    
    # Check for expected content patterns if provided
    if expected_patterns:
        for pattern, description in expected_patterns:
            if not re.search(pattern, extracted_text):
                issues.append(f"Expected pattern not found: {description}")
    
    return issues

def assess_extraction_confidence(extracted_text, original_image):
    """Assess confidence in the extraction results"""
    confidence = "high"  # Default
    warnings = []
    
    # Check for indicators of low confidence
    if "..." in extracted_text or "[...]" in extracted_text:
        confidence = "low"
        warnings.append("Incomplete extraction detected")
    
    if extracted_text.count("\\(") != extracted_text.count("\\)"):
        confidence = "medium" if confidence == "high" else "low"
        warnings.append("Unbalanced LaTeX delimiters")
    
    # Check for very short output relative to image complexity
    if len(extracted_text) < 100 and original_image.size[0] * original_image.size[1] > 250000:
        confidence = "medium" if confidence == "high" else "low"
        warnings.append("Extraction seems shorter than expected for image complexity")
    
    return confidence, warnings

def extract_with_fallback(image_data, prompt_type="standard"):
    """Try extraction with primary model, fall back to alternatives if needed"""
    primary_model = CONFIG["primary_model"]
    fallback_models = CONFIG["fallback_models"]
    
    # Get the appropriate prompt
    prompt_content = EXTRACTION_PROMPTS.get(prompt_type, EXTRACTION_PROMPTS["standard"])
    
    # Try primary model first
    try:
        response = ollama.chat(
            model=primary_model,
            messages=[{
                'role': 'user',
                'content': prompt_content,
                'images': [image_data]
            }],
            options={
                "temperature": CONFIG["temperature"],
                "num_predict": CONFIG["max_tokens"]
            }
        )
        content = response.message.content
        
        # Basic validation
        if content and len(content) > 20:  # Arbitrary minimum length
            return content, None, primary_model
    except Exception as e:
        st.warning(f"Primary model failed: {str(e)}")
    
    # If primary fails, try fallbacks
    for model in fallback_models:
        try:
            st.info(f"Trying fallback model: {model}")
            response = ollama.chat(
                model=model,
                messages=[{
                    'role': 'user',
                    'content': prompt_content,
                    'images': [image_data]
                }],
                options={
                    "temperature": CONFIG["temperature"],
                    "num_predict": CONFIG["max_tokens"]
                }
            )
            content = response.message.content
            
            # Basic validation
            if content and len(content) > 20:  # Arbitrary minimum length
                return content, None, model
        except Exception as e:
            st.warning(f"Fallback model {model} failed: {str(e)}")
    
    # If all models fail
    return None, "All extraction attempts failed", None

def extract_with_verification(image_data, prompt_type="standard"):
    """Two-pass approach for complex equations"""
    # Get the appropriate prompt
    prompt_content = EXTRACTION_PROMPTS.get(prompt_type, EXTRACTION_PROMPTS["standard"])
    
    try:
        # First pass - basic extraction
        first_response = ollama.chat(
            model=CONFIG["primary_model"],
            messages=[{
                'role': 'user',
                'content': prompt_content,
                'images': [image_data]
            }],
            options={
                "temperature": CONFIG["temperature"],
                "num_predict": CONFIG["max_tokens"]
            }
        )
        
        first_content = first_response.message.content
        
        # Second pass - verification with first result as context
        second_response = ollama.chat(
            model=CONFIG["primary_model"],
            messages=[
                {
                    'role': 'user',
                    'content': prompt_content,
                    'images': [image_data]
                },
                {
                    'role': 'assistant',
                    'content': first_content
                },
                {
                    'role': 'user',
                    'content': "Verify your extraction is complete. Check for: 1) Missing equation terms 2) Complete expressions 3) Correct LaTeX syntax. If you find any issues, provide the corrected full extraction."
                }
            ],
            options={
                "temperature": CONFIG["temperature"],
                "num_predict": CONFIG["max_tokens"]
            }
        )
        
        # Use the second response if it's substantially different and longer
        second_content = second_response.message.content
        
        # If the second response is just confirmation without full content
        if len(second_content) < len(first_content) * 0.8 or "looks good" in second_content.lower() or "correct" in second_content.lower():
            return first_content, None, "single_pass"
        else:
            # Extract just the corrected content, not the explanation
            # This is a simple heuristic and might need refinement
            if "here is the corrected" in second_content.lower():
                parts = second_content.split("here is the corrected", 1, re.IGNORECASE)
                if len(parts) > 1:
                    second_content = parts[1].strip()
            
            return second_content, None, "two_pass"
    
    except Exception as e:
        return None, str(e), None

def load_and_validate_image(uploaded_file):
    """Load and validate the uploaded image"""
    if uploaded_file is None:
        return None, None
    
    try:
        image = Image.open(uploaded_file)
        is_valid, message = check_image_quality(image)
        if not is_valid:
            st.warning(message)
        return image, uploaded_file.getvalue()
    except Exception as e:
        st.error(f"Error loading image: {str(e)}")
        return None, None

# Title and description in main area
st.title("🦙 Enhanced Math Equation Extractor")

# Add clear button to top right
col1, col2 = st.columns([6,1])
with col2:
    if st.button("Clear 🗑️"):
        if 'ocr_result' in st.session_state:
            del st.session_state['ocr_result']
        if 'extraction_method' in st.session_state:
            del st.session_state['extraction_method']
        if 'model_used' in st.session_state:
            del st.session_state['model_used']
        if 'processing_time' in st.session_state:
            del st.session_state['processing_time']
        if 'confidence' in st.session_state:
            del st.session_state['confidence']
        if 'warnings' in st.session_state:
            del st.session_state['warnings']
        st.rerun()

st.markdown('<p style="margin-top: -20px;">Extract text and LaTeX code from images using Llama Vision models!</p>', unsafe_allow_html=True)

st.markdown("---")

# Sidebar for settings and upload
with st.sidebar:
    st.header("Upload Image")
    uploaded_file = st.file_uploader("Choose an image...", type=CONFIG["image_types"])
    
    st.header("Extraction Settings")
    
    extraction_method = st.radio(
        "Extraction Method",
        ["Standard", "Example-Based", "Detailed", "Two-Pass Verification"],
        index=3,
        help="Standard: Basic extraction. Example-Based: Uses examples to guide extraction. Detailed: More comprehensive instructions. Two-Pass: Uses verification for complex equations."
    )
    
    # Map the radio selection to prompt types
    prompt_mapping = {
        "Standard": "standard",
        "Example-Based": "example_based",
        "Detailed": "detailed",
        "Two-Pass Verification": "detailed"  # Use detailed prompt for two-pass
    }
    
    selected_prompt = prompt_mapping[extraction_method]
    
    # Preprocessing options
    preprocessing_enabled = st.checkbox("Enable Image Preprocessing", value=CONFIG["preprocessing_enabled"])
    CONFIG["preprocessing_enabled"] = preprocessing_enabled
    
    # Advanced settings in expander
    with st.expander("Advanced Settings"):
        CONFIG["temperature"] = st.slider("Temperature", min_value=0.0, max_value=1.0, value=CONFIG["temperature"], step=0.1)
        CONFIG["max_tokens"] = st.slider("Max Tokens", min_value=500, max_value=4000, value=CONFIG["max_tokens"], step=100)
        
        # Model selection
        available_models = [CONFIG["primary_model"]] + CONFIG["fallback_models"]
        selected_model = st.selectbox("Primary Model", available_models, index=0)
        CONFIG["primary_model"] = selected_model
    
    if uploaded_file is not None:
        # Display the uploaded image
        image, image_data = load_and_validate_image(uploaded_file)
        if image is not None:
            st.image(image, caption="Uploaded Image", use_container_width=True)
            
            # Process button
            if st.button("Extract Content 🔍", type="primary"):
                with st.spinner("Processing image..."):
                    start_time = time.time()
                    
                    # Preprocess image if enabled
                    if CONFIG["preprocessing_enabled"]:
                        processed_image, processing_steps = preprocess_image(image)
                        
                        # Show preprocessing steps in expander
                        with st.expander("View Preprocessing Steps"):
                            st.image(processing_steps["grayscale"], caption="Grayscale", use_container_width=True)
                            st.image(processing_steps["threshold"], caption="Thresholded", use_container_width=True)
                            st.image(processing_steps["denoised"], caption="Denoised", use_container_width=True)
                        
                        # Convert processed image back to bytes for API
                        img_byte_arr = io.BytesIO()
                        processed_image.save(img_byte_arr, format=image.format if image.format else 'PNG')
                        image_data = img_byte_arr.getvalue()
                    
                    try:
                        # Use two-pass verification if selected
                        if extraction_method == "Two-Pass Verification":
                            extracted_content, error, method = extract_with_verification(image_data, selected_prompt)
                            if error:
                                st.error(f"Error: {error}")
                            else:
                                st.session_state['ocr_result'] = extracted_content
                                st.session_state['extraction_method'] = "Two-Pass Verification"
                                st.session_state['model_used'] = CONFIG["primary_model"]
                        else:
                            # Use standard extraction with fallback
                            extracted_content, error, model_used = extract_with_fallback(image_data, selected_prompt)
                            if error:
                                st.error(f"Error: {error}")
                            else:
                                st.session_state['ocr_result'] = extracted_content
                                st.session_state['extraction_method'] = extraction_method
                                st.session_state['model_used'] = model_used
                        
                        # Record processing time
                        st.session_state['processing_time'] = time.time() - start_time
                        
                        # Validate extraction if successful
                        if 'ocr_result' in st.session_state:
                            # Define expected patterns based on the image content
                            # This could be enhanced with more sophisticated pattern matching
                            expected_patterns = [
                                (r"cos.*\+.*cos", "Complete equation with multiple terms"),
                                (r"equal to", "Question completion phrase"),
                                (r"point", "Point value indicator")
                            ]
                            
                            validation_issues = validate_extraction(
                                st.session_state['ocr_result'], 
                                expected_patterns
                            )
                            
                            # Assess confidence
                            confidence, warnings = assess_extraction_confidence(
                                st.session_state['ocr_result'], 
                                image
                            )
                            
                            st.session_state['confidence'] = confidence
                            st.session_state['warnings'] = warnings + validation_issues
                    
                    except Exception as e:
                        st.error(f"Error processing image: {str(e)}")

# Main content area for results
if 'ocr_result' in st.session_state:
    # Display extraction metadata
    st.markdown("### Extraction Details")
    
    # Create columns for metadata
    meta_col1, meta_col2, meta_col3 = st.columns(3)
    
    with meta_col1:
        st.metric("Processing Time", f"{st.session_state['processing_time']:.2f}s")
    
    with meta_col2:
        st.metric("Model Used", st.session_state['model_used'])
    
    with meta_col3:
        # Display confidence with appropriate color
        confidence = st.session_state.get('confidence', 'unknown')
        if confidence == 'high':
            st.markdown("**Confidence:** <span style='color:green'>High</span>", unsafe_allow_html=True)
        elif confidence == 'medium':
            st.markdown("**Confidence:** <span style='color:orange'>Medium</span>", unsafe_allow_html=True)
        else:
            st.markdown("**Confidence:** <span style='color:red'>Low</span>", unsafe_allow_html=True)
    
    # Display warnings if any
    if 'warnings' in st.session_state and st.session_state['warnings']:
        with st.expander("Extraction Warnings"):
            for warning in st.session_state['warnings']:
                st.warning(warning)
    
    # Display the extraction results
    st.markdown("### Extracted Content")
    st.text_area("Copy this text:", st.session_state['ocr_result'], height=200)
    
    # Side-by-side comparison
    st.markdown("### Comparison")
    comp_col1, comp_col2 = st.columns(2)
    
    with comp_col1:
        st.subheader("Original Image")
        if uploaded_file is not None:
            image, _ = load_and_validate_image(uploaded_file)
            if image is not None:
                st.image(image, use_container_width=True)
    
    with comp_col2:
        st.subheader("Rendered Preview")
        # Display as markdown to handle mixed text/math
        st.markdown(st.session_state['ocr_result'])
    
else:
    st.info("Upload an image and click 'Extract Content' to see the results here.")

# Footer
st.markdown("---")
st.markdown("Enhanced Math Equation Extractor | Using Llama Vision Models")

# Add a debug section in an expander
with st.expander("Debug Information"):
    st.write("Current Configuration:")
    st.json(CONFIG)
    
    if 'ocr_result' in st.session_state:
        st.write("Raw Extraction Result Length:", len(st.session_state['ocr_result']))
        
        # Show LaTeX delimiter counts
        st.write("LaTeX Delimiter Counts:")
        st.write("- \\( : ", st.session_state['ocr_result'].count("\\("))
        st.write("- \\) : ", st.session_state['ocr_result'].count("\\)"))
        
        # Show common math symbols counts
        st.write("Math Symbol Counts:")
        st.write("- cos : ", st.session_state['ocr_result'].count("cos"))
        st.write("- sin : ", st.session_state['ocr_result'].count("sin"))
        st.write("- + : ", st.session_state['ocr_result'].count("+"))
        st.write("- = : ", st.session_state['ocr_result'].count("="))
