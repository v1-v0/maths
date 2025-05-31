import streamlit as st
import ollama
from PIL import Image
import numpy as np
import cv2
import re
import io
import time
import base64

# Configuration parameters
CONFIG = {
    "primary_model": "llama3.2-vision",
    "fallback_models": ["llama3-vision", "llama2-vision"],
    "temperature": 0.1,
    "max_tokens": 500,  # Reduced for per-line extraction
    "ollama_base_url": "http://localhost:11434",
    "image_types": ['png', 'jpg', 'jpeg'],
    "max_file_size_mb": 5,
    "preprocessing_enabled": True
}

# Extraction prompts
EXTRACTION_PROMPTS = {
    "strict": r"""
EXTRACT ONLY VISIBLE TEXT AND EQUATIONS FROM THIS LINE IMAGE.

CRITICAL RULES:
1. Output ONLY what is VISIBLY PRESENT in this line image
2. Use minimal LaTeX formatting - only what's needed to represent the math
3. Maintain the EXACT layout and spacing of content
4. For fractions, use simple LaTeX format: \frac{numerator}{denominator} with proper spacing
5. Preserve circle symbols (○) ONLY where they actually appear in the image
6. Do NOT add any formatting not in the original image

ABSOLUTELY DO NOT:
- Add ANY descriptions of the image
- Provide ANY analysis or explanations
- Identify ANY answers or solutions
- Add ANY text not visibly present in the image
- Make ANY inferences or conclusions
- Include the word "ANSWER" anywhere in your output
- Use LaTeX sectioning commands like \section{}
- Use complex LaTeX structures like arrays or tables
- Wrap every math expression in \( \) or $ $ delimiters
- Add bullet points or formatting not in the original image

Your output must be EXACTLY what someone would get if they manually transcribed this line.
""",
    
    "example_based": r"""
EXTRACT ONLY VISIBLE TEXT AND EQUATIONS FROM THIS LINE IMAGE.

Example of CORRECT extraction:
Line image shows: "The expression \frac{1/n!}{1/(n+1)!} is equal to"

Correct output:
The expression \frac{1/n!}{1/(n+1)!} is equal to

Example of INCORRECT extraction (DO NOT DO THIS):
"The expression \( \frac{\frac{1}{n!}}{\frac{1}{(n+1)!}} \) is equal to what value?"

REMEMBER: 
- Output ONLY the visible text and equations in this line
- NO descriptions, analysis, or answers
- Use minimal LaTeX - only what's needed for the math
- NO LaTeX sectioning commands like \section{}
- NO complex structures like arrays or tables
- NO unnecessary delimiters around math expressions
- Preserve circle symbols (○) ONLY where they actually appear
""",
    
    "two_pass": r"""
EXTRACT ONLY VISIBLE TEXT AND EQUATIONS FROM THIS LINE IMAGE.

CRITICAL RULES:
1. Output ONLY what is VISIBLY PRESENT in this line image
2. Use minimal LaTeX formatting - only what's needed to represent the math
3. Maintain the EXACT layout and spacing of content
4. For fractions, use simple LaTeX format: \frac{numerator}{denominator} with proper spacing
5. Preserve circle symbols (○) ONLY where they actually appear in the image
6. Do NOT add any formatting not in the original image

ABSOLUTELY DO NOT:
- Add ANY descriptions of the image
- Provide ANY analysis or explanations
- Identify ANY answers or solutions
- Add ANY text not visibly present in the image
- Make ANY inferences or conclusions
- Include the word "ANSWER" anywhere in your output
- Use LaTeX sectioning commands like \section{}
- Use complex LaTeX structures like arrays or tables
- Wrap every math expression in \( \) or $ $ delimiters
- Add bullet points or formatting not in the original image

Your output must be EXACTLY what someone would get if they manually transcribed this line.
"""
}

# Helper functions for image processing
# FIXED: Removed @st.cache_data decorator to avoid unhashable type error with PIL Image
def segment_image_into_lines(image):
    """
    Segment an image into individual lines using OpenCV
    
    Args:
        image: PIL Image or numpy array
        
    Returns:
        List of line images and their coordinates
    """
    # Convert to numpy array if needed
    if not isinstance(image, np.ndarray):
        image_np = np.array(image)
    
    # Convert to grayscale
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY) if len(image_np.shape) == 3 else image_np
    
    # Apply binary thresholding
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Calculate horizontal projection profile
    h_projection = np.sum(binary, axis=1)
    
    # Find line boundaries using projection profile
    line_boundaries = []
    in_line = False
    start = 0
    
    for i, proj in enumerate(h_projection):
        if not in_line and proj > 0:
            # Start of a new line
            in_line = True
            start = i
        elif in_line and proj == 0:
            # End of a line
            if i - start > 10:  # Minimum line height to avoid noise
                line_boundaries.append((start, i))
            in_line = False
    
    # Handle case where last line extends to bottom of image
    if in_line:
        line_boundaries.append((start, len(h_projection)))
    
    # Extract line images
    line_images = []
    for start, end in line_boundaries:
        # Add padding around lines
        padding = 5
        start_padded = max(0, start - padding)
        end_padded = min(image_np.shape[0], end + padding)
        
        line_img = image_np[start_padded:end_padded, :]
        line_images.append({
            'image': line_img,
            'coordinates': (start_padded, end_padded),
            'original_height': end - start
        })
    
    return line_images

# FIXED: Removed @st.cache_data decorator to avoid unhashable type error with PIL Image
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

# Helper functions for extraction
def clean_latex_formatting(text):
    """Clean up LaTeX formatting to make it minimal"""
    # Remove LaTeX sectioning commands
    text = re.sub(r'\\section\*?\{([^}]+)\}', r'\1', text)
    
    # Remove unnecessary math delimiters
    text = re.sub(r'\\[\(\[]([^\\]+)\\[\)\]]', r'\1', text)
    text = re.sub(r'\$\$([^$]+)\$\$', r'\1', text)
    text = re.sub(r'\$([^$]+)\$', r'\1', text)
    
    # Fix common LaTeX commands to keep them minimal
    # Keep \frac, \cos, \sin, \tan, \sec but remove unnecessary delimiters
    
    # Fix fractions - keep \frac but remove surrounding delimiters
    fraction_pattern = r'\\?\(?\s*\\frac\s*\{\s*([^{}]+)\s*\}\s*\{\s*([^{}]+)\s*\}\s*\\?\)?'
    
    def fix_fraction(match):
        numerator = match.group(1).strip()
        denominator = match.group(2).strip()
        
        # Check if numerator or denominator themselves contain fractions
        if '/' in numerator and not '\\frac' in numerator:
            parts = numerator.split('/')
            if len(parts) == 2:
                numerator = f"\\frac{{{parts[0].strip()}}}{{{parts[1].strip()}}}"
        
        if '/' in denominator and not '\\frac' in denominator:
            parts = denominator.split('/')
            if len(parts) == 2:
                denominator = f"\\frac{{{parts[0].strip()}}}{{{parts[1].strip()}}}"
        
        return f"\\frac{{{numerator}}}{{{denominator}}}"
    
    # Apply the fraction fix
    text = re.sub(fraction_pattern, fix_fraction, text)
    
    # Fix trig functions - add \ if missing
    for func in ['cos', 'sin', 'tan', 'sec', 'csc', 'cot']:
        # Only add \ if it's not already there and it's a standalone function (not part of a word)
        text = re.sub(r'(?<![\\a-zA-Z])' + func + r'(?![a-zA-Z])', r'\\' + func, text)
    
    # Remove array environments
    text = re.sub(r'\\begin\{array\}[^\\]*\\end\{array\}', '', text)
    
    # Remove textbf commands
    text = re.sub(r'\\textbf\{([^}]+)\}', r'\1', text)
    
    # Remove markdown formatting
    text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)  # Bold
    text = re.sub(r'^\s*[\*\-]\s+', r'', text, flags=re.MULTILINE)  # Bullet points
    
    # Ensure proper spacing around LaTeX commands
    text = re.sub(r'([^\s])\\frac', r'\1 \\frac', text)
    text = re.sub(r'\\frac([^\s{])', r'\\frac \1', text)
    
    # Remove extra newlines
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    return text

def fix_spacing_and_symbols(text):
    """Fix spacing issues and ensure symbols are preserved correctly"""
    # Fix spacing around fractions
    text = re.sub(r'([^\s])\\frac', r'\1 \\frac', text)
    text = re.sub(r'\\frac([^\s{])', r'\\frac \1', text)
    
    # Ensure proper spacing after equations
    text = re.sub(r'(\})([a-zA-Z])', r'\1 \2', text)
    
    # Fix spacing around equals sign
    text = re.sub(r'([^\s])=', r'\1 =', text)
    text = re.sub(r'=([^\s])', r'= \1', text)
    
    # Remove any added question marks or periods not in the original
    text = re.sub(r'is equal to\?', r'is equal to', text)
    
    return text

def post_process_line_extraction(text):
    """Post-process the extracted text from a single line"""
    # Remove any "ANSWER" text
    text = re.sub(r'(?i)ANSWER:?', '', text)
    
    # Clean up LaTeX formatting
    text = clean_latex_formatting(text)
    
    # Fix spacing and symbols
    text = fix_spacing_and_symbols(text)
    
    # Remove leading/trailing whitespace
    text = text.strip()
    
    return text

def get_line_extraction_prompt(prompt_type, line_index, previous_results=None):
    """
    Generate a context-aware prompt for line extraction
    
    Args:
        prompt_type: Type of prompt to use
        line_index: Index of current line
        previous_results: Results from previous lines
        
    Returns:
        Prompt string with context
    """
    base_prompt = EXTRACTION_PROMPTS.get(prompt_type, EXTRACTION_PROMPTS["strict"])
    
    # For first line, use base prompt
    if line_index == 0 or previous_results is None:
        return base_prompt + "\n\nThis is the first line of the image. Extract ONLY what is visible in this line segment."
    
    # For subsequent lines, add context from previous lines
    context = "Previous lines extracted:\n"
    
    # Include up to 3 previous lines for context
    start_idx = max(0, line_index - 3)
    for i in range(start_idx, line_index):
        if i < len(previous_results) and 'processed_extraction' in previous_results[i]:
            context += f"Line {i+1}: {previous_results[i]['processed_extraction']}\n"
    
    context += f"\nNow extract ONLY what is visible in line {line_index+1}."
    
    return base_prompt + "\n\n" + context

def extract_line_content(line_image, model, prompt_type, line_index=0, previous_results=None):
    """
    Extract content from a single line image
    
    Args:
        line_image: Image of a single line
        model: Vision model to use
        prompt_type: Type of prompt to use
        line_index: Index of current line
        previous_results: Results from previous lines
        
    Returns:
        Extraction result for the line
    """
    # Convert line image to bytes for API
    if isinstance(line_image, np.ndarray):
        line_img = Image.fromarray(line_image)
    else:
        line_img = line_image
        
    img_byte_arr = io.BytesIO()
    line_img.save(img_byte_arr, format='PNG')
    line_img_bytes = img_byte_arr.getvalue()
    
    # Get appropriate prompt with context from previous lines
    prompt_content = get_line_extraction_prompt(prompt_type, line_index, previous_results)
    
    # Extract content from line
    try:
        response = ollama.chat(
            model=model,
            messages=[{
                'role': 'user',
                'content': prompt_content,
                'images': [line_img_bytes]
            }],
            options={
                "temperature": CONFIG["temperature"],
                "num_predict": CONFIG["max_tokens"]
            }
        )
        
        extracted_content = response.message.content
        
        # Post-process the extraction
        processed_content = post_process_line_extraction(extracted_content)
        
        # Return result with metadata
        return {
            'line_number': line_index,
            'raw_extraction': extracted_content,
            'processed_extraction': processed_content,
            'user_modified': False,
            'user_content': None
        }
        
    except Exception as e:
        # Handle extraction errors
        return {
            'line_number': line_index,
            'error': str(e),
            'user_modified': False,
            'user_content': None
        }

def process_lines_sequentially(line_images, model, prompt_type="strict"):
    """
    Process each line image sequentially through the vision model
    
    Args:
        line_images: List of line image dictionaries
        model: Vision model to use
        prompt_type: Type of prompt to use
        
    Returns:
        List of extraction results for each line
    """
    results = []
    
    # Process each line
    for i, line_data in enumerate(line_images):
        with st.spinner(f"Processing line {i+1} of {len(line_images)}..."):
            # Extract content from line
            result = extract_line_content(
                line_data['image'],
                model,
                prompt_type,
                i,
                results
            )
            
            # Add coordinates to result
            result['coordinates'] = line_data['coordinates']
            
            # Add result to list
            results.append(result)
            
            # Show progress
            st.progress((i + 1) / len(line_images))
    
    return results

def assemble_final_output(extraction_results):
    """
    Assemble the final output from all line extraction results
    
    Args:
        extraction_results: List of extraction results for each line
        
    Returns:
        Final assembled text
    """
    final_text = []
    
    for result in extraction_results:
        # Use user-modified content if available, otherwise use processed extraction
        if result['user_modified'] and result['user_content']:
            line_text = result['user_content']
        elif 'processed_extraction' in result:
            line_text = result['processed_extraction']
        else:
            continue  # Skip lines with errors and no user edits
        
        final_text.append(line_text)
    
    # Join lines with appropriate spacing
    return '\n'.join(final_text)

# UI Components
def create_copy_button_html(text_to_copy):
    """Create HTML for a copy button"""
    # Encode the text for JavaScript
    encoded_text = base64.b64encode(text_to_copy.encode()).decode()
    
    # Create HTML/JavaScript for copy button
    copy_button_html = f"""
    <script>
    function copyToClipboard() {{
        const text = atob("{encoded_text}");
        navigator.clipboard.writeText(text).then(function() {{
            document.getElementById('copy-status').textContent = 'Copied!';
            setTimeout(function() {{
                document.getElementById('copy-status').textContent = '';
            }}, 2000);
        }})
        .catch(function(err) {{
            document.getElementById('copy-status').textContent = 'Failed to copy';
            console.error('Could not copy text: ', err);
        }});
    }}
    </script>
    
    <button 
        onclick="copyToClipboard()" 
        style="background-color: #4CAF50; color: white; padding: 10px 24px; border: none; border-radius: 4px; cursor: pointer;"
    >
        Copy to Clipboard
    </button>
    <span id="copy-status" style="margin-left: 10px; color: green;"></span>
    """
    
    return copy_button_html

def create_format_options_and_copy(extraction_results):
    """
    Create format options and copy button for the final output
    
    Args:
        extraction_results: List of extraction results for each line
        
    Returns:
        None (displays UI elements in the Streamlit app)
    """
    st.markdown("## Final Output")
    
    # Assemble final text
    final_text = assemble_final_output(extraction_results)
    
    # Display the final text
    st.text_area("Extracted Content:", value=final_text, height=300, key="final_output")
    
    # Format options
    format_option = st.radio(
        "Copy Format:",
        ["Plain Text", "LaTeX", "Markdown"],
        horizontal=True
    )
    
    # Format the text based on selection
    if format_option == "Plain Text":
        formatted_text = final_text
    elif format_option == "LaTeX":
        # Convert to LaTeX document format
        formatted_text = "\\documentclass{article}\n\\usepackage{amsmath}\n\\begin{document}\n" + final_text + "\n\\end{document}"
    else:  # Markdown
        # Keep as is since our output is already markdown compatible
        formatted_text = final_text
    
    # Create copy button for the formatted text
    st.components.v1.html(create_copy_button_html(formatted_text), height=50)
    
    # Add download button
    st.download_button(
        label="Download as Text File",
        data=formatted_text,
        file_name=f"extracted_math_{format_option.lower().replace(' ', '_')}.txt",
        mime="text/plain"
    )

def create_line_review_ui(line_images, extraction_results):
    """
    Create an interactive UI for reviewing and editing line extractions
    
    Args:
        line_images: List of line image dictionaries
        extraction_results: List of extraction results for each line
        
    Returns:
        Updated extraction results with user modifications
    """
    st.markdown("## Line-by-Line Review")
    st.info("Review each extracted line and make corrections if needed.")
    
    # Initialize session state for storing edits if not exists
    if 'line_edits' not in st.session_state:
        st.session_state.line_edits = [None] * len(extraction_results)
    
    # Create tabs for navigation between lines
    tab_labels = [f"Line {i+1}" for i in range(len(line_images))]
    tabs = st.tabs(tab_labels)
    
    # Display each line in its own tab
    for i, tab in enumerate(tabs):
        with tab:
            # Display line image
            line_img = Image.fromarray(line_images[i]['image'])
            st.image(line_img, caption=f"Line {i+1}", use_container_width=True)
            
            # Get extraction result
            result = extraction_results[i]
            
            # Check if there was an error
            if 'error' in result:
                st.error(f"Error extracting this line: {result['error']}")
                extracted_text = ""
            else:
                extracted_text = result['processed_extraction']
            
            # Create editable text field with extracted content
            edited_text = st.text_area(
                "Extracted Text (edit if needed):",
                value=st.session_state.line_edits[i] if st.session_state.line_edits[i] is not None else extracted_text,
                key=f"line_edit_{i}",
                height=100
            )
            
            # Store edited text in session state
            st.session_state.line_edits[i] = edited_text
            
            # Update extraction result with user edits
            extraction_results[i]['user_modified'] = (edited_text != extracted_text)
            extraction_results[i]['user_content'] = edited_text
    
    return extraction_results

# Main application
def main():
    """Main application function"""
    # Page configuration
    st.set_page_config(
        page_title="Line-by-Line Math Equation Extractor",
        page_icon="🦙",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Title and description
    st.title("🦙 Line-by-Line Math Equation Extractor")
    st.markdown('<p style="margin-top: -20px;">Extract and edit math equations line by line</p>', unsafe_allow_html=True)
    
    # Initialize session state
    if 'current_step' not in st.session_state:
        st.session_state.current_step = "upload"
    if 'line_images' not in st.session_state:
        st.session_state.line_images = None
    if 'extraction_results' not in st.session_state:
        st.session_state.extraction_results = None
    
    # Sidebar for settings and upload
    with st.sidebar:
        st.header("Upload Image")
        uploaded_file = st.file_uploader("Choose an image...", type=CONFIG["image_types"])
        
        st.header("Extraction Settings")
        extraction_method = st.radio(
            "Extraction Method",
            ["Strict", "Example-Based", "Two-Pass Verification"],
            index=0
        )
        
        # Map the radio selection to prompt types
        prompt_mapping = {
            "Strict": "strict",
            "Example-Based": "example_based",
            "Two-Pass Verification": "two_pass"
        }
        
        selected_prompt = prompt_mapping[extraction_method]
        
        # Preprocessing options
        preprocessing_enabled = st.checkbox("Enable Image Preprocessing", value=CONFIG["preprocessing_enabled"])
        CONFIG["preprocessing_enabled"] = preprocessing_enabled
        
        # Advanced settings in expander
        with st.expander("Advanced Settings"):
            CONFIG["primary_model"] = st.selectbox(
                "Vision Model",
                CONFIG["primary_model"] if isinstance(CONFIG["primary_model"], list) else [CONFIG["primary_model"]] + CONFIG["fallback_models"],
                index=0
            )
            CONFIG["temperature"] = st.slider("Temperature", min_value=0.0, max_value=1.0, value=CONFIG["temperature"], step=0.1)
        
        # Reset button
        if st.button("Reset Process"):
            st.session_state.current_step = "upload"
            st.session_state.line_images = None
            st.session_state.extraction_results = None
            st.session_state.line_edits = None
            st.rerun()
    
    # Main content area - Multi-step workflow
    if st.session_state.current_step == "upload":
        # Step 1: Upload and segment image
        st.markdown("## Step 1: Upload and Segment Image")
        
        if uploaded_file is not None:
            image, image_data = load_and_validate_image(uploaded_file)
            
            if image is not None:
                st.image(image, caption="Uploaded Image", use_container_width=True)
                
                if st.button("Process Image Line by Line", type="primary"):
                    with st.spinner("Segmenting image into lines..."):
                        # Segment image into lines
                        line_images = segment_image_into_lines(image)
                        st.session_state.line_images = line_images
                        
                        # Show preview of segmentation
                        st.success(f"Image segmented into {len(line_images)} lines")
                        
                        # Display segmentation preview
                        cols = st.columns(min(3, len(line_images)))
                        for i, line_data in enumerate(line_images[:3]):
                            with cols[i % 3]:
                                line_img = Image.fromarray(line_data['image'])
                                st.image(line_img, caption=f"Line {i+1}", use_container_width=True)
                        
                        if len(line_images) > 3:
                            st.info(f"... and {len(line_images) - 3} more lines")
                        
                        # Move to next step
                        st.session_state.current_step = "extract"
                        st.rerun()
        else:
            st.info("Please upload an image to begin.")
    
    elif st.session_state.current_step == "extract":
        # Step 2: Extract content from each line
        st.markdown("## Step 2: Extract Content from Lines")
        
        if st.session_state.line_images:
            if st.button("Begin Line-by-Line Extraction", type="primary"):
                with st.spinner("Extracting content from lines..."):
                    # Process each line
                    extraction_results = process_lines_sequentially(
                        st.session_state.line_images,
                        CONFIG["primary_model"],
                        selected_prompt
                    )
                    st.session_state.extraction_results = extraction_results
                    
                    # Show extraction preview
                    st.success(f"Extracted content from {len(extraction_results)} lines")
                    
                    # Move to next step
                    st.session_state.current_step = "review"
                    st.rerun()
            
            # Option to go back
            if st.button("Back to Upload"):
                st.session_state.current_step = "upload"
                st.rerun()
        else:
            st.error("No line images found. Please go back and upload an image.")
            if st.button("Back to Upload"):
                st.session_state.current_step = "upload"
                st.rerun()
    
    elif st.session_state.current_step == "review":
        # Step 3: Review and edit line by line
        st.markdown("## Step 3: Review and Edit")
        
        if st.session_state.line_images and st.session_state.extraction_results:
            updated_results = create_line_review_ui(
                st.session_state.line_images,
                st.session_state.extraction_results
            )
            st.session_state.extraction_results = updated_results
            
            # Button to finalize
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Back to Extraction"):
                    st.session_state.current_step = "extract"
                    st.rerun()
            with col2:
                if st.button("Finalize Extraction", type="primary"):
                    st.session_state.current_step = "finalize"
                    st.rerun()
        else:
            st.error("No extraction results found. Please go back and extract content.")
            if st.button("Back to Upload"):
                st.session_state.current_step = "upload"
                st.rerun()
    
    elif st.session_state.current_step == "finalize":
        # Step 4: Show final output with copy options
        st.markdown("## Step 4: Final Output")
        
        if st.session_state.extraction_results:
            create_format_options_and_copy(st.session_state.extraction_results)
            
            # Button to start over
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Back to Review"):
                    st.session_state.current_step = "review"
                    st.rerun()
            with col2:
                if st.button("Process Another Image"):
                    # Reset session state
                    st.session_state.current_step = "upload"
                    st.session_state.line_images = None
                    st.session_state.extraction_results = None
                    st.session_state.line_edits = None
                    st.rerun()
        else:
            st.error("No extraction results found. Please go back and extract content.")
            if st.button("Back to Upload"):
                st.session_state.current_step = "upload"
                st.rerun()

if __name__ == "__main__":
    main()
