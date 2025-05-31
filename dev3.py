import streamlit as st
import ollama
from PIL import Image
import numpy as np
import cv2
import re
import io
import time
import base64
import os

# Configuration parameters
CONFIG = {
    "primary_model": "llama3.2-vision",
    "fallback_models": ["llama3-vision", "llama2-vision"],
    "temperature": 0.1,
    "max_tokens": 500,
    "ollama_base_url": "http://localhost:11434",
    "image_types": ['png', 'jpg', 'jpeg'],
    "max_file_size_mb": 5,
    "preprocessing_enabled": True
}

# Reference LaTeX template
LATEX_TEMPLATE = """\\documentclass{article}
\\usepackage{enumitem}
\\begin{document}

{content}

\\end{document}"""

# Extraction prompts
EXTRACTION_PROMPTS = {
    "strict": r"""
EXTRACT ONLY VISIBLE TEXT AND EQUATIONS FROM THIS IMAGE.

CRITICAL RULES:
1. Output ONLY what is VISIBLY PRESENT in the image
2. Use minimal LaTeX formatting - only what's needed to represent the math
3. Maintain the EXACT layout, spacing, and sequence of content
4. Include ALL text (headers, questions, options, point values)
5. For fractions, use simple LaTeX format: \frac{numerator}{denominator} with proper spacing
6. Pay special attention to small text like "1 point" - do not miss it
7. Preserve circle symbols (○) ONLY where they actually appear in the image
8. Do NOT add circle symbols to headers, question text, or point values
9. Include ALL answer options - don't skip any options
10. Do NOT add any labels like "Options:" or bullet points that aren't in the image

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
- Add questions like "What is the value of n?" if not in the image
- Add circle symbols (○) to lines that don't have them in the image

Your output must be EXACTLY what someone would get if they manually transcribed all visible text and used minimal LaTeX for math, preserving all spacing and symbols.
""",
    
    "example_based": r"""
EXTRACT ONLY VISIBLE TEXT AND EQUATIONS FROM THIS IMAGE.

Example of CORRECT extraction:
Image shows: "QUESTION 5/6
The expression 1/n! divided by 1/(n+1)! is equal to
1 point
○ n
○ n+1
○ (n+1)/n
○ n/(n+1)"

Correct output:
QUESTION 5/6

The expression \frac{1/n!}{1/(n+1)!} is equal to

1 point

○ n
○ n+1
○ (n+1)/n
○ n/(n+1)

Example of INCORRECT extraction (DO NOT DO THIS):
"○ QUESTION 5/6

○ The expression \frac{ \frac{1}{n}}{ \frac{1}{(n+1)!}} is equal to

○ 1 point

○ n
○ n + 1"

""",
    
    "two_pass": r"""
EXTRACT ONLY VISIBLE TEXT AND EQUATIONS FROM THIS IMAGE.

CRITICAL RULES:
1. Output ONLY what is VISIBLY PRESENT in the image
2. Use minimal LaTeX formatting - only what's needed to represent the math
3. Maintain the EXACT layout, spacing, and sequence of content
4. Include ALL text (headers, questions, options, point values)
5. For fractions, use simple LaTeX format: \frac{numerator}{denominator} with proper spacing
6. Pay special attention to small text like "1 point" - do not miss it
7. Preserve circle symbols (○) ONLY where they actually appear in the image
8. Do NOT add circle symbols to headers, question text, or point values
9. Include ALL answer options - don't skip any options
10. Do NOT add any labels like "Options:" or bullet points that aren't in the image

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
- Add questions like "What is the value of n?" if not in the image
- Add circle symbols (○) to lines that don't have them in the image

Your output must be EXACTLY what someone would get if they manually transcribed all visible text and used minimal LaTeX for math, preserving all spacing and symbols.
"""
}

# Helper functions for image processing
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

def load_and_validate_image(image_path):
    """Load and validate the image from path"""
    if not os.path.exists(image_path):
        return None, f"Image not found: {image_path}"
    
    try:
        image = Image.open(image_path)
        is_valid, message = check_image_quality(image)
        if not is_valid:
            print(f"Warning: {message} for {image_path}")
        return image, None
    except Exception as e:
        return None, f"Error loading image: {str(e)}"

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
    for func in ['cos', 'sin', 'tan', 'sec', 'csc', 'cot', 'log']:
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

def post_process_extraction(text):
    """Post-process the extracted text"""
    # Remove any "ANSWER" text
    text = re.sub(r'(?i)ANSWER:?', '', text)
    
    # Clean up LaTeX formatting
    text = clean_latex_formatting(text)
    
    # Fix spacing and symbols
    text = fix_spacing_and_symbols(text)
    
    # Remove leading/trailing whitespace
    text = text.strip()
    
    return text

def extract_content_from_image(image, model, prompt_type="strict"):
    """
    Extract content from an image
    
    Args:
        image: PIL Image
        model: Vision model to use
        prompt_type: Type of prompt to use
        
    Returns:
        Extraction result
    """
    # Convert image to bytes for API
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='PNG')
    img_bytes = img_byte_arr.getvalue()
    
    # Get prompt
    prompt_content = EXTRACTION_PROMPTS.get(prompt_type, EXTRACTION_PROMPTS["strict"])
    
    # Extract content from image
    try:
        response = ollama.chat(
            model=model,
            messages=[{
                'role': 'user',
                'content': prompt_content,
                'images': [img_bytes]
            }],
            options={
                "temperature": CONFIG["temperature"],
                "num_predict": CONFIG["max_tokens"]
            }
        )
        
        extracted_content = response.message.content
        
        # Post-process the extraction
        processed_content = post_process_extraction(extracted_content)
        
        # Return result with metadata
        return {
            'raw_extraction': extracted_content,
            'processed_extraction': processed_content,
            'success': True
        }
        
    except Exception as e:
        # Handle extraction errors
        return {
            'error': str(e),
            'success': False
        }

def parse_extracted_content(extracted_text):
    """
    Parse extracted text into structured components
    
    Args:
        extracted_text: Extracted text from image
        
    Returns:
        Dictionary with question components
    """
    # Split the text into lines
    lines = extracted_text.strip().split('\n')
    
    # Initialize variables
    question_header = ""
    question_text = ""
    point_value = ""
    options = []
    current_section = "header"
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Check if this is a question header line
        if line.startswith("QUESTION"):
            question_header = line
            current_section = "header"
            continue
            
        # Check if this is the point value line
        if "point" in line.lower():
            point_value = line
            current_section = "points"
            continue
            
        # Check if this is an option line (starts with circle symbol)
        if line.startswith("○"):
            current_section = "options"
            # Remove the circle symbol and clean up
            option = line[1:].strip()
            options.append(option)
        elif current_section == "options" and not line.startswith("QUESTION") and "point" not in line.lower():
            # This might be an option without the circle symbol
            options.append(line)
        elif current_section != "options" and not line.startswith("QUESTION") and "point" not in line.lower():
            # This is likely the question text
            question_text += line + " "
    
    # Clean up question text
    question_text = question_text.strip()
    
    return {
        'header': question_header,
        'question_text': question_text,
        'point_value': point_value,
        'options': options
    }

def convert_to_latex_format(parsed_content):
    """
    Convert parsed content to LaTeX format
    
    Args:
        parsed_content: Dictionary with question components
        
    Returns:
        LaTeX formatted text
    """
    latex_output = f"{parsed_content['header']}\n\n"
    latex_output += f"{parsed_content['question_text']}\n\n"
    latex_output += f"{parsed_content['point_value']}\n\n"
    latex_output += "\\begin{enumerate}[label=]\n"
    
    for option in parsed_content['options']:
        latex_output += f"    \\item ${option}$\n"
    
    latex_output += "\\end{enumerate}\n\n"
    
    return latex_output

def batch_process_images(image_paths, model, prompt_type="strict"):
    """
    Process multiple images and extract content
    
    Args:
        image_paths: List of image paths
        model: Vision model to use
        prompt_type: Type of prompt to use
        
    Returns:
        Dictionary of extraction results
    """
    results = {}
    
    for i, image_path in enumerate(image_paths):
        print(f"Processing image {i+1}/{len(image_paths)}: {image_path}")
        
        # Load and validate image
        image, error = load_and_validate_image(image_path)
        if error:
            results[image_path] = {
                'error': error,
                'success': False
            }
            continue
        
        # Preprocess image if enabled
        if CONFIG["preprocessing_enabled"]:
            processed_image, _ = preprocess_image(image)
        else:
            processed_image = image
        
        # Extract content
        result = extract_content_from_image(processed_image, model, prompt_type)
        
        # Add image path to result
        result['image_path'] = image_path
        
        # Add to results
        results[image_path] = result
        
        # Parse and convert to LaTeX format
        if result['success']:
            parsed_content = parse_extracted_content(result['processed_extraction'])
            latex_text = convert_to_latex_format(parsed_content)
            result['parsed_content'] = parsed_content
            result['latex_text'] = latex_text
        
        # Wait a bit to avoid rate limiting
        time.sleep(1)
    
    return results

def compile_latex_document(extraction_results, template=LATEX_TEMPLATE):
    """
    Compile extraction results into a LaTeX document
    
    Args:
        extraction_results: Dictionary of extraction results
        template: LaTeX template string
        
    Returns:
        Complete LaTeX document
    """
    # Sort results by question number
    sorted_results = []
    for path, result in extraction_results.items():
        if result['success']:
            # Extract question number from filename
            match = re.search(r'DiagnosticQ-(\d+)of6', os.path.basename(path))
            if match:
                question_number = int(match.group(1))
                sorted_results.append((question_number, result))
    
    # Sort by question number
    sorted_results.sort()
    
    # Combine all LaTeX content
    content = ""
    for _, result in sorted_results:
        if 'latex_text' in result:
            content += result['latex_text']
    
    # Insert content into template
    latex_doc = template.format(content=content)
    
    return latex_doc

def validate_latex_document(latex_doc):
    """
    Validate LaTeX document for common issues
    
    Args:
        latex_doc: LaTeX document string
        
    Returns:
        Tuple of (is_valid, issues)
    """
    issues = []
    
    # Check for missing question numbers
    for i in range(1, 7):
        if f"QUESTION {i}/6" not in latex_doc:
            issues.append(f"Missing Question {i}")
    
    # Check for missing point values
    if latex_doc.count("1 point") < 6:
        issues.append("Some questions are missing point values")
    
    # Check for missing enumerate environments
    if latex_doc.count("\\begin{enumerate}") != latex_doc.count("\\end{enumerate}"):
        issues.append("Mismatched enumerate environments")
    
    # Check for missing options
    if latex_doc.count("\\item") < 24:  # 6 questions * 4 options
        issues.append("Some questions are missing options")
    
    return len(issues) == 0, issues

# UI Components for Streamlit app
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

def main():
    """Main Streamlit application"""
    # Page configuration
    st.set_page_config(
        page_title="Math Quiz LaTeX Generator",
        page_icon="🧮",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Title and description
    st.title("🧮 Math Quiz LaTeX Generator")
    st.markdown('<p style="margin-top: -20px;">Extract math equations from images and generate LaTeX documents</p>', unsafe_allow_html=True)
    
    # Initialize session state
    if 'current_step' not in st.session_state:
        st.session_state.current_step = "upload"
    if 'extraction_results' not in st.session_state:
        st.session_state.extraction_results = None
    if 'edited_latex' not in st.session_state:
        st.session_state.edited_latex = {}
    if 'final_latex' not in st.session_state:
        st.session_state.final_latex = None
    if 'reference_latex' not in st.session_state:
        st.session_state.reference_latex = None
    
    # Sidebar for settings and upload
    with st.sidebar:
        st.header("Settings")
        
        # Extraction method
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
        
        # Reference LaTeX file upload
        st.header("Reference LaTeX")
        reference_file = st.file_uploader("Upload reference LaTeX file (optional)", type=["tex"])
        if reference_file:
            reference_content = reference_file.getvalue().decode("utf-8")
            st.session_state.reference_latex = reference_content
            st.success("Reference LaTeX file loaded")
        
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
            st.session_state.extraction_results = None
            st.session_state.edited_latex = {}
            st.session_state.final_latex = None
            st.rerun()
    
    # Main content area - Multi-step workflow
    if st.session_state.current_step == "upload":
        # Step 1: Upload images
        st.markdown("## Step 1: Upload Images")
        st.info("Upload all six diagnostic quiz images to begin the extraction process.")
        
        # File uploader for multiple images
        uploaded_files = st.file_uploader(
            "Upload quiz images", 
            type=CONFIG["image_types"],
            accept_multiple_files=True
        )
        
        if uploaded_files:
            # Save uploaded files to disk
            image_paths = []
            for file in uploaded_files:
                file_path = os.path.join("/tmp", file.name)
                with open(file_path, "wb") as f:
                    f.write(file.getbuffer())
                image_paths.append(file_path)
            
            # Display uploaded images
            st.success(f"Uploaded {len(image_paths)} images")
            
            # Display image previews
            cols = st.columns(3)
            for i, path in enumerate(image_paths[:6]):  # Show up to 6 images
                with cols[i % 3]:
                    image = Image.open(path)
                    st.image(image, caption=f"Image {i+1}", use_container_width=True)
            
            # Process button
            if st.button("Extract Content from Images", type="primary"):
                with st.spinner("Extracting content from images..."):
                    # Process all images
                    results = batch_process_images(
                        image_paths, 
                        CONFIG["primary_model"], 
                        selected_prompt
                    )
                    
                    # Store results in session state
                    st.session_state.extraction_results = results
                    
                    # Initialize edited_latex with the initial LaTeX text
                    for path, result in results.items():
                        if result['success'] and 'latex_text' in result:
                            # Extract question number from filename
                            match = re.search(r'(\d+)of6', os.path.basename(path))
                            if match:
                                question_number = int(match.group(1))
                                st.session_state.edited_latex[question_number] = result['latex_text']
                    
                    # Move to next step
                    st.session_state.current_step = "review"
                    st.rerun()
    
    elif st.session_state.current_step == "review":
        # Step 2: Review and edit extractions
        st.markdown("## Step 2: Review and Edit Extractions")
        st.info("Review each extracted question and make corrections if needed.")
        
        if st.session_state.extraction_results:
            # Create tabs for each question
            tab_labels = [f"Question {i}" for i in range(1, 7)]
            tabs = st.tabs(tab_labels)
            
            # Sort results by question number
            sorted_results = []
            for path, result in st.session_state.extraction_results.items():
                if result['success']:
                    # Extract question number from filename
                    match = re.search(r'(\d+)of6', os.path.basename(path))
                    if match:
                        question_number = int(match.group(1))
                        sorted_results.append((question_number, result))
            
            # Sort by question number
            sorted_results.sort()
            
            # Display each question in its own tab
            for i, (question_number, result) in enumerate(sorted_results):
                with tabs[i]:
                    # Display original image
                    image = Image.open(result['image_path'])
                    st.image(image, caption=f"Question {question_number}", use_container_width=True)
                    
                    # Display extracted text
                    st.subheader("Extracted Text")
                    st.text_area(
                        "Raw Extraction",
                        value=result['processed_extraction'],
                        height=150,
                        key=f"raw_{question_number}",
                        disabled=True
                    )
                    
                    # Display LaTeX format with editing
                    st.subheader("LaTeX Format")
                    edited_latex = st.text_area(
                        "Edit LaTeX if needed",
                        value=st.session_state.edited_latex.get(question_number, result.get('latex_text', '')),
                        height=300,
                        key=f"latex_{question_number}"
                    )
                    
                    # Save edits to session state
                    st.session_state.edited_latex[question_number] = edited_latex
            
            # Buttons for navigation
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Back to Upload"):
                    st.session_state.current_step = "upload"
                    st.rerun()
            with col2:
                if st.button("Generate LaTeX Document", type="primary"):
                    # Compile LaTeX document from edited content
                    if st.session_state.reference_latex:
                        # Use reference LaTeX as template
                        template_parts = st.session_state.reference_latex.split("\\begin{document}")
                        if len(template_parts) > 1:
                            preamble = template_parts[0] + "\\begin{document}\n\n"
                            latex_doc = preamble
                            
                            # Add each question's edited LaTeX content
                            for question_number in sorted(st.session_state.edited_latex.keys()):
                                latex_doc += st.session_state.edited_latex[question_number]
                            
                            # Close document
                            latex_doc += "\\end{document}"
                        else:
                            # Fallback to default template
                            latex_doc = compile_latex_document({
                                f"q{i}": {"success": True, "latex_text": text} 
                                for i, text in st.session_state.edited_latex.items()
                            })
                    else:
                        # Use default template
                        latex_doc = compile_latex_document({
                            f"q{i}": {"success": True, "latex_text": text} 
                            for i, text in st.session_state.edited_latex.items()
                        })
                    
                    # Validate the document
                    is_valid, issues = validate_latex_document(latex_doc)
                    
                    # Save to session state
                    st.session_state.final_latex = latex_doc
                    
                    # Move to next step
                    st.session_state.current_step = "finalize"
                    st.rerun()
        else:
            st.error("No extraction results found. Please go back and extract content.")
            if st.button("Back to Upload"):
                st.session_state.current_step = "upload"
                st.rerun()
    
    elif st.session_state.current_step == "finalize":
        # Step 3: Finalize and download
        st.markdown("## Step 3: Finalize LaTeX Document")
        
        if st.session_state.final_latex:
            # Validate the document
            is_valid, issues = validate_latex_document(st.session_state.final_latex)
            
            # Display validation results
            if not is_valid:
                st.warning("The LaTeX document has some potential issues:")
                for issue in issues:
                    st.write(f"- {issue}")
            else:
                st.success("LaTeX document validation passed!")
            
            # Display final LaTeX
            st.subheader("Final LaTeX Document")
            st.text_area(
                "LaTeX Code",
                value=st.session_state.final_latex,
                height=400,
                key="final_latex"
            )
            
            # Copy button
            st.components.v1.html(create_copy_button_html(st.session_state.final_latex), height=50)
            
            # Download button
            st.download_button(
                label="Download LaTeX File",
                data=st.session_state.final_latex,
                file_name="math_quiz.tex",
                mime="text/plain"
            )
            
            # Button to start over
            if st.button("Process Another Set of Images"):
                st.session_state.current_step = "upload"
                st.session_state.extraction_results = None
                st.session_state.edited_latex = {}
                st.session_state.final_latex = None
                st.rerun()
        else:
            st.error("No LaTeX document generated. Please go back and review extractions.")
            if st.button("Back to Review"):
                st.session_state.current_step = "review"
                st.rerun()

if __name__ == "__main__":
    main()
