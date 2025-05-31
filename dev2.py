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

# Enhanced extraction prompts with stronger answer prevention
EXTRACTION_PROMPTS = {
    "ultra_strict": r"""
YOU ARE A TRANSCRIPTION TOOL, NOT A TUTOR OR SOLVER.

TASK: Type out exactly what you see in the image, character by character.

CRITICAL TRANSCRIPTION RULES:
1. Output ONLY visible text and mathematical expressions
2. Use minimal LaTeX: \cos, \sin, \tan, \frac{numerator}{denominator}
3. Preserve ALL spacing, symbols, and layout exactly as shown
4. Include circle symbols (○) ONLY where they appear in the image
5. Include ALL text: headers, questions, point values, options

ABSOLUTELY FORBIDDEN (CAUSES IMMEDIATE FAILURE):
❌ Solving the problem or providing answers
❌ Writing "The correct answer is..." or similar
❌ Writing "The image shows..." or descriptions
❌ Adding analysis, explanations, or solutions
❌ Writing "Answer:", "Solution:", "Result:"
❌ Adding text not visible in the image
❌ Using phrases like "therefore", "thus", "we can see"
❌ Identifying which option is correct

SUCCESS EXAMPLE:
Input image shows: "QUESTION 1/5 What is 2+2? ○ 3 ○ 4 ○ 5"
Correct output: "QUESTION 1/5\n\nWhat is 2+2?\n\n○ 3\n○ 4\n○ 5"
WRONG output: "The image shows a math question. The correct answer is 4."

YOU ARE A PASSIVE TRANSCRIBER. DO NOT SOLVE OR ANALYZE.
""",
    
    "example_based": r"""
TRANSCRIBE ONLY THE VISIBLE TEXT AND MATH FROM THIS IMAGE.

GOOD EXAMPLE:
Image contains: "QUESTION 2/6
The expression cos(mx + nx) + cos(mx - nx) is equal to
1 point
○ 2 cos mx cos nx
○ 2 sin mx sin nx"

CORRECT transcription:
"QUESTION 2/6

The expression cos (mx + nx) + cos (mx - nx) is equal to

1 point

○ 2 cos mx cos nx
○ 2 sin mx sin nx"

BAD EXAMPLE (DO NOT DO THIS):
"The image shows a trigonometry question. The correct answer is 2 cos mx cos nx because..."

RULES:
- NO descriptions ("The image shows...")
- NO solutions ("The correct answer...")
- NO analysis or explanations
- ONLY transcribe what you see
- Use minimal LaTeX for math expressions
- Preserve exact spacing and symbols
""",
    
    "two_pass": r"""
TRANSCRIBE ALL VISIBLE TEXT AND EQUATIONS FROM THIS IMAGE.

You are a text extraction tool. Your only job is to convert the image to text.

EXTRACT:
✓ All headers (like "QUESTION X/Y")
✓ All question text
✓ All point values (like "1 point")
✓ All answer options with their symbols
✓ Mathematical expressions using minimal LaTeX

DO NOT:
✗ Describe the image
✗ Solve the problem
✗ Identify correct answers
✗ Add explanations
✗ Use analysis language

Output format: Exactly what someone would type if manually copying all visible text.
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

def validate_extraction(extracted_text, original_image_path=None):
    """Enhanced validation to catch answer generation and other issues"""
    issues = []
    
    # Check for common indicators of incomplete extraction
    if "..." in extracted_text or "[...]" in extracted_text:
        issues.append("Incomplete extraction detected")
    
    # Enhanced prohibited content patterns - more comprehensive
    prohibited_patterns = [
        # Answer/solution detection
        (r"(?i)the correct (answer|is|solution)", "Solution/answer identification detected"),
        (r"(?i)(answer|solution|result)\s*[:=]", "Answer provision detected"),
        (r"(?i)correct option is", "Answer identification detected"),
        (r"(?i)the answer is", "Direct answer provision detected"),
        
        # Image description detection
        (r"(?i)the image (shows|displays|contains|presents)", "Image description detected"),
        (r"(?i)this image (shows|contains)", "Image description detected"),
        
        # Analysis language detection
        (r"(?i)(therefore|thus|so|hence)", "Analysis language detected"),
        (r"(?i)(we can see|this means|this shows)", "Interpretation language detected"),
        (r"(?i)(conclusion|concludes)", "Conclusion language detected"),
        (r"(?i)we can (see|observe|note|infer)", "Analysis language detected"),
        (r"(?i)this (question|problem) (asks|requires)", "Question analysis detected"),
        (r"(?i)(because|since).*?(formula|identity)", "Mathematical explanation detected"),
        
        # Specific answer patterns
        (r"(?i)ANSWER", "Answer label detected"),
        (r"(?i)solution:", "Solution label detected"),
        
        # LaTeX formatting issues
        (r"\\section", "LaTeX sectioning command detected"),
        (r"\\begin\{array\}", "LaTeX array environment detected"),
        (r"\\textbf", "LaTeX text formatting command detected"),
        
        # Added content detection
        (r"\*\*Options:\*\*", "Added 'Options:' label detected"),
        (r"What is the value of n\?", "Added question detected"),
        (r"(?i)multiple.?choice question", "Added description detected"),
        (r"(?i)(with|having) \d+ options", "Added option count description detected"),
    ]
    
    for pattern, message in prohibited_patterns:
        matches = re.findall(pattern, extracted_text)
        if matches:
            issues.append(f"{message}: {matches}")
    
    # Check for missing essential elements
    if not re.search(r"(?i)point", extracted_text):
        issues.append("Missing point value text")
    
    if not re.search(r"QUESTION \d+/\d+", extracted_text):
        issues.append("Missing question header")
    
    # Check for circle symbols in wrong places
    if re.search(r'○\s*QUESTION', extracted_text):
        issues.append("Circle symbols incorrectly added to question header")
    
    if re.search(r'○\s*\d+\s*point', extracted_text):
        issues.append("Circle symbols incorrectly added to point value")
    
    # Check for missing circle symbols on options
    option_lines_without_circles = re.findall(r'^(?!○)\s*\d*\s*(cos|sin|tan|\w+)', extracted_text, re.MULTILINE)
    if option_lines_without_circles and len(option_lines_without_circles) > 1:
        issues.append(f"Potential option lines missing circle symbols: {option_lines_without_circles}")
    
    # Check for missing options
    circle_options = len(re.findall(r'○\s*[^○\n]+', extracted_text))
    if circle_options < 2:
        issues.append(f"Insufficient options detected (found {circle_options})")
    
    # Check for markdown formatting that shouldn't be there
    if re.search(r'^\s*[\*\-]\s+', extracted_text, re.MULTILINE):
        issues.append("Added bullet points detected")
    
    if re.search(r'\*\*([^*]+)\*\*', extracted_text):
        issues.append("Added markdown bold formatting detected")
    
    # Specific validation for trigonometry expressions
    if "cos" in extracted_text.lower() and "sin" in extracted_text.lower():
        # Check for proper LaTeX formatting
        if not re.search(r'\\cos', extracted_text) and re.search(r'[^\\]cos', extracted_text):
            issues.append("Missing LaTeX formatting for trigonometric functions")
    
    return issues

def remove_answer_content(text):
    """Aggressively remove any answer or solution content"""
    # Remove lines that provide answers
    lines = text.split('\n')
    clean_lines = []
    
    for line in lines:
        line_lower = line.lower().strip()
        
        # Skip lines that contain answer identification
        skip_patterns = [
            r'(?i)(the\s+)?correct\s+(answer\s+)?is',
            r'(?i)(the\s+)?answer\s*[:=]',
            r'(?i)(the\s+)?solution\s*[:=]',
            r'(?i)therefore',
            r'(?i)the image shows',
            r'(?i)this (image|picture) (shows|contains)',
            r'(?i)multiple.?choice question',
            r'(?i)with \d+ options',
            r'(?i)because.*formula',
            r'(?i)using.*identity'
        ]
        
        should_skip = False
        for pattern in skip_patterns:
            if re.search(pattern, line):
                should_skip = True
                break
        
        if not should_skip:
            clean_lines.append(line)
    
    return '\n'.join(clean_lines)

def clean_latex_formatting(text):
    """Clean up LaTeX formatting to make it minimal and correct"""
    # Remove LaTeX sectioning commands
    text = re.sub(r'\\section\*?\{([^}]+)\}', r'\1', text)
    
    # Remove unnecessary math delimiters
    text = re.sub(r'\\[\(\[]([^\\]+)\\[\)\]]', r'\1', text)
    text = re.sub(r'\$\$([^$]+)\$\$', r'\1', text)
    text = re.sub(r'\$([^$]+)\$', r'\1', text)
    
    # Fix trigonometric functions - ensure they have backslash
    trig_functions = ['cos', 'sin', 'tan', 'sec', 'csc', 'cot']
    for func in trig_functions:
        # Add backslash if missing and it's a standalone function
        text = re.sub(r'(?<![\\a-zA-Z])' + func + r'(?![a-zA-Z])', r'\\' + func, text)
    
    # Fix fractions - keep \frac but clean up formatting
    fraction_pattern = r'\\?\(?\s*\\frac\s*\{\s*([^{}]+)\s*\}\s*\{\s*([^{}]+)\s*\}\s*\\?\)?'
    
    def fix_fraction(match):
        numerator = match.group(1).strip()
        denominator = match.group(2).strip()
        return f"\\frac{{{numerator}}}{{{denominator}}}"
    
    text = re.sub(fraction_pattern, fix_fraction, text)
    
    # Remove array environments and other complex LaTeX
    text = re.sub(r'\\begin\{array\}[^\\]*\\end\{array\}', '', text)
    text = re.sub(r'\\textbf\{([^}]+)\}', r'\1', text)
    
    # Remove markdown formatting
    text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)  # Bold
    text = re.sub(r'^\s*[\*\-]\s+', r'', text, flags=re.MULTILINE)  # Bullet points
    
    # Ensure proper spacing around LaTeX commands
    text = re.sub(r'([^\s])\\(cos|sin|tan|frac)', r'\1 \\\2', text)
    text = re.sub(r'\\(cos|sin|tan)([^\s{])', r'\\\1 \2', text)
    
    # Clean up multiple newlines
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    return text

def fix_spacing_and_symbols(text, image_path=None):
    """Fix spacing issues and ensure symbols are preserved correctly"""
    # Remove circle symbols from non-option lines
    text = re.sub(r'○\s*QUESTION', r'QUESTION', text)
    text = re.sub(r'○\s*The expression', r'The expression', text)
    text = re.sub(r'○\s*\d+\s*point', r'1 point', text)
    
    # Fix spacing around mathematical expressions
    text = re.sub(r'([^\s])\\', r'\1 \\', text)
    text = re.sub(r'\\([a-zA-Z]+)([^\s{])', r'\\\1 \2', text)
    
    # Fix spacing around equals sign
    text = re.sub(r'([^\s])=', r'\1 =', text)
    text = re.sub(r'=([^\s])', r'= \1', text)
    
    # Fix spacing around parentheses in math expressions
    text = re.sub(r'([a-zA-Z])\(', r'\1 (', text)
    text = re.sub(r'\)([a-zA-Z])', r') \1', text)
    
    # Ensure circle symbols are properly formatted for options
    lines = text.split('\n')
    processed_lines = []
    
    for line in lines:
        if not line.strip():
            processed_lines.append(line)
            continue
        
        # Check if this looks like an option line
        clean_line = re.sub(r'^○\s*', '', line.strip())
        
        # Pattern matching for mathematical option lines
        option_patterns = [
            r'^\d+\s*\\(cos|sin|tan)',  # Like "2 \cos mx \cos nx"
            r'^[a-zA-Z]\s*[+\-]?\s*\d*',  # Like "n + 1"
            r'^\([^)]+\)/[^/]+$',  # Like "(n+1)/n"
            r'^[^/]+/\([^)]+\)$',  # Like "n/(n+1)"
        ]
        
        is_option = any(re.match(pattern, clean_line) for pattern in option_patterns)
        
        # Add circle symbol to option lines if missing
        if is_option and not line.strip().startswith('○'):
            processed_lines.append(f"○ {clean_line}")
        elif line.strip().startswith('○') and not is_option:
            # Remove circle from non-option lines
            processed_lines.append(clean_line)
        else:
            processed_lines.append(line)
    
    return '\n'.join(processed_lines)

def post_process_extraction(text, original_image_path=None):
    """Comprehensive post-processing to clean up extraction"""
    # First, aggressively remove any answer content
    text = remove_answer_content(text)
    
    # Clean up LaTeX formatting
    text = clean_latex_formatting(text)
    
    # Fix spacing and symbols
    text = fix_spacing_and_symbols(text, original_image_path)
    
    # Final cleanup
    text = re.sub(r'\n{3,}', '\n\n', text)  # Normalize spacing
    text = text.strip()
    
    return text

def extract_with_enhanced_prompt(image_data, prompt_type="ultra_strict", image_path=None):
    """Extract content using enhanced prompts with strong answer prevention"""
    prompt_content = EXTRACTION_PROMPTS.get(prompt_type, EXTRACTION_PROMPTS["ultra_strict"])
    
    try:
        response = ollama.chat(
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
        
        extracted_content = response.message.content
        
        # Post-process the extraction
        extracted_content = post_process_extraction(extracted_content, image_path)
        
        # Validate for issues
        issues = validate_extraction(extracted_content, image_path)
        
        return extracted_content, issues
        
    except Exception as e:
        return None, [f"Extraction error: {str(e)}"]

def extract_with_multi_pass_verification(image_data, image_path=None):
    """Multi-pass approach with progressive refinement"""
    # First pass with ultra-strict prompt
    first_content, first_issues = extract_with_enhanced_prompt(image_data, "ultra_strict", image_path)
    
    if not first_content:
        return first_content, first_issues
    
    # If first pass has answer/description issues, try second pass with cleanup
    critical_issues = [issue for issue in first_issues if any(keyword in issue.lower() 
                      for keyword in ['answer', 'solution', 'correct', 'image shows', 'description'])]
    
    if critical_issues:
        st.warning(f"Critical issues detected in first pass: {len(critical_issues)} issues. Attempting cleanup...")
        
        # Second pass - verification and cleanup
        try:
            second_response = ollama.chat(
                model=CONFIG["primary_model"],
                messages=[
                    {
                        'role': 'user',
                        'content': f"""
TRANSCRIPTION VERIFICATION TASK:

Original extraction:
{first_content}

PROBLEMS DETECTED:
{chr(10).join(f"- {issue}" for issue in critical_issues)}

YOUR TASK: Fix the extraction by REMOVING all analysis, descriptions, and solutions.

KEEP ONLY:
- Question headers (like "QUESTION 2/6")
- The actual question text
- Point values (like "1 point")  
- Answer options with circle symbols (○)
- Mathematical expressions (using minimal LaTeX like \cos, \sin)

REMOVE COMPLETELY:
- Any text starting with "The image shows..."
- Any text about "correct answer" or solutions
- Any analysis or explanations
- Any descriptions of the image content

Output ONLY the cleaned transcription with no additional text.
""",
                        'images': [image_data]
                    }
                ],
                options={
                    "temperature": 0.0,  # Even lower temperature for cleanup
                    "num_predict": CONFIG["max_tokens"]
                }
            )
            
            second_content = second_response.message.content
            second_content = post_process_extraction(second_content, image_path)
            second_issues = validate_extraction(second_content, image_path)
            
            # Use second pass if it has fewer critical issues
            second_critical = [issue for issue in second_issues if any(keyword in issue.lower() 
                             for keyword in ['answer', 'solution', 'correct', 'image shows', 'description'])]
            
            if len(second_critical) < len(critical_issues):
                return second_content, second_issues
            else:
                return first_content, first_issues
                
        except Exception as e:
            return first_content, first_issues + [f"Verification error: {str(e)}"]
    
    return first_content, first_issues

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
        for key in ['ocr_result', 'extraction_method', 'model_used', 'processing_time', 'issues']:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()

st.markdown('<p style="margin-top: -20px;">Extract text and math with enhanced answer prevention - strict transcription only</p>', unsafe_allow_html=True)

st.markdown("---")

# Sidebar for settings and upload
with st.sidebar:
    st.header("Upload Image")
    uploaded_file = st.file_uploader("Choose an image...", type=CONFIG["image_types"])
    
    st.header("Extraction Settings")
    
    extraction_method = st.radio(
        "Extraction Method",
        ["Ultra Strict", "Example-Based", "Multi-Pass Verification"],
        index=2,
        help="Ultra Strict: Maximum answer prevention. Example-Based: Uses examples. Multi-Pass: Progressive refinement with cleanup."
    )
    
    # Map the radio selection to prompt types
    prompt_mapping = {
        "Ultra Strict": "ultra_strict",
        "Example-Based": "example_based", 
        "Multi-Pass Verification": "multi_pass"
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
                            if processing_steps:
                                st.image(processing_steps["grayscale"], caption="Grayscale", use_container_width=True)
                                st.image(processing_steps["threshold"], caption="Thresholded", use_container_width=True)
                                st.image(processing_steps["denoised"], caption="Denoised", use_container_width=True)
                        
                        # Convert processed image back to bytes for API
                        img_byte_arr = io.BytesIO()
                        processed_image.save(img_byte_arr, format=image.format if image.format else 'PNG')
                        image_data = img_byte_arr.getvalue()
                    
                    try:
                        # Use multi-pass verification if selected
                        if extraction_method == "Multi-Pass Verification":
                            extracted_content, issues = extract_with_multi_pass_verification(image_data, uploaded_file.name)
                        else:
                            # Use enhanced extraction with selected prompt
                            extracted_content, issues = extract_with_enhanced_prompt(image_data, selected_prompt, uploaded_file.name)
                        
                        if extracted_content:
                            st.session_state['ocr_result'] = extracted_content
                            st.session_state['extraction_method'] = extraction_method
                            st.session_state['model_used'] = CONFIG["primary_model"]
                            st.session_state['issues'] = issues
                            st.session_state['processing_time'] = time.time() - start_time
                        
                    except Exception as e:
                        st.error(f"Error processing image: {str(e)}")

# Main content area for results
if 'ocr_result' in st.session_state:
    # Display extraction metadata
    st.markdown("### Extraction Results")
    
    # Create columns for metadata
    meta_col1, meta_col2, meta_col3 = st.columns(3)
    
    with meta_col1:
        st.metric("Processing Time", f"{st.session_state['processing_time']:.2f}s")
    
    with meta_col2:
        st.metric("Model Used", st.session_state['model_used'])
    
    with meta_col3:
        st.metric("Method", st.session_state['extraction_method'])
    
    # Display issues with severity classification
    if 'issues' in st.session_state and st.session_state['issues']:
        st.markdown("#### Validation Results")
        
        critical_issues = []
        warning_issues = []
        
        for issue in st.session_state['issues']:
            if any(keyword in issue.lower() for keyword in ['answer', 'solution', 'correct', 'image shows', 'description']):
                critical_issues.append(issue)
            else:
                warning_issues.append(issue)
        
        if critical_issues:
            st.error("**Critical Issues Found:**")
            for issue in critical_issues:
                st.error(f"🚨 {issue}")
        
        if warning_issues:
            with st.expander("Minor Issues"):
                for issue in warning_issues:
                    st.warning(f"⚠️ {issue}")
    else:
        st.success("✅ No validation issues detected!")
    
    # Display the extraction results
    st.markdown("### Extracted Content")
    st.text_area("Copy this text:", st.session_state['ocr_result'], height=300)
    
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
st.markdown("Enhanced Math Equation Extractor | Anti-Answer Generation System")

# Enhanced debug section
with st.expander("🔍 Detailed Debug Analysis"):
    st.write("### Current Configuration:")
    st.json(CONFIG)
    
    if 'ocr_result' in st.session_state:
        st.write("### Extraction Analysis")
        
        # Basic stats
        st.write(f"**Text Length:** {len(st.session_state['ocr_result'])} characters")
        st.write(f"**Line Count:** {len(st.session_state['ocr_result'].split(chr(10)))}")
        
        # Answer detection analysis
        st.write("### 🚨 Answer Detection Check")
        answer_patterns = [
            (r"(?i)the correct (answer|is|solution)", "Direct answer identification"),
            (r"(?i)(answer|solution|result)\s*[:=]", "Answer provision"),
            (r"(?i)the image shows", "Image description"),
            (r"(?i)(therefore|thus|because)", "Analysis language"),
        ]
        
        answer_detected = False
        for pattern, description in answer_patterns:
            matches = re.findall(pattern, st.session_state['ocr_result'])
            if matches:
                st.error(f"❌ {description}: {matches}")
                answer_detected = True
            else:
                st.success(f"✅ No {description.lower()}")
        
        if not answer_detected:
            st.success("🎉 No answer generation detected!")
        
        # LaTeX formatting check
        st.write("### 📝 LaTeX Formatting Check")
        latex_commands = [
            ("\\cos", "Cosine function", True),
            ("\\sin", "Sine function", True),
            ("\\tan", "Tangent function", True),
            ("\\frac", "Fraction command", True),
            ("\\section", "Sectioning (should be avoided)", False),
            ("\\begin{array}", "Array environment (should be avoided)", False),
            ("\\textbf", "Text formatting (should be avoided)", False),
            ("\\(", "Inline math delimiter (minimal use preferred)", False),
        ]
        
        for command, description, is_good in latex_commands:
            count = st.session_state['ocr_result'].count(command)
            if count > 0:
                if is_good:
                    st.info(f"✓ {description}: {count} occurrences")
                else:
                    st.warning(f"⚠️ {description}: {count} occurrences")
            else:
                if is_good and command in ["\\cos", "\\sin"] and any(func in st.session_state['ocr_result'].lower() for func in ["cos", "sin"]):
                    st.warning(f"⚠️ {description} found without LaTeX formatting")
                elif not is_good:
                    st.success(f"✅ No {description.lower()}")
        
        # Content completeness check
        st.write("### 📋 Content Completeness Check")
        
        # Check for question header
        if re.search(r"QUESTION \d+/\d+", st.session_state['ocr_result']):
            st.success("✅ Question header found")
        else:
            st.error("❌ Question header missing")
        
        # Check for point value
        if re.search(r"\d+\s*point", st.session_state['ocr_result']):
            st.success("✅ Point value found")
        else:
            st.error("❌ Point value missing")
        
        # Check for options with circles
        circle_options = len(re.findall(r'○\s*[^○\n]+', st.session_state['ocr_result']))
        st.info(f"📊 Options with circles found: {circle_options}")
        
        if circle_options >= 4:
            st.success("✅ Sufficient options detected")
        elif circle_options >= 2:
            st.warning("⚠️ Some options detected, but may be incomplete")
        else:
            st.error("❌ Insufficient options detected")
        
        # Symbol placement check
        st.write("### 🎯 Symbol Placement Check")
        
        # Check for misplaced circles
        if re.search(r'○\s*QUESTION', st.session_state['ocr_result']):
            st.error("❌ Circle symbol incorrectly placed on question header")
        else:
            st.success("✅ No circles on question header")
        
        if re.search(r'○\s*\d+\s*point', st.session_state['ocr_result']):
            st.error("❌ Circle symbol incorrectly placed on point value")
        else:
            st.success("✅ No circles on point value")
        
        # Show raw text for manual inspection
        st.write("### 🔤 Raw Text Output")
        st.code(st.session_state['ocr_result'], language="text")
        
        # Character-by-character analysis for debugging
        if st.checkbox("Show Character Analysis"):
            st.write("### Character-by-Character Analysis")
            char_analysis = []
            for i, char in enumerate(st.session_state['ocr_result']):
                if char == '○':
                    char_analysis.append(f"Position {i}: Circle symbol (○)")
                elif char == '\n':
                    char_analysis.append(f"Position {i}: Newline")
                elif char == '\\':
                    char_analysis.append(f"Position {i}: Backslash (\\)")
            
            for analysis in char_analysis[:20]:  # Show first 20 special characters
                st.text(analysis)
            
            if len(char_analysis) > 20:
                st.text(f"... and {len(char_analysis) - 20} more special characters")

    else:
        st.info("No extraction results to analyze. Upload an image and extract content first.")