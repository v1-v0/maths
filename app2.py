import streamlit as st
import ollama
from PIL import Image
# import io # io is imported in the original but not explicitly used. Kept for consistency.

# Page configuration
st.set_page_config(
    page_title="Content Extractor with Llama 3.2 Vision", # Kept generic title
    page_icon="🦙",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title and description in main area
st.title("🦙 Content Extractor with Llama 3.2 Vision")

# Add clear button to top right
col1, col2 = st.columns([6,1])
with col2:
    if st.button("Clear 🗑️"):
        if 'ocr_result' in st.session_state:
            del st.session_state['ocr_result']
        st.rerun()

st.markdown('<p style="margin-top: -20px;">Extract text and LaTeX code from images using Llama 3.2 Vision!</p>', unsafe_allow_html=True)

st.markdown("---")
# Move upload controls to sidebar
with st.sidebar:
    st.header("Upload Image")
    uploaded_file = st.file_uploader("Choose an image...", type=['png', 'jpg', 'jpeg'])
    
    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image")
        
        if st.button("Extract Content 🔍", type="primary"):
            with st.spinner("Processing image..."):
                try:
                    # Refined prompt to extract only text and math LaTeX, no structural commands
                    new_prompt_content = r"""Your task is to extract content from an image. The output must ONLY consist of the visible plain text from the image and the LaTeX representation of any mathematical expressions found in the image.

                    Key Instructions (MUST be followed precisely):

                    1.  **Content Extraction:**
                        *   Extract ALL visible plain text verbatim (e.g., questions, headings, options, instructions, "1 point", etc.).
                        *   Convert ALL visual mathematical notations (symbols, variables, equations) into their exact LaTeX code. For example, if the image shows 'cos(mx+nx)', the LaTeX output for this part must be '\cos(mx+nx)'.

                    2.  **Output Formatting:**
                        *   Plain text should be output as plain text.
                        *   Mathematical LaTeX code should be enclosed in `\(` and `\)` for inline math, or `\[` and `\]` for display math if distinguishable. If you cannot reliably distinguish, use `\(` and `\)` for all mathematical LaTeX. Do NOT use `$` or `$$` delimiters.
                        *   Maintain the top-to-bottom sequence of elements as they appear in the image. Use newlines to separate distinct lines or blocks of content from the image.

                    3.  **Crucial Prohibitions (What NOT to do):**
                        *   **NO LaTeX Structural Commands:** You MUST NOT generate or use any LaTeX document structuring or formatting commands such as `\section`, `\item`, `\begin{enumerate}`, `\begin{itemize}`, `\tightlist`, `\documentclass`, `\usepackage`, `\begin{document}`, `\textbf`, `\textit`, etc. The only LaTeX generated should be for the mathematical content itself (e.g., `\cos`, `\sum`, `\alpha`) and its delimiters (`\(` `\)`, `\[` `\]`), if used.
                        *   **NO Explanations or External Text:** Do NOT add any commentary, summaries, explanations, or any text whatsoever that is not visually present in the original image.
                        *   **NO Simplification or Alteration:** Transcribe text and mathematical notations exactly as they appear. Do not simplify equations or alter the wording of the text.

                    The final output should be a clean sequence of plain text and LaTeX-formatted mathematical expressions, reflecting exactly what is in the image and nothing more.
                                        """

                    
                    response = ollama.chat(
                        model='llama3.2-vision',
                        messages=[{
                            'role': 'user',
                            'content': new_prompt_content,
                            'images': [uploaded_file.getvalue()]
                        }]
                    )
                    st.session_state['ocr_result'] = response.message.content
                except Exception as e:
                    st.error(f"Error processing image: {str(e)}")

# Main content area for results
if 'ocr_result' in st.session_state:
    st.markdown("### Extracted Content")
    st.text_area("Copy this text:", st.session_state['ocr_result'], height=300)
    
    st.markdown("### Preview")
    # Display as markdown to handle mixed text/math
    # With the new prompt, this should render plain text and LaTeX math (if wrapped in \( \))
    st.markdown(st.session_state['ocr_result'])
    
else:
    st.info("Upload an image and click 'Extract Content' to see the results here.")

# Footer
st.markdown("---")
st.markdown("Made with ❤️ using Llama Vision Model | [Report an Issue](https://github.com/patchy631/ai-engineering-hub/issues)")