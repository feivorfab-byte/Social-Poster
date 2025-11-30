import os
import base64
import json
import re
from flask import Flask, request, jsonify
from google import genai
from google.genai import types

app = Flask(__name__)

# --- CONFIGURATION ---
# Gemini 2.5 Flash: Fast for describing images and writing text
ANALYSIS_MODEL = 'gemini-2.5-flash'

# Gemini 3 Pro Image: The ONLY model that can "see" an input image and edit/regenerate it
IMAGE_GEN_MODEL = 'gemini-3-pro-image-preview'

api_key = os.environ.get("GOOGLE_API_KEY")
client = genai.Client(api_key=api_key)

# Helper to strip Markdown formatting from JSON responses
def clean_json_text(text):
    if not text: return "{}"
    text = text.strip()
    # Remove markdown code blocks if present
    if text.startswith("```json"):
        text = text[7:]
    if text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()

@app.route('/')
def home():
    return f"Backend Active. Analysis: {ANALYSIS_MODEL} | Gen: {IMAGE_GEN_MODEL}"

# ==========================================
# 1. STUDIO IMAGE ENDPOINTS
# ==========================================

@app.route('/analyze-image', methods=['POST'])
def analyze_image():
    """Analyze main reference image to get detailed description."""
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400
    
    file = request.files['image']
    prompt = request.form.get('prompt', 'Describe this image in detail.')
    
    try:
        image_bytes = file.read()
        response = client.models.generate_content(
            model=ANALYSIS_MODEL,
            contents=[types.Part.from_bytes(data=image_bytes, mime_type=file.content_type), prompt]
        )
        return jsonify({"description": response.text})
    except Exception as e:
        print(f"!!!!!!!!!!!!!! ANALYSIS ERROR: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/analyze-detail', methods=['POST'])
def analyze_detail():
    """Analyze a detail image using Gemini 2.5 Flash and return a concise label."""
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400
    
    file = request.files['image']
    prompt = request.form.get('prompt', 'Identify the key visual element, texture, or detail shown in this image and describe it in 10 words or less.')
    
    try:
        image_bytes = file.read()
        
        # Force clean output with explicit instruction
        full_prompt = f"{prompt}\n\nIMPORTANT: Output ONLY the descriptive label text. No quotes, no introductory phrases, no punctuation at the end. Just the raw label."
        
        response = client.models.generate_content(
            model=ANALYSIS_MODEL,
            contents=[types.Part.from_bytes(data=image_bytes, mime_type=file.content_type), full_prompt]
        )
        
        # Clean the response - remove quotes, extra whitespace, trailing punctuation
        label = response.text.strip().strip('"\'').rstrip('.')
        
        return jsonify({"label": label})
    except Exception as e:
        print(f"!!!!!!!!!!!!!! DETAIL ANALYSIS ERROR: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/analyze-background', methods=['POST'])
def analyze_background():
    """Analyze a background image and return both a short name and detailed description."""
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400
    
    file = request.files['image']
    
    prompt = """
    Look at this image and analyze the background/surface shown.
    
    Provide TWO things:
    1. NAME: A short 2-4 word name for this background (e.g., "Red Brick Wall", "Rustic Oak Wood", "White Marble")
    2. DESCRIPTION: A detailed description (20-40 words) of the material, texture, color, and characteristics that would help recreate this as a studio backdrop. Focus on: material type, color tones, texture, patterns, weathering, finish.
    
    Output as JSON:
    {"name": "Short Name", "description": "Detailed description..."}
    """
    
    try:
        image_bytes = file.read()
        
        response = client.models.generate_content(
            model=ANALYSIS_MODEL,
            contents=[types.Part.from_bytes(data=image_bytes, mime_type=file.content_type), prompt],
            config=types.GenerateContentConfig(response_mime_type="application/json")
        )
        
        result = json.loads(clean_json_text(response.text))
        
        # Ensure we have both fields
        name = result.get("name", "Custom Background")
        description = result.get("description", name)
        
        # Clean up name if too long
        words = name.split()
        if len(words) > 4:
            name = ' '.join(words[:4])
        
        return jsonify({"name": name, "description": description})
    except Exception as e:
        print(f"!!!!!!!!!!!!!! BACKGROUND ANALYSIS ERROR: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/generate-studio-image', methods=['POST'])
def generate_studio_image():
    """
    Generate studio image using Gemini 3 Pro with multi-reference support.
    
    Implements best practices from Gemini 3 Pro documentation:
    - Explicit image labeling in prompts
    - Main reference first, details in sequence
    - Preservation language for critical features
    - Single-source-of-truth master image approach
    - Optional background reference image for brand consistency
    """
    if 'image' not in request.files:
        return jsonify({"error": "No reference image provided"}), 400
        
    main_file = request.files['image']
    prompt = request.form.get('prompt', 'Turn this into a studio photograph.')
    quality = request.form.get('quality', '1K')
    
    # Check for background reference image
    background_image = None
    background_mime = None
    if 'backgroundImage' in request.files:
        bg_file = request.files['backgroundImage']
        background_image = bg_file.read()
        background_mime = bg_file.content_type
        print(f"--- Background reference image received: {len(background_image)} bytes ---")
    
    # Get detail images (up to 3)
    detail_images = []
    detail_labels = []
    
    for i in range(1, 4):  # detail1, detail2, detail3
        detail_key = f'detail{i}'
        # Support both label key formats
        label_key = f'detail{i}Label'  # iOS sends this
        label_key_alt = f'detail{i}_label'  # Alternative format
        
        if detail_key in request.files:
            detail_file = request.files[detail_key]
            detail_bytes = detail_file.read()
            detail_images.append((detail_bytes, detail_file.content_type))
            # Try both label key formats
            label = request.form.get(label_key) or request.form.get(label_key_alt) or f'Detail {i}'
            detail_labels.append(label)
    
    # Validate quality parameter (only 1K and 2K now)
    if quality not in ['1K', '2K']:
        quality = '1K'
    
    try:
        main_image_bytes = main_file.read()
        has_bg = background_image is not None
        print(f"--- Generating {quality} image with {len(detail_images)} detail ref(s), background ref: {has_bg} ---")
        print(f"--- Prompt snippet: {prompt[:150]}... ---")

        # Build content parts with explicit labeling per Gemini 3 Pro best practices
        # Order: Main reference first (establishes subject), then background, then details
        content_parts = []
        
        # Image 1: Main reference (master image - single source of truth)
        content_parts.append(types.Part.from_bytes(data=main_image_bytes, mime_type=main_file.content_type))
        
        # Image 2 (optional): Background reference
        if background_image:
            content_parts.append(types.Part.from_bytes(data=background_image, mime_type=background_mime))
        
        # Add detail images in sequence
        for detail_bytes, detail_mime in detail_images:
            content_parts.append(types.Part.from_bytes(data=detail_bytes, mime_type=detail_mime))
        
        # Build the explicitly labeled prompt
        labeled_prompt = build_labeled_prompt(prompt, detail_labels, has_background=has_bg)
        content_parts.append(labeled_prompt)

        # Retry logic - try up to 3 times
        max_retries = 3
        last_error = None
        
        for attempt in range(max_retries):
            try:
                response = client.models.generate_content(
                    model=IMAGE_GEN_MODEL,
                    contents=content_parts,
                    config=types.GenerateContentConfig(
                        response_modalities=["TEXT", "IMAGE"],
                        image_config=types.ImageConfig(
                            aspect_ratio="1:1",
                            image_size=quality
                        )
                    )
                )
                
                # Extract the image from the response parts
                if response.candidates and response.candidates[0].content.parts:
                    for part in response.candidates[0].content.parts:
                        if part.inline_data:
                            generated_image_b64 = base64.b64encode(part.inline_data.data).decode('utf-8')
                            print(f"--- {quality} Image generated on attempt {attempt + 1}. Size: {len(part.inline_data.data)} bytes ---")
                            return jsonify({
                                "message": "Image generated successfully",
                                "image": generated_image_b64
                            })
                
                # No image in response, retry
                print(f"--- Attempt {attempt + 1}: No image in response, retrying... ---")
                last_error = "Model returned no image"
                
            except Exception as retry_error:
                print(f"--- Attempt {attempt + 1} failed: {retry_error} ---")
                last_error = str(retry_error)
                continue
        
        # All retries failed
        print(f"!!!!!!!!!!!!!! IMAGE GEN FAILED after {max_retries} attempts: {last_error}")
        return jsonify({"error": f"Failed after {max_retries} attempts: {last_error}"}), 500

    except Exception as e:
        print(f"!!!!!!!!!!!!!! IMAGE GEN ERROR: {e}")
        return jsonify({"error": str(e)}), 500


def build_labeled_prompt(base_prompt, detail_labels, has_background=False):
    """
    Build an explicitly labeled prompt for studio product photography.
    
    Image order:
    - Image 1: Main product reference
    - Image 2: Background/surface material sample (if provided)
    - Image 3+: Detail references
    """
    lines = []
    
    # Label images
    lines.append("IMAGE REFERENCES:")
    lines.append("- Image 1: Product reference (recreate this exact object)")
    
    next_idx = 2
    
    if has_background:
        lines.append(f"- Image {next_idx}: Surface material sample (use this material for the background surface)")
        next_idx += 1
    
    for i, label in enumerate(detail_labels):
        lines.append(f"- Image {next_idx}: {label} (detail reference)")
        next_idx += 1
    
    lines.append("")
    lines.append("TASK:")
    lines.append(base_prompt)
    
    if has_background:
        lines.append("")
        lines.append("BACKGROUND SURFACE:")
        lines.append("Create a TOP-DOWN / FLAT LAY product photograph.")
        lines.append("- The camera is positioned directly above, looking straight down")
        lines.append("- The product sits on a FLAT HORIZONTAL SURFACE (like a table or floor)")
        lines.append("- Use Image 2 as a MATERIAL SAMPLE - match its texture, color, and pattern for the surface")
        lines.append("- Generate a fresh surface using this material type - do not copy Image 2 exactly")
        lines.append("- The surface should extend to fill the entire background")
        lines.append("- Create a natural contact shadow directly beneath/around the product")
        lines.append("- Apply the lighting style specified above consistently to both product and surface")
    
    lines.append("")
    lines.append("Recreate the exact object from Image 1 with high detail and accuracy.")
    
    if detail_labels:
        lines.append("Use detail reference images for accurate textures and fine details.")
    
    return "\n".join(lines)


# ==========================================
# 2. FINISHED PROJECT ENDPOINTS
# ==========================================

@app.route('/generate-interview-questions', methods=['POST'])
def generate_interview_questions():
    if 'image' not in request.files:
        return jsonify({"error": "No image"}), 400
    file = request.files['image']
    
    user_prompt = request.form.get('prompt')
    default_prompt = "Generate 3 interview questions."
    final_prompt = user_prompt if user_prompt else default_prompt
    
    try:
        image_bytes = file.read()
        response = client.models.generate_content(
            model=ANALYSIS_MODEL,
            contents=[types.Part.from_bytes(data=image_bytes, mime_type=file.content_type), final_prompt],
            config=types.GenerateContentConfig(response_mime_type="application/json")
        )
        
        text = clean_json_text(response.text)
        questions = json.loads(text)
        
        # Handle if AI wraps it in a dict key we didn't ask for
        if isinstance(questions, dict) and 'questions' in questions:
            questions = questions['questions']
            
        return jsonify({"questions": questions})
            
    except Exception as e:
        print(f"!!!!!!!!!!!!!! QUESTIONS ERROR: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/generate-captions', methods=['POST'])
def generate_captions():
    try:
        data = request.json
        qa_pairs = data.get('qa_pairs', [])
        tone = data.get('tone', 'balanced')
        length = data.get('length', 'medium')
        
        interview_text = "\n".join([f"Q: {item['question']}\nA: {item['answer']}" for item in qa_pairs])
        
        prompt = f"""
        Based on this artist interview, write 3 distinct Instagram captions.
        INTERVIEW TRANSCRIPT: {interview_text}
        CONFIGURATION: Tone: {tone}, Length: {length}
        
        Output exactly this JSON structure:
        {{
            "storytelling": "Narrative caption...",
            "expert": "Technical/Process caption...",
            "hybrid": "Mixed caption...",
            "hashtags": "#tag1 #tag2..."
        }}
        """
        
        response = client.models.generate_content(
            model=ANALYSIS_MODEL,
            contents=[prompt],
            config=types.GenerateContentConfig(response_mime_type="application/json")
        )
        
        text = clean_json_text(response.text)
        captions = json.loads(text)
        return jsonify(captions)

    except Exception as e:
        print(f"!!!!!!!!!!!!!! CAPTIONS ERROR: {e}")
        return jsonify({"error": str(e)}), 500

# ==========================================
# 3. DAILY POST ENDPOINTS
# ==========================================

@app.route('/analyze-daily-photo', methods=['POST'])
def analyze_daily_photo():
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400
    
    file = request.files['image']
    
    prompt = """
    You are a social media manager for an expert fabricator. Look at this work-in-progress photo.
    Generate 3 short, engaging 'writing prompts' or questions that the fabricator could answer to create a cool post.
    Keep them casual, curious, and specific to what you see in the photo.
    Output ONLY a raw JSON list of strings. Example: ["Prompt 1", "Prompt 2", "Prompt 3"]
    """
    
    try:
        image_bytes = file.read()
        response = client.models.generate_content(
            model=ANALYSIS_MODEL,
            contents=[types.Part.from_bytes(data=image_bytes, mime_type=file.content_type), prompt],
            config=types.GenerateContentConfig(response_mime_type="application/json")
        )
        prompts = json.loads(clean_json_text(response.text))
        
        # Handle wrapping
        if isinstance(prompts, dict) and 'prompts' in prompts:
            prompts = prompts['prompts']
            
        return jsonify({"prompts": prompts})
    except Exception as e:
        print(f"Daily Analysis Error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/generate-daily-caption', methods=['POST'])
def generate_daily_caption():
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400
    
    file = request.files['image']
    selected_prompt = request.form.get('selected_prompt', '')
    user_notes = request.form.get('user_notes', '')
    
    prompt = f"""
    You are a social media manager. Write a polished, engaging Instagram caption (100-150 words) for a fabricator's daily update.
    
    CONTEXT:
    - The fabricator chose this prompt: "{selected_prompt}"
    - Their rough answer/notes: "{user_notes}"
    
    INSTRUCTIONS:
    - Look at the image to add descriptive flair if the notes are brief.
    - Tone: Casual, expert, authentic. Not too "salesy".
    - Include 5-10 relevant hashtags at the bottom.
    
    Output exactly this JSON structure:
    {{
        "caption": "The full caption text...",
        "hashtags": "#tag1 #tag2..."
    }}
    """
    
    try:
        image_bytes = file.read()
        response = client.models.generate_content(
            model=ANALYSIS_MODEL,
            contents=[types.Part.from_bytes(data=image_bytes, mime_type=file.content_type), prompt],
            config=types.GenerateContentConfig(response_mime_type="application/json")
        )
        result = json.loads(clean_json_text(response.text))
        return jsonify(result)
    except Exception as e:
        print(f"Daily Caption Error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)
