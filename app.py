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
    """Analyze main reference image to get detailed description, camera angle, and any visible text."""
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400
    
    file = request.files['image']
    
    prompt = """
    Analyze this product photograph and provide THREE things:
    
    1. DESCRIPTION: A concise, highly specific physical description of the object shown. 
       Focus on geometry, shape, materials, surface textures, and colors. 
       Include any support structure if visible. Do NOT describe background or lighting.
    
    2. CAMERA_ANGLE: Describe the camera's perspective/angle in a short phrase.
       Examples: "top-down flat lay", "3/4 view from above-left", "straight-on front view", 
       "low angle looking up", "eye-level 3/4 view", "overhead slightly angled"
    
    3. VISIBLE_TEXT: List ALL visible text, labels, logos, brand names, numbers, or words 
       that appear anywhere on the object. Be EXACT - include the precise text, spelling, 
       capitalization, and location on the object. If no text is visible, use empty string.
       Examples: "Logo 'HELLO KITTY' on chest, 'Made in Japan' on tag", "Number '42' on side"
    
    Output as JSON:
    {"description": "Object description...", "camera_angle": "angle description", "visible_text": "exact text found or empty string"}
    """
    
    try:
        image_bytes = file.read()
        response = client.models.generate_content(
            model=ANALYSIS_MODEL,
            contents=[types.Part.from_bytes(data=image_bytes, mime_type=file.content_type), prompt],
            config=types.GenerateContentConfig(response_mime_type="application/json")
        )
        
        result = json.loads(clean_json_text(response.text))
        description = result.get("description", "")
        camera_angle = result.get("camera_angle", "3/4 view")
        visible_text = result.get("visible_text", "")
        
        # Append visible text to description if present
        if visible_text:
            description = f"{description} VISIBLE TEXT/LABELS: {visible_text}"
        
        return jsonify({
            "description": description,
            "camera_angle": camera_angle
        })
    except Exception as e:
        print(f"!!!!!!!!!!!!!! ANALYSIS ERROR: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/analyze-detail', methods=['POST'])
def analyze_detail():
    """Analyze a detail image using Gemini 2.5 Flash and return a concise label with any visible text."""
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400
    
    file = request.files['image']
    
    try:
        image_bytes = file.read()
        
        # Updated prompt to capture both the detail and any visible text
        full_prompt = """Analyze this close-up/detail image and provide:

1. A brief label (5-10 words) describing the key visual element, texture, or detail shown.

2. If there is ANY visible text, letters, numbers, logos, or words in this image, include them EXACTLY as they appear.

Output format - just the label, and if text exists, add it after a colon:
- If no text: "Pink fuzzy texture with hearts"
- If text present: "Chest area with embroidered logo: 'HELLO KITTY'"

Output ONLY the label. No quotes, no introductory phrases."""
        
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
    """Analyze a background image and return both a short name and highly detailed description."""
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400
    
    file = request.files['image']
    
    prompt = """
    Analyze this surface/material image in EXTREME detail for reproducible AI image generation.
    
    Your goal: Create a description so detailed and specific that an AI could generate this EXACT surface repeatedly with high consistency.
    
    Provide TWO things:
    
    1. NAME: A descriptive 2-4 word name (e.g., "Weathered Red Brick", "Dark Walnut Planks", "Polished Carrara Marble")
    
    2. DESCRIPTION: A comprehensive specification (100-150 words) covering ALL of these aspects:
    
    === MATERIAL IDENTIFICATION ===
    - Exact material type (e.g., "reclaimed pine wood planks", "handmade clay bricks", "honed granite slab")
    - Material origin/style if apparent (e.g., "industrial", "rustic farmhouse", "modern minimalist")
    
    === PRECISE COLOR SPECIFICATION ===
    - Primary/dominant color with SPECIFIC descriptors (e.g., "warm terracotta red with burnt orange undertones and occasional charcoal-gray fire marks" NOT just "red")
    - Secondary colors (list each with specific descriptors)
    - Color temperature (warm, cool, neutral)
    - Color saturation level (muted/desaturated, vibrant, rich)
    - Any color gradients, variations, or mottling patterns
    
    === TEXTURE CHARACTERISTICS ===
    - Surface roughness (mirror smooth, satin, slightly textured, rough, heavily textured, coarse)
    - Texture type (wood grain, stone pitting, fabric weave, brushed, hammered, sandblasted)
    - Texture depth and scale (fine hairline scratches, deep grooves, subtle undulations)
    - Directional texture (horizontal grain, vertical striations, random, radial)
    
    === PATTERN DETAILS ===
    - Pattern type (linear planks, rectangular bricks, hexagonal tiles, organic veining, random aggregate)
    - Element sizes and proportions (e.g., "4-inch wide planks", "standard brick format 2:1 ratio")
    - Spacing/gaps/joints (tight seams, visible grout lines, natural gaps)
    - Pattern regularity (uniform grid, staggered/offset, deliberately random)
    
    === SURFACE FINISH & LIGHT BEHAVIOR ===
    - Reflectivity level (matte/flat, eggshell, satin, semi-gloss, high-gloss, mirror)
    - How highlights appear (soft diffused, sharp specular, none)
    - Shadow behavior in texture (deep shadows in grooves, subtle shading, minimal shadow)
    
    === AGING & CHARACTER ===
    - Wear indicators (pristine, light wear, moderately distressed, heavily weathered, antique)
    - Specific wear patterns (rounded edges, faded areas, stains, chips, cracks)
    - Patina or finish changes over time
    - Any grout, mortar, filler (color, width, condition)
    
    === VISIBLE TEXT/MARKINGS (if any) ===
    - If there is ANY visible text, words, letters, numbers, logos, stamps, or printed markings on the surface, record them EXACTLY as they appear
    - Include location and style of text (e.g., "faint stamped 'MADE IN USA' in corner", "newspaper headlines visible")
    - If no text is present, omit this section
    
    Be EXTREMELY specific. Instead of "brown wood", say "medium-toned American walnut with prominent dark chocolate grain lines, honey-gold highlights between grain, and a hand-rubbed oil finish giving soft satin sheen."
    
    Output as JSON:
    {"name": "Short Name", "description": "Extremely detailed reproducible description..."}
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
        
        print(f"--- Background analyzed: {name} ---")
        print(f"--- Description length: {len(description)} chars ---")
        
        return jsonify({"name": name, "description": description})
    except Exception as e:
        print(f"!!!!!!!!!!!!!! BACKGROUND ANALYSIS ERROR: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/generate-studio-image', methods=['POST'])
def generate_studio_image():
    """
    Generate studio image using Gemini 3 Pro with multi-reference support.
    
    Implements best practices from Gemini 3 Pro documentation:
    - Indexed image labeling (Image 1, Image 2, etc.)
    - Main reference first, details in sequence
    - Preservation language for critical features
    - Single-source-of-truth master image approach
    - Background via TEXT DESCRIPTION only (not image) for consistency
    """
    if 'image' not in request.files:
        return jsonify({"error": "No reference image provided"}), 400
        
    main_file = request.files['image']
    prompt = request.form.get('prompt', 'Turn this into a studio photograph.')
    quality = request.form.get('quality', '1K')
    
    # Background is now description-only (no image sent)
    background_description = request.form.get('backgroundDescription', '')
    
    # Get detail images (up to 3)
    detail_images = []
    detail_labels = []
    
    for i in range(1, 4):  # detail1, detail2, detail3
        detail_key = f'detail{i}'
        label_key = f'detail{i}Label'
        label_key_alt = f'detail{i}_label'
        
        if detail_key in request.files:
            detail_file = request.files[detail_key]
            detail_bytes = detail_file.read()
            detail_images.append((detail_bytes, detail_file.content_type))
            label = request.form.get(label_key) or request.form.get(label_key_alt) or f'Detail {i}'
            detail_labels.append(label)
    
    # Validate quality parameter (only 1K and 2K now)
    if quality not in ['1K', '2K']:
        quality = '1K'
    
    try:
        main_image_bytes = main_file.read()
        has_bg_desc = bool(background_description)
        print(f"--- Generating {quality} image with {len(detail_images)} detail ref(s) ---")
        print(f"--- Background description: {background_description[:100] if background_description else 'None'}... ---")
        print(f"--- Prompt snippet: {prompt[:150]}... ---")

        # Build content parts with indexed labeling per Gemini 3 Pro best practices
        # Image 1: Main reference (master image - single source of truth)
        content_parts = []
        content_parts.append(types.Part.from_bytes(data=main_image_bytes, mime_type=main_file.content_type))
        
        # Add detail images in sequence (Image 2, 3, 4...)
        for detail_bytes, detail_mime in detail_images:
            content_parts.append(types.Part.from_bytes(data=detail_bytes, mime_type=detail_mime))
        
        # Build the indexed labeled prompt
        labeled_prompt = build_labeled_prompt(prompt, detail_labels, background_description)
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


def build_labeled_prompt(base_prompt, detail_labels, background_description=""):
    """
    Build an indexed labeled prompt for studio product photography.
    
    Following Gemini 3 Pro best practices:
    - Indexed image labeling (Image 1, Image 2, etc.)
    - Single source of truth (master image)
    - Background via detailed text description for consistency
    - Explicit preservation language
    
    Image order:
    - Image 1: Main product reference (master)
    - Image 2+: Detail references
    """
    lines = []
    
    # Indexed image labeling
    lines.append("IMAGE REFERENCES:")
    lines.append("Image 1: Master product reference - this is the EXACT object to recreate. Preserve its shape, proportions, colors, materials, and all visible details with high fidelity.")
    
    next_idx = 2
    for i, label in enumerate(detail_labels):
        lines.append(f"Image {next_idx}: Close-up detail reference showing '{label}' - use this for accurate texture/detail in this area.")
        next_idx += 1
    
    lines.append("")
    lines.append("=" * 50)
    lines.append("TASK: Professional Studio Product Photography")
    lines.append("=" * 50)
    lines.append("")
    lines.append(base_prompt)
    
    # Background handling - detailed description for consistent reproduction
    if background_description:
        lines.append("")
        lines.append("=" * 50)
        lines.append("BACKGROUND SURFACE - CRITICAL")
        lines.append("=" * 50)
        lines.append("")
        lines.append("Generate the background surface using this EXACT specification:")
        lines.append("")
        lines.append(f'"""{background_description}"""')
        lines.append("")
        lines.append("IMPORTANT: Follow the material description PRECISELY. Match the exact colors, textures, patterns, finish, and aging/wear characteristics described. The surface must look like the specific material described, not a generic version of it.")
        lines.append("")
    
    # Core requirements
    lines.append("=" * 50)
    lines.append("REQUIREMENTS")
    lines.append("=" * 50)
    lines.append("")
    lines.append("1. OBJECT FIDELITY")
    lines.append("   - Recreate the EXACT object from Image 1")
    lines.append("   - Match all proportions, shapes, colors, and materials precisely")
    lines.append("   - This is a recreation, not a modification - preserve everything")
    if detail_labels:
        lines.append("   - Use detail reference images to ensure accurate fine details and textures")
    lines.append("")
    
    lines.append("2. UNIFIED LIGHTING")
    lines.append("   - Product and background must share the SAME light source(s)")
    lines.append("   - Light direction must be consistent across entire image")
    lines.append("   - Background surface shows highlights and shadows matching the product")
    lines.append("   - Natural contact shadow where product meets surface (soft, darkest at contact)")
    lines.append("   - Subtle color spill from background onto product edges")
    lines.append("")
    
    lines.append("3. REAL PHOTOGRAPH - NOT 3D RENDER")
    lines.append("   - This MUST look like an actual photograph taken with a real camera")
    lines.append("   - DO NOT create a 3D render, CGI, or computer-generated image")
    lines.append("   - Include natural photographic characteristics:")
    lines.append("     * Subtle film/sensor grain")
    lines.append("     * Natural depth of field (slight softness away from focus plane)")
    lines.append("     * Real material textures (fabric weave, surface imperfections, natural wear)")
    lines.append("     * Authentic lighting falloff (not perfectly uniform)")
    lines.append("     * Micro-details that exist in real objects (dust, fibers, slight irregularities)")
    lines.append("   - Materials should look REAL, not plastic or CG-perfect")
    lines.append("   - Avoid the 'too clean' or 'too perfect' look of 3D renders")
    lines.append("   - No halos, artifacts, or unnatural edges")
    
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
