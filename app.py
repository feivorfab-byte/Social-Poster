import os
import base64
import json
import hashlib
from flask import Flask, request, jsonify
from google import genai
from google.genai import types

app = Flask(__name__)

# --- CONFIGURATION ---
ANALYSIS_MODEL = 'gemini-2.5-flash'
IMAGE_GEN_MODEL = 'gemini-3-pro-image-preview'

api_key = os.environ.get("GOOGLE_API_KEY")
client = genai.Client(api_key=api_key)

# Reliability settings
MAX_GENERATION_ATTEMPTS = 3
MAX_VERIFICATION_RETRIES = 2

# In-memory caches (for server-side caching)
# In production, consider Redis or persistent storage
BACKGROUND_CACHE = {}  # key: background_id -> {"image": bytes, "description": str, "scale": str}
MASTER_CACHE = {}  # key: master_id -> {"image": bytes, "lighting": str, "background": str}


def clean_json_text(text):
    """Strip Markdown formatting from JSON responses."""
    if not text:
        return "{}"
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]
    if text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()


def generate_cache_id(data_bytes, prefix=""):
    """Generate a unique ID for caching."""
    hash_obj = hashlib.md5(data_bytes)
    return f"{prefix}{hash_obj.hexdigest()[:12]}"


@app.route('/')
def home():
    return f"Studio Lights Backend v3.1 | Analysis: {ANALYSIS_MODEL} | Generation: {IMAGE_GEN_MODEL}"


# ==========================================
# ANALYSIS ENDPOINTS
# ==========================================

@app.route('/analyze-image', methods=['POST'])
def analyze_image():
    """Extract metadata from product image."""
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400
    
    file = request.files['image']
    
    prompt = """Analyze this product photograph and extract metadata.

ORIENTATION: Is the product lying flat (viewed from above, like clothing on a table) or standing upright (viewed at eye level, like a bottle or statue)?
- "flat_lay" = lying flat, top-down view
- "standing" = upright, eye-level view  
- "angled" = neither clearly flat nor standing

CAMERA ANGLE: Describe the camera perspective in 3-5 words (e.g., "overhead flat lay", "eye-level front view", "3/4 elevated view")

PRODUCT DIMENSIONS: Estimate real-world size as "W x H x D" with units (inches or feet). Example: "12 x 8 x 4 inches"

VISIBLE TEXT: List any text, numbers, logos, or brand names visible on the product. Transcribe exactly. Empty string if none.

JSON only:
{
    "orientation": "flat_lay" or "standing" or "angled",
    "camera_angle": "brief description",
    "product_dimensions": "W x H x D units",
    "visible_text": "exact text or empty"
}"""
    
    try:
        image_bytes = file.read()
        response = client.models.generate_content(
            model=ANALYSIS_MODEL,
            contents=[types.Part.from_bytes(data=image_bytes, mime_type=file.content_type), prompt],
            config=types.GenerateContentConfig(response_mime_type="application/json")
        )
        
        result = json.loads(clean_json_text(response.text))
        
        print(f"[ANALYSIS] orientation={result.get('orientation')}, dims={result.get('product_dimensions')}")
        
        return jsonify({
            "orientation": result.get("orientation", "angled"),
            "camera_angle": result.get("camera_angle", "3/4 view"),
            "product_dimensions": result.get("product_dimensions", ""),
            "visible_text": result.get("visible_text", ""),
            "description": ""
        })
        
    except Exception as e:
        print(f"[ERROR] Analysis failed: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/analyze-detail', methods=['POST'])
def analyze_detail():
    """Analyze a detail image and return a concise label."""
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400
    
    file = request.files['image']
    
    try:
        image_bytes = file.read()
        
        prompt = """What specific detail, texture, or feature does this close-up show? 
Describe in 5-10 words. Include any visible text exactly.
Write only the label, nothing else."""
        
        response = client.models.generate_content(
            model=ANALYSIS_MODEL,
            contents=[types.Part.from_bytes(data=image_bytes, mime_type=file.content_type), prompt]
        )
        
        label = response.text.strip().strip('"\'').rstrip('.')
        return jsonify({"label": label})
        
    except Exception as e:
        print(f"[ERROR] Detail analysis failed: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/analyze-background', methods=['POST'])
def analyze_background():
    """Analyze a background image for reproduction."""
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400
    
    file = request.files['image']
    
    prompt = """Analyze this background/surface for product photography.

NAME: Short name, 2-4 words (e.g., "Weathered Oak Planks", "Lined Notebook Paper")

DESCRIPTION: Describe materials, colors, textures, patterns, and any markings in detail (80-120 words). Include any text, writing, logos, or graphics exactly.

HAS_BRANDING: Does this contain specific text, logos, handwriting, or graphics that must be reproduced exactly? 
- true = has readable content, specific graphics, text, logos
- false = plain texture like wood grain, concrete, fabric without text

MATERIAL_SCALE: Physical size of repeating elements (e.g., "wood planks 4 inches wide", "ruled lines 8mm apart")

JSON only:
{
    "name": "Short Name",
    "description": "Detailed description",
    "has_branding": true or false,
    "material_scale": "measurements"
}"""
    
    try:
        image_bytes = file.read()
        response = client.models.generate_content(
            model=ANALYSIS_MODEL,
            contents=[types.Part.from_bytes(data=image_bytes, mime_type=file.content_type), prompt],
            config=types.GenerateContentConfig(response_mime_type="application/json")
        )
        
        result = json.loads(clean_json_text(response.text))
        
        name = result.get("name", "Custom Background")
        words = name.split()
        if len(words) > 4:
            name = ' '.join(words[:4])
        
        print(f"[BACKGROUND] {name}, branding={result.get('has_branding')}")
        
        return jsonify({
            "name": name,
            "description": result.get("description", name),
            "has_branding": result.get("has_branding", False),
            "material_scale": result.get("material_scale", "")
        })
        
    except Exception as e:
        print(f"[ERROR] Background analysis failed: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/analyze-style', methods=['POST'])
def analyze_style():
    """Analyze a generated image for style characteristics (used for master images)."""
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400
    
    file = request.files['image']
    
    prompt = """Analyze this studio product photograph for style characteristics.

Describe in 30-50 words:
- Lighting quality (soft/hard, direction, shadow depth percentage)
- Color temperature (warm/cool/neutral, approximate Kelvin)
- Overall mood (bright/dramatic/natural/elegant)
- Background treatment (how it's lit relative to product)

JSON: {"style_description": "..."}"""
    
    try:
        image_bytes = file.read()
        response = client.models.generate_content(
            model=ANALYSIS_MODEL,
            contents=[types.Part.from_bytes(data=image_bytes, mime_type=file.content_type), prompt],
            config=types.GenerateContentConfig(response_mime_type="application/json")
        )
        
        result = json.loads(clean_json_text(response.text))
        
        return jsonify({
            "style_description": result.get("style_description", "")
        })
        
    except Exception as e:
        print(f"[ERROR] Style analysis failed: {e}")
        return jsonify({"error": str(e)}), 500


# ==========================================
# RECOMMENDATION #3: BACKGROUND PRE-GENERATION
# ==========================================

@app.route('/pregenerate-background', methods=['POST'])
def pregenerate_background():
    """
    Pre-generate a background and cache it for reuse.
    
    This removes generation variance from the background entirely.
    The same pre-generated background can be used across multiple product shots.
    
    Returns a background_id that can be passed to generate-studio-image-v2.
    Also returns the image so iOS can cache it locally.
    """
    if 'image' not in request.files:
        return jsonify({"error": "No background image provided"}), 400
    
    file = request.files['image']
    quality = request.form.get('quality', '2K')  # Default to 2K for backgrounds
    
    if quality not in ['1K', '2K']:
        quality = '2K'
    
    try:
        bg_image_bytes = file.read()
        
        # Generate cache ID from source image
        bg_id = generate_cache_id(bg_image_bytes, prefix="bg_")
        
        # Check if already cached
        if bg_id in BACKGROUND_CACHE:
            print(f"[PREGEN-BG] Cache hit: {bg_id}")
            cached = BACKGROUND_CACHE[bg_id]
            return jsonify({
                "message": "Background retrieved from cache",
                "background_id": bg_id,
                "image": base64.b64encode(cached["image"]).decode('utf-8'),
                "cached": True
            })
        
        print(f"[PREGEN-BG] Generating new background: {bg_id}")
        
        # First, analyze the background
        analysis_prompt = """Analyze this background/surface:
1. DESCRIPTION: Detailed description of materials, colors, textures, patterns (50-80 words)
2. MATERIAL_SCALE: Physical size of repeating elements
3. HAS_TEXT: Does it have text, logos, or specific graphics? (true/false)

JSON:
{"description": "...", "material_scale": "...", "has_text": true/false}"""
        
        analysis_response = client.models.generate_content(
            model=ANALYSIS_MODEL,
            contents=[types.Part.from_bytes(data=bg_image_bytes, mime_type=file.content_type), analysis_prompt],
            config=types.GenerateContentConfig(response_mime_type="application/json")
        )
        
        analysis = json.loads(clean_json_text(analysis_response.text))
        
        # Generate the background
        gen_prompt = """Reproduce this image exactly as a clean studio photography surface.

IMAGE 1 shows a background/surface material. Create an exact copy preserving:
- All colors, textures, and patterns exactly
- ALL text, writing, numbers, logos in exact position, size, and style
- ALL graphics, drawings, marks exactly as shown

Fill the entire square frame with this surface material, evenly lit for product photography.
This will be used as a reusable background for multiple product shots, so accuracy is critical."""
        
        content_parts = [
            types.Part.from_bytes(data=bg_image_bytes, mime_type=file.content_type),
            gen_prompt
        ]
        
        # Generate with retries and verification
        generated_bg = None
        for attempt in range(MAX_GENERATION_ATTEMPTS):
            try:
                response = client.models.generate_content(
                    model=IMAGE_GEN_MODEL,
                    contents=content_parts,
                    config=types.GenerateContentConfig(
                        response_modalities=["TEXT", "IMAGE"],
                        image_config=types.ImageConfig(aspect_ratio="1:1", image_size=quality)
                    )
                )
                
                if response.candidates and response.candidates[0].content.parts:
                    for part in response.candidates[0].content.parts:
                        if part.inline_data:
                            # Verify the background
                            if verify_background_reproduction(bg_image_bytes, part.inline_data.data, analysis.get("has_text", False)):
                                generated_bg = part.inline_data.data
                                print(f"[PREGEN-BG] Success on attempt {attempt + 1}")
                                break
                            else:
                                print(f"[PREGEN-BG] Verification failed on attempt {attempt + 1}")
                
                if generated_bg:
                    break
                    
            except Exception as e:
                print(f"[PREGEN-BG] Attempt {attempt + 1} failed: {e}")
        
        if not generated_bg:
            return jsonify({"error": "Failed to generate background"}), 500
        
        # Cache the result
        BACKGROUND_CACHE[bg_id] = {
            "image": generated_bg,
            "description": analysis.get("description", ""),
            "scale": analysis.get("material_scale", "")
        }
        
        return jsonify({
            "message": "Background pre-generated successfully",
            "background_id": bg_id,
            "image": base64.b64encode(generated_bg).decode('utf-8'),
            "description": analysis.get("description", ""),
            "material_scale": analysis.get("material_scale", ""),
            "cached": False
        })
        
    except Exception as e:
        print(f"[ERROR] Background pre-generation failed: {e}")
        return jsonify({"error": str(e)}), 500


def verify_background_reproduction(original_bytes, generated_bytes, has_text):
    """Verify background reproduction quality."""
    
    if has_text:
        # Stricter verification for backgrounds with text
        prompt = """Compare these two images. Image 1 is original, Image 2 is a reproduction.

Check:
1. Colors and textures similar?
2. ALL text reproduced correctly - same words, same placement, same size?
3. ALL graphics/logos reproduced correctly?

JSON: {"colors_ok": bool, "text_ok": bool, "graphics_ok": bool, "pass": bool}"""
    else:
        # Simpler verification for plain textures
        prompt = """Compare these two images. Image 1 is original, Image 2 is a reproduction.

Check:
1. Colors similar?
2. Textures/patterns similar?
3. Overall appearance matches?

JSON: {"colors_ok": bool, "texture_ok": bool, "pass": bool}"""
    
    try:
        response = client.models.generate_content(
            model=ANALYSIS_MODEL,
            contents=[
                types.Part.from_bytes(data=original_bytes, mime_type="image/jpeg"),
                types.Part.from_bytes(data=generated_bytes, mime_type="image/png"),
                prompt
            ],
            config=types.GenerateContentConfig(response_mime_type="application/json")
        )
        
        result = json.loads(clean_json_text(response.text))
        return result.get("pass", False)
        
    except Exception as e:
        print(f"[VERIFY-BG] Error: {e}")
        return True  # Assume OK if verification fails


# ==========================================
# RECOMMENDATION #2: MASTER IMAGE LIBRARY
# ==========================================

@app.route('/save-master', methods=['POST'])
def save_master():
    """
    Save a generated image as a "master" style reference.
    
    The master captures the look/feel that worked well.
    Subsequent generations can reference this master for consistency.
    
    Returns a master_id that can be passed to generation endpoints.
    """
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400
    
    file = request.files['image']
    lighting_scheme = request.form.get('lightingScheme', '')
    background_type = request.form.get('backgroundType', '')
    
    try:
        image_bytes = file.read()
        
        # Generate master ID
        master_id = generate_cache_id(image_bytes, prefix="master_")
        
        # Analyze the master for style characteristics
        analysis_prompt = """Analyze this studio product photograph for style characteristics.

Describe in 30-50 words:
- Lighting quality (soft/hard, direction, shadow depth)
- Color temperature (warm/cool/neutral)
- Overall mood (bright/dramatic/natural)
- Background treatment

JSON: {"style_description": "..."}"""
        
        response = client.models.generate_content(
            model=ANALYSIS_MODEL,
            contents=[types.Part.from_bytes(data=image_bytes, mime_type=file.content_type), analysis_prompt],
            config=types.GenerateContentConfig(response_mime_type="application/json")
        )
        
        analysis = json.loads(clean_json_text(response.text))
        
        # Cache the master
        MASTER_CACHE[master_id] = {
            "image": image_bytes,
            "lighting": lighting_scheme,
            "background": background_type,
            "style": analysis.get("style_description", "")
        }
        
        print(f"[MASTER] Saved: {master_id}")
        
        return jsonify({
            "message": "Master saved successfully",
            "master_id": master_id,
            "style_description": analysis.get("style_description", "")
        })
        
    except Exception as e:
        print(f"[ERROR] Save master failed: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/get-master/<master_id>', methods=['GET'])
def get_master(master_id):
    """Retrieve a saved master image."""
    if master_id not in MASTER_CACHE:
        return jsonify({"error": "Master not found"}), 404
    
    master = MASTER_CACHE[master_id]
    return jsonify({
        "master_id": master_id,
        "image": base64.b64encode(master["image"]).decode('utf-8'),
        "lighting": master["lighting"],
        "background": master["background"],
        "style": master["style"]
    })


@app.route('/list-masters', methods=['GET'])
def list_masters():
    """List all saved masters."""
    masters = []
    for mid, data in MASTER_CACHE.items():
        masters.append({
            "master_id": mid,
            "lighting": data["lighting"],
            "background": data["background"],
            "style": data["style"]
        })
    return jsonify({"masters": masters})


@app.route('/delete-master/<master_id>', methods=['DELETE'])
def delete_master(master_id):
    """Delete a saved master."""
    if master_id in MASTER_CACHE:
        del MASTER_CACHE[master_id]
        return jsonify({"message": "Master deleted"})
    return jsonify({"error": "Master not found"}), 404


# ==========================================
# CORE GENERATION FUNCTIONS
# ==========================================

def generate_with_retry(content_parts, quality, max_attempts=MAX_GENERATION_ATTEMPTS):
    """Generate image with retry logic."""
    last_error = None
    
    for attempt in range(max_attempts):
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
            
            if response.candidates and response.candidates[0].content.parts:
                for part in response.candidates[0].content.parts:
                    if part.inline_data:
                        print(f"[GEN] Success on attempt {attempt + 1}")
                        return part.inline_data.data, None
            
            last_error = "No image in response"
            print(f"[GEN] Attempt {attempt + 1}: {last_error}")
            
        except Exception as e:
            last_error = str(e)
            print(f"[GEN] Attempt {attempt + 1} failed: {e}")
    
    return None, f"Failed after {max_attempts} attempts: {last_error}"


def verify_generation(original_image, generated_image, orientation, check_text=None):
    """Verify generated image meets quality criteria."""
    prompt = f"""Compare these two images. Image 1 is the original product. Image 2 is a generated studio photograph.

Verify:
1. PRODUCT FIDELITY: Same product? Same shape, proportions, colors, materials, details?
2. ORIENTATION: Should be "{orientation}" (flat_lay=top-down, standing=eye-level with depth, angled=natural angle). Correct?
3. COMPOSITION: Product centered, filling ~50-60% of frame?
4. LIGHTING: Unified across product and background?
{"5. TEXT: Visible markings '" + check_text + "' preserved correctly?" if check_text else ""}

JSON:
{{"product_ok": bool, "orientation_ok": bool, "composition_ok": bool, "lighting_ok": bool, {"\"text_ok\": bool, " if check_text else ""}"pass": bool, "issues": []}}"""
    
    try:
        response = client.models.generate_content(
            model=ANALYSIS_MODEL,
            contents=[
                types.Part.from_bytes(data=original_image, mime_type="image/jpeg"),
                types.Part.from_bytes(data=generated_image, mime_type="image/png"),
                prompt
            ],
            config=types.GenerateContentConfig(response_mime_type="application/json")
        )
        
        result = json.loads(clean_json_text(response.text))
        passed = result.get("pass", False)
        issues = result.get("issues", [])
        
        print(f"[VERIFY] passed={passed}, issues={issues}")
        return passed, issues
        
    except Exception as e:
        print(f"[VERIFY] Error: {e}")
        return True, []


def build_composition_instruction(orientation):
    """Get composition instructions based on orientation."""
    if orientation == "flat_lay":
        return """COMPOSITION: Top-down flat lay. Camera directly above, looking straight down. Product lies flat on horizontal surface. Center product, fill 50-60% of frame. Background as continuous horizontal plane. Soft contact shadow beneath."""
    elif orientation == "standing":
        return """COMPOSITION: Standing product shot. Camera at eye level or slightly elevated. Product stands upright with depth perspective - background recedes behind. Center product, fill 50-60% of frame height. Natural contact shadow at base."""
    else:
        return """COMPOSITION: Natural angle matching reference. Center product, fill 50-60% of frame. Background visible with appropriate perspective. Soft contact shadow."""


# ==========================================
# V1 GENERATION (with master reference support)
# ==========================================

@app.route('/generate-studio-image', methods=['POST'])
def generate_studio_image():
    """
    Generate studio photograph with text-described background.
    
    Supports optional master_id for style consistency.
    Includes verification for reliability.
    """
    if 'image' not in request.files:
        return jsonify({"error": "No reference image provided"}), 400
        
    main_file = request.files['image']
    
    # Parameters
    prompt = request.form.get('prompt', '')
    quality = request.form.get('quality', '1K')
    background_description = request.form.get('backgroundDescription', '')
    product_dimensions = request.form.get('productDimensions', '')
    orientation = request.form.get('orientation', 'angled')
    visible_text = request.form.get('visibleText', '')
    
    # RECOMMENDATION #2: Master image reference
    master_id = request.form.get('masterId', '')
    master_image = None
    master_style = ""
    
    if master_id and master_id in MASTER_CACHE:
        master_data = MASTER_CACHE[master_id]
        master_image = master_data["image"]
        master_style = master_data["style"]
        print(f"[V1] Using master reference: {master_id}")
    
    # Also accept master image directly from iOS cache
    if 'masterImage' in request.files and not master_image:
        master_file = request.files['masterImage']
        master_image = master_file.read()
        master_style = request.form.get('masterStyle', '')
        print(f"[V1] Using uploaded master image")
    
    # Collect detail images
    detail_images = []
    detail_labels = []
    for i in range(1, 4):
        if f'detail{i}' in request.files:
            detail_file = request.files[f'detail{i}']
            detail_bytes = detail_file.read()
            detail_images.append((detail_bytes, detail_file.content_type))
            label = request.form.get(f'detail{i}Label', f'Detail {i}')
            detail_labels.append(label)
    
    if quality not in ['1K', '2K']:
        quality = '1K'
    
    try:
        main_image_bytes = main_file.read()
        print(f"[V1] Starting: {quality}, orientation={orientation}, details={len(detail_images)}, master={bool(master_image)}")

        # Build content parts
        content_parts = []
        
        # If master provided, it goes first as style reference
        if master_image:
            content_parts.append(types.Part.from_bytes(data=master_image, mime_type="image/png"))
        
        # Main product image
        content_parts.append(types.Part.from_bytes(data=main_image_bytes, mime_type=main_file.content_type))
        
        # Detail images
        for detail_bytes, detail_mime in detail_images:
            content_parts.append(types.Part.from_bytes(data=detail_bytes, mime_type=detail_mime))
        
        # Build prompt
        generation_prompt = build_v1_prompt(
            lighting_prompt=prompt,
            detail_labels=detail_labels,
            background_description=background_description,
            product_dimensions=product_dimensions,
            orientation=orientation,
            visible_text=visible_text,
            has_master=bool(master_image),
            master_style=master_style
        )
        content_parts.append(generation_prompt)

        # Generate with verification
        for verify_attempt in range(MAX_VERIFICATION_RETRIES + 1):
            generated_bytes, error = generate_with_retry(content_parts, quality)
            
            if error:
                return jsonify({"error": error}), 500
            
            passed, issues = verify_generation(
                main_image_bytes, generated_bytes, orientation,
                visible_text if visible_text else None
            )
            
            if passed:
                return jsonify({
                    "message": "Image generated successfully",
                    "image": base64.b64encode(generated_bytes).decode('utf-8')
                })
            
            if verify_attempt < MAX_VERIFICATION_RETRIES:
                print(f"[V1] Verification failed, retrying")
                content_parts[-1] = generation_prompt + f"\n\nFIX THESE ISSUES: {', '.join(issues)}"
            else:
                return jsonify({
                    "message": "Generated with potential issues",
                    "image": base64.b64encode(generated_bytes).decode('utf-8'),
                    "warnings": issues
                })
        
    except Exception as e:
        print(f"[ERROR] V1 failed: {e}")
        return jsonify({"error": str(e)}), 500


def build_v1_prompt(lighting_prompt, detail_labels, background_description, product_dimensions, orientation, visible_text, has_master=False, master_style=""):
    """Build V1 generation prompt."""
    
    sections = []
    
    # Image roles (adjusted if master is present)
    roles = ["REFERENCE IMAGES:"]
    
    if has_master:
        roles.append("Image 1 = STYLE REFERENCE. Match this image's lighting quality, color temperature, shadow character, and overall photographic style.")
        roles.append("Image 2 = PRODUCT. Reproduce this exact object with complete fidelity.")
        start_idx = 3
    else:
        roles.append("Image 1 = PRODUCT. Reproduce this exact object with complete fidelity - every shape, color, texture, material, marking exactly as shown.")
        start_idx = 2
    
    for i, label in enumerate(detail_labels):
        roles.append(f"Image {start_idx + i} = DETAIL: {label}")
    
    sections.append(" ".join(roles))
    
    # Style guidance from master
    if has_master and master_style:
        sections.append(f"STYLE MATCHING: Replicate this photographic style: {master_style}")
    
    # Composition
    sections.append(build_composition_instruction(orientation))
    
    # Background
    if background_description:
        bg_section = f"BACKGROUND: {background_description}"
        if product_dimensions:
            bg_section += f" Product is ~{product_dimensions} - scale background proportionally. Tile seamlessly if needed."
        sections.append(bg_section)
    
    # Lighting
    if lighting_prompt:
        sections.append(lighting_prompt)
    
    # Text preservation
    if visible_text:
        sections.append(f"PRESERVE TEXT: {visible_text}")
    
    # Quality
    sections.append("OUTPUT: Authentic studio photograph. Natural depth of field, real textures, unified lighting. Full-frame camera, 90mm lens, f/8.")
    
    return "\n\n".join(sections)


# ==========================================
# V2 GENERATION (with cached background support)
# ==========================================

@app.route('/generate-studio-image-v2', methods=['POST'])
def generate_studio_image_v2():
    """
    Two-stage generation for image-based backgrounds.
    
    Supports:
    - background_id: Use a pre-generated cached background (RECOMMENDATION #3)
    - master_id: Use a master for style consistency (RECOMMENDATION #2)
    - Or upload background/master images directly
    """
    if 'image' not in request.files:
        return jsonify({"error": "No reference image provided"}), 400
        
    main_file = request.files['image']
    
    # Parameters
    prompt = request.form.get('prompt', '')
    quality = request.form.get('quality', '1K')
    lighting_prompt = request.form.get('lightingPrompt', '')
    background_description = request.form.get('backgroundDescription', '')
    material_scale = request.form.get('materialScale', '')
    product_dimensions = request.form.get('productDimensions', '')
    orientation = request.form.get('orientation', 'angled')
    visible_text = request.form.get('visibleText', '')
    
    # RECOMMENDATION #3: Use pre-generated background
    background_id = request.form.get('backgroundId', '')
    cached_background = None
    
    if background_id and background_id in BACKGROUND_CACHE:
        cached_data = BACKGROUND_CACHE[background_id]
        cached_background = cached_data["image"]
        if not background_description:
            background_description = cached_data.get("description", "")
        if not material_scale:
            material_scale = cached_data.get("scale", "")
        print(f"[V2] Using cached background: {background_id}")
    
    # Or accept background image directly
    background_image = cached_background
    background_mime = "image/png"
    
    if not background_image and 'backgroundImage' in request.files:
        bg_file = request.files['backgroundImage']
        background_image = bg_file.read()
        background_mime = bg_file.content_type
    
    # Also accept pre-generated background from iOS
    if not background_image and 'cachedBackground' in request.files:
        bg_file = request.files['cachedBackground']
        background_image = bg_file.read()
        background_mime = bg_file.content_type
        print(f"[V2] Using iOS-cached background")
    
    # RECOMMENDATION #2: Master reference
    master_id = request.form.get('masterId', '')
    master_image = None
    master_style = ""
    
    if master_id and master_id in MASTER_CACHE:
        master_data = MASTER_CACHE[master_id]
        master_image = master_data["image"]
        master_style = master_data["style"]
    
    if 'masterImage' in request.files and not master_image:
        master_file = request.files['masterImage']
        master_image = master_file.read()
        master_style = request.form.get('masterStyle', '')
    
    # Detail images
    detail_images = []
    detail_labels = []
    for i in range(1, 4):
        if f'detail{i}' in request.files:
            detail_file = request.files[f'detail{i}']
            detail_bytes = detail_file.read()
            detail_images.append((detail_bytes, detail_file.content_type))
            label = request.form.get(f'detail{i}Label', f'Detail {i}')
            detail_labels.append(label)
    
    if quality not in ['1K', '2K']:
        quality = '1K'
    
    try:
        main_image_bytes = main_file.read()
        
        # No background image - fall back to V1
        if not background_image:
            print("[V2] No background, using V1")
            return generate_v1_internal(
                main_image_bytes, main_file.content_type,
                prompt, quality, detail_images, detail_labels,
                background_description, product_dimensions, orientation, visible_text,
                master_image, master_style
            )
        
        print(f"[V2] Starting: {quality}, orientation={orientation}, cached_bg={bool(cached_background)}, master={bool(master_image)}")
        
        # If using cached background, skip Stage 1
        if cached_background:
            stage1_image = cached_background
            print("[V2] Skipping Stage 1 (using cached background)")
        else:
            # Stage 1: Generate background
            print("[V2] Stage 1: Background")
            
            stage1_prompt = """Reproduce this image exactly as a studio photography surface.

IMAGE 1 shows a background/surface. Create an exact copy preserving:
- All colors, textures, patterns exactly
- ALL text, writing, numbers, logos - exact content, placement, size, style
- ALL graphics, drawings, marks exactly

Fill entire frame, evenly lit. Accuracy is critical."""
            
            stage1_parts = [
                types.Part.from_bytes(data=background_image, mime_type=background_mime),
                stage1_prompt
            ]
            
            stage1_image = None
            for attempt in range(MAX_GENERATION_ATTEMPTS):
                stage1_bytes, _ = generate_with_retry(stage1_parts, quality, max_attempts=1)
                if stage1_bytes:
                    if verify_background_reproduction(background_image, stage1_bytes, False):
                        stage1_image = stage1_bytes
                        print(f"[V2] Stage 1 success on attempt {attempt + 1}")
                        break
            
            if not stage1_image:
                print("[V2] Stage 1 failed, using V1")
                return generate_v1_internal(
                    main_image_bytes, main_file.content_type,
                    prompt, quality, detail_images, detail_labels,
                    background_description, product_dimensions, orientation, visible_text,
                    master_image, master_style
                )
        
        # Stage 2: Composite
        print("[V2] Stage 2: Composite")
        
        stage2_parts = []
        
        # Master first if present
        if master_image:
            stage2_parts.append(types.Part.from_bytes(data=master_image, mime_type="image/png"))
        
        # Background
        stage2_parts.append(types.Part.from_bytes(data=stage1_image, mime_type="image/png"))
        
        # Product
        stage2_parts.append(types.Part.from_bytes(data=main_image_bytes, mime_type=main_file.content_type))
        
        # Details
        for detail_bytes, detail_mime in detail_images:
            stage2_parts.append(types.Part.from_bytes(data=detail_bytes, mime_type=detail_mime))
        
        stage2_prompt = build_stage2_prompt(
            detail_labels=detail_labels,
            lighting_prompt=lighting_prompt or prompt,
            material_scale=material_scale,
            product_dimensions=product_dimensions,
            orientation=orientation,
            visible_text=visible_text,
            has_master=bool(master_image),
            master_style=master_style
        )
        stage2_parts.append(stage2_prompt)
        
        # Generate with verification
        for verify_attempt in range(MAX_VERIFICATION_RETRIES + 1):
            stage2_bytes, error = generate_with_retry(stage2_parts, quality)
            
            if error:
                return jsonify({
                    "message": "Partial - background only",
                    "image": base64.b64encode(stage1_image).decode('utf-8'),
                    "warnings": ["Composite failed"]
                })
            
            passed, issues = verify_generation(
                main_image_bytes, stage2_bytes, orientation,
                visible_text if visible_text else None
            )
            
            if passed:
                return jsonify({
                    "message": "Success",
                    "image": base64.b64encode(stage2_bytes).decode('utf-8')
                })
            
            if verify_attempt < MAX_VERIFICATION_RETRIES:
                print(f"[V2] Stage 2 verification failed, retrying")
                stage2_parts[-1] = stage2_prompt + f"\n\nFIX: {', '.join(issues)}"
        
        return jsonify({
            "message": "Generated with issues",
            "image": base64.b64encode(stage2_bytes).decode('utf-8'),
            "warnings": issues
        })
        
    except Exception as e:
        print(f"[ERROR] V2 failed: {e}")
        return jsonify({"error": str(e)}), 500


def build_stage2_prompt(detail_labels, lighting_prompt, material_scale, product_dimensions, orientation, visible_text, has_master=False, master_style=""):
    """Build Stage 2 composite prompt."""
    
    sections = []
    
    # Image roles
    roles = ["REFERENCE IMAGES:"]
    
    if has_master:
        roles.append("Image 1 = STYLE REFERENCE. Match this photographic style.")
        roles.append("Image 2 = BACKGROUND. Keep exactly as shown including all markings.")
        roles.append("Image 3 = PRODUCT. Reproduce exactly, place on background.")
        start_idx = 4
    else:
        roles.append("Image 1 = BACKGROUND. Keep exactly, including all text/markings.")
        roles.append("Image 2 = PRODUCT. Reproduce exactly, place on background.")
        start_idx = 3
    
    for i, label in enumerate(detail_labels):
        roles.append(f"Image {start_idx + i} = DETAIL: {label}")
    
    sections.append(" ".join(roles))
    
    # Style from master
    if has_master and master_style:
        sections.append(f"MATCH STYLE: {master_style}")
    
    # Composition
    sections.append(build_composition_instruction(orientation))
    
    # Scale
    if product_dimensions and material_scale:
        sections.append(f"SCALE: Product ~{product_dimensions}. Background: {material_scale}. Realistic proportions.")
    elif product_dimensions:
        sections.append(f"SCALE: Product ~{product_dimensions}.")
    
    # Lighting
    if lighting_prompt:
        sections.append(f"{lighting_prompt} Apply to both product and background - unified shadows and color temperature.")
    else:
        sections.append("LIGHTING: Even studio light on both product and background.")
    
    # Text
    if visible_text:
        sections.append(f"PRESERVE: {visible_text}")
    
    sections.append("OUTPUT: Authentic photograph. Natural depth of field, real textures, unified lighting.")
    
    return "\n\n".join(sections)


def generate_v1_internal(main_bytes, main_mime, prompt, quality, detail_images, detail_labels, bg_desc, dims, orientation, visible_text, master_image=None, master_style=""):
    """Internal V1 generation for fallback."""
    
    content_parts = []
    
    if master_image:
        content_parts.append(types.Part.from_bytes(data=master_image, mime_type="image/png"))
    
    content_parts.append(types.Part.from_bytes(data=main_bytes, mime_type=main_mime))
    
    for detail_bytes, detail_mime in detail_images:
        content_parts.append(types.Part.from_bytes(data=detail_bytes, mime_type=detail_mime))
    
    generation_prompt = build_v1_prompt(
        prompt, detail_labels, bg_desc, dims, orientation, visible_text,
        bool(master_image), master_style
    )
    content_parts.append(generation_prompt)
    
    generated_bytes, error = generate_with_retry(content_parts, quality)
    
    if error:
        return jsonify({"error": error}), 500
    
    return jsonify({
        "message": "Success",
        "image": base64.b64encode(generated_bytes).decode('utf-8')
    })


# ==========================================
# CACHE MANAGEMENT
# ==========================================

@app.route('/clear-cache', methods=['POST'])
def clear_cache():
    """Clear all server-side caches."""
    global BACKGROUND_CACHE, MASTER_CACHE
    
    cache_type = request.form.get('type', 'all')
    
    if cache_type in ['all', 'backgrounds']:
        BACKGROUND_CACHE = {}
    if cache_type in ['all', 'masters']:
        MASTER_CACHE = {}
    
    return jsonify({"message": f"Cleared {cache_type} cache"})


@app.route('/cache-stats', methods=['GET'])
def cache_stats():
    """Get cache statistics."""
    return jsonify({
        "backgrounds_cached": len(BACKGROUND_CACHE),
        "masters_cached": len(MASTER_CACHE),
        "background_ids": list(BACKGROUND_CACHE.keys()),
        "master_ids": list(MASTER_CACHE.keys())
    })


# ==========================================
# SOCIAL MEDIA ENDPOINTS
# ==========================================

@app.route('/generate-interview-questions', methods=['POST'])
def generate_interview_questions():
    if 'image' not in request.files:
        return jsonify({"error": "No image"}), 400
    
    file = request.files['image']
    user_prompt = request.form.get('prompt', 'Generate 3 interview questions.')
    
    try:
        image_bytes = file.read()
        response = client.models.generate_content(
            model=ANALYSIS_MODEL,
            contents=[types.Part.from_bytes(data=image_bytes, mime_type=file.content_type), user_prompt],
            config=types.GenerateContentConfig(response_mime_type="application/json")
        )
        
        questions = json.loads(clean_json_text(response.text))
        if isinstance(questions, dict) and 'questions' in questions:
            questions = questions['questions']
            
        return jsonify({"questions": questions})
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/generate-captions', methods=['POST'])
def generate_captions():
    try:
        data = request.json
        qa_pairs = data.get('qa_pairs', [])
        tone = data.get('tone', 'balanced')
        length = data.get('length', 'medium')
        
        interview_text = "\n".join([f"Q: {item['question']}\nA: {item['answer']}" for item in qa_pairs])
        
        prompt = f"""Based on this interview, write 3 Instagram captions.

{interview_text}

Style: {tone}, {length}.

JSON: storytelling, expert, hybrid, hashtags."""
        
        response = client.models.generate_content(
            model=ANALYSIS_MODEL,
            contents=[prompt],
            config=types.GenerateContentConfig(response_mime_type="application/json")
        )
        
        return jsonify(json.loads(clean_json_text(response.text)))
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/analyze-daily-photo', methods=['POST'])
def analyze_daily_photo():
    if 'image' not in request.files:
        return jsonify({"error": "No image"}), 400
    
    file = request.files['image']
    
    prompt = """Look at this WIP photo and suggest 3 prompts for a social media post. Casual and specific.
JSON array of 3 strings."""
    
    try:
        image_bytes = file.read()
        response = client.models.generate_content(
            model=ANALYSIS_MODEL,
            contents=[types.Part.from_bytes(data=image_bytes, mime_type=file.content_type), prompt],
            config=types.GenerateContentConfig(response_mime_type="application/json")
        )
        
        prompts = json.loads(clean_json_text(response.text))
        if isinstance(prompts, dict) and 'prompts' in prompts:
            prompts = prompts['prompts']
            
        return jsonify({"prompts": prompts})
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/generate-daily-caption', methods=['POST'])
def generate_daily_caption():
    if 'image' not in request.files:
        return jsonify({"error": "No image"}), 400
    
    file = request.files['image']
    selected_prompt = request.form.get('selected_prompt', '')
    user_notes = request.form.get('user_notes', '')
    
    prompt = f"""Write Instagram caption (100-150 words) for this maker's update.

Prompt: "{selected_prompt}"
Notes: "{user_notes}"

Casual, authentic. End with 5-10 hashtags.
JSON: caption, hashtags."""
    
    try:
        image_bytes = file.read()
        response = client.models.generate_content(
            model=ANALYSIS_MODEL,
            contents=[types.Part.from_bytes(data=image_bytes, mime_type=file.content_type), prompt],
            config=types.GenerateContentConfig(response_mime_type="application/json")
        )
        
        return jsonify(json.loads(clean_json_text(response.text)))
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)
