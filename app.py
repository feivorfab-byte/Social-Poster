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
    """Analyze a background image and return name, description, and whether it has branding/text."""
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400
    
    file = request.files['image']
    
    prompt = """
    Analyze this image in EXTREME detail for exact reproduction in AI image generation.
    
    Your goal: Describe EVERYTHING in this image so it can be reproduced EXACTLY.
    
    Provide THREE things:
    
    1. NAME: A descriptive 2-4 word name (e.g., "Lined Paper Notes", "Red Brick Wall", "Branded Wood Surface")
    
    2. DESCRIPTION: A comprehensive description (100-150 words) of EVERYTHING visible:
    
    === MATERIAL/SURFACE ===
    - What is the base material? (paper, brick, wood, etc.)
    - Colors, textures, patterns
    - Surface finish and condition
    
    === CONTENT ON THE SURFACE ===
    - Any text, writing, or words - describe EXACTLY what they say
    - Any drawings, sketches, or diagrams
    - Any logos, stamps, or branding
    - Any stains, marks, or intentional markings
    - Position/layout of all elements
    
    === LIGHTING & ATMOSPHERE ===
    - How is it lit?
    - Any shadows or highlights?
    - Overall mood/feel
    
    3. HAS_BRANDING: Set to TRUE if this image contains ANY of the following that must be preserved exactly:
       - Text, words, letters, numbers, or writing of any kind
       - Logos, brand marks, or symbols
       - Specific designs, patterns with meaning, or graphics
       - Handwriting, signatures, or stamps
       Set to FALSE if this is just a plain material/texture (wood, concrete, fabric, paper without writing, plain surfaces)
    
    CRITICAL: Capture EVERY detail. If there is handwriting, describe what it says. If there are logos, describe them. Nothing should be omitted.
    
    Output as JSON:
    {"name": "Short Name", "description": "Complete detailed description...", "has_branding": true/false}
    """
    
    try:
        image_bytes = file.read()
        
        response = client.models.generate_content(
            model=ANALYSIS_MODEL,
            contents=[types.Part.from_bytes(data=image_bytes, mime_type=file.content_type), prompt],
            config=types.GenerateContentConfig(response_mime_type="application/json")
        )
        
        result = json.loads(clean_json_text(response.text))
        
        # Ensure we have all fields
        name = result.get("name", "Custom Background")
        description = result.get("description", name)
        has_branding = result.get("has_branding", False)
        
        # Clean up name if too long
        words = name.split()
        if len(words) > 4:
            name = ' '.join(words[:4])
        
        print(f"--- Background analyzed: {name} ---")
        print(f"--- Description length: {len(description)} chars ---")
        print(f"--- Has branding/text: {has_branding} ---")
        
        return jsonify({"name": name, "description": description, "has_branding": has_branding})
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
    
    lines.append("2. SCALE AND PROPORTION - CRITICAL")
    lines.append("   - First, estimate the real-world size of the product from the reference image")
    lines.append("   - Scale the background texture/pattern to match real-world proportions for that product size")
    lines.append("   - Examples:")
    lines.append("     * 4-inch tall bottle on wood: wood grain lines spaced as they appear next to a 4-inch object")
    lines.append("     * 1.5-inch watch on fabric: fabric weave should be fine/small relative to watch face")
    lines.append("     * 12-inch shoe on concrete: concrete texture appropriately scaled for a foot-long object")
    lines.append("   - DO NOT make background textures too large (zoomed in) or too small relative to product")
    lines.append("   - The background should look like a real surface the product is actually sitting on")
    lines.append("   - Generate/extend background material as needed to fill the frame properly")
    lines.append("   - Product should occupy roughly 40-60% of frame width, with background visible all around")
    lines.append("")
    
    lines.append("3. UNIFIED LIGHTING")
    lines.append("   - Product and background must share the SAME lighting effect")
    lines.append("   - Light direction must be consistent across entire image")
    lines.append("   - Background surface shows highlights and shadows matching the product")
    lines.append("   - Natural contact shadow where product meets surface (soft, darkest at contact)")
    lines.append("")
    
    lines.append("4. REAL PHOTOGRAPH - NOT 3D RENDER")
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
    lines.append("")
    
    lines.append("5. NO EQUIPMENT IN IMAGE")
    lines.append("   - Show ONLY the product and background surface")
    lines.append("   - DO NOT include any lighting equipment, softboxes, reflectors, light stands")
    lines.append("   - DO NOT include windows, window frames, curtains, or any studio equipment")
    lines.append("   - The image should contain nothing but the product on the background")
    
    return "\n".join(lines)


# ==========================================
# 2. TWO-STAGE GENERATION (EXPERIMENTAL)
# ==========================================

@app.route('/generate-studio-image-v2', methods=['POST'])
def generate_studio_image_v2():
    """
    Two-stage generation - Background-first approach.
    
    The insight: Products reproduce well because they're the primary focus.
    So let's make the background the primary focus first, then add the product.
    
    Stage 1: Reproduce the background EXACTLY (like a product photo of the surface)
    Stage 2: Add the product onto that reproduced background
    """
    if 'image' not in request.files:
        return jsonify({"error": "No reference image provided"}), 400
        
    main_file = request.files['image']
    prompt = request.form.get('prompt', 'Turn this into a studio photograph.')
    quality = request.form.get('quality', '1K')
    lighting_prompt = request.form.get('lightingPrompt', '')
    background_description = request.form.get('backgroundDescription', '')
    
    # Background image - this time we USE it
    background_image = None
    background_mime = None
    
    if 'backgroundImage' in request.files:
        bg_file = request.files['backgroundImage']
        background_image = bg_file.read()
        background_mime = bg_file.content_type
        print(f"--- V2: Background image received: {len(background_image)} bytes ---")
    
    # Get detail images
    detail_images = []
    detail_labels = []
    
    for i in range(1, 4):
        detail_key = f'detail{i}'
        label_key = f'detail{i}Label'
        label_key_alt = f'detail{i}_label'
        
        if detail_key in request.files:
            detail_file = request.files[detail_key]
            detail_bytes = detail_file.read()
            detail_images.append((detail_bytes, detail_file.content_type))
            label = request.form.get(label_key) or request.form.get(label_key_alt) or f'Detail {i}'
            detail_labels.append(label)
    
    if quality not in ['1K', '2K']:
        quality = '1K'
    
    try:
        main_image_bytes = main_file.read()
        
        if not background_image:
            # No background image - fall back to V1 behavior
            print("--- V2: No background image, using V1 path ---")
            return generate_studio_image_v1_internal(
                main_image_bytes, main_file.content_type,
                prompt, quality, detail_images, detail_labels, background_description
            )
        
        print(f"--- V2 Background-First Generation: {quality} ---")
        
        # ==========================================
        # STAGE 1: Reproduce background EXACTLY
        # ==========================================
        print("--- Stage 1: Reproducing background exactly ---")
        
        stage1_parts = []
        # Background image is THE ONLY image - treat it like a product
        stage1_parts.append(types.Part.from_bytes(data=background_image, mime_type=background_mime))
        
        stage1_prompt = """Recreate this image EXACTLY as a studio photograph.

IMAGE 1: Reference image to reproduce with PERFECT FIDELITY.

TASK: Create an EXACT reproduction of this image.
- Copy EVERY detail precisely: colors, textures, patterns, text, markings, logos, writing
- This is not inspiration - reproduce it EXACTLY as shown
- Match the exact colors, lighting, and contrast
- If there is text or writing, reproduce it EXACTLY as it appears
- If there are logos or branding, reproduce them EXACTLY

OUTPUT: A perfect reproduction of the reference image, as if photographed in a studio with clean, even lighting.

CRITICAL: Reproduce EVERYTHING in the image exactly. Every mark, every line, every detail."""
        
        stage1_parts.append(stage1_prompt)
        
        stage1_image = None
        max_retries = 3
        
        for attempt in range(max_retries):
            try:
                response = client.models.generate_content(
                    model=IMAGE_GEN_MODEL,
                    contents=stage1_parts,
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
                            stage1_image = part.inline_data.data
                            print(f"--- Stage 1 (background) complete. Size: {len(stage1_image)} bytes ---")
                            break
                
                if stage1_image:
                    break
                    
            except Exception as e:
                print(f"--- Stage 1 attempt {attempt + 1} failed: {e} ---")
                continue
        
        if not stage1_image:
            return jsonify({"error": "Stage 1 failed - could not reproduce background"}), 500
        
        # ==========================================
        # STAGE 2: Add product onto the background
        # ==========================================
        print("--- Stage 2: Adding product to background ---")
        
        stage2_parts = []
        # Image 1: The reproduced background
        stage2_parts.append(types.Part.from_bytes(data=stage1_image, mime_type="image/png"))
        # Image 2: The product to add
        stage2_parts.append(types.Part.from_bytes(data=main_image_bytes, mime_type=main_file.content_type))
        
        # Add detail images
        for detail_bytes, detail_mime in detail_images:
            stage2_parts.append(types.Part.from_bytes(data=detail_bytes, mime_type=detail_mime))
        
        stage2_prompt = build_stage2_add_product_prompt(prompt, detail_labels, lighting_prompt)
        stage2_parts.append(stage2_prompt)
        
        for attempt in range(max_retries):
            try:
                response = client.models.generate_content(
                    model=IMAGE_GEN_MODEL,
                    contents=stage2_parts,
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
                            final_image_b64 = base64.b64encode(part.inline_data.data).decode('utf-8')
                            print(f"--- Stage 2 (add product) complete. Final size: {len(part.inline_data.data)} bytes ---")
                            return jsonify({
                                "message": "Two-stage image generated successfully",
                                "image": final_image_b64
                            })
                
            except Exception as e:
                print(f"--- Stage 2 attempt {attempt + 1} failed: {e} ---")
                continue
        
        # Stage 2 failed - return Stage 1 (just the background)
        print("--- Stage 2 failed, returning background only ---")
        return jsonify({
            "message": "Partial success - could not add product",
            "image": base64.b64encode(stage1_image).decode('utf-8')
        })

    except Exception as e:
        print(f"!!!!!!!!!!!!!! V2 IMAGE GEN ERROR: {e}")
        return jsonify({"error": str(e)}), 500


def generate_studio_image_v1_internal(main_bytes, main_mime, prompt, quality, detail_images, detail_labels, bg_desc):
    """Internal V1 generation for fallback."""
    content_parts = []
    content_parts.append(types.Part.from_bytes(data=main_bytes, mime_type=main_mime))
    
    for detail_bytes, detail_mime in detail_images:
        content_parts.append(types.Part.from_bytes(data=detail_bytes, mime_type=detail_mime))
    
    labeled_prompt = build_labeled_prompt(prompt, detail_labels, bg_desc)
    content_parts.append(labeled_prompt)
    
    for attempt in range(3):
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
                        return jsonify({
                            "message": "Image generated successfully",
                            "image": base64.b64encode(part.inline_data.data).decode('utf-8')
                        })
        except Exception as e:
            continue
    
    return jsonify({"error": "Generation failed"}), 500


def build_stage2_add_product_prompt(base_prompt, detail_labels, lighting_prompt):
    """Stage 2: Add product onto the reproduced background."""
    lines = []
    
    lines.append("You have TWO reference images:")
    lines.append("")
    lines.append("Image 1: BACKGROUND - A studio backdrop surface.")
    lines.append("Image 2: PRODUCT - An object to place on the background. Reproduce this EXACTLY.")
    
    for i, label in enumerate(detail_labels):
        lines.append(f"Image {i + 3}: Detail reference for product - '{label}'")
    
    lines.append("")
    lines.append("=" * 50)
    lines.append("TASK: Place the product onto the background")
    lines.append("=" * 50)
    lines.append("")
    lines.append("Create a photograph where:")
    lines.append("- The product (Image 2) is reproduced EXACTLY and placed on the background surface")
    lines.append("- The product sits naturally with appropriate shadow and lighting")
    lines.append("")
    lines.append(base_prompt)
    lines.append("")
    
    if lighting_prompt:
        lines.append("LIGHTING:")
        lines.append(lighting_prompt)
        lines.append("")
    
    lines.append("SCALE AND PROPORTION - CRITICAL:")
    lines.append("- Estimate the real-world size of the product (e.g., watch ~1.5 inches, shoe ~12 inches, bottle ~8 inches)")
    lines.append("- Scale the background texture to match real-world proportions for that product size")
    lines.append("- If background texture appears too large or zoomed-in relative to product, generate it at proper scale")
    lines.append("- The background should look like a real surface the product would actually sit on")
    lines.append("- Product should occupy 40-60% of frame width with background extending beyond on all sides")
    lines.append("")
    lines.append("CRITICAL REQUIREMENTS:")
    lines.append("- DO NOT modify the product - reproduce it exactly from Image 2")
    lines.append("- Add natural contact shadow where product meets surface")
    lines.append("- Unified lighting across both elements")
    lines.append("- NO lighting equipment, windows, or studio gear visible in image")
    lines.append("")
    lines.append("OUTPUT: A photograph of the exact product on a properly-scaled background surface.")
    
    return "\n".join(lines)


# ==========================================
# 3. FINISHED PROJECT ENDPOINTS
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
