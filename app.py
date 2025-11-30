import os
import base64
import json
import hashlib
import redis
from flask import Flask, request, jsonify
from google import genai
from google.genai import types

app = Flask(__name__)

# --- CONFIGURATION ---
# "Flash" is fast and cheap for logic/verification
ANALYSIS_MODEL = 'gemini-1.5-flash'
# "Pro" or "Imagen" for high-fidelity pixels
IMAGE_GEN_MODEL = 'gemini-3-pro-image-preview'

api_key = os.environ.get("GOOGLE_API_KEY")
client = genai.Client(api_key=api_key)

# --- PROMPT TEMPLATES ---
PROMPTS = {
    "analyze_product": """
        Analyze this product image. Return JSON:
        {
            "orientation": "flat_lay" (top-down) or "standing" (front view) or "angled",
            "camera_angle": "string description",
            "product_dimensions": "W x H x D",
            "visible_text": "text on product",
            "lighting_direction": "e.g., top-left, right, soft-diffused"
        }
    """,
    "verify_match": """
        Compare Image 1 (Reference Product) and Image 2 (Generated Output).
        Does Image 2 accurately represent the product in Image 1?
        - Check geometry/shape.
        - Check logos/text (if any).
        - Check color accuracy.
        
        Return JSON: {"pass": bool, "reason": "short explanation"}
    """,
    "generation_system": """
        You are a professional product photographer using a Phase One XF IQ4 150MP Camera.
        LENS: 80mm f/2.8 Schneider Kreuznach.
        STYLE: Commercial photorealism. 
        NEGATIVE CONSTRAINTS: NO CGI, NO 3D RENDER LOOK, NO CARTOON, NO FLOATING OBJECTS.
    """
}

# --- CACHE SETUP ---
# Tries to connect to Redis. If it fails (local dev without Redis), falls back to memory.
class CacheWrapper:
    def __init__(self):
        self.redis_url = os.environ.get("REDIS_URL")
        self.r = None
        self.local_cache = {}
        if self.redis_url:
            try:
                self.r = redis.from_url(self.redis_url)
                self.r.ping()
                print("✅ Connected to Redis")
            except Exception as e:
                print(f"⚠️ Redis connection failed: {e}. Using in-memory cache.")
                self.r = None
        else:
            print("ℹ️ No REDIS_URL found. Using in-memory cache.")

    def get(self, key):
        if self.r:
            val = self.r.get(key)
            return json.loads(val) if val else None
        return self.local_cache.get(key)

    def set(self, key, value, ex=3600):
        if self.r:
            self.r.set(key, json.dumps(value), ex=ex)
        else:
            self.local_cache[key] = value

cache = CacheWrapper()

def clean_json_text(text):
    """Robust JSON extractor for LLM responses."""
    try:
        text = text.strip()
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            text = text.split("```")[1].split("```")[0]
        return json.loads(text)
    except Exception:
        return {}

# --- HELPER FUNCTIONS ---

def build_photoreal_prompt(user_prompt, orientation, has_bg_ref):
    """Constructs the prompt to enforce perspective and realism."""
    
    # 1. Perspective Control
    if orientation == "flat_lay":
        perspective = "Top-down flat lay photography. The camera is directly above (90 degrees). No horizon line. The product is resting flat on the surface."
        shadows = "Soft ambient occlusion shadows directly beneath the object."
    else:
        # 'standing' or 'angled'
        perspective = "Eye-level product photography. The product is standing upright. Visible horizon line separating surface and background. Shallow depth of field (f/2.8) blurring the background slightly."
        shadows = "Realistic contact shadows grounding the object to the floor."

    # 2. Lighting Cohesion
    lighting = "Soft, unified studio lighting. Match the product lighting to the environment."

    return f"""
    INSTRUCTIONS: {user_prompt}
    
    PERSPECTIVE: {perspective}
    LIGHTING: {lighting}
    SHADOWS: {shadows}
    
    ACTION: Composite the Product from Image 1 into the {"Background from Image 2" if has_bg_ref else "described environment"}.
    ENSURE photorealism.
    """

def verify_generation(original_bytes, generated_bytes, mime_type):
    """Uses a cheap model to verify the expensive generation."""
    try:
        response = client.models.generate_content(
            model=ANALYSIS_MODEL,
            contents=[
                types.Part.from_bytes(data=original_bytes, mime_type=mime_type),
                types.Part.from_bytes(data=generated_bytes, mime_type="image/png"),
                PROMPTS["verify_match"]
            ],
            config=types.GenerateContentConfig(response_mime_type="application/json")
        )
        result = clean_json_text(response.text)
        return result.get("pass", True), result.get("reason", "")
    except Exception as e:
        print(f"Verification error: {e}")
        return True, "Verification failed"

def generate_with_retry(content_parts, quality, original_bytes=None, mime_type=None):
    """Generates image. If verification fails, it retries automatically."""
    MAX_ATTEMPTS = 2
    
    for attempt in range(MAX_ATTEMPTS):
        try:
            print(f"Generating... Attempt {attempt+1}/{MAX_ATTEMPTS}")
            response = client.models.generate_content(
                model=IMAGE_GEN_MODEL,
                contents=content_parts,
                config=types.GenerateContentConfig(
                    response_modalities=["IMAGE"],
                    image_config=types.ImageConfig(aspect_ratio="1:1", image_size=quality)
                )
            )
            
            if response.candidates and response.candidates[0].content.parts:
                for part in response.candidates[0].content.parts:
                    if part.inline_data:
                        gen_data = part.inline_data.data
                        
                        # Verify logic (Server-side Quality Control)
                        if original_bytes:
                            is_good, reason = verify_generation(original_bytes, gen_data, mime_type)
                            if not is_good:
                                print(f"Verification Failed: {reason}")
                                continue # Retry loop
                            
                        return gen_data
                        
        except Exception as e:
            print(f"Generation error on attempt {attempt}: {e}")
            
    return None

# --- ENDPOINTS ---

@app.route('/')
def home():
    return "Studio Lights Unified Backend v4.0 (Redis + Verification)"

@app.route('/analyze-image', methods=['POST'])
def analyze_image():
    if 'image' not in request.files:
        return jsonify({"error": "No image"}), 400
    
    file = request.files['image']
    image_bytes = file.read()
    
    try:
        response = client.models.generate_content(
            model=ANALYSIS_MODEL,
            contents=[
                types.Part.from_bytes(data=image_bytes, mime_type=file.content_type),
                PROMPTS["analyze_product"]
            ],
            config=types.GenerateContentConfig(response_mime_type="application/json")
        )
        return jsonify(clean_json_text(response.text))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/generate-studio-image', methods=['POST'])
def generate_studio_image():
    """
    The Single Source of Truth for generation.
    Handles: V1 (Description), V2 (Bg Image), and Cached Bg.
    """
    if 'image' not in request.files:
        return jsonify({"error": "No reference image"}), 400
        
    try:
        # 1. Gather Inputs
        main_file = request.files['image']
        main_bytes = main_file.read()
        
        # Check for background (uploaded file OR pre-generated/cached file)
        bg_file = request.files.get('backgroundImage')
        cached_bg_file = request.files.get('cachedBackground')
        
        # Decide which background bytes to use (if any)
        bg_bytes = None
        bg_mime = None
        
        if bg_file:
            bg_bytes = bg_file.read()
            bg_mime = bg_file.content_type
        elif cached_bg_file:
            bg_bytes = cached_bg_file.read()
            bg_mime = cached_bg_file.content_type
            
        # Params
        prompt = request.form.get('prompt', '')
        orientation = request.form.get('orientation', 'standing') # Default to standing
        quality = request.form.get('quality', '1K')
        
        # 2. Build Content Parts
        content_parts = []
        
        # System Instruction
        content_parts.append(PROMPTS["generation_system"])
        
        # Image 1: Product
        content_parts.append(types.Part.from_bytes(data=main_bytes, mime_type=main_file.content_type))
        content_parts.append("REFERENCE IMAGE 1 (Product). Reproduce exactly.")
        
        # Image 2: Background (Optional)
        if bg_bytes:
            content_parts.append(types.Part.from_bytes(data=bg_bytes, mime_type=bg_mime))
            content_parts.append("REFERENCE IMAGE 2 (Background). Use this environment/surface.")
            
        # Prompt
        final_prompt = build_photoreal_prompt(prompt, orientation, has_bg_ref=bool(bg_bytes))
        content_parts.append(final_prompt)
        
        # 3. Generate
        generated_bytes = generate_with_retry(
            content_parts,
            quality,
            original_bytes=main_bytes,
            mime_type=main_file.content_type
        )
        
        if not generated_bytes:
            return jsonify({"error": "Failed to generate acceptable image"}), 500
            
        return jsonify({
            "image": base64.b64encode(generated_bytes).decode('utf-8'),
            "message": "Success"
        })

    except Exception as e:
        print(f"Error in generation: {e}")
        return jsonify({"error": str(e)}), 500

# Keep other minor endpoints (analyze-background, etc.) if needed for specific UI features,
# but the above covers the core workflow.

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)
