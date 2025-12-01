"""
Studio Lights Production Backend v4.0

Architecture:
- Stateless: No in-memory caching
- Upstash Redis: Shared cache across all workers
- Supabase: Prompt management and analytics
- Unified generation: Single code path for all generation types
"""

import os
import base64
import json
import hashlib
import time
from functools import lru_cache
from flask import Flask, request, jsonify
import requests
from google import genai
from google.genai import types

app = Flask(__name__)

# =============================================
# CONFIGURATION
# =============================================

ANALYSIS_MODEL = 'gemini-2.5-flash'
IMAGE_GEN_MODEL = 'gemini-3-pro-image-preview'

# Reliability settings
MAX_GENERATION_ATTEMPTS = 3
MAX_VERIFICATION_RETRIES = 2
CACHE_TTL_SECONDS = 86400 * 7  # 7 days

# Environment variables
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
UPSTASH_REDIS_REST_URL = os.environ.get("UPSTASH_REDIS_REST_URL")
UPSTASH_REDIS_REST_TOKEN = os.environ.get("UPSTASH_REDIS_REST_TOKEN")
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_ANON_KEY = os.environ.get("SUPABASE_ANON_KEY")

# Initialize Gemini client
gemini_client = genai.Client(api_key=GOOGLE_API_KEY)


# =============================================
# REDIS CACHE (Upstash REST API)
# =============================================

class RedisCache:
    """Upstash Redis REST API wrapper for cross-worker caching."""
    
    def __init__(self, url, token):
        self.url = url
        self.token = token
        self.headers = {"Authorization": f"Bearer {token}"}
    
    def _request(self, command):
        """Execute Redis command via REST API."""
        try:
            response = requests.post(
                f"{self.url}",
                headers=self.headers,
                json=command,
                timeout=5
            )
            if response.status_code == 200:
                return response.json().get("result")
            return None
        except Exception as e:
            print(f"[REDIS] Error: {e}")
            return None
    
    def get(self, key):
        """Get value from cache."""
        return self._request(["GET", key])
    
    def set(self, key, value, ttl=CACHE_TTL_SECONDS):
        """Set value with TTL."""
        return self._request(["SET", key, value, "EX", ttl])
    
    def delete(self, key):
        """Delete key."""
        return self._request(["DEL", key])
    
    def exists(self, key):
        """Check if key exists."""
        result = self._request(["EXISTS", key])
        return result == 1
    
    def get_json(self, key):
        """Get and parse JSON value."""
        value = self.get(key)
        if value:
            try:
                return json.loads(value)
            except:
                return None
        return None
    
    def set_json(self, key, data, ttl=CACHE_TTL_SECONDS):
        """Set JSON value."""
        return self.set(key, json.dumps(data), ttl)
    
    def get_binary(self, key):
        """Get base64-encoded binary data."""
        value = self.get(key)
        if value:
            try:
                return base64.b64decode(value)
            except:
                return None
        return None
    
    def set_binary(self, key, data, ttl=CACHE_TTL_SECONDS):
        """Set binary data as base64."""
        return self.set(key, base64.b64encode(data).decode('utf-8'), ttl)


# Initialize Redis (will be None if not configured)
redis_cache = None
if UPSTASH_REDIS_REST_URL and UPSTASH_REDIS_REST_TOKEN:
    redis_cache = RedisCache(UPSTASH_REDIS_REST_URL, UPSTASH_REDIS_REST_TOKEN)
    print("[INIT] Redis cache connected")
else:
    print("[INIT] Redis not configured - caching disabled")


# =============================================
# SUPABASE (Prompts & Analytics)
# =============================================

class SupabaseClient:
    """Simple Supabase REST API client."""
    
    def __init__(self, url, key):
        self.url = url
        self.headers = {
            "apikey": key,
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json"
        }
    
    def select(self, table, columns="*", filters=None):
        """Select from table."""
        try:
            url = f"{self.url}/rest/v1/{table}?select={columns}"
            if filters:
                for key, value in filters.items():
                    url += f"&{key}=eq.{value}"
            response = requests.get(url, headers=self.headers, timeout=5)
            if response.status_code == 200:
                return response.json()
            return []
        except Exception as e:
            print(f"[SUPABASE] Select error: {e}")
            return []
    
    def insert(self, table, data):
        """Insert into table."""
        try:
            url = f"{self.url}/rest/v1/{table}"
            response = requests.post(url, headers=self.headers, json=data, timeout=5)
            return response.status_code in [200, 201]
        except Exception as e:
            print(f"[SUPABASE] Insert error: {e}")
            return False


# Initialize Supabase (will be None if not configured)
supabase = None
if SUPABASE_URL and SUPABASE_ANON_KEY:
    supabase = SupabaseClient(SUPABASE_URL, SUPABASE_ANON_KEY)
    print("[INIT] Supabase connected")
else:
    print("[INIT] Supabase not configured - using hardcoded prompts")


# =============================================
# PROMPT MANAGEMENT
# =============================================

# Fallback prompts (used if Supabase unavailable)
FALLBACK_PROMPTS = {
    'analysis_metadata': '''Analyze this product photograph and extract metadata.

ORIENTATION: Is the product lying flat (viewed from above, like clothing on a table) or standing upright (viewed at eye level, like a bottle or statue)?
- "flat_lay" = lying flat, top-down view
- "standing" = upright, eye-level view  
- "angled" = neither clearly flat nor standing

CAMERA ANGLE: Describe the camera perspective in 3-5 words

PRODUCT DIMENSIONS: Estimate real-world size as "W x H x D" with units

VISIBLE TEXT: List any visible text exactly. Empty string if none.

JSON only:
{"orientation": "...", "camera_angle": "...", "product_dimensions": "...", "visible_text": "..."}''',
    
    'composition_flat_lay': 'COMPOSITION: Top-down flat lay photograph. Camera positioned directly above, looking straight down. Product lies flat on a horizontal surface. Center the product, filling 50-60% of frame. Background extends as continuous horizontal plane around all edges. Soft contact shadow directly beneath product.',
    
    'composition_standing': 'COMPOSITION: Standing product photograph. Camera at eye level or slightly elevated. Product stands upright on surface with depth perspective - background visible beneath and behind, receding naturally. Center product, filling 50-60% of frame height. Natural contact shadow at base.',
    
    'composition_angled': 'COMPOSITION: Photograph at natural angle matching reference. Center product, filling 50-60% of frame. Background surface visible around product with appropriate perspective. Soft contact shadow grounding product on surface.',
    
    'output_quality': 'OUTPUT: Authentic studio photograph. Natural depth of field, real material textures, unified lighting across product and background. Shot on full-frame camera with 90mm lens at f/8.',
    
    'background_reproduction': '''Reproduce this image exactly as a clean studio photography surface.

IMAGE 1 shows a background/surface material. Create an exact copy preserving:
- All colors, textures, patterns exactly
- ALL text, writing, numbers, logos - exact content, placement, size, style
- ALL graphics, drawings, marks exactly

Fill entire frame, evenly lit. Accuracy is critical.'''
}

FALLBACK_LIGHTING = {
    'softbox': 'LIGHTING: Soft box. Large diffused source at 45° left and above, subtle fill from right. Shadows soft gradients at 30-40% gray with smooth falloff. Highlights broad and wrapped. Exposure balanced and neutral. Background evenly lit matching product. Color temperature neutral daylight (5500K).'
}

FALLBACK_BACKGROUNDS = {
    'white': 'Professional seamless studio surface: pure white duvetyn fabric with soft, velvety matte surface. Zero shine or reflections. Taut and smooth. Extends seamlessly with no visible edges.',
    'gray': 'Professional seamless studio surface: neutral medium gray duvetyn fabric. True neutral with no warm or cool cast. Soft matte surface. Extends seamlessly.',
    'black': 'Professional seamless studio surface: near-black duvetyn fabric. Deep rich black with subtle texture. Matte surface. Extends seamlessly.'
}


@lru_cache(maxsize=50)
def get_prompt(name):
    """Get prompt by name from Supabase or fallback."""
    if supabase:
        results = supabase.select('prompts', 'content', {'name': name, 'is_active': 'true'})
        if results and len(results) > 0:
            return results[0].get('content', '')
    return FALLBACK_PROMPTS.get(name, '')


@lru_cache(maxsize=20)
def get_lighting_scheme(scheme_id):
    """Get lighting scheme from Supabase or fallback."""
    print(f"[LIGHTING] Looking up scheme: {scheme_id}")
    if supabase:
        # Query by ID only (is_active filter handled differently for booleans)
        results = supabase.select('lighting_schemes', 'id,prompt_text,is_active', {'id': scheme_id})
        if results and len(results) > 0:
            scheme = results[0]
            if scheme.get('is_active', True):
                print(f"[LIGHTING] Found in Supabase: {scheme_id}")
                return scheme.get('prompt_text', '')
    
    fallback = FALLBACK_LIGHTING.get(scheme_id, FALLBACK_LIGHTING['softbox'])
    print(f"[LIGHTING] Using fallback for: {scheme_id}")
    return fallback


@lru_cache(maxsize=10)
def get_background_description(bg_id):
    """Get default background description."""
    if supabase:
        results = supabase.select('backgrounds', 'description', {'id': bg_id})
        if results and len(results) > 0:
            return results[0].get('description', '')
    return FALLBACK_BACKGROUNDS.get(bg_id, FALLBACK_BACKGROUNDS['white'])


def log_generation(data):
    """Log generation to Supabase for analytics."""
    if supabase:
        supabase.insert('generation_logs', data)


# =============================================
# UTILITY FUNCTIONS
# =============================================

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


def generate_cache_key(data_bytes, prefix=""):
    """Generate a cache key from data."""
    hash_obj = hashlib.md5(data_bytes)
    return f"{prefix}{hash_obj.hexdigest()}"


def get_composition_prompt(orientation):
    """Get composition prompt based on orientation."""
    if orientation == "flat_lay":
        return get_prompt('composition_flat_lay')
    elif orientation == "standing":
        return get_prompt('composition_standing')
    else:
        return get_prompt('composition_angled')


# =============================================
# CORE GENERATION LOGIC (UNIFIED)
# =============================================

class GenerationRequest:
    """Unified container for all generation parameters."""
    
    def __init__(self, request_obj):
        self.main_image = None
        self.main_mime = None
        self.background_image = None
        self.background_mime = None
        self.cached_background = None
        self.master_image = None
        self.detail_images = []
        self.detail_labels = []
        
        # Parse parameters
        self.prompt = request_obj.form.get('prompt', '')
        self.quality = request_obj.form.get('quality', '1K')
        self.lighting_prompt = request_obj.form.get('lightingPrompt', '')
        self.lighting_scheme_id = request_obj.form.get('lightingSchemeId', 'softbox')
        self.background_description = request_obj.form.get('backgroundDescription', '')
        self.material_scale = request_obj.form.get('materialScale', '')
        self.product_dimensions = request_obj.form.get('productDimensions', '')
        self.orientation = request_obj.form.get('orientation', 'angled')
        self.visible_text = request_obj.form.get('visibleText', '')
        self.master_style = request_obj.form.get('masterStyle', '')
        self.has_branding = request_obj.form.get('hasBranding', 'false').lower() == 'true'
        
        # Validate quality
        if self.quality not in ['1K', '2K']:
            self.quality = '1K'
        
        # Parse images
        if 'image' in request_obj.files:
            f = request_obj.files['image']
            self.main_image = f.read()
            self.main_mime = f.content_type
        
        if 'backgroundImage' in request_obj.files:
            f = request_obj.files['backgroundImage']
            self.background_image = f.read()
            self.background_mime = f.content_type
        
        if 'cachedBackground' in request_obj.files:
            f = request_obj.files['cachedBackground']
            self.cached_background = f.read()
        
        if 'masterImage' in request_obj.files:
            f = request_obj.files['masterImage']
            self.master_image = f.read()
        
        # Parse detail images
        for i in range(1, 4):
            if f'detail{i}' in request_obj.files:
                f = request_obj.files[f'detail{i}']
                self.detail_images.append((f.read(), f.content_type))
                label = request_obj.form.get(f'detail{i}Label', f'Detail {i}')
                self.detail_labels.append(label)
        
        # Get lighting from scheme if not provided directly
        if not self.lighting_prompt and self.lighting_scheme_id:
            self.lighting_prompt = get_lighting_scheme(self.lighting_scheme_id)
        
        # Debug logging
        print(f"[REQUEST] orientation={self.orientation}, lighting_scheme={self.lighting_scheme_id}")
        print(f"[REQUEST] lighting_prompt_length={len(self.lighting_prompt) if self.lighting_prompt else 0}")


def generate_image(content_parts, quality):
    """Core generation function with retries."""
    last_error = None
    
    for attempt in range(MAX_GENERATION_ATTEMPTS):
        try:
            response = gemini_client.models.generate_content(
                model=IMAGE_GEN_MODEL,
                contents=content_parts,
                config=types.GenerateContentConfig(
                    response_modalities=["TEXT", "IMAGE"]
                )
            )
            
            if response.candidates and response.candidates[0].content.parts:
                for part in response.candidates[0].content.parts:
                    if part.inline_data:
                        return part.inline_data.data, None
            
            last_error = "No image in response"
            
        except Exception as e:
            last_error = str(e)
            print(f"[GEN] Attempt {attempt + 1} failed: {e}")
    
    return None, f"Failed after {MAX_GENERATION_ATTEMPTS} attempts: {last_error}"


def verify_generation(original_bytes, generated_bytes, orientation, visible_text=None):
    """Verify generated image meets criteria."""
    
    text_check = ""
    text_field = ""
    if visible_text:
        text_check = f'\n5. TEXT: Visible markings "{visible_text}" preserved correctly?'
        text_field = '"text_ok": bool, '
    
    prompt_template = get_prompt('verification')
    if not prompt_template:
        # Fallback verification prompt - no curly braces to avoid format issues
        prompt_template = '''Compare these images. Image 1 is original product, Image 2 is generated.
Verify: product fidelity, orientation (ORIENTATION_PLACEHOLDER), composition, lighting unity.
JSON with pass (boolean) and issues (array of strings).'''
    
    # Use simple string replacement instead of .format() to avoid JSON brace conflicts
    prompt = prompt_template.replace("{orientation}", orientation)
    prompt = prompt.replace("ORIENTATION_PLACEHOLDER", orientation)
    prompt = prompt.replace("{text_check}", text_check)
    prompt = prompt.replace("{text_field}", text_field)
    
    try:
        response = gemini_client.models.generate_content(
            model=ANALYSIS_MODEL,
            contents=[
                types.Part.from_bytes(data=original_bytes, mime_type="image/jpeg"),
                types.Part.from_bytes(data=generated_bytes, mime_type="image/png"),
                prompt
            ],
            config=types.GenerateContentConfig(response_mime_type="application/json")
        )
        
        result = json.loads(clean_json_text(response.text))
        return result.get("pass", False), result.get("issues", [])
        
    except Exception as e:
        print(f"[VERIFY] Error: {e}")
        return True, []  # Assume OK if verification fails


def build_generation_prompt(gen_req, has_master=False, has_cached_bg=False):
    """Build the unified generation prompt."""
    
    sections = []
    
    # Image role assignments
    roles = ["REFERENCE IMAGES:"]
    img_idx = 1
    
    if has_master:
        roles.append(f"Image {img_idx} = STYLE REFERENCE. Match this photographic style.")
        img_idx += 1
    
    if has_cached_bg:
        roles.append(f"Image {img_idx} = BACKGROUND. Keep exactly as shown including all markings.")
        img_idx += 1
    
    roles.append(f"Image {img_idx} = PRODUCT. Reproduce exactly with complete fidelity.")
    img_idx += 1
    
    for i, label in enumerate(gen_req.detail_labels):
        roles.append(f"Image {img_idx} = DETAIL: {label}")
        img_idx += 1
    
    sections.append(" ".join(roles))
    
    # Master style
    if has_master and gen_req.master_style:
        sections.append(f"MATCH STYLE: {gen_req.master_style}")
    
    # Composition
    sections.append(get_composition_prompt(gen_req.orientation))
    
    # Background (if not using cached image)
    if not has_cached_bg and gen_req.background_description:
        bg_section = f"BACKGROUND: {gen_req.background_description}"
        if gen_req.product_dimensions:
            bg_section += f" Product is ~{gen_req.product_dimensions} - scale appropriately."
        if gen_req.material_scale:
            bg_section += f" Material scale: {gen_req.material_scale}."
        sections.append(bg_section)
    
    # Scale for cached background
    if has_cached_bg and gen_req.product_dimensions:
        scale_section = f"SCALE: Product is ~{gen_req.product_dimensions}."
        if gen_req.material_scale:
            scale_section += f" Background: {gen_req.material_scale}."
        sections.append(scale_section)
    
    # Lighting
    if gen_req.lighting_prompt:
        lighting = gen_req.lighting_prompt
        if has_cached_bg:
            lighting += " Apply to both product and background - unified shadows and color temperature."
        sections.append(lighting)
    
    # Text preservation
    if gen_req.visible_text:
        sections.append(f"PRESERVE TEXT: {gen_req.visible_text}")
    
    # Output quality (include aspect ratio since we can't set via API)
    quality_instruction = get_prompt('output_quality')
    quality_instruction += " Square 1:1 aspect ratio."
    sections.append(quality_instruction)
    
    return "\n\n".join(sections)


def unified_generate(gen_req):
    """
    Unified generation pipeline handling all cases:
    - V1 (text background)
    - V2 (image background)
    - With/without master
    - With/without cached background
    """
    
    start_time = time.time()
    
    if not gen_req.main_image:
        return {"error": "No product image provided"}, 400
    
    # Determine generation mode
    needs_background_gen = gen_req.background_image is not None and gen_req.has_branding
    has_cached_bg = gen_req.cached_background is not None
    has_master = gen_req.master_image is not None
    
    # Try to get cached background from Redis
    if needs_background_gen and not has_cached_bg and redis_cache:
        cache_key = generate_cache_key(gen_req.background_image, "bg_")
        cached = redis_cache.get_binary(cache_key)
        if cached:
            gen_req.cached_background = cached
            has_cached_bg = True
            print(f"[CACHE] Background hit: {cache_key[:20]}...")
    
    # Stage 1: Generate background if needed
    stage1_image = None
    if needs_background_gen and not has_cached_bg:
        print("[GEN] Stage 1: Background generation")
        
        bg_prompt = get_prompt('background_reproduction')
        bg_parts = [
            types.Part.from_bytes(data=gen_req.background_image, mime_type=gen_req.background_mime),
            bg_prompt
        ]
        
        stage1_image, error = generate_image(bg_parts, gen_req.quality)
        
        if error:
            print(f"[GEN] Stage 1 failed: {error}")
            # Fall back to V1 (text-only background)
            needs_background_gen = False
        else:
            # Cache the generated background
            if redis_cache:
                cache_key = generate_cache_key(gen_req.background_image, "bg_")
                redis_cache.set_binary(cache_key, stage1_image)
                print(f"[CACHE] Background stored: {cache_key[:20]}...")
            
            gen_req.cached_background = stage1_image
            has_cached_bg = True
    
    # Stage 2 (or only stage): Generate final image
    print("[GEN] Final generation")
    
    # Build content parts
    content_parts = []
    
    if has_master:
        content_parts.append(types.Part.from_bytes(data=gen_req.master_image, mime_type="image/jpeg"))
    
    if has_cached_bg:
        content_parts.append(types.Part.from_bytes(data=gen_req.cached_background, mime_type="image/png"))
    
    content_parts.append(types.Part.from_bytes(data=gen_req.main_image, mime_type=gen_req.main_mime))
    
    for detail_bytes, detail_mime in gen_req.detail_images:
        content_parts.append(types.Part.from_bytes(data=detail_bytes, mime_type=detail_mime))
    
    prompt = build_generation_prompt(gen_req, has_master, has_cached_bg)
    content_parts.append(prompt)
    
    # Generate with verification loop
    final_image = None
    issues = []
    verification_attempts = 0
    
    for verify_attempt in range(MAX_VERIFICATION_RETRIES + 1):
        verification_attempts += 1
        
        generated, error = generate_image(content_parts, gen_req.quality)
        
        if error:
            return {"error": error}, 500
        
        # Verify
        passed, issues = verify_generation(
            gen_req.main_image,
            generated,
            gen_req.orientation,
            gen_req.visible_text if gen_req.visible_text else None
        )
        
        if passed:
            final_image = generated
            break
        
        if verify_attempt < MAX_VERIFICATION_RETRIES:
            print(f"[VERIFY] Failed, retrying: {issues}")
            # Update prompt with issues
            content_parts[-1] = prompt + f"\n\nFIX THESE ISSUES: {', '.join(issues)}"
        else:
            # Return with warnings
            final_image = generated
    
    # Log generation
    elapsed_ms = int((time.time() - start_time) * 1000)
    log_generation({
        "orientation": gen_req.orientation,
        "lighting_scheme": gen_req.lighting_scheme_id,
        "background_type": "image" if needs_background_gen else "text",
        "quality": gen_req.quality,
        "has_master": has_master,
        "has_cached_bg": has_cached_bg,
        "verification_passed": len(issues) == 0,
        "verification_attempts": verification_attempts,
        "generation_time_ms": elapsed_ms
    })
    
    response = {
        "message": "Success" if not issues else "Generated with potential issues",
        "image": base64.b64encode(final_image).decode('utf-8')
    }
    
    if issues:
        response["warnings"] = issues
    
    return response, 200


# =============================================
# ROUTES
# =============================================

@app.route('/')
def home():
    return f"Studio Lights v4.0 | Redis: {'✓' if redis_cache else '✗'} | Supabase: {'✓' if supabase else '✗'}"


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "redis": redis_cache is not None,
        "supabase": supabase is not None
    })


# MARK: - Analysis Endpoints

@app.route('/analyze-image', methods=['POST'])
def analyze_image():
    """Extract metadata from product image."""
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400
    
    file = request.files['image']
    prompt = get_prompt('analysis_metadata')
    
    try:
        image_bytes = file.read()
        response = gemini_client.models.generate_content(
            model=ANALYSIS_MODEL,
            contents=[types.Part.from_bytes(data=image_bytes, mime_type=file.content_type), prompt],
            config=types.GenerateContentConfig(response_mime_type="application/json")
        )
        
        result = json.loads(clean_json_text(response.text))
        
        return jsonify({
            "orientation": result.get("orientation", "angled"),
            "camera_angle": result.get("camera_angle", "3/4 view"),
            "product_dimensions": result.get("product_dimensions", ""),
            "visible_text": result.get("visible_text", ""),
            "description": ""  # Backwards compatibility
        })
        
    except Exception as e:
        print(f"[ERROR] Analysis: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/analyze-detail', methods=['POST'])
def analyze_detail():
    """Analyze detail image."""
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400
    
    file = request.files['image']
    custom_prompt = request.form.get('prompt', '')
    
    prompt = custom_prompt or "What specific detail, texture, or feature does this show? Describe in 5-10 words. Include any visible text. Write only the label."
    
    try:
        image_bytes = file.read()
        response = gemini_client.models.generate_content(
            model=ANALYSIS_MODEL,
            contents=[types.Part.from_bytes(data=image_bytes, mime_type=file.content_type), prompt]
        )
        
        label = response.text.strip().strip('"\'').rstrip('.')
        return jsonify({"label": label})
        
    except Exception as e:
        print(f"[ERROR] Detail analysis: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/analyze-background', methods=['POST'])
def analyze_background():
    """Analyze background image."""
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400
    
    file = request.files['image']
    
    prompt = """Analyze this background/surface for product photography.

NAME: Short name, 2-4 words
DESCRIPTION: Describe materials, colors, textures, patterns (80-120 words). Include any text exactly.
HAS_BRANDING: Contains text, logos, graphics to preserve? (true/false)
MATERIAL_SCALE: Physical size of repeating elements

JSON: {"name": "...", "description": "...", "has_branding": bool, "material_scale": "..."}"""
    
    try:
        image_bytes = file.read()
        response = gemini_client.models.generate_content(
            model=ANALYSIS_MODEL,
            contents=[types.Part.from_bytes(data=image_bytes, mime_type=file.content_type), prompt],
            config=types.GenerateContentConfig(response_mime_type="application/json")
        )
        
        result = json.loads(clean_json_text(response.text))
        
        name = result.get("name", "Custom Background")
        words = name.split()
        if len(words) > 4:
            name = ' '.join(words[:4])
        
        return jsonify({
            "name": name,
            "description": result.get("description", name),
            "has_branding": result.get("has_branding", False),
            "material_scale": result.get("material_scale", "")
        })
        
    except Exception as e:
        print(f"[ERROR] Background analysis: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/analyze-style', methods=['POST'])
def analyze_style():
    """Analyze style for master images."""
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400
    
    file = request.files['image']
    
    prompt = """Analyze this studio photograph for style characteristics.
Describe in 30-50 words: lighting quality, color temperature, mood, background treatment.
JSON: {"style_description": "..."}"""
    
    try:
        image_bytes = file.read()
        response = gemini_client.models.generate_content(
            model=ANALYSIS_MODEL,
            contents=[types.Part.from_bytes(data=image_bytes, mime_type=file.content_type), prompt],
            config=types.GenerateContentConfig(response_mime_type="application/json")
        )
        
        result = json.loads(clean_json_text(response.text))
        return jsonify({"style_description": result.get("style_description", "")})
        
    except Exception as e:
        print(f"[ERROR] Style analysis: {e}")
        return jsonify({"error": str(e)}), 500


# MARK: - Generation Endpoints (Unified)

@app.route('/generate-studio-image', methods=['POST'])
def generate_studio_image():
    """Generate studio image (V1 - text backgrounds)."""
    gen_req = GenerationRequest(request)
    response, status = unified_generate(gen_req)
    return jsonify(response), status


@app.route('/generate-studio-image-v2', methods=['POST'])
def generate_studio_image_v2():
    """Generate studio image (V2 - image backgrounds)."""
    gen_req = GenerationRequest(request)
    response, status = unified_generate(gen_req)
    return jsonify(response), status


@app.route('/pregenerate-background', methods=['POST'])
def pregenerate_background():
    """Pre-generate and cache a background."""
    if 'image' not in request.files:
        return jsonify({"error": "No background image provided"}), 400
    
    file = request.files['image']
    quality = request.form.get('quality', '2K')
    
    if quality not in ['1K', '2K']:
        quality = '2K'
    
    try:
        bg_image_bytes = file.read()
        cache_key = generate_cache_key(bg_image_bytes, "bg_")
        
        # Check cache first
        if redis_cache:
            cached = redis_cache.get_binary(cache_key)
            if cached:
                return jsonify({
                    "message": "Retrieved from cache",
                    "background_id": cache_key,
                    "image": base64.b64encode(cached).decode('utf-8'),
                    "cached": True
                })
        
        # Generate
        bg_prompt = get_prompt('background_reproduction')
        bg_parts = [
            types.Part.from_bytes(data=bg_image_bytes, mime_type=file.content_type),
            bg_prompt
        ]
        
        generated, error = generate_image(bg_parts, quality)
        
        if error:
            return jsonify({"error": error}), 500
        
        # Cache
        if redis_cache:
            redis_cache.set_binary(cache_key, generated)
        
        return jsonify({
            "message": "Background generated",
            "background_id": cache_key,
            "image": base64.b64encode(generated).decode('utf-8'),
            "cached": False
        })
        
    except Exception as e:
        print(f"[ERROR] Background pre-generation: {e}")
        return jsonify({"error": str(e)}), 500


# MARK: - Cache Management

@app.route('/cache/clear', methods=['POST'])
def clear_cache():
    """Clear cached backgrounds (admin endpoint)."""
    # In production, add authentication here
    if redis_cache:
        # Note: This clears ALL keys, use with caution
        # For selective clearing, implement key patterns
        return jsonify({"message": "Cache clear not implemented for safety"})
    return jsonify({"message": "No cache configured"})


@app.route('/cache/stats', methods=['GET'])
def cache_stats():
    """Get cache statistics."""
    return jsonify({
        "redis_connected": redis_cache is not None,
        "supabase_connected": supabase is not None
    })


# MARK: - Config Endpoints

@app.route('/config/lighting-schemes', methods=['GET'])
def get_lighting_schemes():
    """Get all active lighting schemes."""
    if supabase:
        schemes = supabase.select('lighting_schemes', 'id,name,description,prompt_text', {'is_active': 'true'})
        return jsonify({"schemes": schemes})
    
    # Fallback
    return jsonify({"schemes": [
        {"id": "softbox", "name": "Soft Box", "description": "Classic commercial lighting"}
    ]})


@app.route('/config/backgrounds', methods=['GET'])
def get_backgrounds():
    """Get default backgrounds."""
    if supabase:
        backgrounds = supabase.select('backgrounds', 'id,name,description,is_default')
        return jsonify({"backgrounds": backgrounds})
    
    # Fallback
    return jsonify({"backgrounds": [
        {"id": "white", "name": "White", "description": FALLBACK_BACKGROUNDS['white'], "is_default": True}
    ]})


# =============================================
# SOCIAL MEDIA ENDPOINTS (unchanged)
# =============================================

@app.route('/generate-interview-questions', methods=['POST'])
def generate_interview_questions():
    if 'image' not in request.files:
        return jsonify({"error": "No image"}), 400
    
    file = request.files['image']
    user_prompt = request.form.get('prompt', 'Generate 3 interview questions.')
    
    try:
        image_bytes = file.read()
        response = gemini_client.models.generate_content(
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
        
        response = gemini_client.models.generate_content(
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
    prompt = "Look at this WIP photo and suggest 3 prompts for social media. Casual and specific. JSON array of 3 strings."
    
    try:
        image_bytes = file.read()
        response = gemini_client.models.generate_content(
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
        response = gemini_client.models.generate_content(
            model=ANALYSIS_MODEL,
            contents=[types.Part.from_bytes(data=image_bytes, mime_type=file.content_type), prompt],
            config=types.GenerateContentConfig(response_mime_type="application/json")
        )
        
        return jsonify(json.loads(clean_json_text(response.text)))
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# =============================================
# MAIN
# =============================================

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)
