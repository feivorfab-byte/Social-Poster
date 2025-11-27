import os
import base64
import json
from flask import Flask, request, jsonify
from google import genai
from google.genai import types

app = Flask(__name__)

# --- CONFIGURATION ---
# Gemini 2.5 Pro: Excellent for describing images and writing text
ANALYSIS_MODEL = 'gemini-2.5-pro'

# Gemini 3 Pro Image: The ONLY model that can "see" an input image and edit/regenerate it
IMAGE_GEN_MODEL = 'gemini-3-pro-image-preview'

api_key = os.environ.get("GOOGLE_API_KEY")
client = genai.Client(api_key=api_key)

@app.route('/')
def home():
    return f"Backend Active. Analysis: {ANALYSIS_MODEL} | Gen: {IMAGE_GEN_MODEL}"

# --- Endpoint 1: Analyze Image ---
@app.route('/analyze-image', methods=['POST'])
def analyze_image():
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

# --- Endpoint 2: Generate Studio Image (FIXED: Uses Input Image) ---
@app.route('/generate-studio-image', methods=['POST'])
def generate_studio_image():
    if 'image' not in request.files:
        return jsonify({"error": "No reference image provided"}), 400
        
    file = request.files['image']
    # The prompt comes from your Swift app (includes the description + background request)
    prompt = request.form.get('prompt', 'Turn this into a studio photograph.')
    
    try:
        image_bytes = file.read()
        
        print(f"--- Generating with Image Input. Prompt snippet: {prompt[:50]}... ---")

        # We use 'generate_content' because we are passing an input image (multimodal)
        # This allows the AI to "see" your product and preserve it
        response = client.models.generate_content(
            model=IMAGE_GEN_MODEL,
            contents=[
                # 1. The Image
                types.Part.from_bytes(data=image_bytes, mime_type=file.content_type),
                # 2. The Instructions
                prompt
            ],
            config=types.GenerateContentConfig(
                response_modalities=["TEXT", "IMAGE"],
                # 'safety_settings' can be added here if images get blocked often
            )
        )

        generated_image_b64 = None
        
        # Extract the image from the response parts
        if response.candidates and response.candidates[0].content.parts:
            for part in response.candidates[0].content.parts:
                if part.inline_data:
                    generated_image_b64 = base64.b64encode(part.inline_data.data).decode('utf-8')
                    break
        
        if generated_image_b64:
            return jsonify({
                "message": "Image generated successfully",
                "image_base64": generated_image_b64
            })
        else:
            # Sometimes the model refuses and just returns text explaining why
            print(f"API returned text but no image: {response.text}")
            return jsonify({"error": "Model returned text but no image (Safety or Instruction issue)."}), 500

    except Exception as e:
        print(f"!!!!!!!!!!!!!! IMAGE GEN ERROR: {e}")
        return jsonify({"error": str(e)}), 500

# --- Endpoint 3: Generate Interview Questions ---
@app.route('/generate-interview-questions', methods=['POST'])
def generate_interview_questions():
    if 'image' not in request.files:
        return jsonify({"error": "No image"}), 400
    file = request.files['image']
    
    # We enforce a new "Fabricator" prompt here, overriding the client if necessary
    # to ensure questions are short, casual, and about the BUILD (not design).
    fabricator_prompt = """
    Analyze this image. You are interviewing the FABRICATOR (builder/maker), NOT the designer.
    Generate 3 short, casual, and direct interview questions.
    
    Constraints:
    - Questions must be SHORT (max 15 words).
    - Tone: Casual, specific, technical but friendly (like two makers talking in a shop).
    - Focus ONLY on: Materials, Techniques, Assembly, Finishes, or Fabrication Challenges.
    - Do NOT ask about: Inspiration, Meaning, Design Concept, or "Why".
    
    Output exactly this JSON structure (List of Objects):
    [
        {"id": 1, "category": "Technique", "text": "(Question about a specific joinery, weld, or method seen in the image)"},
        {"id": 2, "category": "Materials", "text": "(Question about the specific wood/metal/material choice)"},
        {"id": 3, "category": "The Build", "text": "(Question about the trickiest part of putting it together)"}
    ]
    """
    
    try:
        image_bytes = file.read()
        response = client.models.generate_content(
            model=ANALYSIS_MODEL,
            contents=[types.Part.from_bytes(data=image_bytes, mime_type=file.content_type), fabricator_prompt],
            config=types.GenerateContentConfig(response_mime_type="application/json")
        )
        
        try:
            questions = json.loads(response.text)
            # Handle case where AI wraps it in a dict key we didn't ask for
            if isinstance(questions, dict) and 'questions' in questions:
                questions = questions['questions']
            return jsonify({"questions": questions})
        except:
            # Fallback questions if JSON fails
            fallback = [
                {"id": 1, "category": "Process", "text": "What was the hardest part of this build?"},
                {"id": 2, "category": "Materials", "text": "What materials did you use here?"},
                {"id": 3, "category": "Finish", "text": "How did you get that finish?"}
            ]
            return jsonify({"questions": fallback})
            
    except Exception as e:
        print(f"!!!!!!!!!!!!!! QUESTIONS ERROR: {e}")
        return jsonify({"error": str(e)}), 500

# --- Endpoint 4: Generate Captions ---
@app.route('/generate-captions', methods=['POST'])
def generate_captions():
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
    
    try:
        response = client.models.generate_content(
            model=ANALYSIS_MODEL,
            contents=[prompt],
            config=types.GenerateContentConfig(response_mime_type="application/json")
        )
        captions = json.loads(response.text)
        return jsonify(captions)
    except Exception as e:
        print(f"!!!!!!!!!!!!!! CAPTIONS ERROR: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)
