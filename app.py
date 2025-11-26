import os
import base64
import json
from flask import Flask, request, jsonify
from google import genai
from google.genai import types

app = Flask(__name__)

# --- CONFIGURATION ---
ANALYSIS_MODEL = 'gemini-2.5-pro'
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

# --- Endpoint 2: Generate Studio Image ---
@app.route('/generate-studio-image', methods=['POST'])
def generate_studio_image():
    if 'image' not in request.files:
        return jsonify({"error": "No reference image provided"}), 400
        
    file = request.files['image']
    prompt = request.form.get('prompt', 'Turn this into a studio photograph.')
    
    try:
        image_bytes = file.read()
        response = client.models.generate_content(
            model=IMAGE_GEN_MODEL,
            contents=[
                types.Part.from_bytes(data=image_bytes, mime_type=file.content_type),
                prompt
            ],
            config=types.GenerateContentConfig(response_modalities=["TEXT", "IMAGE"])
        )

        generated_image_b64 = None
        if response.candidates and response.candidates[0].content.parts:
            for part in response.candidates[0].content.parts:
                if part.inline_data:
                    generated_image_b64 = base64.b64encode(part.inline_data.data).decode('utf-8')
                    break
        
        if generated_image_b64:
            return jsonify({"message": "Success", "image_base64": generated_image_b64})
        else:
            return jsonify({"error": "Model returned text but no image."}), 500
    except Exception as e:
        print(f"!!!!!!!!!!!!!! IMAGE GEN ERROR: {e}")
        return jsonify({"error": str(e)}), 500

# --- Endpoint 3: Generate Interview Questions (UPDATED) ---
@app.route('/generate-interview-questions', methods=['POST'])
def generate_interview_questions():
    if 'image' not in request.files:
        return jsonify({"error": "No image"}), 400
    file = request.files['image']
    
    # 1. Look for a prompt from the app.
    # 2. If none, use a SIMPLE default prompt.
    user_prompt = request.form.get('prompt')
    
    default_prompt = """
    Analyze this artwork. Generate 3 short, fun, and casual interview questions that I (the artist) can answer quickly for social media.
    1. One simple question about materials or process.
    2. One fun question about the idea/vibe.
    3. One easy question about what I love about it.
    Keep the tone light, friendly, and easy. No complex art jargon.
    Return ONLY a raw JSON array of strings: ["Q1", "Q2", "Q3"]
    """
    
    final_prompt = user_prompt if user_prompt else default_prompt
    
    try:
        image_bytes = file.read()
        response = client.models.generate_content(
            model=ANALYSIS_MODEL,
            contents=[types.Part.from_bytes(data=image_bytes, mime_type=file.content_type), final_prompt],
            config=types.GenerateContentConfig(response_mime_type="application/json")
        )
        questions = json.loads(response.text)
        return jsonify({"questions": questions})
    except Exception as e:
        print(f"!!!!!!!!!!!!!! QUESTIONS ERROR: {e}")
        return jsonify({"error": str(e)}), 500

# --- Endpoint 4: Generate Captions ---
@app.route('/generate-captions', methods=['POST'])
def generate_captions():
    data = request.json
    qa_pairs = data.get('qa_pairs', [])
    interview_text = "\n".join([f"Q: {item['question']}\nA: {item['answer']}" for item in qa_pairs])
    
    prompt = f"""
    Write 3 Instagram captions (Storytelling, Expert, Hybrid) based on this:
    {interview_text}
    Output JSON.
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
