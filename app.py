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

# --- Endpoint 3: Generate Interview Questions ---
@app.route('/generate-interview-questions', methods=['POST'])
def generate_interview_questions():
    if 'image' not in request.files:
        return jsonify({"error": "No image"}), 400
    file = request.files['image']
    
    user_prompt = request.form.get('prompt')
    default_prompt = """
    Analyze this artwork. Generate 3 distinct, engaging interview questions...
    Return raw JSON: [{"id": 1, "category": "...", "text": "..."}]
    """
    final_prompt = user_prompt if user_prompt else default_prompt
    
    try:
        image_bytes = file.read()
        response = client.models.generate_content(
            model=ANALYSIS_MODEL,
            contents=[types.Part.from_bytes(data=image_bytes, mime_type=file.content_type), final_prompt],
            config=types.GenerateContentConfig(response_mime_type="application/json")
        )
        
        # Handle response wrapping
        try:
            # First try to parse as pure list
            questions = json.loads(response.text)
            if isinstance(questions, dict) and 'questions' in questions:
                questions = questions['questions']
            
            # Ensure it's a list for the app
            return jsonify({"questions": questions})
        except:
            return jsonify({"questions": []}) # Fallback
            
    except Exception as e:
        print(f"!!!!!!!!!!!!!! QUESTIONS ERROR: {e}")
        return jsonify({"error": str(e)}), 500

# --- Endpoint 4: Generate Captions (UPDATED) ---
@app.route('/generate-captions', methods=['POST'])
def generate_captions():
    data = request.json
    qa_pairs = data.get('qa_pairs', [])
    tone = data.get('tone', 'balanced')   # Read Tone
    length = data.get('length', 'medium') # Read Length
    
    interview_text = "\n".join([f"Q: {item['question']}\nA: {item['answer']}" for item in qa_pairs])
    
    # Updated Prompt with Tone/Length instructions
    prompt = f"""
    Based on this artist interview, write 3 distinct Instagram captions.
    
    INTERVIEW TRANSCRIPT:
    {interview_text}
    
    CONFIGURATION:
    - Tone: {tone} (Adjust the voice accordingly)
    - Length: {length} (Adjust word count)
    
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
