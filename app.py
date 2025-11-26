import os
import base64
import json
from flask import Flask, request, jsonify
from google import genai
from google.genai import types

app = Flask(__name__)

# --- CONFIGURATION ---
# We use 1.5 Pro for analysis because it excels at Vision-to-Text understanding.
ANALYSIS_MODEL = 'gemini-1.5-pro'

# We use Gemini 3 Pro (Nano Banana) for generation because it is the SOTA image generator.
IMAGE_GEN_MODEL = 'gemini-3-pro-image-preview'

# Initialize the Google GenAI Client
api_key = os.environ.get("GOOGLE_API_KEY")
if not api_key:
    print("Warning: GOOGLE_API_KEY not found in environment variables.")

client = genai.Client(api_key=api_key)

@app.route('/')
def home():
    return f"Social Poster Backend Active. Analysis: {ANALYSIS_MODEL} | Gen: {IMAGE_GEN_MODEL}"

# --- Endpoint 1: Analyze Image (Vision) ---
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
            contents=[
                types.Part.from_bytes(data=image_bytes, mime_type=file.content_type),
                prompt
            ]
        )
        return jsonify({"description": response.text})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# --- Endpoint 2: Generate Studio Image (Nano Banana/Gemini 3) ---
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
            config=types.GenerateContentConfig(
                response_modalities=["TEXT", "IMAGE"]
            )
        )

        generated_image_b64 = None
        
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
            return jsonify({"error": "Model returned text but no image."}), 500

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# --- Endpoint 3: Generate Interview Questions ---
@app.route('/generate-interview-questions', methods=['POST'])
def generate_interview_questions():
    if 'image' not in request.files:
        return jsonify({"error": "No image"}), 400
    file = request.files['image']
    
    # Prompt to get a JSON array back
    prompt = """
    Analyze this artwork/sculpture. Generate 3 distinct, engaging interview questions that I (the artist) can answer to tell the story of this piece. 
    1. One specific question about the technique/materials.
    2. One conceptual question about the inspiration.
    3. One question about the challenges faced.
    
    Return ONLY a raw JSON array of strings, like this: 
    ["Question 1", "Question 2", "Question 3"]
    """
    
    try:
        image_bytes = file.read()
        response = client.models.generate_content(
            model=ANALYSIS_MODEL,
            contents=[types.Part.from_bytes(data=image_bytes, mime_type=file.content_type), prompt],
            config=types.GenerateContentConfig(response_mime_type="application/json")
        )
        
        # Parse the JSON string from Gemini into a real Python list
        questions = json.loads(response.text)
        return jsonify({"questions": questions})
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# --- Endpoint 4: Generate Captions ---
@app.route('/generate-captions', methods=['POST'])
def generate_captions():
    # Expecting JSON data: {"qa_pairs": [{"question": "...", "answer": "..."}, ...]}
    data = request.json
    qa_pairs = data.get('qa_pairs', [])
    
    # Format the transcript for the AI
    interview_text = "\n".join([f"Q: {item['question']}\nA: {item['answer']}" for item in qa_pairs])
    
    prompt = f"""
    Based on the following artist interview about a new piece, write 3 distinct Instagram captions.
    
    INTERVIEW TRANSCRIPT:
    {interview_text}
    
    Tone: Professional but accessible expert.
    
    Output exactly this JSON structure:
    {{
        "storytelling": "A narrative caption focusing on the personal journey...",
        "expert": "A technical caption focusing on materials and method...",
        "hybrid": "A balanced mix of story and technique...",
        "hashtags": "#tag1 #tag2 #tag3..."
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
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)
