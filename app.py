import os
import base64
import json
import re
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
# 1. FINISHED PROJECT ENDPOINTS
# ==========================================

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

@app.route('/generate-studio-image', methods=['POST'])
def generate_studio_image():
    if 'image' not in request.files:
        return jsonify({"error": "No reference image provided"}), 400
        
    file = request.files['image']
    prompt = request.form.get('prompt', 'Turn this into a studio photograph.')
    
    try:
        image_bytes = file.read()
        print(f"--- Generating image. Prompt snippet: {prompt[:100]}... ---")

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
        
        # Extract the image from the response parts
        if response.candidates and response.candidates[0].content.parts:
            for part in response.candidates[0].content.parts:
                if part.inline_data:
                    # Return the full base64 data without any compression
                    generated_image_b64 = base64.b64encode(part.inline_data.data).decode('utf-8')
                    print(f"--- Image generated. Size: {len(part.inline_data.data)} bytes ---")
                    break
        
        if generated_image_b64:
            return jsonify({
                "message": "Image generated successfully",
                "image_base64": generated_image_b64
            })
        else:
            print(f"API returned text but no image: {response.text}")
            return jsonify({"error": "Model returned text but no image (Safety or Instruction issue)."}), 500

    except Exception as e:
        print(f"!!!!!!!!!!!!!! IMAGE GEN ERROR: {e}")
        return jsonify({"error": str(e)}), 500

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
# 2. DAILY POST ENDPOINTS
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
