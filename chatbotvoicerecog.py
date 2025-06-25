from flask import Flask, request, jsonify, render_template
import os
import time
from PIL import Image as PILImage
import fitz 
import google.generativeai as genai
from diffusers import StableDiffusionPipeline
import torch
import whisper
import tempfile
import librosa
import soundfile as sf
from huggingface_hub import snapshot_download

# === Flask App Setup ===
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['GENERATED_FOLDER'] = 'static/generated'
app.config['AUDIO_FOLDER'] = 'audio_uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['GENERATED_FOLDER'], exist_ok=True)
os.makedirs(app.config['AUDIO_FOLDER'], exist_ok=True)
os.makedirs('templates', exist_ok=True)  # Ensure templates directory exists

# === Configure Gemini ===
try:
    genai.configure(api_key=YOUR API KEY) #change to your Gemini API
    model = genai.GenerativeModel('gemini-1.5-flash')
    print("Gemini model configured successfully")
except Exception as e:
    print(f"Error configuring Gemini: {e}")
    model = None

# === Load Models ===
print("Loading models...")

# Load Phi-4 model (optional - can be disabled if causing issues)
model_path = None
try:
    print("Downloading Phi-4 model...")
    model_path = snapshot_download(repo_id="microsoft/Phi-4-multimodal-instruct")
    speech_lora_path = os.path.join(model_path, "speech-lora")
    vision_lora_path = os.path.join(model_path, "vision-lora")
    print(f"Phi-4 model downloaded to: {model_path}")
except Exception as e:
    print(f"Error loading Phi-4 (skipping): {e}")
    model_path = None

# Load Whisper for speech recognition
whisper_model = None
try:
    print("Loading Whisper model...")
    whisper_model = whisper.load_model("base")
    print("Whisper model loaded successfully")
except Exception as e:
    print(f"Error loading Whisper (skipping): {e}")
    whisper_model = None

# Load Stable Diffusion (optional - can be disabled if causing issues)
sd_pipe = None
try:
    print("Loading Stable Diffusion...")
    sd_model_id = "sd-legacy/stable-diffusion-v1-5"
    sd_pipe = StableDiffusionPipeline.from_pretrained(sd_model_id)
    sd_pipe = sd_pipe.to("cpu")
    print("Stable Diffusion loaded successfully")
except Exception as e:
    print(f"Error loading Stable Diffusion (skipping): {e}")
    sd_pipe = None

print("Model loading complete!")

# === Helper Functions ===
def process_audio_file(audio_path):
    """Process audio file and return transcription"""
    try:
        if whisper_model is None:
            return "Speech recognition model not available"
        
        # Use Whisper for transcription
        result = whisper_model.transcribe(audio_path)
        transcription = result["text"]
        
        return transcription
    except Exception as e:
        return f"Error processing audio: {str(e)}"

def convert_audio_format(input_path, output_path):
    """Convert audio to WAV format if needed"""
    try:
        audio, sr = librosa.load(input_path, sr=16000)
        sf.write(output_path, audio, sr)
        return True
    except Exception as e:
        print(f"Error converting audio: {e}")
        return False

# === Routes ===
@app.route('/')
def index():
    try:
        return render_template('index1.html') #change to your template path
    except Exception as e:
        print(f"Error rendering template: {e}")
        # Return a simple HTML page if template is missing
        return '''
        <!DOCTYPE html>
        <html>
        <head>
            <title>Lorry Crane Assistant</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                .error { color: red; background: #ffe6e6; padding: 20px; border-radius: 5px; }
            </style>
        </head>
        <body>
            <h1>Lorry Crane Assistant</h1>
            <div class="error">
                <h3>Template Error</h3>
                <p>The HTML template file is missing. Please ensure 'templates/index1.html' exists.</p>
                <p>Error: ''' + str(e) + '''</p>
            </div>
        </body>
        </html>
        '''

@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        text_prompt = request.form.get('prompt', '').strip()
        generate_image = request.form.get('generateImage', '') == 'true'
        file = request.files.get('file')
        audio_file = request.files.get('audio')

        # Handle audio input
        transcribed_text = ""
        if audio_file:
            try:
                # Save uploaded audio file
                audio_filename = f"audio_{int(time.time())}.wav"
                audio_filepath = os.path.join(app.config['AUDIO_FOLDER'], audio_filename)
                
                # Save the original file
                temp_path = os.path.join(app.config['AUDIO_FOLDER'], f"temp_{audio_filename}")
                audio_file.save(temp_path)
                
                # Convert to WAV if needed
                if convert_audio_format(temp_path, audio_filepath):
                    transcribed_text = process_audio_file(audio_filepath)
                    
                    # Clean up temporary files
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                    if os.path.exists(audio_filepath):
                        os.remove(audio_filepath)
                else:
                    transcribed_text = "Error: Could not process audio file"
                    
            except Exception as e:
                transcribed_text = f"Error processing audio: {str(e)}"

        # Combine text prompt with transcribed audio
        if transcribed_text:
            if text_prompt:
                combined_prompt = f"{text_prompt}\n\n[Audio transcription: \"{transcribed_text}\"]"
            else:
                combined_prompt = f"I heard you say: \"{transcribed_text}\""
            final_prompt = combined_prompt
        else:
            final_prompt = text_prompt

        if not final_prompt and not file:
            return jsonify({'error': 'Please provide a prompt, audio, or file.'}), 400

        # === Image Generation Mode ===
        if generate_image:
            if sd_pipe is None:
                return jsonify({'error': 'Image generation model not available'}), 500
            
            image = sd_pipe(final_prompt).images[0]
            filename = f"gen_{int(time.time())}.png"
            filepath = os.path.join(app.config['GENERATED_FOLDER'], filename)
            image.save(filepath)
            image_url = f"/static/generated/{filename}"
            
            response_text = f"Here is the image you requested"
            if transcribed_text:
                response_text += f" (based on your speech: \"{transcribed_text}\")"
            response_text += f":<br><img src='{image_url}' class='img-fluid rounded' />"
            
            return jsonify({'response': response_text})

        # === Gemini Prompt Mode (Text/Image/PDF/Audio) ===
        system_prompt = (
            "You are an expert assistant for lorry crane operators. "
            "Provide accurate, practical, and safety-focused guidance related to lorry crane usage, safety checks, load limits, "
            "regulations, daily inspections, maintenance, signaling, and best practices. Respond in a helpful and professional tone. "
            "Use plain text formatting without asterisks or markdown. Use clear paragraphs and bullet points with dashes (-) instead of asterisks. "
            "If you receive transcribed speech, acknowledge what you heard and respond accordingly."
        )

        if model is None:
            return jsonify({'error': 'Gemini model not available. Please check your API key.'}), 500

        if file:
            filename = file.filename
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                pil_image = PILImage.open(filepath).resize((640, 640))
                response = model.generate_content([system_prompt, final_prompt, pil_image])
                pil_image.close()

            elif filename.lower().endswith('.pdf'):
                doc = fitz.open(filepath)
                extracted_text = "".join(page.get_text() for page in doc)
                doc.close()
                full_prompt = f"{system_prompt}\n\n{final_prompt}\n\nPDF Content:\n{extracted_text}"
                response = model.generate_content(full_prompt)

            else:
                return jsonify({'error': 'Unsupported file format.'}), 400

            os.remove(filepath)  

        else:
            full_prompt = [system_prompt, final_prompt]
            response = model.generate_content(full_prompt)

        # Add transcription acknowledgment to response if audio was processed
        final_response = response.text
        if transcribed_text:
            final_response = f"I heard you say: \"{transcribed_text}\"\n\n{response.text}"

        return jsonify({'response': final_response})

    except Exception as e:
        print(f"Error in chat endpoint: {e}")
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@app.route('/api/transcribe', methods=['POST'])
def transcribe_audio():
    """Dedicated endpoint for audio transcription"""
    try:
        audio_file = request.files.get('audio')
        
        if not audio_file:
            return jsonify({'error': 'No audio file provided'}), 400
        
        if whisper_model is None:
            return jsonify({'error': 'Speech recognition model not available'}), 500
        
        # Save and process audio
        audio_filename = f"transcribe_{int(time.time())}.wav"
        audio_filepath = os.path.join(app.config['AUDIO_FOLDER'], audio_filename)
        
        temp_path = os.path.join(app.config['AUDIO_FOLDER'], f"temp_{audio_filename}")
        audio_file.save(temp_path)
        
        if convert_audio_format(temp_path, audio_filepath):
            transcription = process_audio_file(audio_filepath)
            
            # Clean up
            if os.path.exists(temp_path):
                os.remove(temp_path)
            if os.path.exists(audio_filepath):
                os.remove(audio_filepath)
            
            return jsonify({'transcription': transcription})
        else:
            return jsonify({'error': 'Could not process audio file'}), 500
            
    except Exception as e:
        print(f"Error in transcribe endpoint: {e}")
        return jsonify({'error': f'Server error: {str(e)}'}), 500

# Add a test route for debugging
@app.route('/test')
def test():
    return "Flask app is running!"

if __name__ == '__main__':
    print("Starting Flask application...")
    print("Available routes:")
    print("- / (Main page)")
    print("- /test (Test endpoint)")
    print("- /api/chat (Chat API)")
    print("- /api/transcribe (Audio transcription)")
    app.run(debug=True, host='0.0.0.0', port=5000)
