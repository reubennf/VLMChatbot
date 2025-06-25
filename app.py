from flask import Flask, request, jsonify, render_template
import os
import time
from PIL import Image as PILImage
import fitz 
import google.generativeai as genai
from diffusers import StableDiffusionPipeline
import torch

# === Flask App Setup ===
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['GENERATED_FOLDER'] = 'static/generated'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['GENERATED_FOLDER'], exist_ok=True)

# === Configure Gemini ===
genai.configure(api_key=YOUR API KEY) #fill with your gemini api 
model = genai.GenerativeModel('gemini-1.5-flash')

# === Load Stable Diffusion Once ===
sd_model_id = "sd-legacy/stable-diffusion-v1-5"
sd_pipe = StableDiffusionPipeline.from_pretrained(sd_model_id, 
                                                #   torch_dtype=torch.float16
                                                )
sd_pipe = sd_pipe.to("cpu")

# === Routes ===
@app.route('/')
def index():
    return render_template('index.html') #change with your actual template path

@app.route('/api/chat', methods=['POST'])
def chat():
    text_prompt = request.form.get('prompt', '').strip()
    generate_image = request.form.get('generateImage', '') == 'true'
    file = request.files.get('file')

    if not text_prompt and not file:
        return jsonify({'error': 'Please provide a prompt or file.'}), 400

    try:
        # === Image Generation Mode ===
        if generate_image:
            image = sd_pipe(text_prompt).images[0]
            filename = f"gen_{int(time.time())}.png"
            filepath = os.path.join(app.config['GENERATED_FOLDER'], filename)
            image.save(filepath)
            image_url = f"/static/generated/{filename}"
            return jsonify({
                'response': f"Here is the image you requested:<br><img src='{image_url}' class='img-fluid rounded' />"
            })

        # === Gemini Prompt Mode (Text/Image/PDF) ===
        system_prompt = (
            "You are an expert assistant for lorry crane operators. "
            "Provide accurate, practical, and safety-focused guidance related to lorry crane usage, safety checks, load limits, "
            "regulations, daily inspections, maintenance, signaling, and best practices. Respond in a helpful and professional tone."
        )

        if file:
            filename = file.filename
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                pil_image = PILImage.open(filepath).resize((640, 640))
                response = model.generate_content([system_prompt, text_prompt, pil_image])
                pil_image.close()

            elif filename.lower().endswith('.pdf'):
                doc = fitz.open(filepath)
                extracted_text = "".join(page.get_text() for page in doc)
                doc.close()
                full_prompt = f"{system_prompt}\n\n{text_prompt}\n\nPDF Content:\n{extracted_text}"
                response = model.generate_content(full_prompt)

            else:
                return jsonify({'error': 'Unsupported file format.'}), 400

            os.remove(filepath)  

        else:
            full_prompt = [system_prompt, text_prompt]
            response = model.generate_content(full_prompt)

        return jsonify({'response': response.text})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
