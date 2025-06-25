# VLMChatbot
## Lorry Crane Assistant 
<br>
An intelligent multimodal assistant to support lorry crane operators with safety guidance, speech-to-text instructions, PDF/image processing, and AI-generated image visualizations.

### Features
- 🎙️ Speech Recognition: Transcribe audio using OpenAI Whisper.
- 💬 Gemini AI Chat: Ask questions related to lorry crane operations and receive expert responses.
- 🖼️ Image Generation: Generate visuals based on prompts using Stable Diffusion.
- 📄 Multimodal Input: Supports text, image, PDF, and audio inputs.
- 🔊 Audio Transcription Endpoint: Dedicated API route for converting speech to text.

### Requirements
Ensure you have Python 3.8+ and pip installed
<br>
**Python Dependencies**
<br>
> pip install -r requirements.txt

### Installation
**1. Clone this repository**
>git clone https://github.com/yourusername/lorry-crane-assistant.git
<br>cd lorry-crane-assistant

**2. Install requirements**
> pip install -r requirements.txt

**3. Run the application**
<br>
If you want a chatbot without voice feature
<br>
> python chatbot1.py

If you want a chatbot with voice feature
> python chatbotvoice.py
<br>

**4. Access locally**
Access the app at http://localhost:5000

### Notes
Ensure you have a Gemini API Key and put it inside the code
