from openai import OpenAI

# Initialize client with custom base URL
client = OpenAI(
    base_url="http://localhost:8000/v1",  # Custom endpoint
    api_key="sk-your-key"  # Optional, depending on the endpoint
)

# Transcribe audio
file_path = '002007.mp3'
with open(file_path, "rb") as audio_file:
    transcript = client.audio.transcriptions.create(
        model="tarteel-ai/whisper-base-ar-quran",  # Use the appropriate model name
        file=audio_file
    )

print("Transcription:", transcript.text)
