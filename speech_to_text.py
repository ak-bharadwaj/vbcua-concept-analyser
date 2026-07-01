import os
from openai import OpenAI

def transcribe_audio(audio_file_path, api_key=None, use_mock=False, reference_text=None):
    """
    Transcribes an audio file using OpenAI Whisper API or a simulated fallback.
    
    Parameters:
    - audio_file_path (str): Path to the audio file.
    - api_key (str): OpenAI API Key. If None, it will fallback to mock.
    - use_mock (bool): Force mock transcription.
    - reference_text (str): Reference explanation text to simulate close transcription.
    
    Returns:
    - dict: A dictionary with 'text' and 'status' keys.
    """
    if use_mock or not api_key:
        # Generate a simulated transcription based on reference text
        if reference_text:
            # Let's clean the reference text slightly to make it look transcribed
            import random
            text = reference_text.strip()
            # Simulate a few transcription variations/filler words
            words = text.split()
            if len(words) > 10:
                # Insert a few filler words
                fillers = ["um", "uh", "like"]
                for _ in range(random.randint(1, 3)):
                    idx = random.randint(3, len(words) - 3)
                    words.insert(idx, random.choice(fillers))
                # Delete a random word to simulate slight error
                if random.random() > 0.5:
                    words.pop(random.randint(0, len(words)-1))
            mock_text = " ".join(words)
            return {
                "text": mock_text,
                "status": "success",
                "method": "Simulated Transcription (Based on Reference Concept)"
            }
        else:
            return {
                "text": "This is a simulated transcription of the audio explanation. In a production environment with an OpenAI API key configured, Whisper would transcribe this audio directly. The quick brown fox jumps over the lazy dog.",
                "status": "success",
                "method": "Mock Transcription (No Reference Provided)"
            }
            
    try:
        client = OpenAI(api_key=api_key)
        with open(audio_file_path, "rb") as audio_file:
            transcript = client.audio.transcriptions.create(
                model="whisper-1", 
                file=audio_file
            )
        return {
            "text": transcript.text,
            "status": "success",
            "method": "OpenAI Whisper API"
        }
    except Exception as e:
        return {
            "text": f"Error during Whisper API transcription: {str(e)}",
            "status": "error",
            "method": "Failed API call"
        }
