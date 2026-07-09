import os
from gtts import gTTS

def generate_voice_file():
    text = (
        "An E R Diagram is a visual representation of a database's logical structure, "
        "defining entities, attributes, and relationships. It optimizes data organization, "
        "supports normalization and integrity, and simplifies query design. Key entities "
        "include User, Audio File, Transcript, Audio Feature, Evaluation Result, Reference Concept, "
        "Semantic Similarity, Report, and Session."
    )
    print("Generating spoken voice explanation...")
    tts = gTTS(text=text, lang='en', slow=False)
    # Save as MP3
    tts.save("sample_explanation.mp3")
    print("Success! Spoken explanation saved as sample_explanation.mp3")

if __name__ == "__main__":
    generate_voice_file()
