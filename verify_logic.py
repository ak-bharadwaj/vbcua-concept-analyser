import os
import audio_utils
import speech_to_text
import semantic_eval
import scoring_engine
import report_generator

def run_tests():
    print("--- Starting VBCUA Functional Tests ---")
    
    # 1. Predefined concept reference
    reference_concept = (
        "An ER Diagram (ERD) is a visual representation of a database's logical structure, "
        "defining entities, attributes, and relationships. It optimizes data organization, "
        "supports normalization and integrity, and simplifies query design."
    )
    
    # 2. Mock speech transcription
    user_speech_simulated = (
        "An ER Diagram is a visual representation of a database logical structure. "
        "It defines entities, attributes, and relationships, optimizing data organization. "
        "Um, it also supports database normalization and integrity, simplifying query design."
    )
    
    print("\n[Test 1] Testing Speech Transcription Fallback...")
    transcription = speech_to_text.transcribe_audio(
        "sample_explanation.wav",
        api_key=None,
        use_mock=True,
        reference_text=user_speech_simulated
    )
    print(f"Transcribed Text: {transcription['text']}")
    print(f"Status: {transcription['status']} via {transcription['method']}")
    
    # 3. Semantic Similarity Eval
    print("\n[Test 2] Testing Semantic Evaluation...")
    similarity = semantic_eval.evaluate_semantic_similarity(
        transcription['text'],
        reference_concept
    )
    print(f"Similarity Score: {similarity['similarity_score']}")
    print(f"Method: {similarity['method']}")
    print(f"Keywords Covered: {similarity['covered_keywords']}")
    print(f"Keywords Missed: {similarity['missed_keywords']}")
    
    # 4. Audio analysis
    print("\n[Test 3] Testing Audio Analytics...")
    # First ensure sample wav is created
    import streamlit as st
    import app
    sample_wav = app.ensure_sample_wav()
    
    audio_metrics = audio_utils.analyze_audio_features(sample_wav, transcription['text'])
    print(f"Audio Duration: {audio_metrics['duration']:.2f}s")
    print(f"Pacing (WPM): {audio_metrics['speaking_pace_wpm']:.1f}")
    print(f"Pause Ratio: {audio_metrics['pause_ratio']*100:.1f}%")
    print(f"Filler Counts: {audio_metrics['filler_counts']}")
    
    # 5. Scoring Engine
    print("\n[Test 4] Testing Scoring Engine...")
    scores = scoring_engine.calculate_scores(similarity, audio_metrics)
    print(f"Overall Score: {scores['overall_score']}")
    print(f"Classification: {scores['classification']}")
    print(f"Feedback Points:")
    for f in scores['feedback_points']:
        print(f" - {f}")
        
    # 6. PDF Report Generation
    print("\n[Test 5] Testing Report Generator...")
    pdf_path = "test_output_report.pdf"
    report_generator.generate_pdf_report(
        scores,
        audio_metrics,
        transcription['text'],
        reference_concept,
        pdf_path
    )
    if os.path.exists(pdf_path):
        print(f"Success! PDF report generated at: {os.path.abspath(pdf_path)} (size: {os.path.getsize(pdf_path)} bytes)")
    else:
        print("Error: PDF report was not generated.")

if __name__ == "__main__":
    run_tests()
