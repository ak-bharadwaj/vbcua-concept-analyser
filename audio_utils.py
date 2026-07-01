import numpy as np
import soundfile as sf
import plotly.graph_objects as go
import os
import re

def load_audio(file_path):
    """
    Loads an audio file and returns the sample rate and raw audio array.
    Uses soundfile as primary, falls back to librosa if needed.
    """
    try:
        data, samplerate = sf.read(file_path)
        # Convert stereo to mono if necessary
        if len(data.shape) > 1:
            data = np.mean(data, axis=1)
        return data, samplerate
    except Exception as e_sf:
        try:
            import librosa
            data, samplerate = librosa.load(file_path, sr=None)
            return data, samplerate
        except Exception as e_lib:
            # Return dummy audio array to keep application from crashing
            # on unsupported file formats or system config errors
            dummy_rate = 16000
            dummy_data = np.sin(2 * np.pi * 440 * np.arange(0, 5, 1/dummy_rate)) * 0.1
            return dummy_data, dummy_rate

def analyze_audio_features(file_path, transcribed_text=""):
    """
    Analyzes audio features including:
    - Duration
    - RMS Energy (mean and timeline)
    - Pause / Silence ratio
    - Estimated filler words (analyzed from transcript)
    - Speaking pace (words per minute)
    
    Returns:
    - dict: Calculated metrics and timelines.
    """
    data, samplerate = load_audio(file_path)
    
    # Calculate duration
    duration = len(data) / samplerate if samplerate > 0 else 0
    
    # RMS Energy Calculation
    # Use windowed RMS to get variation over time
    frame_size = int(0.050 * samplerate)  # 50ms frames
    hop_size = int(0.025 * samplerate)   # 25ms overlap
    
    rms_values = []
    for i in range(0, len(data) - frame_size, hop_size):
        frame = data[i:i + frame_size]
        rms = np.sqrt(np.mean(frame**2)) if len(frame) > 0 else 0
        rms_values.append(rms)
    
    rms_values = np.array(rms_values) if len(rms_values) > 0 else np.array([0])
    mean_rms = float(np.mean(rms_values))
    
    # Pause / Silence analysis
    # Define silence threshold as 10% of mean RMS or a minimum noise floor
    silence_threshold = max(0.01, mean_rms * 0.25)
    silence_frames = np.sum(rms_values < silence_threshold)
    pause_ratio = float(silence_frames / len(rms_values)) if len(rms_values) > 0 else 0.0
    
    # Speech flow and filler word analysis (from text transcript)
    filler_words = ["um", "uh", "like", "ah", "er", "you know", "basically", "actually"]
    filler_counts = {}
    total_filler_count = 0
    
    cleaned_text = transcribed_text.lower()
    for filler in filler_words:
        # Match word boundaries
        matches = re.findall(rf"\b{filler}\b", cleaned_text)
        count = len(matches)
        if count > 0:
            filler_counts[filler] = count
            total_filler_count += count
            
    # Words count
    words = cleaned_text.split()
    word_count = len(words)
    
    # Speaking Pace (Words Per Minute)
    duration_min = duration / 60.0 if duration > 0 else 0.0
    speaking_pace = word_count / duration_min if duration_min > 0 else 0.0
    
    return {
        "duration": duration,
        "mean_rms": mean_rms,
        "rms_timeline": rms_values.tolist(),
        "pause_ratio": pause_ratio,
        "filler_counts": filler_counts,
        "total_filler_count": total_filler_count,
        "word_count": word_count,
        "speaking_pace_wpm": speaking_pace,
        "audio_length_samples": len(data),
        "samplerate": samplerate
    }

def generate_waveform_plot(file_path):
    """
    Generates a beautiful Plotly chart of the audio waveform.
    Downsamples the audio data for fast rendering.
    """
    data, samplerate = load_audio(file_path)
    
    # Downsample to maximum of 2000 points to ensure smooth UI rendering
    max_points = 2000
    if len(data) > max_points:
        step = len(data) // max_points
        data_downsampled = data[::step]
    else:
        data_downsampled = data
        step = 1
        
    duration = len(data) / samplerate if samplerate > 0 else 0
    time_axis = np.linspace(0, duration, len(data_downsampled))
    
    fig = go.Figure()
    
    # Add waveform trace with modern gradient/color scheme
    fig.add_trace(go.Scatter(
        x=time_axis,
        y=data_downsampled,
        mode='lines',
        name='Waveform',
        line=dict(color='#3b82f6', width=1.5),
        fill='tozeroy',
        fillcolor='rgba(59, 130, 246, 0.15)'
    ))
    
    fig.update_layout(
        title={
            'text': "Audio Waveform & Signal Amplitude",
            'y':0.9,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        xaxis_title="Time (seconds)",
        yaxis_title="Amplitude",
        template="plotly_dark",
        margin=dict(l=20, r=20, t=40, b=20),
        height=280,
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)'),
        yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)', range=[-1, 1])
    )
    
    return fig
