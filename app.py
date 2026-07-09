import streamlit as st
import os
import numpy as np
import scipy.io.wavfile as wav
import audio_utils
import speech_to_text
import semantic_eval
import scoring_engine
import report_generator

# Set page config for beautiful layout
st.set_page_config(
    page_title="Voice-Based Concept Understanding Analyser",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Predefined Reference Concepts
CONCEPTS = {
    "Entity Relationship Diagram (ERD)": (
        "An ER Diagram (ERD) is a visual representation of a database's logical structure, "
        "defining entities, attributes, and relationships. It optimizes data organization, "
        "supports normalization and integrity, and simplifies query design. Key entities "
        "include User, Audio File, Transcript, Audio Feature, Evaluation Result, Reference Concept, "
        "Semantic Similarity, Report, and Session."
    ),
    "Machine Learning (ML)": (
        "Machine learning is a subfield of artificial intelligence that involves training algorithms "
        "to learn patterns from data and make predictions or decisions without being explicitly programmed. "
        "Key aspects include supervised learning, unsupervised learning, model training, feature extraction, "
        "generalization, testing data, training data, and neural networks."
    ),
    "Cloud Computing": (
        "Cloud computing is the delivery of computing services—including servers, storage, databases, "
        "networking, software, and analytics—over the Internet. It provides rapid innovation, "
        "flexible resources, and economies of scale, typically utilizing models like Infrastructure as a "
        "Service (IaaS), Platform as a Service (PaaS), and Software as a Service (SaaS)."
    )
}

# Helper function to generate spoken or mock audio files for each concept if not exists
def ensure_sample_audios():
    samples = {
        "Entity Relationship Diagram (ERD)": ("sample_erd.mp3", CONCEPTS["Entity Relationship Diagram (ERD)"]),
        "Machine Learning (ML)": ("sample_ml.mp3", CONCEPTS["Machine Learning (ML)"]),
        "Cloud Computing": ("sample_cloud.mp3", CONCEPTS["Cloud Computing"])
    }
    
    for key, (filename, text) in samples.items():
        if not os.path.exists(filename):
            try:
                from gtts import gTTS
                tts = gTTS(text=text, lang='en', slow=False)
                tts.save(filename)
            except Exception:
                # Fallback to standard sine wave WAV if offline or gTTS fails
                wav_filename = filename.replace('.mp3', '.wav')
                if not os.path.exists(wav_filename):
                    sample_rate = 16000
                    t = np.linspace(0, 8, 8 * sample_rate)
                    data = np.sin(2 * np.pi * 180 * t) * 0.2
                    data[int(sample_rate * 1.5):int(sample_rate * 2.5)] = 0.0
                    data[int(sample_rate * 4.8):int(sample_rate * 5.6)] = 0.0
                    data += np.random.normal(0, 0.005, len(t))
                    data = np.clip(data, -1.0, 1.0)
                    wav.write(wav_filename, sample_rate, data.astype(np.float32))

# Ensure all sample audios exist
ensure_sample_audios()

# Helper to get the correct audio file path for the active concept
def get_sample_audio_path(selected_concept):
    mapping = {
        "Entity Relationship Diagram (ERD)": ("sample_erd.mp3", "sample_erd.wav"),
        "Machine Learning (ML)": ("sample_ml.mp3", "sample_ml.wav"),
        "Cloud Computing": ("sample_cloud.mp3", "sample_cloud.wav")
    }
    mp3_path, wav_path = mapping.get(selected_concept, ("sample_erd.mp3", "sample_erd.wav"))
    return mp3_path if os.path.exists(mp3_path) else (wav_path if os.path.exists(wav_path) else "sample_erd.mp3")


# --- Custom Styling ---
st.markdown("""
<style>
    .main-title {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #1e3a8a 0%, #2563eb 50%, #3b82f6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        font-size: 1.1rem;
        color: #4b5563;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1e3a8a;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #64748b;
        text-transform: uppercase;
        font-weight: 600;
        letter-spacing: 0.05em;
    }
    .badge {
        padding: 0.4rem 0.8rem;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 0.9rem;
        display: inline-block;
    }
    .badge-green { background-color: #dcfce7; color: #15803d; }
    .badge-orange { background-color: #ffedd5; color: #c2410c; }
    .badge-red { background-color: #fee2e2; color: #b91c1c; }
    
    .card {
        background: white;
        border-radius: 12px;
        border: 1px solid #e2e8f0;
        padding: 1.5rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        margin-bottom: 1.5rem;
    }
    .entity-tag {
        background-color: #e0f2fe;
        color: #0369a1;
        padding: 0.2rem 0.6rem;
        border-radius: 6px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-right: 0.5rem;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)

# Title & Description
st.markdown("<div class='main-title'>🎙️ Voice-Based Concept Understanding Analyser</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Evaluate conceptual understanding and speech fluency using AI-driven audio & semantic assessment.</div>", unsafe_allow_html=True)

# --- Session State Initialization ---
if 'audio_file' not in st.session_state:
    st.session_state.audio_file = None
if 'transcription' not in st.session_state:
    st.session_state.transcription = None
if 'audio_metrics' not in st.session_state:
    st.session_state.audio_metrics = None
if 'scores' not in st.session_state:
    st.session_state.scores = None
if 'concept_selected' not in st.session_state:
    st.session_state.concept_selected = "Entity Relationship Diagram (ERD)"
if 'reference_concept_text' not in st.session_state:
    st.session_state.reference_concept_text = CONCEPTS["Entity Relationship Diagram (ERD)"]
if 'custom_concept_text' not in st.session_state:
    st.session_state.custom_concept_text = ""

# --- Sidebar Configuration ---
with st.sidebar:
    st.markdown("### ⚙️ System Settings")
    
    # Transcription Settings
    transcribe_mode = st.radio(
        "Speech-to-Text Mode",
        ["Simulated (No API Key)", "OpenAI Whisper API"],
        help="Use Whisper API with your key, or run a simulated/offline transcription."
    )
    
    openai_key = ""
    # Check if key is configured in Streamlit Secrets
    try:
        if "OPENAI_API_KEY" in st.secrets:
            openai_key = st.secrets["OPENAI_API_KEY"]
    except Exception:
        # st.secrets might raise an exception if configuration folder doesn't exist
        pass

    if transcribe_mode == "OpenAI Whisper API":
        key_placeholder = "Configured in Secrets" if openai_key else "Enter API Key starting with sk-"
        user_key = st.text_input("OpenAI API Key", type="password", placeholder=key_placeholder, help="Enter API Key starting with sk-")
        if user_key.strip():
            openai_key = user_key.strip()

        
    st.markdown("---")
    st.markdown("### 📚 Reference Concept Selection")
    
    selected_concept = st.selectbox(
        "Choose a Concept Template",
        list(CONCEPTS.keys()) + ["Custom Concept"]
    )
    
    if selected_concept != "Custom Concept":
        reference_text = CONCEPTS[selected_concept]
        st.session_state.concept_selected = selected_concept
        st.session_state.reference_concept_text = reference_text
    else:
        st.session_state.concept_selected = "Custom Concept"
        custom_ref = st.text_area(
            "Write Custom Reference Concept",
            value=st.session_state.custom_concept_text,
            placeholder="Type core definitions, terms, and requirements here..."
        )
        st.session_state.custom_concept_text = custom_ref
        st.session_state.reference_concept_text = custom_ref

    st.markdown("---")
    st.markdown("### ℹ️ About VBCUA")
    st.info(
        "VBCUA analyses content accuracy via Sentence-BERT / semantic matching and delivery quality via signal pauses, WPM, and filler counts."
    )

# --- Layout Configuration ---
tab_analysis, tab_erd = st.tabs(["📊 Concept Analysis Dashboard", "🧬 ERD Schema & Session Entities"])

with tab_analysis:
    col_left, col_right = st.columns([1, 1.2])
    
    with col_left:
        with st.container(border=True):
            st.markdown("#### 📝 Selected Reference Concept")
            st.caption(f"Topic: **{st.session_state.concept_selected}**")
            
            # Display the reference text in a nice clean block
            st.write(st.session_state.reference_concept_text)
            
            # Extracted Keywords Highlight
            keywords = semantic_eval.extract_keywords(st.session_state.reference_concept_text)
            if keywords:
                st.markdown("**Expected Key Terms:**")
                kw_html = " ".join([f"<span style='background-color:#f1f5f9; color:#475569; padding: 2px 8px; border-radius: 4px; margin-right: 5px; font-size: 0.85rem; display: inline-block; margin-bottom: 5px;'>{k}</span>" for k in keywords[:12]])
                st.markdown(kw_html, unsafe_allow_html=True)
        
        st.write("") # Spacer
        
        with st.container(border=True):
            st.markdown("#### 🎙️ Step 2: Provide Spoken Explanation")
            
            input_type = st.radio(
                "Audio Input Method",
                ["Use Pre-Recorded Sample", "Upload Audio File"],
                horizontal=True
            )
            
            audio_file_path = None
            
            if input_type == "Use Pre-Recorded Sample":
                st.success("Sample audio initialized. Click 'Run Analysis' to test the engine instantly!")
                active_sample_path = get_sample_audio_path(st.session_state.concept_selected)
                st.audio(active_sample_path)
                audio_file_path = active_sample_path
            else:
                uploaded_file = st.file_uploader(
                    "Upload audio file (WAV, MP3, or M4A)", 
                    type=["wav", "mp3", "m4a"],
                    help="Make sure the audio file is clear and audible."
                )
                if uploaded_file is not None:
                    # Save uploaded file temporarily
                    temp_filename = f"temp_upload.{uploaded_file.name.split('.')[-1]}"
                    with open(temp_filename, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    audio_file_path = temp_filename
                    st.audio(uploaded_file)
                    st.session_state.audio_file = temp_filename
                    
            st.markdown("---")
            # Action button to trigger processing
            run_analysis = st.button("🚀 Run Analysis & Evaluate", use_container_width=True)

        
    with col_right:
        # Check if we should process
        if run_analysis:
            if not audio_file_path:
                st.error("Please record or upload an audio file first.")
            elif not st.session_state.reference_concept_text.strip():
                st.error("Reference concept cannot be empty. Select or type a concept first.")
            else:
                with st.spinner("Processing explanation, transcribing and evaluating..."):
                    # 1. Speech-to-Text Transcription
                    use_mock_transcription = (transcribe_mode == "Simulated (No API Key)")
                    
                    transcription_results = speech_to_text.transcribe_audio(
                        audio_file_path, 
                        api_key=openai_key if transcribe_mode == "OpenAI Whisper API" else None,
                        use_mock=use_mock_transcription,
                        reference_text=st.session_state.reference_concept_text
                    )
                    
                    if transcription_results["status"] == "error":
                        st.error(transcription_results["text"])
                    else:
                        transcribed_text = transcription_results["text"]
                        
                        # 2. Extract Audio features
                        audio_metrics = audio_utils.analyze_audio_features(
                            audio_file_path, 
                            transcribed_text=transcribed_text
                        )
                        
                        # 3. Semantic Similarity Evaluation
                        semantic_results = semantic_eval.evaluate_semantic_similarity(
                            transcribed_text,
                            st.session_state.reference_concept_text
                        )
                        
                        # 4. Scoring Engine
                        scores = scoring_engine.calculate_scores(semantic_results, audio_metrics)
                        
                        # Save in session state
                        st.session_state.transcription = transcribed_text
                        st.session_state.audio_metrics = audio_metrics
                        st.session_state.scores = scores
                        
                        # Clean up temporary uploaded file if we created one
                        if audio_file_path != get_sample_audio_path(st.session_state.concept_selected) and os.path.exists(audio_file_path):
                            try:
                                pass # Keep it for the session or delete after use
                            except:
                                pass

        # Render Results (if present in session state)
        if st.session_state.scores is not None:
            scores = st.session_state.scores
            audio_metrics = st.session_state.audio_metrics
            transcription = st.session_state.transcription
            
            st.markdown("### 📊 Evaluation Results")
            
            # Badge for Classification
            badge_class = "badge-green" if scores["classification_color"] == "green" else ("badge-orange" if scores["classification_color"] == "orange" else "badge-red")
            st.markdown(f"Assessment Class: <span class='badge {badge_class}'>{scores['classification']}</span>", unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Score Metric Grid
            m_col1, m_col2, m_col3 = st.columns(3)
            with m_col1:
                st.markdown(f"""
                <div class='metric-card'>
                    <div class='metric-label'>Overall Score</div>
                    <div class='metric-value'>{scores['overall_score']}</div>
                    <div style='font-size:0.8rem; color:#64748b;'>weighted accuracy + fluency</div>
                </div>
                """, unsafe_allow_html=True)
            with m_col2:
                st.markdown(f"""
                <div class='metric-card'>
                    <div class='metric-label'>Semantic Match</div>
                    <div class='metric-value'>{scores['semantic_score']}%</div>
                    <div style='font-size:0.8rem; color:#64748b;'>Sentence-BERT similarity</div>
                </div>
                """, unsafe_allow_html=True)
            with m_col3:
                st.markdown(f"""
                <div class='metric-card'>
                    <div class='metric-label'>Speech Fluency</div>
                    <div class='metric-value'>{scores['fluency_score']}%</div>
                    <div style='font-size:0.8rem; color:#64748b;'>silence + filler word penalty</div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Waveform Plot
            st.markdown("#### 📈 Waveform Visualization")
            fig = audio_utils.generate_waveform_plot(
                st.session_state.audio_file if st.session_state.audio_file else get_sample_audio_path(st.session_state.concept_selected)
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Transcription Output
            st.markdown("#### 🗣️ Transcribed Explanation")
            st.info(f'"{transcription}"')
            
            # Audio Feature Details Table
            st.markdown("#### 🔬 Detailed Speech Metrics")
            det_col1, det_col2 = st.columns(2)
            with det_col1:
                st.write(f"⏱️ **Total Speaking Duration:** {audio_metrics['duration']:.2f} seconds")
                st.write(f"⚡ **Speaking Pace (Speed):** {audio_metrics['speaking_pace_wpm']:.0f} WPM")
            with det_col2:
                st.write(f"🛑 **Pause/Silence Ratio:** {audio_metrics['pause_ratio'] * 100:.1f}%")
                st.write(f"💬 **Filler Words Count:** {audio_metrics['total_filler_count']} instances")
                
            # Feedback Bullets
            st.markdown("#### 💡 AI Feedback & Recommendations")
            for bullet in scores["feedback_points"]:
                st.markdown(f"- {bullet}")
                
            # PDF Report Generation Button
            st.markdown("---")
            
            pdf_filename = "vbcua_assessment_report.pdf"
            try:
                report_generator.generate_pdf_report(
                    scores=scores,
                    audio_metrics=audio_metrics,
                    transcription=transcription,
                    reference_concept=st.session_state.reference_concept_text,
                    output_path=pdf_filename
                )
                
                with open(pdf_filename, "rb") as pdf_file:
                    st.download_button(
                        label="📥 Download Detailed PDF Report",
                        data=pdf_file,
                        file_name="VBCUA_Evaluation_Report.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )
            except Exception as e:
                st.error(f"Could not generate PDF report: {e}")
        else:
            st.info("👈 Set your reference concept, upload/select audio, and click 'Run Analysis' to see evaluation outputs here.")

with tab_erd:
    st.markdown("### 🧬 ER Diagram Entities and Live Session Mappings")
    st.markdown(
        "VBCUA's logical structure maps directly to the entity relationships designed for session management and reporting. "
        "Below are the main database entities and their mapped runtime state variables."
    )
    
    erd_data = [
        {"Entity": "User", "Description": "Represents the student, presenter, or educator.", "Current Session State Mapping": "None (Anonymous local execution)"},
        {"Entity": "Audio File", "Description": "Holds the raw bytes, path, sample rate, and duration of explanation speech.", "Current Session State Mapping": f"Path: '{st.session_state.audio_file or get_sample_audio_path(st.session_state.concept_selected)}'"},
        {"Entity": "Transcript", "Description": "Stores the speech-to-text converted words.", "Current Session State Mapping": f"Text size: {len(st.session_state.transcription) if st.session_state.transcription else 0} chars"},
        {"Entity": "Audio Feature", "Description": "RMS energy profiles, silence ratios, and verbal pacing (WPM).", "Current Session State Mapping": f"Pace: {st.session_state.audio_metrics['speaking_pace_wpm']:.1f} WPM, Silence: {st.session_state.audio_metrics['pause_ratio']*100:.1f}%" if st.session_state.audio_metrics else "Empty"},
        {"Entity": "Evaluation Result", "Description": "Comprehension score, fluency score, and qualitative categorization.", "Current Session State Mapping": f"Score: {st.session_state.scores['overall_score'] if st.session_state.scores else 'N/A'}"},
        {"Entity": "Reference Concept", "Description": "Expected description template and key vocabulary checklist.", "Current Session State Mapping": f"Selected Template: '{st.session_state.concept_selected}'"},
        {"Entity": "Semantic Similarity", "Description": "Vector/Cosine comparison results and key term match lists.", "Current Session State Mapping": f"Score: {st.session_state.scores['semantic_score']}%" if st.session_state.scores else "Empty"},
        {"Entity": "Report", "Description": "Compiled PDF documents containing analytics, graphs, and suggestions.", "Current Session State Mapping": "File: 'vbcua_assessment_report.pdf' (if generated)"},
        {"Entity": "Session", "Description": "User's current state and history tracking token.", "Current Session State Mapping": "streamlit.session_state"}
    ]
    
    st.table(erd_data)
    
    # Display the visual flow diagram using Mermaid
    st.markdown("#### Logical Flow Relationship Diagram")
    st.markdown("""
    ```mermaid
    erDiagram
        User ||--o{ Session : "initiates"
        Session ||--o{ "Audio File" : "records"
        "Audio File" ||--|| Transcript : "produces"
        "Audio File" ||--|| "Audio Feature" : "extracts"
        Transcript ||--|| "Semantic Similarity" : "compares"
        "Reference Concept" ||--o{ "Semantic Similarity" : "compares-to"
        "Semantic Similarity" ||--|| "Evaluation Result" : "calculates"
        "Audio Feature" ||--|| "Evaluation Result" : "calculates"
        "Evaluation Result" ||--|| Report : "compiles"
    ```
    """)

