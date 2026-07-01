import os
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.graphics.shapes import Drawing, Rect, Line

def generate_pdf_report(scores, audio_metrics, transcription, reference_concept, output_path):
    """
    Generates a professional, structured evaluation report PDF.
    
    Parameters:
    - scores (dict): Evaluation scores from scoring_engine.py
    - audio_metrics (dict): Audio metrics from audio_utils.py
    - transcription (str): The transcribed text.
    - reference_concept (str): Reference explanation text.
    - output_path (str): File path where the PDF will be saved.
    
    Returns:
    - str: Output path if successful, otherwise raises an exception.
    """
    doc = SimpleDocTemplate(
        output_path, 
        pagesize=letter,
        rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40
    )
    
    styles = getSampleStyleSheet()
    
    # Custom Styles for modern design
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=24,
        leading=28,
        textColor=colors.HexColor('#1e3a8a'), # Deep Navy
        spaceAfter=15
    )
    
    section_style = ParagraphStyle(
        'SectionHeading',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=14,
        leading=18,
        textColor=colors.HexColor('#1e40af'), # Blue
        spaceBefore=12,
        spaceAfter=8,
        keepWithNext=True
    )
    
    body_style = ParagraphStyle(
        'BodyDark',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        leading=14,
        textColor=colors.HexColor('#374151') # Charcoal
    )
    
    bold_body_style = ParagraphStyle(
        'BoldBody',
        parent=body_style,
        fontName='Helvetica-Bold'
    )
    
    callout_style = ParagraphStyle(
        'CalloutText',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=10.5,
        leading=15,
        textColor=colors.HexColor('#1f2937')
    )
    
    story = []
    
    # 1. Header Banner & Title
    story.append(Paragraph("Voice-Based Concept Understanding Analyser (VBCUA)", title_style))
    story.append(Paragraph("<b>Concept Assessment & Fluency Evaluation Report</b>", styles['Normal']))
    
    # Decorative line
    d = Drawing(532, 5)
    d.add(Rect(0, 0, 532, 3, fillColor=colors.HexColor('#2563eb'), strokeColor=None))
    story.append(d)
    story.append(Spacer(1, 15))
    
    # 2. Score Summary Panel (using Table for neat grid formatting)
    classification_color = {
        "Strong Understanding": colors.HexColor('#16a34a'),   # Green
        "Moderate Understanding": colors.HexColor('#ea580c'), # Orange
        "Poor Understanding": colors.HexColor('#dc2626')      # Red
    }.get(scores['classification'], colors.HexColor('#4b5563'))
    
    summary_data = [
        [
            Paragraph("<b>Overall Score</b>", body_style),
            Paragraph(f"<font size=14 color='#1e3a8a'><b>{scores['overall_score']}/100</b></font>", body_style),
            Paragraph("<b>Assessment Class</b>", body_style),
            Paragraph(f"<font size=11 color='{classification_color.hexval()}'><b>{scores['classification']}</b></font>", body_style)
        ],
        [
            Paragraph("<b>Semantic Match</b>", body_style),
            Paragraph(f"{scores['semantic_score']}%", body_style),
            Paragraph("<b>Speech Fluency</b>", body_style),
            Paragraph(f"{scores['fluency_score']}%", body_style)
        ],
        [
            Paragraph("<b>Keyword Coverage</b>", body_style),
            Paragraph(f"{scores['keyword_coverage']}%", body_style),
            Paragraph("<b>Speech Duration</b>", body_style),
            Paragraph(f"{audio_metrics['duration']:.2f}s", body_style)
        ]
    ]
    
    summary_table = Table(summary_data, colWidths=[120, 140, 120, 150])
    summary_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#f8fafc')),
        ('BOX', (0, 0), (-1, -1), 1, colors.HexColor('#e2e8f0')),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cbd5e1')),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('PADDING', (0, 0), (-1, -1), 8),
    ]))
    
    story.append(summary_table)
    story.append(Spacer(1, 15))
    
    # 3. Quick Summary Paragraph
    story.append(Paragraph("<b>Evaluation Summary:</b>", bold_body_style))
    story.append(Paragraph(scores['summary'], body_style))
    story.append(Spacer(1, 15))
    
    # 4. Transcribed Explanation
    story.append(Paragraph("Transcribed Explanation", section_style))
    transcription_box_data = [[Paragraph(f'"{transcription}"', callout_style)]]
    transcription_table = Table(transcription_box_data, colWidths=[530])
    transcription_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#eff6ff')),
        ('BOX', (0, 0), (-1, -1), 1.5, colors.HexColor('#bfdbfe')),
        ('PADDING', (0, 0), (-1, -1), 10),
    ]))
    story.append(transcription_table)
    story.append(Spacer(1, 15))
    
    # 5. Audio Analytics & Speech Quality Features
    story.append(Paragraph("Audio Analytics & Speech Quality Features", section_style))
    
    # Render a clean miniature vector schematic waveform in PDF to show signal amplitude variation
    waveform_drawing = Drawing(530, 40)
    waveform_drawing.add(Rect(0, 0, 530, 40, fillColor=colors.HexColor('#f1f5f9'), strokeColor=colors.HexColor('#e2e8f0')))
    
    # Draw simple waveform bars based on actual RMS timeline or simulated variance
    rms_timeline = audio_metrics.get("rms_timeline", [])
    if len(rms_timeline) > 0:
        # Downsample timeline to fit 80 bars
        num_bars = 80
        step = max(1, len(rms_timeline) // num_bars)
        bars_rms = rms_timeline[::step][:num_bars]
        max_rms = max(bars_rms) if max(bars_rms) > 0 else 1.0
        
        col_width = 530.0 / len(bars_rms)
        for idx, val in enumerate(bars_rms):
            bar_height = (val / max_rms) * 30
            # Center vertically
            y_pos = (40 - bar_height) / 2
            x_pos = idx * col_width + (col_width * 0.1)
            # Add line or rect
            waveform_drawing.add(Rect(
                x_pos, y_pos, col_width * 0.8, bar_height,
                fillColor=colors.HexColor('#3b82f6'), strokeColor=None
            ))
    story.append(waveform_drawing)
    story.append(Spacer(1, 10))
    
    # Fluency stats grid
    filler_counts = audio_metrics.get("filler_counts", {})
    filler_str = ", ".join([f"'{k}': {v}" for k, v in filler_counts.items()]) if filler_counts else "None detected"
    
    fluency_data = [
        [
            Paragraph("<b>Pacing (WPM)</b>", body_style),
            Paragraph(f"{audio_metrics['speaking_pace_wpm']:.0f} WPM", body_style),
            Paragraph("<b>Pause Ratio</b>", body_style),
            Paragraph(f"{audio_metrics['pause_ratio'] * 100:.1f}%", body_style)
        ],
        [
            Paragraph("<b>Total Filler Words</b>", body_style),
            Paragraph(str(audio_metrics['total_filler_count']), body_style),
            Paragraph("<b>Specific Fillers</b>", body_style),
            Paragraph(filler_str, body_style)
        ]
    ]
    fluency_table = Table(fluency_data, colWidths=[120, 140, 120, 150])
    fluency_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#fafafa')),
        ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor('#e5e5e5')),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e5e5e5')),
        ('PADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(fluency_table)
    story.append(Spacer(1, 15))
    
    # 6. Detailed Recommendations and Feedback
    story.append(Paragraph("Detailed Recommendations & Feedback", section_style))
    feedback_bullets = scores.get("feedback_points", [])
    for bullet in feedback_bullets:
        story.append(Paragraph(f"• {bullet}", body_style))
        story.append(Spacer(1, 4))
        
    story.append(Spacer(1, 20))
    story.append(Paragraph("<b>Reference Concept Model:</b>", bold_body_style))
    story.append(Paragraph(reference_concept, ParagraphStyle('RefText', parent=body_style, fontName='Helvetica-Oblique', textColor=colors.HexColor('#6b7280'))))
    
    # Build Document
    doc.build(story)
    return output_path
