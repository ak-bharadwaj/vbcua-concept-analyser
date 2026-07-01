def calculate_scores(semantic_results, audio_metrics):
    """
    Calculates comprehensive scoring metrics for user explanation.
    
    Parameters:
    - semantic_results (dict): Output from semantic_eval.py
    - audio_metrics (dict): Output from audio_utils.py
    
    Returns:
    - dict: Detailed scoring breakdown.
    """
    # 1. Semantic Similarity Score (0 to 100)
    semantic_score = round(semantic_results.get("similarity_score", 0.0) * 100, 1)
    keyword_coverage = round(semantic_results.get("keyword_coverage", 0.0) * 100, 1)
    
    # 2. Speech Fluency Score (0 to 100)
    # Start at 100, deduct points for excessive filler words and suboptimal pause ratio
    fluency_score = 100.0
    
    # Filler words penalty: deduct 5 points per filler word, capped at 30 points penalty
    total_fillers = audio_metrics.get("total_filler_count", 0)
    filler_penalty = min(30.0, total_fillers * 5.0)
    fluency_score -= filler_penalty
    
    # Pause ratio penalty: ideal pause ratio is between 0.10 and 0.25 (10% to 25% silence)
    pause_ratio = audio_metrics.get("pause_ratio", 0.0)
    pause_penalty = 0.0
    if pause_ratio > 0.30:
        # Penalize excessive pauses (hesitation)
        pause_penalty = min(30.0, (pause_ratio - 0.30) * 100.0)
    elif pause_ratio < 0.05:
        # Penalize rushing (no natural pauses)
        pause_penalty = 15.0
        
    fluency_score -= pause_penalty
    fluency_score = max(0.0, min(100.0, fluency_score))
    fluency_score = round(fluency_score, 1)
    
    # 3. Overall Comprehension Score
    # Content is king: 70% weight on semantic understanding, 30% on fluency
    overall_score = round((semantic_score * 0.7) + (fluency_score * 0.3), 1)
    
    # 4. Classification
    if overall_score >= 80.0:
        classification = "Strong Understanding"
        color = "green"
        summary_desc = "The explanation covers the concept accurately, displaying strong subject matter grasp and fluent delivery."
    elif overall_score >= 50.0:
        classification = "Moderate Understanding"
        color = "orange"
        summary_desc = "The explanation covers key points but lacks full coverage or fluency. There are minor gaps or hesitations."
    else:
        classification = "Poor Understanding"
        color = "red"
        summary_desc = "The explanation shows major gaps in concept comprehension, or the presentation has severe communication barriers."
        
    # 5. Feedback Generation
    feedback_points = []
    
    # Accuracy Feedback
    covered = semantic_results.get("covered_keywords", [])
    missed = semantic_results.get("missed_keywords", [])
    
    if len(covered) > 0:
        feedback_points.append(f"**Strengths**: You successfully explained core points including: {', '.join(covered[:4])}.")
    if len(missed) > 0:
        feedback_points.append(f"**Gaps**: You missed several key related ideas or terms: {', '.join(missed[:4])}. Try to incorporate these to demonstrate deeper understanding.")
        
    # Fluency Feedback
    speaking_pace = audio_metrics.get("speaking_pace_wpm", 0.0)
    if speaking_pace > 160:
        feedback_points.append("**Pacing**: Your speaking pace is quite fast (approx. WPM of {:.0f}). Try speaking a bit slower to help your audience digest the points.".format(speaking_pace))
    elif speaking_pace < 90 and speaking_pace > 0:
        feedback_points.append("**Pacing**: Your speaking pace is slow (approx. WPM of {:.0f}). Consider speaking slightly more dynamically to keep listeners engaged.".format(speaking_pace))
    else:
        feedback_points.append("**Pacing**: Excellent pace (approx. WPM of {:.0f}). Very clear and natural rate.".format(speaking_pace))
        
    if total_fillers > 3:
        feedback_points.append("**Fluency**: Frequent filler words detected (e.g., 'um', 'like'). Pausing silently instead of using verbal fillers can boost professionalism.")
        
    if pause_ratio > 0.35:
        feedback_points.append("**Articulation**: Long or frequent pauses detected (silence ratio of {:.0f}%). Practice speaking in complete, connected sentences to reduce hesitation.".format(pause_ratio * 100))
    elif pause_ratio < 0.05:
        feedback_points.append("**Articulation**: Very few natural pauses detected. Remember to breathe and pause between punctuation marks for structured delivery.")
        
    return {
        "overall_score": overall_score,
        "semantic_score": semantic_score,
        "keyword_coverage": keyword_coverage,
        "fluency_score": fluency_score,
        "classification": classification,
        "classification_color": color,
        "summary": summary_desc,
        "feedback_points": feedback_points
    }
