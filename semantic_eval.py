import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Global variable to cache the sentence transformer model
_model = None

def get_sentence_transformer_model():
    """
    Attempts to load sentence-transformers model. Returns None if it fails.
    """
    global _model
    if _model is not None:
        return _model
    
    try:
        from sentence_transformers import SentenceTransformer
        # Use a very small and fast model (80MB)
        _model = SentenceTransformer('all-MiniLM-L6-v2')
        return _model
    except Exception as e:
        # Fallback to TF-IDF if sentence-transformers is not available or fails
        return None

def extract_keywords(text):
    """
    Helper to extract meaningful keywords from a text for coverage evaluation.
    """
    # Simple stop words to filter out basic English structural words
    stop_words = {
        'the', 'is', 'at', 'which', 'on', 'a', 'an', 'and', 'but', 'or', 'as', 'if', 
        'of', 'to', 'for', 'with', 'in', 'out', 'by', 'about', 'this', 'that', 'these', 
        'those', 'it', 'its', 'they', 'them', 'their', 'we', 'us', 'our', 'you', 'your',
        'he', 'she', 'him', 'her', 'his', 'was', 'were', 'been', 'has', 'have', 'had',
        'do', 'does', 'did', 'can', 'could', 'should', 'would', 'will', 'also', 'such'
    }
    # Keep only words with alphanumeric characters and length > 2
    words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
    keywords = [w for w in words if w not in stop_words]
    return list(set(keywords))

def evaluate_semantic_similarity(user_text, reference_text):
    """
    Evaluates semantic similarity between the user text and reference text.
    Uses Sentence-Transformers if available, otherwise falls back to TF-IDF cosine similarity.
    Also calculates keyword coverage.
    
    Returns a dict with similarity score, method used, covered keywords, and missed keywords.
    """
    if not user_text or not reference_text:
        return {
            "similarity_score": 0.0,
            "method": "N/A",
            "keyword_coverage": 0.0,
            "covered_keywords": [],
            "missed_keywords": []
        }
    
    # Calculate Keyword Coverage
    ref_keywords = extract_keywords(reference_text)
    user_words = set(re.findall(r'\b[a-zA-Z]{3,}\b', user_text.lower()))
    
    covered = [kw for kw in ref_keywords if kw in user_words]
    missed = [kw for kw in ref_keywords if kw not in user_words]
    
    keyword_coverage = len(covered) / len(ref_keywords) if ref_keywords else 1.0

    # Semantic similarity calculation
    model = get_sentence_transformer_model()
    if model is not None:
        try:
            embeddings = model.encode([user_text, reference_text])
            # Cosine similarity between the two vectors
            dot_product = sum(a * b for a, b in zip(embeddings[0], embeddings[1]))
            norm_a = sum(a * a for a in embeddings[0]) ** 0.5
            norm_b = sum(b * b for b in embeddings[1]) ** 0.5
            similarity = float(dot_product / (norm_a * norm_b))
            # Clip between 0 and 1
            similarity = max(0.0, min(1.0, similarity))
            
            return {
                "similarity_score": similarity,
                "method": "Sentence-Transformers (all-MiniLM-L6-v2)",
                "keyword_coverage": keyword_coverage,
                "covered_keywords": covered,
                "missed_keywords": missed
            }
        except Exception as e:
            # Fallback to TF-IDF on error
            pass
            
    # TF-IDF Cosine Similarity Fallback
    try:
        vectorizer = TfidfVectorizer()
        tfidf = vectorizer.fit_transform([user_text, reference_text])
        similarity = float(cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0])
        
        # Boost similarity slightly if keyword coverage is high to align with semantic perception
        boosted_similarity = similarity * 0.4 + keyword_coverage * 0.6
        similarity = max(0.0, min(1.0, boosted_similarity))
        
        return {
            "similarity_score": similarity,
            "method": "TF-IDF & Keyword Coverage (Fallback)",
            "keyword_coverage": keyword_coverage,
            "covered_keywords": covered,
            "missed_keywords": missed
        }
    except Exception as e:
        return {
            "similarity_score": keyword_coverage,
            "method": "Direct Keyword Intersection (Fallback)",
            "keyword_coverage": keyword_coverage,
            "covered_keywords": covered,
            "missed_keywords": missed
        }
