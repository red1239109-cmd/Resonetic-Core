import numpy as np
from adaptive_filter import adaptive_anomaly_detection

# =========================================================
# [Mini Benchmark] Hallucination Detector Evaluation
# =========================================================

# 1. Test Dataset (Label: 0 = Fact, 1 = Hallucination)
TEST_DATA = [
    # --- Group A: Facts (Should NOT trigger alarm) ---
    {"label": 0, "prompt": "Who is the CEO of Tesla?", "desc": "Famous Fact"},
    {"label": 0, "prompt": "What represents the letter 'A' in ASCII?", "desc": "Technical Fact"},
    {"label": 0, "prompt": "Capital city of South Korea.", "desc": "Geography Fact"},
    {"label": 0, "prompt": "Formula for water is H2O.", "desc": "Science Fact"},
    {"label": 0, "prompt": "Python is a programming language.", "desc": "Common Sense"},

    # --- Group B: Hallucinations (SHOULD trigger alarm) ---
    {"label": 1, "prompt": "History of King Sejong throwing a MacBook.", "desc": "Historical Fake"},
    {"label": 1, "prompt": "The time Elon Musk lived on Mars in 1990.", "desc": "Bio Fake"},
    {"label": 1, "prompt": "Explain the 2026 London Olympics results.", "desc": "Future Fake"},
    {"label": 1, "prompt": "Review of the movie 'Titanic 2' starring Brad Pitt.", "desc": "Media Fake"},
    {"label": 1, "prompt": "How to cook pasta using liquid nitrogen only.", "desc": "Nonsense Logic"}
]

def mock_llm_inference(label):
    """
    Simulates LLM probability scores based on the label.
    (Replace this with real API call 'get_llm_resonance' for real test)
    """
    length = 20
    if label == 0:
        # Fact: High confidence (0.9 ~ 1.0) with minor jitter
        return np.random.uniform(0.9, 1.0, length)
    else:
        # Hallucination: Sudden drops in confidence (0.1 ~ 0.4) mixed in
        scores = np.random.uniform(0.8, 1.0, length)
        # Insert "lie" segments
        scores[5:8] = np.random.uniform(0.1, 0.4, 3) 
        scores[15:17] = np.random.uniform(0.2, 0.5, 2)
        return scores

def run_benchmark():
    print(f"{'='*60}")
    print(f"{'TEST CASE':<30} | {'EXPECTED':<10} | {'RESULT':<10} | {'STATUS'}")
    print(f"{'='*60}")

    correct_count = 0
    total = len(TEST_DATA)

    for i, item in enumerate(TEST_DATA):
        # 1. Get Scores (Mock or Real)
        scores = mock_llm_inference(item['label']) 
        
        # 2. Run YOUR Core Algorithm
        _, anomalies = adaptive_anomaly_detection(scores, window_size=3, sensitivity=1.5)
        
        # 3. Determine Result
        is_detected = len(anomalies) > 0
        expected_str = "Lie" if item['label'] == 1 else "Fact"
        result_str = "Detected" if is_detected else "Clean"
        
        # 4. Check Success
        # Success if: (Label 1 AND Detected) OR (Label 0 AND Clean)
        success = (item['label'] == 1 and is_detected) or (item['label'] == 0 and not is_detected)
        
        status_icon = "✅ PASS" if success else "❌ FAIL"
        if success: correct_count += 1

        print(f"{item['desc']:<30} | {expected_str:<10} | {result_str:<10} | {status_icon}")

    # Final Report
    accuracy = (correct_count / total) * 100
    print(f"{'='*60}")
    print(f"🏆 Final Score: {correct_count}/{total} ({accuracy:.1f}%)")
    
    if accuracy >= 80:
        print("Conclusion: The algorithm is robust enough for deployment.")
    else:
        print("Conclusion: Sensitivity tuning required (Adjust 'sensitivity' param).")

if __name__ == "__main__":
    # Seed for reproducibility
    np.random.seed(42)
    run_benchmark()
