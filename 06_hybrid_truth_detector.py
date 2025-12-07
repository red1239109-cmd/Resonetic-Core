# Copyright (c) 2025 red1239109-cmd
# Licensed under AGPL-3.0. See LICENSE file for details.

import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

# =========================================================
# 1. Phase Vortex Truth Detector (Pure NumPy Implementation)
# =========================================================

class PhaseVortexTruthDetector:
    def __init__(self, vortex_threshold=0.7):
        self.vortex_threshold = vortex_threshold
        
        # Emotion dictionary (English)
        self.emotion_dict = {
            'love': 0.9, 'happy': 0.8, 'joy': 0.85, 'hope': 0.75,
            'sad': -0.7, 'angry': -0.6, 'worry': -0.5, 'disappoint': -0.65,
            'truth': 0.85, 'certain': 0.7, 'clear': 0.6,
            'lie': -0.9, 'maybe': -0.3, 'perhaps': -0.35, 'possibly': -0.4,
            'good': 0.6, 'bad': -0.6, 'many': 0.4, 'few': -0.4,
            'always': 0.7, 'never': -0.8, 'every': 0.5, 'none': -0.7,
            'beautiful': 0.8, 'ugly': -0.7, 'strong': 0.6, 'weak': -0.6,
            'success': 0.8, 'fail': -0.7, 'win': 0.7, 'lose': -0.6
        }
        
        # Riemann phase vortex pattern (simulated with NumPy)
        self.riemann_pattern = self._generate_riemann_pattern()
    
    def _generate_riemann_pattern(self):
        """Generate Riemann ζ(s) phase vortex pattern using only NumPy"""
        t = np.linspace(0, 2*np.pi, 100)
        # Stable truth pattern: smooth periodic function
        pattern = np.sin(t) + 0.3 * np.sin(2*t) + 0.1 * np.sin(3*t)
        return pattern
    
    def _interpolate_to_match_length(self, original, target_length):
        """Linear interpolation without scipy"""
        if len(original) == target_length:
            return original
        
        # Simple linear interpolation
        x_old = np.linspace(0, 1, len(original))
        x_new = np.linspace(0, 1, target_length)
        interpolated = np.interp(x_new, x_old, original)
        return interpolated
    
    def text_to_complex_embedding(self, text):
        """Convert text to complex number sequence"""
        words = text.split()
        complex_seq = []
        
        for i, word in enumerate(words):
            # Clean and normalize word
            clean_word = ''.join(c for c in word if c.isalnum()).lower()
            if not clean_word:
                continue
            
            # Emotion score (real part)
            emotion_score = 0.0
            # Check for emotion words in dictionary
            for key in self.emotion_dict:
                if key in clean_word or clean_word in key:
                    emotion_score = self.emotion_dict[key]
                    break
            
            # Logic consistency (imaginary part)
            position_factor = 1.0 - (i / max(len(words), 1)) * 0.3
            length_factor = min(len(clean_word) / 15, 1.0)
            
            # Common words get consistency bonus
            common_words = ['the', 'is', 'are', 'in', 'on', 'at', 'and', 'of', 'to', 'a', 'an']
            common_bonus = 0.1 if clean_word in common_words else 0.0
            
            logic_score = position_factor * length_factor + common_bonus
            
            # Create complex number
            z = complex(emotion_score, logic_score)
            complex_seq.append(z)
        
        return np.array(complex_seq)
    
    def detect_vortex_pattern(self, complex_seq):
        """Detect phase vortex patterns in complex sequence"""
        if len(complex_seq) < 3:
            return [], 0.0
        
        phases = np.angle(complex_seq)
        unwrapped = np.unwrap(phases)
        
        # Analyze phase changes
        phase_diffs = np.diff(unwrapped)
        
        # Detect vortex points (rapid phase changes)
        vortex_indices = []
        vortex_intensities = []
        
        for i in range(len(phase_diffs) - 1):
            # Magnitude of consecutive changes
            change_magnitude = abs(phase_diffs[i]) + abs(phase_diffs[i+1])
            direction_change = abs(np.sign(phase_diffs[i]) - np.sign(phase_diffs[i+1]))
            
            # Vortex condition: large change + direction change
            if change_magnitude > 1.5 and direction_change > 0.5:
                vortex_indices.append(i+1)
                vortex_intensities.append(change_magnitude)
        
        avg_intensity = np.mean(vortex_intensities) if vortex_intensities else 0.0
        return vortex_indices, avg_intensity
    
    def calculate_riemann_similarity(self, phase_pattern):
        """Calculate similarity between text phase pattern and Riemann pattern"""
        if len(phase_pattern) < 10:
            return 0.5  # Default for short texts
        
        # Match lengths
        if len(phase_pattern) != len(self.riemann_pattern):
            phase_pattern = self._interpolate_to_match_length(
                phase_pattern, len(self.riemann_pattern)
            )
        
        # Cosine similarity
        similarity = np.dot(phase_pattern, self.riemann_pattern) / (
            np.linalg.norm(phase_pattern) * np.linalg.norm(self.riemann_pattern)
        )
        
        # Normalize to 0-1 range
        return (similarity + 1) / 2
    
    def analyze_truthfulness(self, text):
        """Analyze text truthfulness using phase vortex method"""
        complex_seq = self.text_to_complex_embedding(text)
        
        if len(complex_seq) == 0:
            return {
                'truth_score': 0.5,
                'vortex_indices': [],
                'vortex_intensity': 0.0,
                'riemann_similarity': 0.5,
                'complex_sequence': complex_seq
            }
        
        # Detect vortex patterns
        vortex_indices, vortex_intensity = self.detect_vortex_pattern(complex_seq)
        
        # Extract phase pattern
        phases = np.angle(complex_seq)
        unwrapped_phases = np.unwrap(phases)
        
        # Calculate Riemann pattern similarity
        riemann_similarity = self.calculate_riemann_similarity(unwrapped_phases)
        
        # Calculate truth score
        vortex_penalty = min(vortex_intensity * len(vortex_indices) / max(len(complex_seq), 1), 1.0)
        base_score = riemann_similarity * (1 - vortex_penalty)
        
        truth_score = max(0.0, min(1.0, base_score))
        
        return {
            'truth_score': float(truth_score),
            'vortex_indices': vortex_indices,
            'vortex_intensity': float(vortex_intensity),
            'riemann_similarity': float(riemann_similarity),
            'complex_sequence': complex_seq
        }

# =========================================================
# 2. Adaptive Anomaly Detector
# =========================================================

class AdaptiveAnomalyDetector:
    def __init__(self, window_size=5, sensitivity=2.0):
        self.window_size = window_size
        self.sensitivity = sensitivity
    
    def adaptive_anomaly_detection(self, scores):
        """Adaptive anomaly detection (original algorithm)"""
        thresholds = []
        anomalies = []
        
        for i in range(len(scores)):
            # Local context window
            start = max(0, i - self.window_size)
            end = min(len(scores), i + self.window_size + 1)
            context = scores[start:end]
            
            # Local statistics
            local_mean = np.mean(context)
            local_std = np.std(context) if len(context) > 1 else 0.1
            
            # Dynamic threshold
            threshold = local_mean - self.sensitivity * local_std
            threshold = max(threshold, 0.3)  # Minimum safety net
            
            thresholds.append(threshold)
            
            # Anomaly detection
            if scores[i] < threshold:
                anomalies.append(i)
        
        return np.array(thresholds), anomalies
    
    def analyze_truthfulness(self, text_or_scores):
        """
        Analyze truthfulness of text or score array
        """
        if isinstance(text_or_scores, str):
            # Text input: generate word confidence scores
            words = text_or_scores.split()
            word_scores = []
            
            for word in words:
                clean_word = ''.join(c for c in word if c.isalnum()).lower()
                if not clean_word:
                    continue
                
                # Word length based score (longer words = more specific)
                length_score = min(len(clean_word) / 20, 1.0)
                
                # Common word bonus
                common_words = ['the', 'is', 'are', 'in', 'on', 'at', 'and', 'of', 'to', 'a', 'an']
                frequency_bonus = 0.1 if clean_word in common_words else 0.0
                
                # Combined score
                score = length_score * 0.7 + 0.3 + frequency_bonus
                word_scores.append(min(score, 1.0))
            
            scores = np.array(word_scores) if word_scores else np.array([0.5])
        else:
            # Score array input
            scores = np.array(text_or_scores)
        
        # Run anomaly detection
        thresholds, anomalies = self.adaptive_anomaly_detection(scores)
        
        # Calculate truth score
        if len(scores) == 0:
            truth_score = 0.5
        else:
            # Score based on anomaly ratio
            anomaly_ratio = len(anomalies) / len(scores)
            truth_score = 1.0 - min(anomaly_ratio, 1.0)
            
            # Adjust with mean score
            mean_score = np.mean(scores)
            truth_score = (truth_score + mean_score) / 2
        
        return {
            'truth_score': float(truth_score),
            'anomalies': anomalies,
            'thresholds': thresholds,
            'scores': scores,
            'mean_score': float(np.mean(scores)) if len(scores) > 0 else 0.5
        }

# =========================================================
# 3. Hybrid Riemann Adaptive Truth Detector
# =========================================================

class RiemannAdaptiveTruthDetector:
    """
    Hybrid Truth Detection System combining:
    1. Phase Vortex Detection (Riemann-inspired mathematical analysis)
    2. Adaptive Anomaly Detection (real-time engineering analysis)
    
    This system provides epistemological framework for analyzing
    textual truthfulness from multiple perspectives.
    """
    
    def __init__(self, vortex_thresh=0.7, adapt_window=5, adapt_sens=2.0):
        self.vortex_detector = PhaseVortexTruthDetector(vortex_thresh)
        self.adaptive_detector = AdaptiveAnomalyDetector(adapt_window, adapt_sens)
        
        # Weights (learnable)
        self.vortex_weight = 0.6  # Phase vortex weight
        self.adaptive_weight = 0.4  # Adaptive detection weight
        
        # Analysis history
        self.analysis_history = []
    
    def analyze(self, text):
        """
        Perform hybrid text analysis
        
        Args:
            text: Text to analyze
            
        Returns:
            dict: Analysis results
        """
        # Step 1: Phase vortex analysis
        vortex_result = self.vortex_detector.analyze_truthfulness(text)
        
        # Step 2: Adaptive analysis (using phase scores as input)
        complex_seq = vortex_result['complex_sequence']
        phase_scores = []
        
        if len(complex_seq) > 0:
            # Combine magnitude (confidence) and phase stability
            for z in complex_seq:
                magnitude = abs(z)  # Complex magnitude = confidence
                phase = np.angle(z)
                phase_stability = 1.0 - (abs(phase) / np.pi)
                
                combined_score = magnitude * 0.6 + phase_stability * 0.4
                phase_scores.append(min(combined_score, 1.0))
        else:
            phase_scores = [0.5]
        
        adaptive_result = self.adaptive_detector.analyze_truthfulness(phase_scores)
        
        # Step 3: Hybrid score calculation
        vortex_score = vortex_result['truth_score']
        adaptive_score = adaptive_result['truth_score']
        
        # Weighted average
        final_score = (
            vortex_score * self.vortex_weight + 
            adaptive_score * self.adaptive_weight
        )
        
        # Step 4: False positive correction
        agreement = 1.0 - abs(vortex_score - adaptive_score)
        confidence = min(vortex_score, adaptive_score) * agreement
        
        # Final adjusted score
        adjusted_score = final_score * 0.7 + confidence * 0.3
        
        # Result
        result = {
            'text': text,
            'vortex': vortex_result,
            'adaptive': adaptive_result,
            'final_score': float(adjusted_score),
            'confidence': float(confidence),
            'agreement': float(agreement),
            'verdict': 'TRUTHFUL' if adjusted_score > 0.65 else 'SUSPICIOUS'
        }
        
        self.analysis_history.append(result)
        return result
    
    def update_weights(self, feedback):
        """
        Update weights based on feedback
        
        Args:
            feedback: {'vortex_correct': bool, 'adaptive_correct': bool}
        """
        if 'vortex_correct' in feedback and 'adaptive_correct' in feedback:
            vortex_correct = feedback['vortex_correct']
            adaptive_correct = feedback['adaptive_correct']
            
            if vortex_correct and not adaptive_correct:
                # Phase vortex was more accurate
                self.vortex_weight = min(self.vortex_weight + 0.05, 0.9)
                self.adaptive_weight = max(self.adaptive_weight - 0.05, 0.1)
            elif adaptive_correct and not vortex_correct:
                # Adaptive detection was more accurate
                self.adaptive_weight = min(self.adaptive_weight + 0.05, 0.9)
                self.vortex_weight = max(self.vortex_weight - 0.05, 0.1)
    
    def visualize_analysis(self, result, save_path=None):
        """
        Visualize analysis results
        
        Args:
            result: Output from analyze() method
            save_path: Path to save visualization (None to display)
        """
        fig = plt.figure(figsize=(15, 10))
        
        # 1. Complex Plane Visualization
        ax1 = plt.subplot(2, 3, 1)
        complex_seq = result['vortex']['complex_sequence']
        if len(complex_seq) > 0:
            real_parts = np.real(complex_seq)
            imag_parts = np.imag(complex_seq)
            colors = range(len(complex_seq))
            
            scatter = ax1.scatter(real_parts, imag_parts, c=colors, cmap='viridis', 
                                 alpha=0.6, s=50)
            plt.colorbar(scatter, ax=ax1, label='Word Position')
            
            vortex_idx = result['vortex']['vortex_indices']
            if vortex_idx:
                ax1.scatter(real_parts[vortex_idx], imag_parts[vortex_idx], 
                           color='red', s=100, label='Vortex Points', zorder=5)
        
        ax1.set_xlabel('Real Part (Emotion)')
        ax1.set_ylabel('Imaginary Part (Logic)')
        ax1.set_title('Text Representation in Complex Plane')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # 2. Phase Change Graph
        ax2 = plt.subplot(2, 3, 2)
        if len(complex_seq) > 0:
            phases = np.angle(complex_seq)
            unwrapped = np.unwrap(phases)
            
            ax2.plot(unwrapped, 'b-', label='Phase Change', linewidth=2)
            if vortex_idx:
                ax2.scatter(vortex_idx, unwrapped[vortex_idx], 
                           color='red', s=50, label='Vortex', zorder=5)
            
            riemann_pattern = self.vortex_detector.riemann_pattern[:len(unwrapped)]
            ax2.plot(riemann_pattern, 'g--', label='Riemann Pattern', alpha=0.7)
        
        ax2.set_xlabel('Word Position')
        ax2.set_ylabel('Phase (radians)')
        ax2.set_title('Phase Change Pattern')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Adaptive Detection Results
        ax3 = plt.subplot(2, 3, 3)
        scores = result['adaptive']['scores']
        thresholds = result['adaptive']['thresholds']
        
        if len(scores) > 0:
            x = range(len(scores))
            ax3.plot(x, scores, 'b-', label='Confidence Scores', linewidth=2)
            ax3.plot(x, thresholds, 'r--', label='Adaptive Threshold', linewidth=2, alpha=0.7)
            
            anomalies = result['adaptive']['anomalies']
            if anomalies:
                ax3.scatter(anomalies, scores[anomalies], 
                           color='red', s=50, label='Anomalies', zorder=5)
            
            ax3.fill_between(x, thresholds, scores, where=(scores < thresholds),
                            color='red', alpha=0.2, label='Anomaly Region')
        
        ax3.set_xlabel('Word Position')
        ax3.set_ylabel('Score')
        ax3.set_title('Adaptive Anomaly Detection')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Score Comparison Chart
        ax4 = plt.subplot(2, 3, 4)
        scores_comparison = {
            'Phase Vortex': result['vortex']['truth_score'],
            'Adaptive': result['adaptive']['truth_score'],
            'Hybrid': result['final_score'],
            'Confidence': result['confidence']
        }
        
        colors = ['blue', 'orange', 'green', 'purple']
        bars = ax4.bar(range(len(scores_comparison)), 
                      list(scores_comparison.values()),
                      color=colors, alpha=0.7)
        
        ax4.set_xticks(range(len(scores_comparison)))
        ax4.set_xticklabels(list(scores_comparison.keys()), rotation=45)
        ax4.set_ylabel('Score (0~1)')
        ax4.set_title('Truthfulness Scores by Method')
        ax4.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, value in zip(bars, scores_comparison.values()):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{value:.3f}', ha='center', va='bottom')
        
        # 5. Weight Information
        ax5 = plt.subplot(2, 3, 5)
        weights = {
            'Phase Vortex': self.vortex_weight,
            'Adaptive': self.adaptive_weight
        }
        
        ax5.pie(list(weights.values()), labels=list(weights.keys()),
               autopct='%1.1f%%', colors=['blue', 'orange'])
        ax5.set_title('Algorithm Weights')
        
        # 6. Final Verdict Display
        ax6 = plt.subplot(2, 3, 6)
        ax6.axis('off')
        
        info_text = f"""
        📊 Analysis Summary
        {'='*40}
        
        📝 Input Text:
        \"{result['text'][:100]}{'...' if len(result['text']) > 100 else ''}\"
        
        🎯 Final Score: {result['final_score']:.3f}
        ⚖️ Verdict: {result['verdict']}
        
        📈 Component Scores:
        • Phase Vortex: {result['vortex']['truth_score']:.3f}
        • Adaptive Detection: {result['adaptive']['truth_score']:.3f}
        • Agreement: {result['agreement']:.3f}
        • Confidence: {result['confidence']:.3f}
        
        🔍 Detection Details:
        • Vortex Points: {len(result['vortex']['vortex_indices'])} locations
        • Anomalies: {len(result['adaptive']['anomalies'])} locations
        • Riemann Similarity: {result['vortex']['riemann_similarity']:.3f}
        """
        
        ax6.text(0.05, 0.95, info_text, fontsize=10, 
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.suptitle('🌀 Riemann Adaptive Hybrid Truth Detector', 
                    fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✅ Visualization saved: {save_path}")
        else:
            plt.show()
        
        plt.close(fig)

# =========================================================
# 4. Demo Execution
# =========================================================

def run_demo():
    """Run demonstration of the hybrid truth detector"""
    print("🌀 Riemann Adaptive Hybrid Truth Detector")
    print("="*60)
    
    test_texts = [
        # 1. Truthful statement
        "I read a book and drank coffee this morning. The weather was nice so I took a walk in the park.",
        
        # 2. Contradictory statement (possible lie)
        "Yesterday it rained heavily, but the sky was clear with no clouds at all.",
        
        # 3. Emotionally unstable statement
        "I'm so happy and joyful too excited feels like crazy why so sad also angry",
        
        # 4. Logically discontinuous statement
        "Apples are fruits. Cats live in water. The Earth is square shaped.",
        
        # 5. Neutral factual statements
        "Python is a programming language. The capital of Korea is Seoul. Water consists of hydrogen and oxygen."
    ]
    
    # Create detector
    detector = RiemannAdaptiveTruthDetector(
        vortex_thresh=0.7,
        adapt_window=5,
        adapt_sens=2.0
    )
    
    # Analyze each text
    for i, text in enumerate(test_texts, 1):
        print(f"\n{'='*60}")
        print(f"Test {i}:")
        print(f"Text: {text[:80]}...")
        
        # Perform analysis
        result = detector.analyze(text)
        
        # Print results
        print(f"📊 Final Score: {result['final_score']:.3f}")
        print(f"⚖️ Verdict: {result['verdict']}")
        print(f"🔍 Phase Vortex Score: {result['vortex']['truth_score']:.3f}")
        print(f"📈 Adaptive Detection Score: {result['adaptive']['truth_score']:.3f}")
        print(f"🤝 Agreement: {result['agreement']:.3f}")
        
        # Save visualization
        detector.visualize_analysis(result, save_path=f'hybrid_analysis_{i}.png')
    
    print("\n" + "="*60)
    print("✅ All analyses completed!")
    print("📁 Visualization files saved in current directory.")
    
    # Print statistics
    print("\n📊 Overall Statistics:")
    print(f"• Texts analyzed: {len(detector.analysis_history)}")
    
    verdicts = [r['verdict'] for r in detector.analysis_history]
    verdict_counts = Counter(verdicts)
    
    for verdict, count in verdict_counts.items():
        percentage = (count / len(detector.analysis_history)) * 100
        print(f"• {verdict}: {count} texts ({percentage:.1f}%)")
    
    avg_score = np.mean([r['final_score'] for r in detector.analysis_history])
    print(f"• Average truth score: {avg_score:.3f}")

if __name__ == "__main__":
    run_demo()


            

