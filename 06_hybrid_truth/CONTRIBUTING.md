🧠 Resonance-based Hallucination Detector

Project Resonetics

This project analyzes the output probability scores (logprobs) of Large Language Models (LLMs) to detect periods of low confidence, which often correlate with factual inaccuracies or "hallucination." We treat the LLM's certainty as a time-series signal, applying an adaptive filter to detect anomalies (lies).

🛠 Installation

Clone the repository:

git clone [Your Repository URL]


Install dependencies:

pip install -r requirements.txt


Run the detector (Requires OpenAI API Key):

python resonance_detector.py


© 2025 red1239109-cmd
