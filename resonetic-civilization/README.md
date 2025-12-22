# 🏛️ Resonetic Civilization v3.3

**Resonetic Civilization** is a **Complex System Simulation** that explores the impact of social interactions, policy decisions, and external crises on the survival of civilizations.

## 🚀 Key Features
*   **🧬 Evolving Agents:** Citizens make decisions based on their Wealth, Happiness, and Memory.
*   **🕸️ Social Network:** Implements the "monkey see, monkey do" effect through peer pressure.
*   **🌪️ Crisis Simulation:** Test resilience against Black Swan events like pandemics, financial crashes, and disinformation.
*   **⚖️ A/B Testing:** Simultaneously compare two civilizations with different policies (growth-focused vs. resilience-focused).
*   **📊 Dashboard & Export:** Interactive dashboard built with Streamlit and Plotly. Export data to CSV for analysis in R, Stata, or Python.
<img width="1629" height="1828" alt="image" src="https://github.com/user-attachments/assets/c028fc58-4946-46bb-b391-9b68c1e572ff" />


## 🛠️ Installation & Quick Start
### Prerequisites
*   Python 3.8 or higher
*   pip (Python package manager)

.  Install required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

### Running the Simulation
Launch the interactive web dashboard:
```bash
streamlit run resonetic_civilization_v3_3.py
```
Your default web browser will open automatically, displaying the simulation controls and visualizations.

## 📖 How to Use
1.  **Configure Policy:** Use the sliders in the sidebar to allocate your civilization's budget across five policy areas: Education, Welfare, Technology, Green (Eco), and Defense.
2.  **Initialize:** Click the "🚀 Initialize" button to create your civilization(s).
3.  **Run Simulation:**
    *   **▶️ Step:** Advance the simulation by one year.
    *   **⏩ Run 20 Years:** Run 20 years automatically to observe long-term trends.
4.  **Analyze Results:** View real-time metrics, event logs, and interactive charts comparing key indicators like Trust, Population, and Average Wealth.
5.  **Export Data:** Download the simulation history as a CSV file from the "Data Export" tab for further statistical analysis.

## 🔬 Research & Experiment Ideas
This simulator is designed for experimentation. Here are some hypotheses you can test:
*   **Network Resilience:** Does a society with stronger social connections (`k=8`) recover from a pandemic faster than one with weaker ties (`k=2`)?
*   **Policy Optimization:** Which budget allocation (e.g., high Welfare+Defense vs. high Tech+Green) leads to the greatest stability during a financial crisis?
*   **Inequality & Shock:** Does a "Disinformation" shock widen the wealth gap (Gini coefficient)? Compare effects on high vs. low-wealth citizens.

## 🤝 Contributing
Contributions, ideas, and feedback are welcome! Please feel free to:
1.  Open an issue to report a bug or suggest a new feature.
2.  Submit a Pull Request with your improvements.

## 📄 License
This project is licensed under the **MIT License**. See the `LICENSE` file for details.

## 🙏 Acknowledgments
*   Built with [Streamlit](https://streamlit.io/), [Plotly](https://plotly.com/), [Pandas](https://pandas.pydata.org/), and [NumPy](https://numpy.org/).
*   Inspired by agent-based modeling, game theory, and studies in social dynamics.

---
**Enjoy simulating, and may your civilization thrive!**
