# 🧠🧬 Quantum Neuro-Persona (QNP) RAG Explorer 🎭🎓

Welcome to the Quantum Neuro-Persona (QNP) RAG Explorer! This project implements and tests a **hybrid AI system** that combines classical AI techniques (RAG, neural networks) with quantum-inspired processing mechanisms, adaptive **emotional modulation (Limbus Affektus)**, and **cognitive meta-control (Meta-Node)**.

The goal is the **simulation and exploration** of a **coherent artificial persona** that not only answers questions but demonstrably exhibits contextual depth, emotional nuances, logical connections, and signs of self-modulation (creativity, criticism, metacognition). It is a **functional system** designed to resonate semantically, emotionally, and cognitively, and over time, develop its own understandable "personality" through learning and interaction.

---

## ✨ Key Features

*   **🧠 Semantic Network:** Builds an adaptive network of nodes (representing concepts) and their learned associations based on processed text data.
*   **⚛️ Quantum-Inspired Nodes:** Utilize simulated Parameterized Quantum Circuits (PQC) for probabilistic activation dynamics. Analysis of internal quantum states (variance, jumps) provides additional information that influences system processes.
*   **📜 Retrieval-Augmented Generation (RAG):** A two-stage process:
    1.  **Modulated Retrieval:** Finds relevant text chunks from a knowledge base. This search and ranking process is **actively modulated** by the internal emotional (Limbus) and cognitive (Meta-Node) state, as well as by quantum-based score adjustments (activation boost, variance penalty, consistency bias).
    2.  **State-Dependent Generation:** Uses the retrieved chunks and the **entire internal system state** (activated concepts, Limbus emotions, Meta-Node states, quantum metrics) as dynamic context for a Large Language Model (LLM, e.g., Google Gemini). The goal is to generate a coherent, nuanced response from the **perspective of the QNP persona**. The LLM temperature is also dynamically modulated by Limbus and Meta-Nodes.
*   **🎭 Limbus Affektus (Emotional Modulator):** A specialized node simulates a global emotional state (PAD model) based on network activity, actively influencing retrieval parameters, LLM temperature, learning rate, and quantum effect modulation.
*   **💡 Cognitive Meta-Modulators (Implemented):** Specialized nodes that simulate higher cognitive dynamics and further adjust system behavior:
    *   `CreativusNode`: Promotes exploration (influences Temp, LR), activated by Arousal & Quantum Variance.
    *   `CortexCriticusNode`: Promotes coherence/focus (influences Temp, LR, RAG consistency bias), activated by Dominance & Quantum Stability.
    *   `MetaCognitioNode`: Observes global activity & quantum jumps and informs the LLM context (simulates rudimentary self-awareness of the system state).
*   **🎓 Self-Learning with Memory:** Generated, valid responses are persisted, processed as new data, and fed back into the chunk corpus and network training. The system **demonstrably** builds a history of its own "thoughts" and learns from them.
*   **⚙️ Configurable:** Almost all system parameters (Network, Qubits, n_shots, Learning Rates, RAG, Modulation influences for Limbus & Meta-Nodes) are controllable via a JSON file.
*   **💾 State Persistence:** The entire complex state (Nodes, Quantum parameters, Connections, Chunks, Modulator states, Config) is saved and can be loaded to ensure continuity.
*   **🌐 Interactive UI:** A Streamlit web interface allows interaction, detailed state observation (including Limbus & Meta-Nodes), and temporary adjustment of parameters at runtime.

---

## 💡 How the System Works

1.  **Build Knowledge & Structure (Training):**
    *   Text documents are split into chunks.
    *   Chunks are associated with predefined semantic nodes (concepts like "Ethics").
    *   When multiple concepts appear together in a chunk, the **connection** between the corresponding nodes in the network is strengthened. The **learning rate** for this strengthening is modulated by the current state of Limbus and the `Creativus`/`CortexCriticus` Meta-Nodes. This happens over several epochs.
    *   Quantum nodes initialize their internal states and parameters.
    *   A TF-IDF index is built over all chunks.
2.  **Process Queries & Generate Responses (Inference/RAG):**
    *   **Input (Prompt):** A user query is received.
    *   **Network Simulation:**
        *   The prompt initially activates semantic nodes.
        *   Activation spreads through the network.
        *   Quantum nodes perform their PQC simulation; activation, variance, and jumps are calculated and stored (`last_measurement_analysis`).
        *   The Limbus Affektus calculates the current PAD state based on network activity.
        *   The Meta-Nodes (`Creativus`, `CortexCriticus`, `MetaCognitio`) calculate their activation based on the states of Limbus, relevant node quantum metrics, and global activity.
    *   **Modulated Retrieval:**
        *   The system identifies candidate chunks based on the activated semantic nodes.
        *   The **relevance scoring** of these chunks is done by a combination of:
            *   Textual similarity to the prompt (TF-IDF).
            *   **Modulation by Limbus:** Adjustment of the relevance threshold and adding a Pleasure-based bias.
            *   **Modulation by Quantum Effects:** Applying a (Limbus-influenced) activation boost and variance penalty based on the state of associated quantum nodes.
            *   **Modulation by Meta-Nodes:** Adding a consistency bias (based on CortexCriticus and Quantum Stability).
    *   **Context Building:** The top-ranked chunks, activated concepts, the (scaled) Limbus state, and the state of the Meta-Nodes are compiled into a detailed internal system state.
    *   **State-Dependent Generation:**
        *   The LLM (Gemini) receives the user prompt and the compiled internal state. It is given **explicit instructions** to respond as the QNP persona and **authentically use all aspects of the state** to shape tone, focus, content, creativity, and criticality, **without** explaining its technical nature or internal process.
        *   The LLM **temperature** is dynamically adjusted by the state of Limbus *and* the `Creativus`/`CortexCriticus` Meta-Nodes.
    *   **Self-Learning & Memory:** Valid, generated responses are saved to `learn.txt`, processed during the next training run, and thus influence the future knowledge base and network structure.
3.  **Persistence:** The current state of all components is saved to a JSON file.

---

## 📁 Project Structure

```markdown
.
├── quantum_neuropersona_hybrid_llm.py # Main logic: Classes for Nodes, QNP, Processor, RAG, Limbus, Meta-Nodes etc.
├── qllm_train_hybrid.py # Script for training/initializing the network. (Possibly rename to qnp_train.py)
├── qnp_streamlit_ui.py # Interactive web interface with Streamlit. (Adapted name)
├── config_qllm.json # Configuration file. (Possibly rename to config_qnp.json)
├── qnp_state.json # Saved network state. (Adapted name)
├── requirements.txt # Python dependencies.
├── training_data/ # Directory for training documents.
│ ├── ... (Text files)
│ └── learn.txt # File for self-learned responses (will be created).
└── README.md # This file.
```
*(Note: File names in code and for execution may need adjustment)*

---

## 🚀 Setup & Installation

1.  **Clone Repository:**
    ```bash
    git clone https://github.com/CipherCorePro/Quantum-Augmented-Retrieval.git
    cd Quantum-Augmented-Retrieval
    ```
2.  **Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    # Windows: .\venv\Scripts\activate
    # Linux/Mac: source venv/bin/activate
    ```
3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Ensure `requirements.txt` exists and contains: `numpy`, `scikit-learn`, `google-generativeai`, `streamlit`, `pandas`, `tqdm` (optional))*
4.  **Google Gemini API Key:** Set as environment variable `GEMINI_API_KEY` or provide as a Streamlit Secret.
5.  **Training Data:** Place text files in `training_data/` and enter their paths in the config file under `training_files`.

---

## ▶️ Usage

1.  **(Optional) Adjust Configuration:** Edit the `config_*.json` file.
2.  **Train/Initialize Network:**
    ```bash
    python qllm_train_hybrid.py -c config_qllm.json -s qnp_state.json [-f]
    ```
    *(Adjust file names and options. `-f` forces rebuild.)*
3.  **Start Interactive UI:**
    ```bash
    streamlit run qnp_streamlit_ui.py --server.address 0.0.0.0 --server.port 787
    ```
    *(Adjust script names and port. Open the URL in your browser.)*
    *   In the UI: Load state (`qnp_state.json`), enter prompts, observe states (Limbus, Meta-Nodes), temporarily adjust parameters, inspect connections.

---

## ⚙️ Configuration (`config_qllm.json`)

Important parameters (see `DEFAULT_CONFIG` in code for all):

*   `training_files`, `semantic_nodes`, `chunk_size`, `chunk_overlap`
*   `connection_learning_rate`, `connection_decay_rate`
*   `use_quantum_nodes`, `default_num_qubits`, `simulation_n_shots`
*   `enable_rag`, `generator_model_name`, `generator_temperature` (Base value)
*   `enable_self_learning`, `self_learning_file_path`
*   **`limbus_...` parameters:** Control the Limbus node and its influence on Retrieval, Temperature, Learning Rate, Quantum Effects.
*   **`meta_nodes_enabled`**: Global switch for Meta-Nodes.
*   **`creativus_num_qubits`**, **`cortex_criticus_num_qubits`**: Qubit count.
*   **`creativus_influence_...`**, **`criticus_influence_...`**, **`metacognitio_influence_...`**: Control the influence of Meta-Nodes on Temperature, Learning Rate, RAG bias, and Prompt context.

---

## 💾 State File (`qnp_state.json`)

Saves the **entire learnable state** of the system, including:
*   All nodes (semantic, Limbus, Meta) with their attributes and quantum parameters.
*   All learned connections and weights.
*   All processed text chunks and their node associations.
*   State information of the modulators (e.g., `emotion_state`, `last_total_jumps`).
*   The configuration valid at runtime.
*   Metadata (processed sources, TF-IDF mapping).

---

## 🔭 Future Ideas

*   Refining Meta-Node logic and interactions (e.g., mutual influence).
*   Implementing more complex RAG biases (e.g., genuine novelty bias).
*   Real-time adjustment of modulation parameters based on dialogue history.
*   Evaluation metrics for coherence, persona stability, and response quality.
*   Investigation of emergent long-term dynamics and "personality development".
*   Connecting to real quantum processors for parts of the simulation.

---

## 🤝 Contributing

Contributions are welcome! Please create an Issue to report bugs or discuss new features. Pull Requests are also greatly appreciated.

---

## 📜 License

[MIT License](LICENSE).
