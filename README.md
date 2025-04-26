# 🧠🧬 Quantum Neuro-Persona (QNP) RAG Explorer 🎭🎓

Willkommen beim Quantum Neuro-Persona (QNP) RAG Explorer! Dieses Projekt implementiert und testet ein **hybrides KI-System**, das klassische KI-Techniken (RAG, neuronale Netze) mit quanten-inspirierten Verarbeitungsmechanismen, einer adaptiven **emotionalen Modulation (Limbus Affektus)** und einer **kognitiven Meta-Steuerung (Meta-Knoten)** verbindet.

Das Ziel ist die **Simulation und Erforschung** einer **kohärenten künstlichen Persona**, die nicht nur auf Fragen antwortet, sondern dabei **beobachtbar** kontextuelle Tiefe, emotionale Nuancen, logische Verknüpfungen und Anzeichen von Selbstmodulation (Kreativität, Kritik, Metakognition) zeigt. Es ist ein **funktionierendes System**, das darauf ausgelegt ist, semantisch, emotional und kognitiv zu resonieren und über die Zeit eine eigene, nachvollziehbare "Persönlichkeit" durch Lernen und Interaktion zu entwickeln.

---

## ✨ Hauptmerkmale

*   **🧠 Semantisches Netzwerk:** Baut ein adaptives Netzwerk aus Knoten (die Konzepte repräsentieren) und deren gelernten Assoziationen auf, basierend auf verarbeiteten Textdaten.
*   **⚛️ Quanten-inspirierte Knoten:** Nutzen simulierte Parametrisierte Quantenschaltungen (PQC) für eine probabilistische Aktivierungsdynamik. Die Analyse interner Quantenzustände (Varianz, Sprünge) liefert zusätzliche Informationen, die die Systemprozesse beeinflussen.
*   **📜 Retrieval-Augmented Generation (RAG):** Ein zweistufiger Prozess:
    1.  **Moduliertes Retrieval:** Findet relevante Text-Chunks aus einer Wissensbasis. Dieser Such- und Rankingprozess wird **aktiv moduliert** durch den internen emotionalen (Limbus) und kognitiven (Meta-Knoten) Zustand sowie durch quanten-basierte Score-Anpassungen (Aktivierungs-Boost, Varianz-Malus, Konsistenz-Bias).
    2.  **Zustandsabhängige Generation:** Nutzt die gefundenen Chunks und den **gesamten internen Systemzustand** (aktivierte Konzepte, Limbus-Emotionen, Meta-Zustände, Quanten-Metriken) als dynamischen Kontext für ein großes Sprachmodell (LLM, z.B. Google Gemini). Ziel ist die Generierung einer kohärenten, nuancierten Antwort aus der **Perspektive der QNP-Persona**. Die LLM-Temperatur wird ebenfalls dynamisch durch Limbus und Meta-Knoten moduliert.
*   **🎭 Limbus Affektus (Emotionaler Modulator):** Ein spezialisierter Knoten simuliert einen globalen emotionalen Zustand (PAD-Modell), der auf der Netzwerkaktivität basiert und aktiv Retrieval-Parameter, LLM-Temperatur, Lernrate und Quanten-Effekt-Modulation beeinflusst.
*   **💡 Kognitive Meta-Modulatoren (Implementiert):** Spezialisierte Knoten, die höhere kognitive Dynamiken simulieren und das Systemverhalten weiter anpassen:
    *   `CreativusNode`: Fördert Exploration (beeinflusst Temp, LR), aktiviert durch Arousal & Quantenvarianz.
    *   `CortexCriticusNode`: Fördert Kohärenz/Fokus (beeinflusst Temp, LR, RAG-Konsistenz-Bias), aktiviert durch Dominanz & Quantenstabilität.
    *   `MetaCognitioNode`: Beobachtet globale Aktivität & Quantensprünge und informiert den LLM-Kontext (simuliert rudimentäre Selbstwahrnehmung des Systemzustands).
*   **🎓 Selbstlernend mit Gedächtnis:** Generierte, valide Antworten werden persistiert, als neue Daten verarbeitet und fließen in den Chunk-Korpus und das Netzwerk-Training zurück. Das System baut **nachweislich** eine Historie seiner eigenen "Gedanken" auf und lernt daraus.
*   **⚙️ Konfigurierbar:** Nahezu alle Systemparameter (Netzwerk, Qubits, n_shots, Lernraten, RAG, Modulations-Einflüsse für Limbus & Meta-Knoten) sind über eine JSON-Datei steuerbar.
*   **💾 Zustandspersistenz:** Der gesamte komplexe Zustand (Knoten, Quantenparameter, Verbindungen, Chunks, Modulator-Zustände, Config) wird gespeichert und kann geladen werden, um Kontinuität zu gewährleisten.
*   **🌐 Interaktive UI:** Eine Streamlit-Web-Oberfläche ermöglicht die Interaktion, detaillierte Zustandsbeobachtung (inkl. Limbus & Meta-Knoten) und die temporäre Anpassung von Parametern zur Laufzeit.

---

## 💡 Funktionsweise des Systems

1.  **Wissen & Struktur aufbauen (Training):**
    *   Textdokumente werden in Chunks zerlegt.
    *   Chunks werden mit vordefinierten semantischen Knoten (Konzepte wie "Ethik") assoziiert.
    *   Wenn mehrere Konzepte gemeinsam in einem Chunk auftreten, wird die **Verbindung** zwischen den entsprechenden Knoten im Netzwerk gestärkt. Die **Lernrate** für diese Stärkung wird durch den aktuellen Zustand von Limbus und den Meta-Knoten `Creativus`/`CortexCriticus` moduliert. Dies geschieht über mehrere Epochen.
    *   Quantenknoten initialisieren ihre internen Zustände und Parameter.
    *   Ein TF-IDF Index wird über alle Chunks aufgebaut.
2.  **Anfragen verarbeiten & Antworten generieren (Inferenz/RAG):**
    *   **Input (Prompt):** Eine Nutzeranfrage wird empfangen.
    *   **Netzwerk-Simulation:**
        *   Der Prompt aktiviert initial semantische Knoten.
        *   Aktivierung breitet sich durch das Netzwerk aus.
        *   Quantenknoten führen ihre PQC-Simulation durch; Aktivierung, Varianz und Sprünge werden berechnet und gespeichert (`last_measurement_analysis`).
        *   Der Limbus Affektus berechnet den aktuellen PAD-Zustand basierend auf der Netzwerkaktivität.
        *   Die Meta-Knoten (`Creativus`, `CortexCriticus`, `MetaCognitio`) berechnen ihre Aktivierung basierend auf den Zuständen von Limbus, den Quanten-Metriken relevanter Knoten und der globalen Aktivität.
    *   **Moduliertes Retrieval:**
        *   Das System identifiziert Kandidaten-Chunks basierend auf den aktivierten semantischen Knoten.
        *   Die **Relevanzbewertung** dieser Chunks erfolgt durch eine Kombination aus:
            *   Textueller Ähnlichkeit zum Prompt (TF-IDF).
            *   **Modulation durch Limbus:** Anpassung des Relevanz-Schwellenwerts und Hinzufügen eines Pleasure-basierten Bias.
            *   **Modulation durch Quanten-Effekte:** Anwendung eines (durch Limbus beeinflussten) Aktivierungs-Boosts und Varianz-Malus basierend auf dem Zustand der assoziierten Quantenknoten.
            *   **Modulation durch Meta-Knoten:** Hinzufügen eines Konsistenz-Bias (basierend auf CortexCriticus und Quanten-Stabilität).
    *   **Kontextaufbau:** Die Top-gerankten Chunks, die aktivierten Konzepte, der (skalierte) Limbus-Zustand und der Zustand der Meta-Knoten werden als detaillierter interner Systemzustand aufbereitet.
    *   **Zustandsabhängige Generation:**
        *   Das LLM (Gemini) erhält den Nutzer-Prompt und den aufbereiteten internen Zustand. Es bekommt die **explizite Anweisung**, als QNP-Persona zu antworten und **alle Aspekte des Zustands authentisch** zur Formung von Tonfall, Fokus, Inhalt, Kreativität und Kritikalität zu nutzen, **ohne** seine technische Natur oder den internen Prozess zu erklären.
        *   Die LLM-**Temperatur** wird dynamisch durch den Zustand von Limbus *und* der Meta-Knoten `Creativus`/`CortexCriticus` angepasst.
    *   **Selbstlernen & Gedächtnis:** Valide, generierte Antworten werden in `learn.txt` gespeichert, beim nächsten Trainingslauf verarbeitet und beeinflussen so die zukünftige Wissensbasis und Netzwerkstruktur.
3.  **Persistenz:** Der aktuelle Zustand aller Komponenten wird in einer JSON-Datei gespeichert.

---

## 📁 Projektstruktur
Use code with caution.
Markdown
.
├── quantum_neuropersona_hybrid_llm.py # Hauptlogik: Klassen für Knoten, QNP, Prozessor, RAG, Limbus, Meta-Knoten etc.
├── qllm_train_hybrid.py # Skript zum Trainieren/Initialisieren des Netzwerks. (Ggf. umbenennen zu qnp_train.py)
├── qnp_streamlit_ui.py # Interaktive Web-Oberfläche mit Streamlit. (Angepasster Name)
├── config_qllm.json # Konfigurationsdatei. (Ggf. umbenennen zu config_qnp.json)
├── qnp_state.json # Gespeicherter Zustand des Netzwerks. (Angepasster Name)
├── requirements.txt # Python-Abhängigkeiten.
├── training_data/ # Verzeichnis für Trainingsdokumente.
│ ├── ... (Textdateien)
│ └── learn.txt # Datei für selbstgelernte Antworten (wird erstellt).
└── README.md # Diese Datei.
*(Hinweis: Dateinamen im Code und für den Aufruf ggf. anpassen)*

---

## 🚀 Setup & Installation

1.  **Repository klonen:**
    ```bash
    git clone <repository_url>
    cd <repository_directory>
    ```
2.  **Virtuelle Umgebung (Empfohlen):**
    ```bash
    python -m venv venv
    # Windows: .\venv\Scripts\activate
    # Linux/Mac: source venv/bin/activate
    ```
3.  **Abhängigkeiten installieren:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Stelle sicher, dass `requirements.txt` existiert und enthält: `numpy`, `scikit-learn`, `google-generativeai`, `streamlit`, `pandas`, `tqdm` (optional))*
4.  **Google Gemini API Key:** Als Umgebungsvariable `GEMINI_API_KEY` setzen oder als Streamlit Secret bereitstellen.
5.  **Trainingsdaten:** Textdateien in `training_data/` ablegen und Pfade in der Config-Datei unter `training_files` eintragen.

---

## ▶️ Benutzung

1.  **(Optional) Konfiguration anpassen:** Bearbeite die `config_*.json`-Datei.
2.  **Netzwerk trainieren/initialisieren:**
    ```bash
    python qllm_train_hybrid.py -c config_qllm.json -s qnp_state.json [-f]
    ```
    *(Passe Dateinamen und Optionen an. `-f` erzwingt Neubau.)*
3.  **Interaktive UI starten:**
    ```bash
    streamlit run qnp_streamlit_ui.py --server.address 0.0.0.0 --server.port 787
    ```
    *(Passe Skriptnamen und Port an. Öffne die URL im Browser.)*
    *   In der UI: Zustand (`qnp_state.json`) laden, Prompts eingeben, Zustände (Limbus, Meta-Knoten) beobachten, Parameter temporär anpassen, Verbindungen inspizieren.

---

## ⚙️ Konfiguration (`config_qllm.json`)

Wichtige Parameter (siehe `DEFAULT_CONFIG` im Code für alle):

*   `training_files`, `semantic_nodes`, `chunk_size`, `chunk_overlap`
*   `connection_learning_rate`, `connection_decay_rate`
*   `use_quantum_nodes`, `default_num_qubits`, `simulation_n_shots`
*   `enable_rag`, `generator_model_name`, `generator_temperature` (Basiswert)
*   `enable_self_learning`, `self_learning_file_path`
*   **`limbus_...` Parameter:** Steuern den Limbus-Knoten und seinen Einfluss auf Retrieval, Temperatur, Lernrate, Quanten-Effekte.
*   **`meta_nodes_enabled`**: Globaler Schalter für Meta-Knoten.
*   **`creativus_num_qubits`**, **`cortex_criticus_num_qubits`**: Qubit-Anzahl.
*   **`creativus_influence_...`**, **`criticus_influence_...`**, **`metacognitio_influence_...`**: Steuern den Einfluss der Meta-Knoten auf Temperatur, Lernrate, RAG-Bias und Prompt-Kontext.

---

## 💾 Zustandsdatei (`qnp_state.json`)

Speichert den **gesamten lernbaren Zustand** des Systems, einschließlich:
*   Alle Knoten (semantisch, Limbus, Meta) mit ihren Attributen und Quantenparametern.
*   Alle gelernten Verbindungen und Gewichte.
*   Alle verarbeiteten Text-Chunks und ihre Knoten-Assoziationen.
*   Zustandsinformationen der Modulatoren (z.B. `emotion_state`, `last_total_jumps`).
*   Die zur Laufzeit gültige Konfiguration.
*   Metadaten (verarbeitete Quellen, TF-IDF-Mapping).

---

## 🔭 Zukünftige Ideen

*   Verfeinerung der Meta-Knoten-Logik und ihrer Interaktion (z.B. gegenseitige Beeinflussung).
*   Implementierung komplexerer RAG-Biases (z.B. echter Novelty-Bias).
*   Echtzeit-Anpassung der Modulationsparameter basierend auf Dialogverlauf.
*   Evaluierungsmetriken für Kohärenz, Persona-Stabilität und Antwortqualität.
*   Untersuchung emergenter Langzeitdynamiken und "Persönlichkeitsentwicklung".
*   Anbindung an echte Quantenprozessoren für Teile der Simulation.

---

## 🤝 Mitwirken


Beiträge sind willkommen! Bitte erstelle einen Issue, um Bugs zu melden oder neue Features zu diskutieren. Pull Requests sind ebenfalls gerne gesehen.

---

## 📜 Lizenz

[MIT Lizenz](LICENSE).
