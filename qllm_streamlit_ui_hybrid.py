# -- coding: utf-8 --

# Filename: qnp_streamlit_ui.py # Neuer Dateiname zur Verdeutlichung
# Description: Interaktives Interface für Quantum Neuro-Persona Hybrid LLM (RAG)
#              mit Self-Learning, Limbus- und Meta-Knoten-Modulation.
# Version: 1.4 - Integrated Meta Node Display & Parameter Tuning
# Author: [CipherCore Technology] & Gemini & Your Input & History Maker

import streamlit as st
import json
import os
import sys
import pandas as pd
import traceback
import time
import numpy as np
from collections import defaultdict, deque, Counter # Counter hinzugefügt
from typing import List, Optional, Dict, Any

# --- WICHTIG: Import aus der NEUEN Datei ---
try:
    from quantum_neuropersona_hybrid_llm import (
        QuantumEnhancedTextProcessor,
        TextChunk,
        Node, # Basisklasse
        Connection,
        QuantumNodeSystem,
        LimbusAffektus,
        CreativusNode,        # NEU
        CortexCriticusNode,   # NEU
        MetaCognitioNode      # NEU
    )
    from quantum_neuropersona_hybrid_llm import GEMINI_AVAILABLE
except ImportError as e:
    st.error(
        f"""
    **FEHLER: Konnte 'QuantumEnhancedTextProcessor' oder abhängige Klassen nicht importieren.**

    Fehlerdetails: {e}

    Stellen Sie sicher, dass:
    1. Die Datei `quantum_neuropersona_hybrid_llm.py` (Version 1.3+) existiert und alle Klassen enthält.
    2. Sie sich im selben Verzeichnis wie dieses Skript befindet ODER im Python-Pfad liegt.
        """
    )
    st.stop()
except Exception as import_err:
     st.error(f"Anderer Importfehler: {import_err}")
     st.stop()


# === Hilfsfunktion für Verbindungsanzeige ===
def show_connections_table(connections_data: List[Dict[str, Any]]) -> None:
    """Zeigt Verbindungsdaten als Tabelle an."""
    if connections_data:
        df_connections = pd.DataFrame(connections_data)
        if "Gewicht" in df_connections.columns:
             df_connections["Gewicht"] = df_connections["Gewicht"].round(4)
        st.dataframe(df_connections, use_container_width=True, hide_index=True)
    else:
        st.info("Keine Verbindungen mit den aktuellen Filterkriterien gefunden.")

# === Zustandslade-Funktion ===
def load_processor_state(state_path: str) -> Optional[QuantumEnhancedTextProcessor]:
    """Lädt den Zustand des QuantumEnhancedTextProcessor."""
    if not os.path.exists(state_path):
        st.error(f"❌ Zustandsdatei nicht gefunden: `{state_path}`.")
        return None
    try:
        # Nutzt die load_state Methode der (neuen) Klasse
        processor = QuantumEnhancedTextProcessor.load_state(state_path)
        if processor:
            return processor
        else:
            st.error(f"❌ Fehler beim Laden des Zustands aus `{state_path}` (Methode gab None zurück).")
            return None
    except Exception as e:
        st.error(f"❌ Unerwarteter Fehler beim Laden des Zustands: {e}")
        st.error("Traceback:")
        st.code(traceback.format_exc())
        return None

# === Streamlit GUI Setup ===
st.set_page_config(page_title="QNP RAG Explorer", layout="wide", initial_sidebar_state="expanded") # Titel angepasst

# --- Header ---
col1_title, col2_title = st.columns([1, 9])
with col1_title:
     st.markdown("🧬", unsafe_allow_html=True) # Anderes Emoji
with col2_title:
     st.title("Quantum Neuro-Persona RAG Explorer") # Titel angepasst
     st.caption("Hybrides Interface mit Quanten-Retrieval, LLM, Lernzyklus, Limbus & Meta-Modulation") # Caption angepasst

# --- Session State Initialization ---
if 'processor' not in st.session_state: st.session_state['processor'] = None
# --- NEUER Default State-Dateiname ---
if 'state_file_path' not in st.session_state: st.session_state['state_file_path'] = "qnp_state.json"
if 'last_retrieved_chunks' not in st.session_state: st.session_state['last_retrieved_chunks'] = []
if 'last_generated_response' not in st.session_state: st.session_state['last_generated_response'] = None
if 'last_prompt' not in st.session_state: st.session_state['last_prompt'] = ""

# --- Initial State Load if not already loaded ---
if st.session_state.processor is None:
    with st.spinner(f"Lade initialen Zustand von `{st.session_state.state_file_path}`..."):
        st.session_state.processor = load_processor_state(st.session_state.state_file_path)
        # Fallback, wenn Laden fehlschlägt und Config existiert -> Neu initialisieren? (Optional)
        # if st.session_state.processor is None and os.path.exists("config_qllm.json"):
        #     st.warning("Konnte Zustand nicht laden, versuche Neuinitialisierung mit config_qllm.json...")
        #     try:
        #         st.session_state.processor = QuantumEnhancedTextProcessor(config_path="config_qllm.json")
        #         st.success("Neuinitialisierung erfolgreich (keine Daten geladen).")
        #     except Exception as init_err:
        #         st.error(f"Neuinitialisierung fehlgeschlagen: {init_err}")

# === Sidebar: Control & Info ===
with st.sidebar:
    st.header("⚙️ Steuerung & Status")

    # --- State Load/Save ---
    current_state_path = st.text_input(
        "Pfad zur Zustandsdatei",
        value=st.session_state['state_file_path'],
        key="state_path_input",
        help="JSON-Datei mit dem Netzwerkzustand (Nodes, Chunks, Config)."
    )
    col1_load, col2_save = st.columns(2)
    with col1_load:
        if st.button("🔄 Laden", key="load_state_button", help="Lädt den Zustand aus der Datei."):
            with st.spinner(f"Lade Zustand aus `{current_state_path}`..."):
                # Reset state before loading
                st.session_state['processor'] = None
                st.session_state['last_retrieved_chunks'] = []
                st.session_state['last_generated_response'] = None
                processor_instance = load_processor_state(current_state_path)
                if processor_instance:
                    st.session_state['processor'] = processor_instance
                    st.session_state['state_file_path'] = current_state_path
                    st.success("Zustand geladen.")
                    st.rerun()
                # Fehlermeldung wird in load_processor_state angezeigt

    processor = st.session_state.get('processor') # Holen des aktuellen Prozessors

    with col2_save:
        save_disabled = processor is None
        if st.button("💾 Speichern", key="save_state_button", disabled=save_disabled, help="Speichert den aktuellen Netzwerkzustand."):
            if processor:
                with st.spinner("Speichere Zustand..."):
                    try:
                         processor.save_state(st.session_state['state_file_path'])
                         st.success("Zustand gespeichert.")
                    except Exception as save_err:
                         st.error(f"Fehler beim Speichern: {save_err}")
            else: st.warning("Kein Prozessor zum Speichern geladen.")

    st.markdown("---")

    # --- Network Status & Simulation ---
    if processor is not None:
        st.subheader("📊 Netzwerk Übersicht")
        try:
            summary = processor.get_network_state_summary()
            col1_info, col2_info, col3_info = st.columns(3)
            with col1_info: st.metric("Knoten", summary.get('num_nodes', 0))
            with col2_info: st.metric("Verbindungen", summary.get('total_connections', 0))
            with col3_info: st.metric("Chunks", summary.get('num_chunks', 0))

            # Zeige Knotentypen
            node_types_dict = summary.get('node_types', {})
            if node_types_dict:
                node_types_str = ", ".join([f"{k}: {v}" for k, v in node_types_dict.items()])
                st.caption(f"Typen: {node_types_str}")

            with st.expander("Netzwerk Details (JSON)", expanded=False):
                st.json(summary) # summary enthält jetzt mehr Infos

            # Status indicators
            if not getattr(processor, 'rag_enabled', False): st.warning("RAG (Gemini) ist deaktiviert.", icon="⚠️")
            if getattr(processor, 'self_learning_enabled', False): st.success("Self-Learning ist aktiviert.", icon="🎓")
            else: st.info("Self-Learning ist deaktiviert.", icon="❌")
            # --- NEU: Meta-Knoten Status ---
            if getattr(processor, 'meta_nodes_enabled', False) and any(k in processor.nodes for k in ['Creativus', 'Cortex Criticus', 'MetaCognitio']):
                st.success("Meta-Knoten sind aktiviert.", icon="🧠")
            else:
                st.info("Meta-Knoten sind deaktiviert.", icon="❌")

            # Simulation Button
            if st.button("➡️ Schritt simulieren", key="simulate_button", help="Führt einen Simulationsschritt durch (Aktivierung, Decay, Emotion, Meta)."):
                with st.spinner("Simuliere Netzwerk..."):
                    try:
                         processor.simulate_network_step(decay_connections=True)
                         st.success("Simulationsschritt abgeschlossen.")
                    except Exception as sim_err: st.error(f"Simulationsfehler: {sim_err}")
                st.rerun()

        except Exception as e: st.error(f"Fehler beim Abrufen der Netzwerkinfo: {e}")

        # --- Parameters & Metrics ---
        st.markdown("---")
        st.subheader("🔧 Parameter & Metriken")

        # --- Parameter Tuning Expander ---
        with st.expander("Konfiguration anpassen (Temporär)", expanded=False):
            st.caption("Änderungen hier sind nur für die aktuelle Sitzung wirksam. '💾 Speichern' macht sie persistent.")

            # -- Allgemeine Parameter --
            st.markdown("**Allgemein**")
            n_shots_val = processor.config.get("simulation_n_shots", 50)
            n_shots_new = st.slider("Quanten-Messungen (n_shots)", 1, 200, n_shots_val, 1, key="n_shots_slider_sidebar", help="Stabilität der Quantenknoten-Aktivierung.")
            if n_shots_new != n_shots_val: processor.config["simulation_n_shots"] = n_shots_new

            # -- Limbus Modulation --
            st.markdown("**Limbus Modulation**")
            # (Slider für Limbus bleiben wie gehabt, evtl. leicht umstrukturieren)
            st.caption("Retrieval")
            col_t1, col_t2 = st.columns(2)
            with col_t1:
                thr_a_val = processor.config.get("limbus_influence_threshold_arousal", -0.03)
                thr_a_new = st.slider("Thr(Arousal)", -0.1, 0.1, thr_a_val, 0.005, "%.3f", key="thr_a_slider", help="Einfluss Arousal auf Retrieval-Threshold")
                if thr_a_new != thr_a_val: processor.config["limbus_influence_threshold_arousal"] = thr_a_new
            with col_t2:
                thr_p_val = processor.config.get("limbus_influence_threshold_pleasure", 0.03)
                thr_p_new = st.slider("Thr(Pleasure)", -0.1, 0.1, thr_p_val, 0.005, "%.3f", key="thr_p_slider", help="Einfluss Pleasure auf Retrieval-Threshold")
                if thr_p_new != thr_p_val: processor.config["limbus_influence_threshold_pleasure"] = thr_p_new
            rank_p_val = processor.config.get("limbus_influence_ranking_bias_pleasure", 0.02)
            rank_p_new = st.slider("Rank Bias(Pleasure)", -0.1, 0.1, rank_p_val, 0.005, "%.3f", key="rank_bias_p_slider", help="Einfluss Pleasure auf Chunk-Ranking")
            if rank_p_new != rank_p_val: processor.config["limbus_influence_ranking_bias_pleasure"] = rank_p_new

            st.caption("LLM Temperatur")
            col_temp1, col_temp2 = st.columns(2)
            with col_temp1:
                temp_a_val = processor.config.get("limbus_influence_temperature_arousal", 0.1)
                temp_a_new = st.slider("Temp(Arousal)", -0.5, 0.5, temp_a_val, 0.01, "%.2f", key="temp_a_slider", help="Einfluss Arousal auf LLM-Temperatur")
                if temp_a_new != temp_a_val: processor.config["limbus_influence_temperature_arousal"] = temp_a_new
            with col_temp2:
                temp_d_val = processor.config.get("limbus_influence_temperature_dominance", -0.1)
                temp_d_new = st.slider("Temp(Dominance)", -0.5, 0.5, temp_d_val, 0.01, "%.2f", key="temp_d_slider", help="Einfluss Dominance auf LLM-Temperatur")
                if temp_d_new != temp_d_val: processor.config["limbus_influence_temperature_dominance"] = temp_d_new

            st.caption("Lernrate")
            lr_limbus_val = processor.config.get("limbus_influence_learning_rate_multiplier", 0.1)
            lr_limbus_new = st.slider("LR-Mult Limbus(P+A)/2", -0.5, 0.5, lr_limbus_val, 0.01, "%.2f", key="lr_mult_limbus_slider", help="Einfluss Limbus auf Lernraten-Multiplikator")
            if lr_limbus_new != lr_limbus_val: processor.config["limbus_influence_learning_rate_multiplier"] = lr_limbus_new

            st.caption("Quanten Effekte (Retrieval)")
            col_q1, col_q2 = st.columns(2)
            with col_q1:
                var_pen_val = processor.config.get("limbus_influence_variance_penalty", 0.1)
                var_pen_new = st.slider("VarPenalty Limbus(A-P)/2", -0.5, 0.5, var_pen_val, 0.01, "%.2f", key="var_penalty_slider", help="Einfluss Limbus auf Varianz-Malus")
                if var_pen_new != var_pen_val: processor.config["limbus_influence_variance_penalty"] = var_pen_new
            with col_q2:
                act_boost_val = processor.config.get("limbus_influence_activation_boost", 0.05)
                act_boost_new = st.slider("ActBoost Limbus(P-A)/2", -0.5, 0.5, act_boost_val, 0.01, "%.2f", key="act_boost_slider", help="Einfluss Limbus auf Aktivierungs-Bonus")
                if act_boost_new != act_boost_val: processor.config["limbus_influence_activation_boost"] = act_boost_new

            # --- NEU: Meta-Knoten Modulation ---
            st.markdown("**Meta-Knoten Modulation**")
            meta_enabled_val = processor.config.get("meta_nodes_enabled", True)
            meta_enabled_new = st.toggle("Meta-Knoten aktiviert", value=meta_enabled_val, key="meta_enabled_toggle")
            if meta_enabled_new != meta_enabled_val: processor.config["meta_nodes_enabled"] = meta_enabled_new

            if meta_enabled_new: # Zeige Slider nur wenn aktiviert
                st.caption("Temperatur Einfluss")
                col_meta_t1, col_meta_t2 = st.columns(2)
                with col_meta_t1:
                    temp_creat_val = processor.config.get("creativus_influence_temperature", 0.15)
                    temp_creat_new = st.slider("Temp(Creativus)", -0.5, 0.5, temp_creat_val, 0.01, "%.2f", key="temp_creat_slider", help="Einfluss Creativus auf LLM-Temperatur")
                    if temp_creat_new != temp_creat_val: processor.config["creativus_influence_temperature"] = temp_creat_new
                with col_meta_t2:
                    temp_crit_val = processor.config.get("criticus_influence_temperature", -0.15)
                    temp_crit_new = st.slider("Temp(Criticus)", -0.5, 0.5, temp_crit_val, 0.01, "%.2f", key="temp_crit_slider", help="Einfluss CortexCriticus auf LLM-Temperatur")
                    if temp_crit_new != temp_crit_val: processor.config["criticus_influence_temperature"] = temp_crit_new

                st.caption("Lernrate Einfluss")
                col_meta_lr1, col_meta_lr2 = st.columns(2)
                with col_meta_lr1:
                    lr_creat_val = processor.config.get("creativus_influence_learning_rate", 0.1)
                    lr_creat_new = st.slider("LR-Mult(Creativus)", -0.5, 0.5, lr_creat_val, 0.01, "%.2f", key="lr_creat_slider", help="Einfluss Creativus auf Lernrate")
                    if lr_creat_new != lr_creat_val: processor.config["creativus_influence_learning_rate"] = lr_creat_new
                with col_meta_lr2:
                    lr_crit_val = processor.config.get("criticus_influence_learning_rate", -0.1)
                    lr_crit_new = st.slider("LR-Mult(Criticus)", -0.5, 0.5, lr_crit_val, 0.01, "%.2f", key="lr_crit_slider", help="Einfluss CortexCriticus auf Lernrate")
                    if lr_crit_new != lr_crit_val: processor.config["criticus_influence_learning_rate"] = lr_crit_new

                st.caption("Retrieval Bias Einfluss")
                # consistency_val = processor.config.get("criticus_influence_rag_consistency_bias", 0.03)
                # consistency_new = st.slider("Rank Bias(Consistency)", -0.1, 0.1, consistency_val, 0.005, "%.3f", key="rank_bias_consistency_slider", help="Einfluss Criticus auf Chunk-Ranking (Stabilität)")
                # if consistency_new != consistency_val: processor.config["criticus_influence_rag_consistency_bias"] = consistency_new
                # Novelty Bias ist komplexer, vorerst kein Slider

                st.caption("Prompt Einfluss")
                mc_prompt_val = processor.config.get("metacognitio_influence_prompt_level", 1.0)
                mc_prompt_new = st.slider("Prompt(MetaCognitio)", 0.0, 1.0, mc_prompt_val, 0.05, "%.2f", key="mc_prompt_slider", help="Stärke des MetaCognitio-Einflusses auf Prompt-Kontext")
                if mc_prompt_new != mc_prompt_val: processor.config["metacognitio_influence_prompt_level"] = mc_prompt_new


        # --- Current Metrics Display ---
        st.markdown("---")
        st.subheader("📈 Aktuelle Metriken")

        # 1. Durchschnittliche Aktivierung (alle Knoten)
        all_activations = [getattr(n, 'activation', 0.0) for n in processor.nodes.values()]
        valid_all_activations = [a for a in all_activations if isinstance(a, (float, np.number)) and np.isfinite(a)]
        if valid_all_activations:
            avg_all_act = sum(valid_all_activations) / len(valid_all_activations)
            st.metric("Ø Aktivierung (Alle)", f"{avg_all_act:.4f}")
        else: st.info("Keine gültigen Aktivierungsdaten.")

        # 2. Limbus Affektus Zustand
        limbus_node = processor.nodes.get("Limbus Affektus")
        if isinstance(limbus_node, LimbusAffektus):
             st.caption("Globaler emotionaler Zustand (PAD)")
             emotion_state = getattr(limbus_node, 'emotion_state', {})
             if emotion_state:
                  col_p, col_a, col_d = st.columns(3)
                  with col_p: st.metric("Pleasure", f"{emotion_state.get('pleasure', 0.0):.3f}", delta=None)
                  with col_a: st.metric("Arousal", f"{emotion_state.get('arousal', 0.0):.3f}", delta=None)
                  with col_d: st.metric("Dominance", f"{emotion_state.get('dominance', 0.0):.3f}", delta=None)
             else: st.info("Emotionszustand nicht verfügbar.")

        # --- NEU: Meta-Knoten Zustand ---
        if getattr(processor, 'meta_nodes_enabled', False):
            st.caption("Meta-Modulatoren Zustand")
            meta_nodes_data = {}
            for label in ["Creativus", "Cortex Criticus", "MetaCognitio"]:
                node = processor.nodes.get(label)
                if node:
                    meta_nodes_data[label] = {
                        "Activation": getattr(node, 'activation', 0.0),
                        "Type": type(node).__name__
                    }
                    if isinstance(node, MetaCognitioNode):
                        meta_nodes_data[label]["Jumps"] = getattr(node, 'last_total_jumps', 'N/A')

            if meta_nodes_data:
                # Kompakte Darstellung mit Metriken
                cols_meta = st.columns(len(meta_nodes_data))
                i = 0
                for label, data in meta_nodes_data.items():
                    with cols_meta[i]:
                        help_text = f"Typ: {data.get('Type', 'N/A')}"
                        if "Jumps" in data: help_text += f", Letzte Sprünge: {data['Jumps']}"
                        st.metric(label=label, value=f"{data.get('Activation', 0.0):.3f}", delta=None, help=help_text)
                    i += 1
            else:
                st.info("Meta-Knoten aktiviert, aber nicht gefunden.")

        # Optional: Expander für detaillierte Knotenliste
        with st.expander("Alle Knoten Zustände", expanded=False):
            node_states = []
            for node in processor.nodes.values():
                try:
                    node_states.append(node.get_state_representation())
                except Exception as repr_err:
                    node_states.append({"label": node.label, "error": str(repr_err)})
            if node_states:
                 # Zeige als Tabelle für bessere Übersicht
                 try:
                     df_nodes = pd.DataFrame(node_states)
                     # Relevante Spalten auswählen und anordnen
                     cols_order = ["label", "type", "activation", "smoothed_activation", "is_quantum", "num_qubits", "num_connections"]
                     # Füge optionale Spalten hinzu, falls vorhanden
                     if "emotion_state" in df_nodes.columns: cols_order.append("emotion_state")
                     if "last_total_jumps_detected" in df_nodes.columns: cols_order.append("last_total_jumps_detected")
                     # Filtere nur vorhandene Spalten
                     final_cols = [c for c in cols_order if c in df_nodes.columns]
                     st.dataframe(df_nodes[final_cols], hide_index=True, use_container_width=True)
                 except Exception as df_err:
                     st.error(f"Fehler bei Anzeige der Knoten-Tabelle: {df_err}")
                     st.json(node_states) # Fallback zu JSON
            else: st.info("Keine Knotenzustände verfügbar.")

    else: # Kein Prozessor geladen
        st.info("ℹ️ Kein Prozessor-Zustand geladen.")
        st.warning("Bitte laden Sie eine Zustandsdatei (z.B. `qnp_state.json`) oder führen Sie das Trainingsskript aus.")

# === Main Area: Prompt & Results ===
processor = st.session_state.get('processor') # Sicherstellen, dass der aktuelle Prozessor verwendet wird
if processor is not None:
    st.header("💬 Prompt & Antwort")
    prompt = st.text_area(
        "Geben Sie einen Prompt oder eine Frage ein:",
        height=100,
        key="prompt_input_main",
        value=st.session_state.get("last_prompt", ""),
        placeholder="Was möchtest du wissen oder diskutieren?",
        help="Ihre Frage oder Ihr Thema für das RAG-System."
    )

    # Generate Button
    generate_disabled = not processor.rag_enabled or not prompt.strip()
    if st.button("🚀 Antwort generieren", key="generate_button", disabled=generate_disabled, type="primary"):
        st.session_state['last_prompt'] = prompt
        st.session_state['last_retrieved_chunks'] = []
        st.session_state['last_generated_response'] = None
        st.rerun()

    # --- Generation Logic ---
    if st.session_state.last_prompt and st.session_state.last_generated_response is None:
        if processor.rag_enabled and st.session_state.last_prompt.strip():
            start_process_time = time.time()
            success_flag = False
            with st.spinner("🧠 Generiere Antwort (Retrieval + LLM)..."):
                try:
                    current_prompt = st.session_state.last_prompt
                    # generate_response nutzt jetzt intern Limbus UND Meta-Knoten
                    generated_response = processor.generate_response(current_prompt)
                    st.session_state['last_generated_response'] = generated_response
                    # respond_to_prompt (für Kontextanzeige) nutzt auch Modulation
                    st.session_state['last_retrieved_chunks'] = processor.respond_to_prompt(current_prompt)

                    is_valid = (generated_response and not generated_response.startswith("[Fehler") and not generated_response.startswith("[Antwort blockiert"))
                    if is_valid: success_flag = True
                    st.success(f"Antwort generiert in {time.time() - start_process_time:.2f}s")

                    if processor.self_learning_enabled and success_flag:
                         # Speichern passiert jetzt in _save_and_reprocess_response
                         # processor.save_state(st.session_state['state_file_path']) # Wird indirekt aufgerufen
                         pass # Kein explizites Speichern mehr hier nötig

                except Exception as e:
                    st.error(f"Fehler bei der Verarbeitung des Prompts: {e}")
                    st.error("Traceback:"); st.code(traceback.format_exc())
                    st.session_state['last_generated_response'] = "[Fehler bei der Generierung]"
            st.rerun()

        elif not processor.rag_enabled:
             st.error("Textgenerierung (RAG) ist nicht aktiviert.")

    # --- Results Display ---
    if st.session_state.get('last_generated_response'):
         st.markdown("---"); st.subheader("💡 Generierte Antwort")
         st.markdown(st.session_state['last_generated_response'])

    retrieved_chunks = st.session_state.get('last_retrieved_chunks', [])
    if retrieved_chunks:
         st.markdown("---")
         with st.expander(f"📚 Kontext ({len(retrieved_chunks)} Textabschnitt{'e' if len(retrieved_chunks) != 1 else ''})", expanded=False):
              for i, chunk in enumerate(retrieved_chunks):
                  st.markdown(f"**[{i+1}] Quelle:** `{chunk.source}` (`Index: {chunk.index}`)")
                  if hasattr(chunk, 'activated_node_labels') and chunk.activated_node_labels:
                      nodes_str = ", ".join(f"`{lbl}`" for lbl in chunk.activated_node_labels)
                      st.markdown(f"**Assoz. Knoten:** {nodes_str}") # Angepasst
                  # Optional: Zeige Score des Chunks (wenn verfügbar - müssten wir in respond_to_prompt zurückgeben)
                  # score = getattr(chunk, 'retrieval_score', None)
                  # if score is not None: st.markdown(f"**Score:** {score:.4f}")
                  st.markdown(f"> _{chunk.text[:300]}..._")
                  cols_btn = st.columns([1, 5])
                  with cols_btn[0]:
                        if st.button(f"Volltext {i+1}", key=f"chunk_text_{i}"):
                             st.markdown(f"**Volltext {i+1}:**\n{chunk.text}")
                  if i < len(retrieved_chunks) - 1: st.markdown("---")

    # --- Network Connection Display ---
    st.markdown("---")
    if st.checkbox("🕸️ Zeige Netzwerkverbindungen", key="show_connections_main", value=False):
        st.subheader("Gelernte Verbindungen (Top 50)")
        processor_ui_main = st.session_state.get('processor')
        if processor_ui_main and hasattr(processor_ui_main, 'nodes'):
            min_weight_thr = st.slider("Mindestgewicht", 0.0, 1.0, 0.1, 0.01, key="weight_slider_main")
            connections_data = []
            node_uuid_map = {n.uuid: n for n in processor_ui_main.nodes.values()}
            for node in processor_ui_main.nodes.values():
                if isinstance(getattr(node, 'connections', None), dict):
                    for conn_uuid, conn in node.connections.items():
                        if conn is None: continue
                        target_node = node_uuid_map.get(getattr(conn, 'target_node_uuid', None))
                        weight = getattr(conn, 'weight', None)
                        if (target_node and isinstance(weight, (float, np.number)) and np.isfinite(weight) and weight >= min_weight_thr):
                            connections_data.append({"Quelle": node.label, "Ziel": target_node.label, "Gewicht": weight, "Typ": getattr(conn, 'conn_type', 'N/A')})
            connections_data.sort(key=lambda x: x["Gewicht"], reverse=True)
            show_connections_table(connections_data[:50])
            if len(connections_data) > 50: st.caption(f"Zeige Top 50 von {len(connections_data)} Verbindungen ≥ {min_weight_thr:.2f}.")
        else: st.warning("Prozessor oder Knoten nicht verfügbar.")

else: # Kein Prozessor im Session State
    st.info("ℹ️ Bitte laden Sie einen Prozessor-Zustand über die Seitenleiste, um zu beginnen.")

# --- Footer ---
st.sidebar.markdown("---")
st.sidebar.caption(f"QNP Interface v1.4") # Version angepasst