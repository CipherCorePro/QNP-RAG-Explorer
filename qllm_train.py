# -- coding: utf-8 --

# Filename: qllm_train_neuropersona.py # Angepasster Name im Kommentar
# Version: 1.3 - Adapted for Meta Nodes
# Author: [CipherCore Technology] & Gemini & Your Input

import os
import sys
import time
import json
import argparse
import copy
import random
import traceback
from typing import Optional, List, Dict, Any
from collections import deque, Counter # Counter hinzugefügt für Node-Typ-Analyse
from datetime import datetime

# Required for quantum system parameters and other operations
import numpy as np

# --- WICHTIG: Import aus der NEUEN Datei ---
try:
    from quantum_neuropersona_hybrid_llm import (
        QuantumEnhancedTextProcessor, TextChunk, Node, Connection, QuantumNodeSystem,
        LimbusAffektus, CreativusNode, CortexCriticusNode, MetaCognitioNode # Importiere auch Meta-Knoten-Klassen
    )
    # Prüfe Gemini Verfügbarkeit
    from quantum_neuropersona_hybrid_llm import GEMINI_AVAILABLE
    try:
        from tqdm import tqdm
        TQDM_AVAILABLE = True
    except ImportError:
        TQDM_AVAILABLE = False
        def tqdm(iterable, *args, **kwargs): return iterable
except ImportError as e:
    print(f"FEHLER: Konnte 'QuantumEnhancedTextProcessor' oder abhängige Klassen nicht importieren: {e}")
    print("Stelle sicher, dass 'quantum_neuropersona_hybrid_llm.py' (Version 1.3+) im selben Verzeichnis oder im Python-Pfad liegt.")
    sys.exit(1)
except Exception as import_err:
     print(f"Anderer Importfehler: {import_err}")
     sys.exit(1)


def train_hybrid_model(config_path: str, state_path: str, force_rebuild: bool = False):
    """Main function to train/process data with the hybrid model over epochs."""
    print("="*50)
    print(" Starte Training/Datenverarbeitung für Quantum Neuro-Persona Hybrid LLM (v1.3)") # Angepasst
    print(f" - Konfiguration: {config_path}")
    print(f" - Zustandsdatei: {state_path}")
    print(f" - Neubau erzwingen: {force_rebuild}")
    print("="*50)

    start_time = time.time()

    # Load current configuration from file first
    config_from_file = None
    print(f"INFO: Lese aktuelle Konfiguration aus {config_path}...")
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config_from_file = json.load(f)
            # Wichtig: Stelle sicher, dass alle Default-Werte (auch die neuen) vorhanden sind
            # Dies geschieht jetzt in QuantumEnhancedTextProcessor.__init__, was besser ist.
    except Exception as e:
        print(f"FATALER FEHLER: Konnte Konfigurationsdatei '{config_path}' nicht laden: {e}")
        sys.exit(1)

    # 1. Load state or initialize new
    processor: Optional[QuantumEnhancedTextProcessor] = None
    if not force_rebuild and os.path.exists(state_path):
        print(f"\nVersuche, Zustand aus '{state_path}' zu laden...")
        # load_state verwendet jetzt die Klassen aus der neuen Datei
        processor = QuantumEnhancedTextProcessor.load_state(state_path)
        if processor:
            print("INFO: Zustand geladen. Versuche Konfiguration zu aktualisieren...")
            try:
                # Update config im Prozessor mit Werten aus der Datei
                # Fehlende Keys in der Datei werden nicht hinzugefügt, aber __init__ hat Defaults gesetzt
                processor.config.update(config_from_file)

                # Update processor attributes dependent on the potentially changed config
                processor.self_learning_enabled = processor.config.get("enable_self_learning", False)
                processor.learn_file_path = processor.config.get("self_learning_file_path", "./training_data/learn.txt")
                processor.learn_source_name = processor.config.get("self_learning_source_name", "Generated Responses")
                processor.rag_enabled = processor.config.get("enable_rag", False) and GEMINI_AVAILABLE
                processor.meta_nodes_enabled = processor.config.get("meta_nodes_enabled", True) # Update Meta-Node Status

                # Aktualisiere Limbus und Meta-Knoten Config-Referenz und Parameter (obwohl __init__ das auch tun sollte)
                # Dies stellt sicher, dass auch bei geladenem Zustand die aktuelle Config aktiv ist
                for node_label in ["Limbus Affektus", "Creativus", "Cortex Criticus", "MetaCognitio"]:
                    node = processor.nodes.get(node_label)
                    if node and hasattr(node, 'config'):
                        node.config = processor.config # Referenz aktualisieren
                        if isinstance(node, LimbusAffektus):
                           node._update_params_from_config() # Sicherstellen, dass Limbus-Parameter aktuell sind
                        # Meta-Knoten lesen ihre Params direkt aus config, kein explizites Update nötig

                print(" -> Konfiguration im geladenen Prozessor mit aktueller Datei aktualisiert.")
            except Exception as e:
                print(f"WARNUNG: Konnte Konfiguration im geladenen Prozessor nicht vollständig aktualisieren: {e}")
                traceback.print_exc(limit=1)
        else:
            print(f" -> Laden des Zustands aus '{state_path}' fehlgeschlagen oder Datei leer.")

    # If no processor was loaded, initialize new
    if processor is None:
        if force_rebuild: print(f"\nNeuerstellung erzwungen.")
        print(f"\nInitialisiere neues Modell mit Konfiguration aus '{config_path}'.")
        try:
            # Init mit aktueller Config (inkl. Defaults für neue Meta-Parameter)
            processor = QuantumEnhancedTextProcessor(config_dict=config_from_file)
        except Exception as e:
             print(f"\nFATALER FEHLER: Initialisierung fehlgeschlagen: {e}"); traceback.print_exc(limit=1); sys.exit(1)
        if not processor or not hasattr(processor, 'config'):
            print("\nFATALER FEHLER: Prozessor-Objekt konnte nicht korrekt initialisiert werden."); sys.exit(1)

    # Ensure a valid processor exists
    if processor is None: print("\nFATALER FEHLER: Konnte keinen Prozessor laden oder initialisieren."); sys.exit(1)

    # --- Final checks before training ---
    print(f"\n--- Status vor dem Training ---")
    node_types = Counter(type(n).__name__ for n in processor.nodes.values())
    print(f" -> Knoten-Typen: {dict(node_types)}")
    print(f" -> Qubit-Anzahl (Default): {processor.config.get('default_num_qubits')}")
    print(f" -> Self-Learning: {'Aktiviert' if processor.self_learning_enabled else 'Deaktiviert'}")
    print(f" -> RAG Status: {'Aktiviert' if processor.rag_enabled else 'Deaktiviert'}")
    print(f" -> Limbus Modulation: {'Aktiviert' if 'Limbus Affektus' in processor.nodes else 'Deaktiviert'}")
    print(f" -> Meta-Knoten Modulation: {'AKTIVIERT' if processor.meta_nodes_enabled and any(k in processor.nodes for k in ['Creativus', 'Cortex Criticus', 'MetaCognitio']) else 'DEAKTIVIERT'}")
    print(f" -> Simulation Steps After Training: {processor.config.get('simulation_steps_after_training')}")
    print("-" * 28)


    # Get number of epochs from configuration
    num_epochs = processor.config.get("training_epochs", 1)
    print(f"\nAnzahl der Trainingsepochen: {num_epochs}")

    # 2. Process training files (chunking & initial processing)
    training_files = processor.config.get("training_files", [])
    if not training_files:
        print("\nWARNUNG: Keine Trainingsdateien in der Konfiguration gefunden ('training_files').")
    else:
        print(f"\n--- Schritt 1: Verarbeite/Aktualisiere Chunks aus {len(training_files)} Trainingsdatei(en) ---")
        files_processed_count = 0
        # Verwende eine Kopie der Liste, falls sie während der Iteration geändert wird (sollte nicht passieren)
        file_iterator = tqdm(list(training_files), desc="Verarbeite Dateien", leave=False) if TQDM_AVAILABLE else list(training_files)
        for file_path in file_iterator:
            if os.path.exists(file_path):
                 try:
                     # load_and_process_file ruft intern process_chunk auf,
                     # welches jetzt die Meta-Knoten für die LR-Modulation nutzt
                     processor.load_and_process_file(file_path)
                     files_processed_count += 1
                 except Exception as load_err:
                      print(f"FEHLER beim Verarbeiten von Datei '{file_path}': {load_err}")
                      traceback.print_exc(limit=1)
            else:
                 print(f"WARNUNG: Trainingsdatei '{file_path}' nicht gefunden. Übersprungen.")

        if files_processed_count > 0:
            print(f" -> {files_processed_count} vorhandene Trainingsdateien verarbeitet/aktualisiert.")
            processor.update_tfidf_index() # Wichtig nach dem Hinzufügen von Chunks
        else:
            print(" -> Keine vorhandenen Trainingsdateien gefunden oder verarbeitet.")

    # 3. Run training epochs (Stärkung der Verbindungen)
    if not processor.chunks:
        print("\nWARNUNG: Keine Chunks zum Trainieren vorhanden. Überspringe Epochen-Training.")
    else:
        all_chunk_ids = list(processor.chunks.keys())
        print(f"\n--- Schritt 2: Beginne Epochen-Training über {num_epochs} Epoche(n) für {len(all_chunk_ids)} Chunks ---")
        for epoch in range(1, num_epochs + 1):
            print(f"\n🚀 Epoche {epoch}/{num_epochs}")
            random.shuffle(all_chunk_ids)
            epoch_chunk_iterator = tqdm(all_chunk_ids, desc=f"Epoch {epoch}", leave=False) if TQDM_AVAILABLE else all_chunk_ids
            chunks_processed_in_epoch = 0
            # --- Netzwerk-Simulation VOR jeder Epoche? ---
            # Optional: Einen Simulationsschritt machen, um Limbus/Meta-Zustände zu aktualisieren,
            # bevor die Lernraten in process_chunk berechnet werden.
            # print(f"   -> Führe Simulationsschritt vor Epoche {epoch} durch...")
            # processor.simulate_network_step(decay_connections=False) # Ohne Decay hier

            for chunk_uuid in epoch_chunk_iterator:
                 if chunk_uuid in processor.chunks:
                      try:
                          # process_chunk nutzt jetzt Limbus UND Meta-Knoten Aktivierungen
                          # (die im vorherigen Schritt oder der letzten Iteration berechnet wurden)
                          # um die Lernrate zu modulieren.
                          processor.process_chunk(processor.chunks[chunk_uuid])
                          chunks_processed_in_epoch += 1
                      except Exception as process_err:
                           print(f"FEHLER beim Verarbeiten von Chunk UUID {chunk_uuid}: {process_err}")
                           traceback.print_exc(limit=1)

            print(f"   -> Epoche {epoch} abgeschlossen ({chunks_processed_in_epoch} Chunks verarbeitet).")
            # --- Netzwerk-Simulation NACH jeder Epoche? ---
            # Optional: Decay & Zustandsentwicklung nach jeder Epoche simulieren
            # print(f"   -> Führe Simulationsschritt nach Epoche {epoch} durch...")
            # processor.simulate_network_step(decay_connections=True) # Mit Decay hier


    # 4. (Optional) Run network simulation steps after ALL training epochs
    simulation_steps = processor.config.get("simulation_steps_after_training", 0)
    if simulation_steps > 0 and processor.nodes:
        print(f"\n--- Schritt 3: Führe {simulation_steps} Netzwerk-Simulationsschritte nach Training durch ---")
        sim_iterator = tqdm(range(simulation_steps), desc="Simulationsschritte", leave=False) if TQDM_AVAILABLE else range(simulation_steps)
        for i in sim_iterator:
             try:
                 # simulate_network_step enthält jetzt Meta-Knoten-Updates
                 processor.simulate_network_step(decay_connections=True)
             except Exception as sim_err:
                 print(f"FEHLER während Simulationsschritt {i+1}: {sim_err}")
                 traceback.print_exc(limit=1); break
        print("--- Simulation abgeschlossen ---")
    elif simulation_steps > 0:
         print("\nWARNUNG: Simulation übersprungen, da keine Knoten im Netzwerk vorhanden sind.")
    else:
         print("\nINFO: Keine Simulationsschritte nach dem Training konfiguriert.")

    # Calculate final activations without decay before saving/summary
    print("\n--- Berechne finale Knotenaktivierungen für Zustandsspeicherung/Summary ---")
    if processor.nodes:
        try:
            processor.simulate_network_step(decay_connections=False)
        except Exception as final_sim_err:
             print(f"FEHLER während finaler Aktivierungsberechnung: {final_sim_err}")
             traceback.print_exc(limit=1)
    else: print("   -> Übersprungen, keine Knoten vorhanden.")

    # Calculate Summary BEFORE Saving
    final_summary = {}
    print("\n--- Berechne finale Summary (aus In-Memory Zustand) ---")
    try:
        final_summary = processor.get_network_state_summary() # Enthält jetzt Meta-Knoten Infos
        print("   -> Summary erfolgreich berechnet.")
    except Exception as summary_err:
         print(f"FEHLER beim Erstellen der Netzwerk-Summary: {summary_err}")
         traceback.print_exc(limit=1)

    # Save state afterwards
    print(f"\n--- Schritt 4: Speichere finalen Zustand nach '{state_path}' ---")
    processor.save_state(state_path) # save_state ist angepasst

    # --- Abschluss ---
    end_time = time.time()
    print("\n--- Zusammenfassung des Laufs ---")
    print(f" - Gesamtdauer: {end_time - start_time:.2f} Sekunden")
    print(" - Finaler Netzwerkstatus (aus In-Memory, VOR dem Speichern berechnet):")
    if isinstance(final_summary, dict) and final_summary:
        print(json.dumps(final_summary, indent=2, ensure_ascii=False))
    else: print("   -> Konnte keine gültige Summary erstellen.")
    print("="*50)
    print(" Training/Datenverarbeitung beendet.")
    print("="*50)

# --- End train_hybrid_model ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trainiert das Quantum Neuro-Persona Hybrid LLM.") # Angepasst
    parser.add_argument(
        "-c", "--config",
        default="config_qllm.json", # Behalte Standard bei, kann überschrieben werden
        help="Pfad zur JSON-Konfigurationsdatei (default: config_qllm.json)"
    )
    parser.add_argument(
        "-s", "--state",
        default="qnp_state.json", # Neuer Default-State-Name
        help="Pfad zur Zustandsdatei für Speichern/Laden (default: qnp_state.json)" # Angepasst
    )
    parser.add_argument(
        "-f", "--force-rebuild",
        action="store_true",
        help="Ignoriert einen vorhandenen Zustand und baut das Modell neu auf."
    )

    args = parser.parse_args()

    if not os.path.exists(args.config):
        print(f"FEHLER: Konfigurationsdatei nicht gefunden: {args.config}")
        sys.exit(1)

    print(f"INFO: Verwende Konfigurationsdatei: {args.config}")
    print(f"INFO: Verwende Zustandsdatei: {args.state}")
    if args.force_rebuild:
        print("INFO: Neuerstellung des Zustands ist erzwungen (--force-rebuild).")

    try:
        train_hybrid_model(args.config, args.state, args.force_rebuild)
    except Exception as main_err:
        print(f"\nFATALER FEHLER im Haupt-Trainingsprozess: {main_err}")
        traceback.print_exc()
        sys.exit(1)