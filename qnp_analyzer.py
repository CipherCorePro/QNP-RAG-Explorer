# -- coding: utf-8 --

# Filename: qnp_simulation_analyzer.py
# Description: Lädt einen QNP-Zustand und simuliert N Schritte,
#              zeichnet Metriken auf und erstellt Verlaufsplots, Heatmaps und Korrelationen.
# Version: 1.1 - Added Heatmap, Correlation, Moving Avg, Derivatives plots
# Author: [CipherCore Technology] & Gemini & Your Input

import json
import argparse
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker # Für Formatierung der Korrelationsmatrix
import os
import numpy as np
import pandas as pd
import traceback
from datetime import datetime
from collections import Counter, defaultdict
from typing import Optional, Dict, Any, List, Tuple
import seaborn as sns  # Zusätzlicher Import für Heatmaps


# --- Importiere die notwendigen Klassen aus dem Hauptmodul ---
try:
    from quantum_neuropersona_hybrid_llm import (
        QuantumEnhancedTextProcessor,
        LimbusAffektus,
        CreativusNode,
        CortexCriticusNode,
        MetaCognitioNode
    )
    try:
        from tqdm import tqdm
        TQDM_AVAILABLE = True
    except ImportError:
        TQDM_AVAILABLE = False
        def tqdm(iterable, *args, **kwargs): return iterable

except ImportError as e:
    print(f"FEHLER: Konnte 'QuantumEnhancedTextProcessor' oder abhängige Klassen nicht importieren: {e}")
    print("Stelle sicher, dass 'quantum_neuropersona_hybrid_llm.py' (Version 1.3+) im selben Verzeichnis oder im Python-Pfad liegt.")
    import sys
    sys.exit(1)
except Exception as import_err:
     print(f"Anderer Importfehler: {import_err}")
     import sys
     sys.exit(1)

# --- Konstanten ---
ANALYSIS_OUTPUT_BASE_DIR = "qnp_simulation_analysis"

# --- Hilfsfunktionen (unverändert) ---

def load_processor_state(state_path: str) -> Optional[QuantumEnhancedTextProcessor]:
    """Lädt den Zustand des QuantumEnhancedTextProcessor aus einer Datei."""
    print(f"INFO: Lade Prozessorzustand von: {state_path}")
    if not os.path.exists(state_path):
        print(f"FEHLER: Zustandsdatei nicht gefunden: {state_path}")
        return None
    try:
        processor = QuantumEnhancedTextProcessor.load_state(state_path)
        if processor:
            print("INFO: Prozessorzustand erfolgreich geladen.")
            return processor
        else:
            print(f"FEHLER: Laden des Zustands aus '{state_path}' fehlgeschlagen (Methode gab None zurück).")
            return None
    except Exception as e:
        print(f"FEHLER: Unerwarteter Fehler beim Laden des Zustands: {e}")
        traceback.print_exc(limit=2)
        return None

def get_simulation_subdir(state_filename_base: str) -> str:
    """Erstellt einen eindeutigen Unterordner für die Simulationsanalyse."""
    zeitstempel = datetime.now().strftime("%Y%m%d_%H%M%S")
    subdir_name = f"sim_{state_filename_base}_{zeitstempel}"
    full_path = os.path.join(ANALYSIS_OUTPUT_BASE_DIR, subdir_name)
    os.makedirs(full_path, exist_ok=True)
    print(f"INFO: Ergebnisse werden gespeichert in: {full_path}")
    return full_path

def speichere_plot(fig, dateiname: str, analyse_subdir: str):
    """Speichert eine Matplotlib-Figur im Analyse-Unterordner."""
    try:
        pfad = os.path.join(analyse_subdir, dateiname)
        fig.savefig(pfad, bbox_inches='tight')
        plt.close(fig)
        print(f"  -> Plot gespeichert: {pfad}")
    except Exception as e:
        print(f"FEHLER beim Speichern des Plots {dateiname}: {e}")
        plt.close(fig)

# --- Metrik-Aufzeichnung (unverändert) ---

def record_metrics(processor: QuantumEnhancedTextProcessor, step: int) -> Dict[str, Any]:
    """Zeichnet relevante Metriken nach einem Simulationsschritt auf."""
    metrics = {"step": step}
    if not processor or not hasattr(processor, 'nodes'): return metrics
    all_activations = [getattr(n, 'activation', np.nan) for n in processor.nodes.values()]; valid_activations = [a for a in all_activations if isinstance(a, (float, int)) and np.isfinite(a)]; metrics["avg_activation_all"] = np.mean(valid_activations) if valid_activations else np.nan
    limbus = processor.nodes.get("Limbus Affektus")
    if isinstance(limbus, LimbusAffektus) and hasattr(limbus, 'emotion_state'): metrics["limbus_pleasure"] = limbus.emotion_state.get('pleasure', np.nan); metrics["limbus_arousal"] = limbus.emotion_state.get('arousal', np.nan); metrics["limbus_dominance"] = limbus.emotion_state.get('dominance', np.nan); metrics["limbus_activation"] = getattr(limbus, 'activation', np.nan)
    else: metrics["limbus_pleasure"] = metrics["limbus_arousal"] = metrics["limbus_dominance"] = metrics["limbus_activation"] = np.nan
    meta_nodes_enabled = getattr(processor, 'meta_nodes_enabled', False); metrics["meta_creativus_activation"] = np.nan; metrics["meta_criticus_activation"] = np.nan; metrics["meta_metacognitio_activation"] = np.nan; metrics["meta_metacognitio_jumps"] = np.nan
    if meta_nodes_enabled:
        creativus = processor.nodes.get("Creativus"); criticus = processor.nodes.get("Cortex Criticus"); metacognitio = processor.nodes.get("MetaCognitio")
        if isinstance(creativus, CreativusNode): metrics["meta_creativus_activation"] = getattr(creativus, 'activation', np.nan)
        if isinstance(criticus, CortexCriticusNode): metrics["meta_criticus_activation"] = getattr(criticus, 'activation', np.nan)
        if isinstance(metacognitio, MetaCognitioNode): metrics["meta_metacognitio_activation"] = getattr(metacognitio, 'activation', np.nan); metrics["meta_metacognitio_jumps"] = getattr(metacognitio, 'last_total_jumps', np.nan)
    total_q_variance = 0.0; total_q_jumps = 0; num_valid_q_nodes = 0
    for node in processor.nodes.values():
        if getattr(node, 'is_quantum', False) and hasattr(node, 'last_measurement_analysis'):
             analysis = getattr(node, 'last_measurement_analysis', {});
             if isinstance(analysis, dict) and analysis.get("error_count", 1) == 0:
                 variance = analysis.get("state_variance"); jump_detected = analysis.get("jump_detected")
                 if isinstance(variance, (float, int)) and np.isfinite(variance): total_q_variance += variance; num_valid_q_nodes += 1
                 if jump_detected: total_q_jumps += 1
    metrics["avg_q_variance"] = (total_q_variance / num_valid_q_nodes) if num_valid_q_nodes > 0 else np.nan; metrics["total_q_jumps"] = total_q_jumps
    return metrics

# --- Plotting der Zeitreihen & Matrizen ---

def plot_time_series(df: pd.DataFrame, columns: List[str], title: str, ylabel: str, dateiname: str, analyse_subdir: str, ylim: Optional[Tuple]=None):
    """Erstellt einen Linienplot für eine oder mehrere Spalten eines DataFrames über die Schritte."""
    plot_df = df[columns].dropna(axis=0, how='all') # Nur Zeilen behalten, wo mind. 1 Wert da ist
    if plot_df.empty or not any(col in plot_df.columns for col in columns):
        print(f"INFO: Keine gültigen Daten zum Plotten für '{title}'.")
        return

    fig, ax = plt.subplots(figsize=(12, 6))
    for col in columns:
        if col in plot_df.columns and plot_df[col].notna().any(): # Nur plotten wenn Daten vorhanden
            ax.plot(plot_df.index, plot_df[col], marker='.', linestyle='-', label=col) # df.index sind die Schritte

    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Simulationsschritt", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    if len(columns) > 1 or any('_' in col for col in columns): # Bessere Bedingung für Legende
        ax.legend(fontsize=10)
    if ylim:
        ax.set_ylim(ylim)
    ax.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    speichere_plot(fig, dateiname, analyse_subdir)


def plot_heatmap(df: pd.DataFrame, columns_to_plot: List[str], title: str, dateiname: str, analyse_subdir: str):
    """Erstellt eine Heatmap für ausgewählte Spalten eines DataFrames."""
    # Wähle nur Spalten aus, die existieren und numerisch sind
    valid_columns = [col for col in columns_to_plot if col in df.columns and pd.api.types.is_numeric_dtype(df[col])]
    if not valid_columns:
        print(f"INFO: Keine gültigen numerischen Spalten für Heatmap '{title}' gefunden.")
        return

    data = df[valid_columns].copy()
    # Optional: Normalisieren für bessere Farbverteilung, wenn Werte sehr unterschiedlich sind
    # from sklearn.preprocessing import MinMaxScaler
    # scaler = MinMaxScaler()
    # data_scaled = scaler.fit_transform(data.fillna(0)) # NaN mit 0 füllen für Skalierung
    # data = pd.DataFrame(data_scaled, index=data.index, columns=data.columns)

    # Schönere Labels für die Y-Achse
    data.columns = [col.replace('_', ' ').replace('activation', 'Act.').replace('limbus', 'L.').replace('meta ', 'M. ').replace('criticus', 'Crit.').replace('creativus', 'Creat.').replace('metacognitio', 'MetaCog.').replace('variance', 'Var.').replace('jumps', 'Jumps').replace('dominance', 'Dom.').replace('pleasure','Pleas.').replace('arousal', 'Arous.') for col in data.columns]


    fig, ax = plt.subplots(figsize=(14, 8)) # Angepasste Größe
    sns.heatmap(data.transpose(), cmap="coolwarm", cbar=True, annot=False, linewidths=0.5, linecolor='lightgrey', ax=ax) # Transponiert für Metriken auf Y-Achse
    ax.set_title(title, fontsize=16)
    ax.set_xlabel("Simulationsschritte", fontsize=12)
    ax.set_ylabel("Metriken", fontsize=12)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10, rotation=0) # Keine Rotation für bessere Lesbarkeit
    plt.tight_layout(rect=[0, 0, 1, 0.97]) # Platz für Titel
    speichere_plot(fig, dateiname, analyse_subdir)


def plot_correlation_matrix(df: pd.DataFrame, columns_to_correlate: List[str], title: str, dateiname: str, analyse_subdir: str):
    """Berechnet und plottet die Korrelationsmatrix für ausgewählte Spalten."""
    # Wähle nur Spalten aus, die existieren und numerisch sind
    valid_columns = [col for col in columns_to_correlate if col in df.columns and pd.api.types.is_numeric_dtype(df[col])]
    if len(valid_columns) < 2:
        print(f"INFO: Nicht genügend gültige numerische Spalten für Korrelationsmatrix '{title}'.")
        return

    data = df[valid_columns].copy()
    correlation_matrix = data.corr()

    # Schönere Labels
    axis_labels = [col.replace('_', ' ').replace('activation', 'Act.').replace('limbus', 'L.').replace('meta ', 'M. ').replace('criticus', 'Crit.').replace('creativus', 'Creat.').replace('metacognitio', 'MetaCog.').replace('variance', 'Var.').replace('jumps', 'Jumps').replace('dominance', 'Dom.').replace('pleasure','Pleas.').replace('arousal', 'Arous.') for col in correlation_matrix.columns]

    fig, ax = plt.subplots(figsize=(12, 10)) # Angepasste Größe
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f",
                linewidths=.5, linecolor='lightgrey', ax=ax,
                xticklabels=axis_labels, yticklabels=axis_labels, # Verwende angepasste Labels
                annot_kws={"size": 8}) # Kleinere Schrift für Annot.
    ax.set_title(title, fontsize=16)
    plt.xticks(rotation=90, fontsize=9) # Rotiere X-Labels
    plt.yticks(rotation=0, fontsize=9) # Y-Labels horizontal
    plt.tight_layout(rect=[0, 0.05, 1, 0.97]) # Mehr Platz unten/oben
    speichere_plot(fig, dateiname, analyse_subdir)


# --- Haupt-Simulations- und Analysefunktion ---
def run_simulation_and_analyze(state_path: str, num_steps: int, moving_avg_window: int = 5): # Moving Avg Fenstergröße hinzugefügt
    """Lädt den Zustand, führt Simulationen durch, zeichnet Metriken auf, inklusive Heatmaps und Korrelationen."""

    processor = load_processor_state(state_path)
    if not processor: return

    print(f"\nINFO: Starte Simulation für {num_steps} Schritte...")
    all_metrics = []

    # Schritt 0: Initialer Zustand
    print("Zeichne Metriken für Startzustand (Schritt 0) auf...")
    try: processor.simulate_network_step(decay_connections=False); initial_metrics = record_metrics(processor, 0); all_metrics.append(initial_metrics); print(" -> Metriken für Schritt 0 erfasst.")
    except Exception as e: print(f"FEHLER beim Erfassen der initialen Metriken: {e}"); traceback.print_exc(limit=1)

    # Simulationsschleife
    step_iterator = range(1, num_steps + 1)
    if TQDM_AVAILABLE: step_iterator = tqdm(step_iterator, desc="Simulationsschritte")
    for step in step_iterator:
        try:
            processor.simulate_network_step(decay_connections=True); step_metrics = record_metrics(processor, step); all_metrics.append(step_metrics)
            if TQDM_AVAILABLE:
                pad_str = f"P:{step_metrics.get('limbus_pleasure', 0):.2f} A:{step_metrics.get('limbus_arousal', 0):.2f} D:{step_metrics.get('limbus_dominance', 0):.2f}"
                meta_str = f"Cr:{step_metrics.get('meta_creativus_activation', 0):.2f} Ct:{step_metrics.get('meta_criticus_activation', 0):.2f} M:{step_metrics.get('meta_metacognitio_activation', 0):.2f}"
                step_iterator.set_postfix_str(f"{pad_str} | {meta_str}")
        except Exception as e: print(f"\nFEHLER in Simulationsschritt {step}: {e}"); traceback.print_exc(limit=1); print("Breche Simulation ab."); break

    print(f"\nINFO: Simulation abgeschlossen nach {len(all_metrics)} erfassten Schritten.")
    if len(all_metrics) < 2: print("FEHLER: Nicht genügend Metriken für Analyse aufgezeichnet."); return

    # --- Ergebnisse verarbeiten ---
    print("INFO: Bereite Metrikdaten auf...")
    metrics_df = pd.DataFrame(all_metrics); metrics_df = metrics_df.set_index('step')
    basis_dateiname = os.path.splitext(os.path.basename(state_path))[0]; analyse_subdir = get_simulation_subdir(basis_dateiname)
    csv_path = os.path.join(analyse_subdir, "simulation_metrics.csv")
    try: metrics_df.to_csv(csv_path); print(f"  -> Metrikdaten gespeichert: {csv_path}")
    except Exception as e: print(f"FEHLER beim Speichern der CSV-Daten: {e}")

    # --- Erstelle Plots ---
    print("\nINFO: Erstelle Verlaufsplots...")
    # Definiere die Spalten, die geplottet werden sollen
    limbus_cols = ["limbus_pleasure", "limbus_arousal", "limbus_dominance"]
    meta_cols = ["meta_creativus_activation", "meta_criticus_activation", "meta_metacognitio_activation"]
    quant_cols = ["avg_q_variance", "total_q_jumps"]
    activation_cols = ["avg_activation_all"]

    plot_time_series(metrics_df, activation_cols, "Durchschnittliche Knotenaktivierung über Simulationsschritte", "Ø Aktivierung", "avg_activation_timeseries.png", analyse_subdir, ylim=(0, 1))
    plot_time_series(metrics_df, limbus_cols, "Limbus Affektus (PAD) Zustand über Simulationsschritte", "PAD Wert", "limbus_pad_timeseries.png", analyse_subdir, ylim=(-1.1, 1.1))
    if processor.meta_nodes_enabled: plot_time_series(metrics_df, meta_cols, "Meta-Knoten Aktivierung über Simulationsschritte", "Aktivierungslevel", "meta_nodes_activation_timeseries.png", analyse_subdir, ylim=(0, 1.1))
    plot_time_series(metrics_df, ["avg_q_variance"], "Durchschnittliche Quanten-Varianz über Simulationsschritte", "Ø Varianz", "q_variance_timeseries.png", analyse_subdir)
    plot_time_series(metrics_df, ["total_q_jumps"], "Anzahl Quanten-Sprünge pro Simulationsschritt", "Anzahl Sprünge", "q_jumps_timeseries.png", analyse_subdir)

    # --- Erstelle Heatmap und Korrelationsmatrix ---
    print("\nINFO: Erstelle Heatmap und Korrelationsmatrix...")
    all_plot_columns = activation_cols + limbus_cols + meta_cols + quant_cols
    # Entferne Spalten, die nur NaN enthalten könnten
    columns_for_analysis = [col for col in all_plot_columns if col in metrics_df.columns and metrics_df[col].notna().any()]

    plot_heatmap(metrics_df, columns_for_analysis, "Heatmap: Verlauf der Netzwerkmetriken", "metrics_heatmap.png", analyse_subdir)
    plot_correlation_matrix(metrics_df, columns_for_analysis, "Korrelationsmatrix der Metriken", "correlation_heatmap.png", analyse_subdir)

    # --- Erstelle Moving Average Plot ---
    print("\nINFO: Erstelle Moving Average Plot...")
    moving_avg_df = metrics_df.rolling(window=moving_avg_window, min_periods=1).mean()
    plot_time_series(moving_avg_df, activation_cols + limbus_cols, # Beispiel: Akt. & Limbus
                     f"Moving Average ({moving_avg_window} Schritte) - Aktivierung und Limbus",
                     "Geglättete Werte", "moving_average_timeseries.png", analyse_subdir)

    # --- Erstelle Ableitungsplot ---
    print("\nINFO: Erstelle Ableitungsplot...")
    derivatives_df = metrics_df.diff().dropna(how='all') # Erste Zeile ist NaN
    # Beispiel: Pleasure, Arousal, Varianz
    cols_for_derivative_plot = ["limbus_pleasure", "limbus_arousal", "avg_q_variance"]
    valid_derivative_cols = [col for col in cols_for_derivative_plot if col in derivatives_df.columns]
    if valid_derivative_cols:
        plot_time_series(derivatives_df, valid_derivative_cols,
                         "1. Ableitung: Dynamik von Limbus und Quanten-Varianz",
                         "Änderung zum vorherigen Schritt", "derivatives_timeseries.png", analyse_subdir)
    else:
        print("INFO: Nicht genügend Daten für Ableitungsplot vorhanden.")


    print(f"\n===== Simulationsanalyse vollständig abgeschlossen =====")
    print(f"Ergebnisse (Plots, Heatmaps, CSV) in: {analyse_subdir}")


# --- Kommandozeilen-Interface (unverändert) ---
def cli():
    parser = argparse.ArgumentParser(
        description="Führt N Simulationsschritte auf einem QNP-Zustand aus und plottet Metrikverläufe.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "state_file",
        type=str,
        help="Pfad zur QNP-Zustandsdatei (.json), die geladen werden soll."
    )
    parser.add_argument(
        "-n", "--num_steps",
        type=int,
        default=50,
        help="Anzahl der durchzuführenden Simulationsschritte."
    )
    parser.add_argument(
        "-w", "--moving_avg_window",
        type=int,
        default=5,
        help="Fenstergröße für den Moving Average Plot."
    )
    args = parser.parse_args()

    if args.num_steps <= 0:
        print("FEHLER: Anzahl der Schritte muss positiv sein.")
        return
    if args.moving_avg_window <= 0:
         print("FEHLER: Fenstergröße für Moving Average muss positiv sein.")
         return

    if os.path.exists(args.state_file):
        run_simulation_and_analyze(args.state_file, args.num_steps, args.moving_avg_window)
    else:
        print(f"FEHLER: Zustandsdatei nicht gefunden: {args.state_file}")

if __name__ == "__main__":
    # Beispielaufruf: python qnp_simulation_analyzer.py qnp_state.json -n 10 -w 10
    cli()