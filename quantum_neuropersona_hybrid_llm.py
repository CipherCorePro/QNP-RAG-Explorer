# -- coding: utf-8 --

# Filename: quantum_neuropersona_hybrid_llm.py
# Version: 1.3 - Integrated Meta Nodes (Creativus, CortexCriticus, MetaCognitio)
# Author: [CipherCore Technology] & Gemini & Your Input & History Maker

import numpy as np
import pandas as pd
import random
from collections import deque, Counter, defaultdict
import json
# import sqlite3 # Vorerst nicht verwendet
import os
import time
import traceback
from typing import Optional, Callable, List, Tuple, Dict, Any, Generator
from datetime import datetime
import math # Hinzugefügt für LimbusAffektus (tanh)
import uuid as uuid_module
import re

# Text Processing / Retrieval specific imports
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- Imports für Gemini API ---
try:
    import google.generativeai as genai
    # Optional: Importiere Typen für Fehlerbehandlung
    from google.api_core.exceptions import GoogleAPIError
    GEMINI_AVAILABLE = True
    print("INFO: Google Generative AI SDK gefunden.")
except ImportError:
    GEMINI_AVAILABLE = False
    print("WARNUNG: 'google-generativeai' nicht gefunden. RAG-Funktionalität (Textgenerierung) ist deaktiviert.")
    print("Installieren Sie es mit: pip install google-generativeai")
    genai = None
    GoogleAPIError = None
# --- Ende Imports ---

# Optional: Netzwerk-Visualisierung
try: import networkx as nx; NETWORKX_AVAILABLE = True
except ImportError: NETWORKX_AVAILABLE = False

# Optional: Fortschrittsbalken
try: from tqdm import tqdm; TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    def tqdm(iterable, *args, **kwargs): return iterable

# --- HILFSFUNKTIONEN & BASIS-GATES ---
# Definiere Quantengates
I = np.array([[1, 0], [0, 1]], dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
H = (1/np.sqrt(2)) * np.array([[1, 1], [1, -1]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
P0 = np.array([[1, 0], [0, 0]], dtype=complex) # Projektor |0><0|
P1 = np.array([[0, 0], [0, 1]], dtype=complex) # Projektor |1><1|

def _ry(theta: float) -> np.ndarray:
    """Erzeugt eine RY-Rotationsmatrix."""
    if not np.isfinite(theta): theta = 0.0 # Fallback für ungültige Winkel
    cos_t = np.cos(theta / 2); sin_t = np.sin(theta / 2)
    return np.array([[cos_t, -sin_t], [sin_t, cos_t]], dtype=complex)

def _rz(phi: float) -> np.ndarray:
    """Erzeugt eine RZ-Rotationsmatrix."""
    if not np.isfinite(phi): phi = 0.0 # Fallback für ungültige Winkel
    exp_m = np.exp(-1j * phi / 2); exp_p = np.exp(1j * phi / 2)
    return np.array([[exp_m, 0], [0, exp_p]], dtype=complex)

def _apply_gate(state_vector: np.ndarray, gate: np.ndarray, target_qubit: int, num_qubits: int) -> np.ndarray:
    """Wendet ein Single-Qubit-Gate auf den Zustandsvektor an."""
    if gate.shape != (2, 2): raise ValueError("Gate muss 2x2 sein.")
    if not (0 <= target_qubit < num_qubits): raise ValueError(f"Target qubit {target_qubit} out of range [0, {num_qubits-1}].")

    expected_len = 2**num_qubits; current_len = len(state_vector)
    if current_len != expected_len:
        state_vector = np.zeros(expected_len, dtype=complex); state_vector[0] = 1.0

    op_list = [I] * num_qubits
    op_list[target_qubit] = gate
    full_matrix = op_list[0]
    for i in range(1, num_qubits):
        full_matrix = np.kron(full_matrix, op_list[i])
    new_state = np.dot(full_matrix, state_vector)

    if not np.all(np.isfinite(new_state)):
        new_state = np.zeros(expected_len, dtype=complex); new_state[0] = 1.0
    return new_state

def _apply_cnot(state_vector: np.ndarray, control_qubit: int, target_qubit: int, num_qubits: int) -> np.ndarray:
    """Wendet ein CNOT-Gate auf den Zustandsvektor an."""
    if not (0 <= control_qubit < num_qubits and 0 <= target_qubit < num_qubits): raise ValueError("Qubit index out of range.")
    if control_qubit == target_qubit: raise ValueError("Control and target must be different.")

    expected_len = 2**num_qubits; current_len = len(state_vector)
    if current_len != expected_len:
        state_vector = np.zeros(expected_len, dtype=complex); state_vector[0] = 1.0

    op_list_p0 = [I] * num_qubits
    op_list_p1 = [I] * num_qubits
    op_list_p0[control_qubit] = P0
    op_list_p1[control_qubit] = P1
    op_list_p1[target_qubit] = X
    term0_matrix = op_list_p0[0]; term1_matrix = op_list_p1[0]
    for i in range(1, num_qubits):
        term0_matrix = np.kron(term0_matrix, op_list_p0[i])
        term1_matrix = np.kron(term1_matrix, op_list_p1[i])
    cnot_matrix = term0_matrix + term1_matrix
    new_state = np.dot(cnot_matrix, state_vector)

    if not np.all(np.isfinite(new_state)):
        new_state = np.zeros(expected_len, dtype=complex); new_state[0] = 1.0
    return new_state

# --- QUANTEN-ENGINE ---
class QuantumNodeSystem:
    """Simuliert das quantenbasierte Verhalten eines Knotens via PQC."""
    def __init__(self, num_qubits: int, initial_params: Optional[np.ndarray] = None):
        if not isinstance(num_qubits, int) or num_qubits <= 0: raise ValueError("num_qubits must be a positive integer.")
        self.num_qubits = num_qubits
        self.num_params = num_qubits * 2 # Je ein RY und ein RZ Parameter pro Qubit
        self.state_vector_size = 2**self.num_qubits

        if initial_params is None:
            self.params = np.random.rand(self.num_params) * 2 * np.pi
        elif isinstance(initial_params, np.ndarray) and initial_params.shape == (self.num_params,):
             safe_params = np.nan_to_num(initial_params, nan=np.pi, posinf=2*np.pi, neginf=0.0)
             self.params = np.clip(safe_params, 0, 2 * np.pi)
        else:
            print(f"WARNUNG: Ungültige initial_params (Typ: {type(initial_params)}, Shape: {getattr(initial_params, 'shape', 'N/A')}). Initialisiere zufällig.")
            self.params = np.random.rand(self.num_params) * 2 * np.pi

        self.state_vector = np.zeros(self.state_vector_size, dtype=complex); self.state_vector[0] = 1.0 + 0j
        self.last_measurement_results: List[Dict] = []
        self.last_applied_ops: List[Tuple] = []

    def _build_pqc_ops(self, input_strength: float) -> List[Tuple]:
        """Erstellt die Liste der Gate-Operationen für den PQC."""
        ops = []
        scaled_input_angle = np.tanh(input_strength) * np.pi
        if not np.isfinite(scaled_input_angle): scaled_input_angle = 0.0

        for i in range(self.num_qubits): ops.append(('H', i))
        for i in range(self.num_qubits):
            theta = scaled_input_angle * self.params[2 * i]
            ops.append(('RY', i, theta if np.isfinite(theta) else 0.0))
        for i in range(self.num_qubits):
            phi = self.params[2 * i + 1]
            ops.append(('RZ', i, phi if np.isfinite(phi) else 0.0))
        if self.num_qubits > 1:
            for i in range(self.num_qubits - 1): ops.append(('CNOT', i, i + 1))
        return ops

    def activate(self, input_strength: float, n_shots: int = 100) -> Tuple[float, np.ndarray, List[Dict]]:
        """Führt den PQC aus und gibt die Aktivierungswahrscheinlichkeit zurück."""
        if not np.isfinite(input_strength): input_strength = 0.0
        if n_shots <= 0: n_shots = 1

        pqc_ops = self._build_pqc_ops(input_strength)
        self.last_applied_ops = pqc_ops

        current_state = self.state_vector.copy()
        if not np.isclose(np.linalg.norm(current_state), 1.0):
            current_state = np.zeros(self.state_vector_size, dtype=complex); current_state[0] = 1.0

        gate_application_successful = True
        for op_index, op in enumerate(pqc_ops):
            try:
                op_type = op[0]
                if op_type == 'H': current_state = _apply_gate(current_state, H, op[1], self.num_qubits)
                elif op_type == 'RY': current_state = _apply_gate(current_state, _ry(op[2]), op[1], self.num_qubits)
                elif op_type == 'RZ': current_state = _apply_gate(current_state, _rz(op[2]), op[1], self.num_qubits)
                elif op_type == 'CNOT': current_state = _apply_cnot(current_state, op[1], op[2], self.num_qubits)

                if not np.all(np.isfinite(current_state)): raise ValueError(f"Non-finite state after {op}")
                norm = np.linalg.norm(current_state)
                if norm > 1e-9: current_state /= norm
                else: raise ValueError(f"Zero state after {op}")
            except Exception as e:
                current_state = np.zeros(self.state_vector_size, dtype=complex); current_state[0] = 1.0
                gate_application_successful = False; break

        self.state_vector = current_state
        total_hamming_weight = 0
        measurement_log = []
        activation_prob = 0.0

        if n_shots > 0 and gate_application_successful and self.num_qubits > 0:
            probabilities = np.abs(current_state)**2
            probabilities = np.maximum(0, probabilities)
            prob_sum = np.sum(probabilities)

            if not np.isclose(prob_sum, 1.0, atol=1e-7):
                if prob_sum < 1e-9: probabilities.fill(0.0); probabilities[0] = 1.0
                else: probabilities /= prob_sum
                probabilities = np.maximum(0, probabilities)
                probabilities /= np.sum(probabilities)

            try:
                measured_indices = np.random.choice(self.state_vector_size, size=n_shots, p=probabilities)
                for shot_idx, measured_index in enumerate(measured_indices):
                    state_idx_int = int(measured_index)
                    binary_repr = format(state_idx_int, f'0{self.num_qubits}b')
                    hamming_weight = binary_repr.count('1')
                    total_hamming_weight += hamming_weight
                    measurement_log.append({
                        "shot": shot_idx, "index": state_idx_int,
                        "binary": binary_repr, "hamming": hamming_weight,
                        "probability": probabilities[state_idx_int]
                    })
            except ValueError as e:
                 print(f"WARNUNG: np.random.choice Fehler in QNS ({e}). Fallback zu argmax.");
                 if np.any(probabilities):
                     measured_index = np.argmax(probabilities)
                     state_idx_int = int(measured_index)
                     binary_repr = format(state_idx_int, f'0{self.num_qubits}b')
                     hamming_weight = binary_repr.count('1')
                     total_hamming_weight = hamming_weight * n_shots
                     measurement_log.append({"shot": 0, "index": state_idx_int, "binary": binary_repr, "hamming": hamming_weight, "error": "ValueError, used argmax", "probability": probabilities[state_idx_int]})
                 else:
                     measurement_log.append({"shot": 0, "index": 0, "binary": '0'*self.num_qubits, "hamming": 0, "error": "All probabilities zero", "probability": 0.0})

            if n_shots > 0 and self.num_qubits > 0:
                activation_prob = float(np.clip(total_hamming_weight / (n_shots * self.num_qubits), 0.0, 1.0))
                if not np.isfinite(activation_prob): activation_prob = 0.0

        elif not gate_application_successful:
             activation_prob = 0.0
             measurement_log = [{"error": "PQC execution failed"}]

        self.last_measurement_results = measurement_log
        if not isinstance(activation_prob, (float, np.number)) or not np.isfinite(activation_prob):
            activation_prob = 0.0
        return activation_prob, self.state_vector, measurement_log

    def get_params(self) -> np.ndarray:
        """Gibt eine sichere Kopie der aktuellen Parameter zurück."""
        safe_params = np.nan_to_num(self.params.copy(), nan=np.pi, posinf=2*np.pi, neginf=0.0)
        return np.clip(safe_params, 0, 2 * np.pi)

    def set_params(self, params: np.ndarray):
        """Setzt die Parameter des Systems sicher."""
        if isinstance(params, np.ndarray) and params.shape == self.params.shape:
            safe_params = np.nan_to_num(params, nan=np.pi, posinf=2*np.pi, neginf=0.0)
            self.params = np.clip(safe_params, 0, 2 * np.pi)

    def update_internal_params(self, delta_params: np.ndarray):
        """Aktualisiert die internen Parameter um delta_params."""
        if not isinstance(delta_params, np.ndarray) or delta_params.shape != self.params.shape: return
        safe_delta = np.nan_to_num(delta_params, nan=0.0, posinf=0.0, neginf=0.0)
        new_params = self.params + safe_delta
        new_params_safe = np.nan_to_num(new_params, nan=np.pi, posinf=2*np.pi, neginf=0.0)
        self.params = np.clip(new_params_safe, 0, 2 * np.pi)

# --- NETZWERK-STRUKTUR & TEXT-CHUNKS ---
class Connection:
    """Repräsentiert eine gerichtete, gewichtete Verbindung zwischen zwei Knoten."""
    DEFAULT_WEIGHT_RANGE = (0.01, 0.5)
    DEFAULT_LEARNING_RATE = 0.05
    DEFAULT_DECAY_RATE = 0.001

    def __init__(self, target_node: 'Node', weight: Optional[float] = None, source_node_label: Optional[str] = None, conn_type: str = "associative"):
        if target_node is None or not hasattr(target_node, 'uuid'): raise ValueError("Target node invalid or missing UUID.")
        self.target_node_uuid: str = target_node.uuid
        self.source_node_label: Optional[str] = source_node_label
        self.conn_type: str = conn_type
        raw_weight = weight if weight is not None else random.uniform(*self.DEFAULT_WEIGHT_RANGE)
        self.weight: float = float(np.clip(raw_weight, 0.0, 1.0))
        self.last_transmitted_signal: float = 0.0
        self.transmission_count: int = 0
        self.created_at: datetime = datetime.now()
        self.last_update_at: datetime = datetime.now()

    def update_weight(self, delta_weight: float, learning_rate: Optional[float] = None):
        lr = learning_rate if learning_rate is not None else self.DEFAULT_LEARNING_RATE
        new_weight = self.weight + (delta_weight * lr)
        self.weight = float(np.clip(new_weight, 0.0, 1.0))
        self.last_update_at = datetime.now()

    def decay(self, decay_rate: Optional[float] = None):
        dr = decay_rate if decay_rate is not None else self.DEFAULT_DECAY_RATE
        self.weight = max(0.0, self.weight * (1.0 - dr))
        self.last_update_at = datetime.now()

    def transmit(self, source_activation: float) -> float:
        transmitted_signal = source_activation * self.weight
        self.last_transmitted_signal = transmitted_signal
        self.transmission_count += 1
        return transmitted_signal

    def __repr__(self) -> str:
        target_info = f"to_UUID:{self.target_node_uuid[:8]}..."
        source_info = f" from:{self.source_node_label}" if self.source_node_label else ""
        weight_info = f"W:{self.weight:.3f}" if hasattr(self, 'weight') else "W:N/A"
        count_info = f"Cnt:{self.transmission_count}" if hasattr(self, 'transmission_count') else "Cnt:N/A"
        return f"<Conn {target_info} {weight_info} {count_info}{source_info}>"

class Node:
    """Basisklasse für alle Knoten im Netzwerk."""
    DEFAULT_NUM_QUBITS = 10
    DEFAULT_ACTIVATION_HISTORY_LEN = 20
    DEFAULT_N_SHOTS = 50

    def __init__(self, label: str, num_qubits: Optional[int] = None, is_quantum: bool = True, neuron_type: str = "excitatory",
                 initial_params: Optional[np.ndarray] = None, uuid: Optional[str] = None):
        if not label: raise ValueError("Node label cannot be empty.")
        self.label: str = label
        self.uuid: str = uuid if uuid else str(uuid_module.uuid4())
        self.neuron_type: str = neuron_type # 'semantic', 'affective_modulator', 'creative_modulator', 'critical_modulator', 'meta_cognitive'
        self.is_quantum = is_quantum
        self.num_qubits = num_qubits if num_qubits is not None else self.DEFAULT_NUM_QUBITS

        if self.is_quantum and (not isinstance(self.num_qubits, int) or self.num_qubits <= 0):
            print(f"WARNUNG: Ungültige num_qubits ({self.num_qubits}) für Quantenknoten '{self.label}'. Setze auf klassisch.")
            self.is_quantum = False; self.num_qubits = 0
        elif not self.is_quantum:
            self.num_qubits = 0

        self.connections: Dict[str, Optional[Connection]] = {}
        self.incoming_connections_info: List[Tuple[str, str]] = []
        self.activation: float = 0.0
        self.activation_sum: float = 0.0
        self.activation_history: deque = deque(maxlen=self.DEFAULT_ACTIVATION_HISTORY_LEN)

        self.q_system: Optional[QuantumNodeSystem] = None
        if self.is_quantum:
            try:
                self.q_system = QuantumNodeSystem(num_qubits=self.num_qubits, initial_params=initial_params)
            except Exception as e:
                print(f"FEHLER bei Initialisierung des Quantensystems für Knoten '{self.label}': {e}. Setze auf klassisch.")
                self.is_quantum = False; self.num_qubits = 0

        self.last_measurement_log: List[Dict] = []
        self.last_state_vector: Optional[np.ndarray] = None
        # NEU: Store last analysis result
        self.last_measurement_analysis: Dict[str, Any] = {}

    def add_connection(self, target_node: 'Node', weight: Optional[float] = None, conn_type: str = "associative") -> Optional[Connection]:
        if target_node is None or not hasattr(target_node, 'uuid') or target_node.uuid == self.uuid: return None
        target_uuid = target_node.uuid
        if target_uuid not in self.connections:
            try:
                conn = Connection(target_node=target_node, weight=weight, source_node_label=self.label, conn_type=conn_type)
                self.connections[target_uuid] = conn
                if hasattr(target_node, 'add_incoming_connection_info'):
                    target_node.add_incoming_connection_info(self.uuid, self.label)
                return conn
            except Exception as e:
                 print(f"FEHLER Erstellen/Hinzufügen Verbindung {self.label} -> {target_node.label}: {e}"); return None
        else: return self.connections.get(target_uuid)

    def add_incoming_connection_info(self, source_uuid: str, source_label: str):
        if not any(info[0] == source_uuid for info in self.incoming_connections_info):
             self.incoming_connections_info.append((source_uuid, source_label))

    def strengthen_connection(self, target_node: 'Node', learning_signal: float = 0.1, learning_rate: Optional[float] = None):
        if target_node is None or not hasattr(target_node, 'uuid'): return
        connection = self.connections.get(target_node.uuid)
        if connection is not None:
             connection.update_weight(delta_weight=learning_signal, learning_rate=learning_rate)

    def calculate_activation(self, n_shots: Optional[int] = None):
        """Standard-Aktivierungsberechnung für semantische und affektive Knoten."""
        current_n_shots = n_shots if n_shots is not None else self.DEFAULT_N_SHOTS
        new_activation: float = 0.0
        self.last_measurement_log = []
        self.last_measurement_analysis = {}

        if self.is_quantum and self.q_system:
            try:
                q_activation, q_state_vector, q_measure_log = self.q_system.activate(self.activation_sum, current_n_shots)
                new_activation = q_activation
                self.last_state_vector = q_state_vector
                self.last_measurement_log = q_measure_log
                # Berechne und speichere Analyse direkt hier
                self.last_measurement_analysis = self.analyze_jumps(q_measure_log)
            except Exception as e:
                new_activation = 0.0; self.last_state_vector = None
                self.last_measurement_analysis = {"error": f"Activation failed: {e}"}
        else: # Klassisch
            activation_sum_float = float(self.activation_sum) if isinstance(self.activation_sum, (float, np.number)) and np.isfinite(self.activation_sum) else 0.0
            safe_activation_sum = np.clip(activation_sum_float, -700, 700)
            try: new_activation = 1 / (1 + np.exp(-safe_activation_sum))
            except FloatingPointError: new_activation = 1.0 if safe_activation_sum > 0 else 0.0
            self.last_state_vector = None
            self.last_measurement_analysis = {} # Keine Analyse für klassische Knoten

        if not isinstance(new_activation, (float, np.number)) or not np.isfinite(new_activation):
            self.activation = 0.0
        else: self.activation = float(np.clip(new_activation, 0.0, 1.0))

        self.activation_history.append(self.activation)
        self.activation_sum = 0.0 # Reset für nächsten Schritt

    # calculate_meta_activation wird in Subklassen definiert

    def get_smoothed_activation(self, window: int = 3) -> float:
        if not self.activation_history: return self.activation
        hist = list(self.activation_history)[-window:]
        valid_hist = [a for a in hist if isinstance(a, (float, np.number)) and np.isfinite(a)]
        if not valid_hist: return self.activation
        else: return float(np.mean(valid_hist))

    def get_state_representation(self) -> Dict[str, Any]:
        state = {
            "label": self.label, "uuid": self.uuid,
            "activation": round(self.activation, 4),
            "smoothed_activation": round(self.get_smoothed_activation(), 4),
            "type": type(self).__name__, "neuron_type": self.neuron_type,
            "is_quantum": self.is_quantum,
            "num_connections": len(self.connections) if hasattr(self, 'connections') and isinstance(self.connections, dict) else 0
        }
        if self.is_quantum and self.q_system:
            state["num_qubits"] = self.num_qubits
            # Verwende gespeicherte Analyse
            state["last_measurement_analysis"] = getattr(self, 'last_measurement_analysis', {})
        # Spezifische Zustände
        if isinstance(self, LimbusAffektus): state["emotion_state"] = getattr(self, 'emotion_state', {}).copy()
        # Meta-Knoten könnten hier auch spezifische Zustände hinzufügen, falls nötig
        return state

    # Innerhalb der Klasse Node in quantum_neuropersona_hybrid_llm.py

    def analyze_jumps(self, measurement_log: List[Dict]) -> Dict[str, Any]:
        """
        Analysiert die Messprotokolle auf Zustandsvarianz und signifikante
        Zustandsänderungen (Jumps). Stellt sicher, dass boolesche Rückgabewerte
        Python-Standardtypen sind.

        Args:
            measurement_log: Eine Liste von Dictionaries, wobei jedes Dict
                             eine Messung repräsentiert (sollte 'index'
                             und optional 'error' enthalten).

        Returns:
            Ein Dictionary mit Analyse-Metriken:
            - shots_recorded (int): Anzahl gültiger Messungen berücksichtigt.
            - jump_detected (bool): Ob ein Sprung über dem Schwellenwert erkannt wurde (Python bool).
            - max_jump_abs (int): Der größte absolute Sprung zwischen zwei Messungen.
            - avg_jump_abs (float): Der durchschnittliche absolute Sprung.
            - state_variance (float): Die Varianz der gemessenen Zustandsindizes.
            - significant_threshold (float): Der Schwellenwert für signifikante Sprünge.
            - error_count (int): Anzahl der Messungen mit Fehlermarkierung.
        """
        # Zähle Fehler und Gesamtzahl der Einträge im Log
        total_entries = len(measurement_log)
        error_count = sum(1 for m in measurement_log if m.get("error"))

        # Extrahiere gültige gemessene Zustandsindizes (int oder numpy integer), ignoriere Fehler
        valid_indices = [
            m.get('index') for m in measurement_log
            if isinstance(m.get('index'), (int, np.integer)) and not m.get("error")
        ]
        # valid_shots_analyzed ist die Anzahl der *gültigen* Messungen für die Analyse
        valid_shots_analyzed = len(valid_indices)

        # Initialisiere Standard-Rückgabewerte
        max_jump = 0.0
        avg_jump = 0.0
        state_variance = 0.0
        jump_detected_py: bool = False # Python bool als Default
        significant_threshold = 0.0

        # Definiere Schwellenwert auch wenn nicht genug Daten für Sprungberechnung da sind (für Info)
        if self.is_quantum and self.q_system and self.num_qubits > 0:
            # Schwellwert z.B. 1/4 des Zustandsraums
            significant_threshold = (2**self.num_qubits) / 4.0
        else:
            significant_threshold = 1.0 # Standard für klassisch oder Fallback

        # Berechnungen nur durchführen, wenn mindestens 2 gültige Messungen vorhanden sind
        if valid_shots_analyzed >= 2:
            try:
                indices_array = np.array(valid_indices, dtype=float)

                # Berechne absolute Differenzen zwischen aufeinanderfolgenden Messungen
                jumps = np.abs(np.diff(indices_array))

                # Berechne Varianz der gemessenen Zustände
                # ddof=0 ist die Populationsvarianz
                state_variance = np.var(indices_array, ddof=0) if valid_shots_analyzed > 0 else 0.0

                if jumps.size > 0: # Nur wenn Sprünge berechnet werden konnten
                    max_jump = np.max(jumps)
                    avg_jump = np.mean(jumps)

                    # Prüfe auf signifikanten Sprung (Vergleich muss gültig sein)
                    if np.isfinite(max_jump) and np.isfinite(significant_threshold):
                        # --- EXPLIZITE KONVERTIERUNG ZU PYTHON BOOL ---
                        jump_detected_py = bool(max_jump > significant_threshold)

            except Exception as e:
                print(f"WARNUNG: Fehler bei der Jump/Varianz-Berechnung für Knoten {self.label}: {e}")
                # Setze Werte auf Default zurück oder markiere als Fehler
                max_jump = 0.0
                avg_jump = 0.0
                state_variance = 0.0
                jump_detected_py = False
                # Optional: error_count erhöhen oder speziellen Fehlerflag setzen
                error_count += 1 # Zähle Berechnungsfehler als Fehler

        # Erstelle das finale Ergebnis-Dictionary
        # Verwende die Anzahl der tatsächlich analysierten gültigen Shots
        # und die oben berechneten oder Standardwerte.
        return {
            "shots_recorded": total_entries,              # Ursprüngliche Anzahl an Messungen im Log
            "valid_shots_analyzed": valid_shots_analyzed, # Anzahl der für Analyse genutzten Messungen
            "jump_detected": jump_detected_py,            # Sicher ein Python bool
            "max_jump_abs": int(max_jump),                # Als Integer für Klarheit
            "avg_jump_abs": round(avg_jump, 3),           # Runde für Lesbarkeit
            "state_variance": round(state_variance, 3),   # Runde für Lesbarkeit
            "significant_threshold": round(significant_threshold, 1), # Runde für Lesbarkeit
            "error_count": error_count                    # Anzahl fehlerhafter Einträge/Berechnungsfehler
        }
    def __repr__(self) -> str:
        act_str = f"Act:{self.activation:.3f}"
        q_info = f" Q:{self.num_qubits}" if self.is_quantum and self.q_system else " (Cls)"
        conn_count = len(self.connections) if hasattr(self, 'connections') and isinstance(self.connections, dict) else 0
        conn_info = f" Conns:{conn_count}"
        return f"<{type(self).__name__} '{self.label}' {act_str}{q_info}{conn_info}>"

    # --- State Management (__getstate__ / __setstate__) ---
    def __getstate__(self):
        state_to_return = {}
        # Basisattribute
        for key in ['label', 'uuid', 'neuron_type', 'is_quantum', 'num_qubits', 'activation', 'activation_sum']:
             if hasattr(self, key): state_to_return[key] = getattr(self, key)
        state_to_return['incoming_connections_info'] = getattr(self, 'incoming_connections_info', [])
        state_to_return['activation_history'] = list(getattr(self, 'activation_history', deque()))

        # Quantenparameter
        q_system = getattr(self, 'q_system', None)
        if q_system is not None and hasattr(q_system, 'get_params'):
            try:
                q_params = q_system.get_params()
                state_to_return['q_system_params'] = q_params.tolist() if isinstance(q_params, np.ndarray) else q_params
            except Exception as e_q:
                print(f"    ERROR getting/converting q_system_params for {self.label}: {e_q}"); state_to_return['q_system_params'] = None
        else: state_to_return['q_system_params'] = None

        # Verbindungen
        connections_serializable = {}
        live_connections = getattr(self, 'connections', None)
        if isinstance(live_connections, dict):
            for target_uuid, conn in live_connections.items():
                if conn is None: continue
                try:
                    target_uuid_in_conn = getattr(conn, 'target_node_uuid', target_uuid)
                    if not target_uuid_in_conn: continue
                    conn_data = {'target_node_uuid': target_uuid_in_conn}
                    for attr in ['weight', 'source_node_label', 'conn_type', 'last_transmitted_signal', 'transmission_count', 'created_at', 'last_update_at']:
                        val = getattr(conn, attr, None)
                        if isinstance(val, (datetime)): val = str(val)
                        elif isinstance(val, (float, np.number)): val = float(val)
                        elif isinstance(val, (int, np.integer)): val = int(val)
                        conn_data[attr] = val
                    connections_serializable[target_uuid_in_conn] = conn_data
                except Exception as e_ser: print(f"    ERROR serializing connection {target_uuid} from {self.label}: {e_ser}")
        state_to_return['connections_serializable'] = connections_serializable

        # Spezifische Attribute von Subklassen
        if isinstance(self, LimbusAffektus): state_to_return['emotion_state'] = getattr(self, 'emotion_state', INITIAL_EMOTION_STATE).copy()
        # Meta-Knoten: Fügen Sie hier ggf. spezifische Zustände hinzu

        state_to_return['type'] = type(self).__name__ # Wichtig für Wiederherstellung
        return state_to_return

    def __setstate__(self, state: Dict[str, Any]):
        # Aktivierungsverlauf
        history_len = getattr(type(self), 'DEFAULT_ACTIVATION_HISTORY_LEN', 20)
        self.activation_history = deque(state.get('activation_history', []), maxlen=history_len)

        # Quantensystem
        q_params_list = state.pop('q_system_params', None)
        num_qbits = state.get('num_qubits', getattr(type(self), 'DEFAULT_NUM_QUBITS', 10))
        is_q = state.get('is_quantum', True)
        self.q_system = None
        q_params_np = None
        if q_params_list is not None and isinstance(q_params_list, list):
             try:
                  q_params_np = np.array(q_params_list, dtype=float)
                  expected_shape = (num_qbits * 2,)
                  if num_qbits > 0 and (q_params_np.shape != expected_shape or not np.all(np.isfinite(q_params_np))): q_params_np = None
                  elif num_qbits == 0 and q_params_np.size != 0: q_params_np = None
             except Exception: q_params_np = None
        if is_q and num_qbits > 0:
             try: self.q_system = QuantumNodeSystem(num_qubits=num_qbits, initial_params=q_params_np)
             except Exception as e: print(f"FEHLER (__setstate__) Restore QNS für '{state.get('label', '?')}': {e}"); state['is_quantum'] = False; state['num_qubits'] = 0
        else: state['is_quantum'] = False; state['num_qubits'] = 0

        # Temporär gespeicherte Verbindungsdaten
        self.connections_serializable_temp = state.pop('connections_serializable', {})
        self.connections: Dict[str, Optional[Connection]] = {} # Wird später gefüllt

        # Spezifische Attribute für Subklassen
        if state.get('type') == 'LimbusAffektus': self.emotion_state = state.pop('emotion_state', INITIAL_EMOTION_STATE.copy())
        # Meta-Knoten: Fügen Sie hier ggf. spezifische Zustände hinzu

        # Restliche Attribute setzen
        valid_attrs = ['label', 'uuid', 'neuron_type', 'is_quantum', 'num_qubits', 'activation', 'activation_sum', 'incoming_connections_info']
        for key, value in state.items():
             if key in valid_attrs: setattr(self, key, value)
        if not hasattr(self, 'uuid') or not self.uuid: self.uuid = str(uuid_module.uuid4())
        if not hasattr(self, 'incoming_connections_info') or not isinstance(self.incoming_connections_info, list): self.incoming_connections_info = []
        if not hasattr(self, 'last_measurement_analysis'): self.last_measurement_analysis = {}


# --- Emotionale Konstanten und LimbusAffektus Klasse ---
EMOTION_DIMENSIONS = ["pleasure", "arousal", "dominance"]
INITIAL_EMOTION_STATE = {dim: 0.0 for dim in EMOTION_DIMENSIONS}

class LimbusAffektus(Node):
    """Modelliert globalen emotionalen Zustand (PAD) und moduliert."""
    def __init__(self, label: str = "Limbus Affektus", num_qubits: Optional[int] = None, is_quantum: bool = True, config: Optional[Dict] = None, **kwargs):
        actual_num_qubits = num_qubits if is_quantum else 0
        if actual_num_qubits is None and is_quantum: actual_num_qubits = 4 # Default für Limbus
        super().__init__(label=label, num_qubits=actual_num_qubits, is_quantum=is_quantum, neuron_type="affective_modulator", **kwargs)
        self.emotion_state = INITIAL_EMOTION_STATE.copy()
        self.config = config if config else {}
        self._update_params_from_config() # Initialisiere Parameter aus Config
        self.last_input_sum_for_pleasure = 0.0

    def _update_params_from_config(self):
        """Aktualisiert interne Parameter aus dem Config-Dict."""
        self.decay = self.config.get("limbus_emotion_decay", 0.95)
        self.arousal_sens = self.config.get("limbus_arousal_sensitivity", 1.5)
        self.pleasure_sens = self.config.get("limbus_pleasure_sensitivity", 1.0)
        self.dominance_sens = self.config.get("limbus_dominance_sensitivity", 1.0)

    def calculate_activation(self, n_shots: Optional[int] = None):
        """Berechnet Aktivierung und speichert Input-Summe für Pleasure-Update."""
        self.last_input_sum_for_pleasure = float(self.activation_sum) if isinstance(self.activation_sum, (float, np.number)) and np.isfinite(self.activation_sum) else 0.0
        # Ruft die Standard-Aktivierungsberechnung der Node-Klasse auf
        super().calculate_activation(n_shots=n_shots)

    def update_emotion_state(self, all_nodes: List['Node']):
        """Aktualisiert den internen emotionalen Zustand (PAD)."""
        if not all_nodes: return
        # Aktualisiere Sensitivitäten etc. falls Config geändert wurde
        self._update_params_from_config()

        other_node_activations = [n.activation for n in all_nodes if n.uuid != self.uuid and hasattr(n, 'activation') and isinstance(n.activation, (float, np.number)) and np.isfinite(n.activation)]
        avg_activation = np.mean(other_node_activations) if other_node_activations else 0.0

        # Berechne Updates basierend auf Sensitivitäten
        arousal_update = (avg_activation * 2 - 1) * self.arousal_sens
        # Verwende gespeicherte Summe von calculate_activation
        pleasure_update = math.tanh(self.last_input_sum_for_pleasure * self.pleasure_sens)
        # Eigene Aktivierung beeinflusst Dominanz
        dominance_update = (self.activation * 2 - 1) * self.dominance_sens

        # Update PAD Werte mit Decay und Clipping
        for dim, update in [("pleasure", pleasure_update), ("arousal", arousal_update), ("dominance", dominance_update)]:
            current_val = self.emotion_state.get(dim, 0.0)
            new_val = current_val * self.decay + update
            self.emotion_state[dim] = float(np.clip(new_val, -1.0, 1.0))


# --- NEUE META-KNOTEN KLASSEN ---

class CreativusNode(Node):
    """Modelliert Kreativität, beeinflusst durch Arousal und Quantenvarianz."""
    def __init__(self, label: str = "Creativus", num_qubits: Optional[int] = None, is_quantum: bool = True, config: Optional[Dict] = None, **kwargs):
        actual_num_qubits = num_qubits if is_quantum else 0
        if actual_num_qubits is None and is_quantum: actual_num_qubits = 6 # Eigener Default
        super().__init__(label=label, num_qubits=actual_num_qubits, is_quantum=is_quantum, neuron_type="creative_modulator", **kwargs)
        self.config = config if config else {}

    def calculate_meta_activation(self, relevant_nodes: List[Node], limbus_state: Dict[str, float]):
        """Berechnet Aktivierung basierend auf Arousal und Varianz relevanter Quantenknoten."""
        arousal = limbus_state.get("arousal", 0.0)
        # Aktivierung steigt mit Arousal (transformiert auf 0-1)
        arousal_component = (arousal + 1) / 2.0

        # Berechne durchschnittliche Varianz der *relevanten* Quantenknoten
        variances = []
        for node in relevant_nodes:
            if node.is_quantum and hasattr(node, 'last_measurement_analysis'):
                analysis = node.last_measurement_analysis
                if isinstance(analysis, dict) and "state_variance" in analysis and analysis.get("error_count", 1) == 0:
                     # Normalisiere Varianz grob (Annahme: max Varianz ~ (2^N)/4 ? Sehr grob!)
                     # Oder einfacher: nutze tanh, um hohe Varianz zu belohnen
                     variance_val = analysis.get("state_variance", 0.0)
                     # Normiere Varianz auf ca. 0-1 Bereich (sehr heuristisch!)
                     norm_variance = np.tanh(variance_val / (2**(node.num_qubits-2)) if node.num_qubits >= 2 else variance_val)
                     variances.append(norm_variance)

        variance_component = np.mean(variances) if variances else 0.0 # Durchschnittliche (normalisierte) Varianz

        # Kombiniere: Hohes Arousal UND hohe Varianz -> hohe Kreativität
        # Gewichtung kann angepasst werden
        combined_signal = (0.6 * arousal_component + 0.4 * variance_component)

        # Verwende Sigmoid oder Tanh für finale Aktivierung
        # Hier einfacher linearer Clip, könnte verfeinert werden
        new_activation = float(np.clip(combined_signal, 0.0, 1.0))

        # Standard-Aktivierungslogik (für Historie etc.)
        if not isinstance(new_activation, (float, np.number)) or not np.isfinite(new_activation): self.activation = 0.0
        else: self.activation = new_activation
        self.activation_history.append(self.activation)
        # Meta-Knoten nutzen `activation_sum` nicht direkt für ihre Kernlogik
        self.activation_sum = 0.0


class CortexCriticusNode(Node):
    """Modelliert kritische Bewertung, beeinflusst durch Dominanz und Quantenstabilität."""
    def __init__(self, label: str = "Cortex Criticus", num_qubits: Optional[int] = None, is_quantum: bool = True, config: Optional[Dict] = None, **kwargs):
        actual_num_qubits = num_qubits if is_quantum else 0
        if actual_num_qubits is None and is_quantum: actual_num_qubits = 6 # Eigener Default
        super().__init__(label=label, num_qubits=actual_num_qubits, is_quantum=is_quantum, neuron_type="critical_modulator", **kwargs)
        self.config = config if config else {}

    def calculate_meta_activation(self, relevant_nodes: List[Node], limbus_state: Dict[str, float]):
        """Berechnet Aktivierung basierend auf Dominanz und *niedriger* Varianz."""
        dominance = limbus_state.get("dominance", 0.0)
        # Aktivierung steigt mit Dominanz (transformiert auf 0-1)
        dominance_component = (dominance + 1) / 2.0

        # Berechne durchschnittliche Varianz der *relevanten* Quantenknoten
        variances = []
        for node in relevant_nodes:
            if node.is_quantum and hasattr(node, 'last_measurement_analysis'):
                analysis = node.last_measurement_analysis
                if isinstance(analysis, dict) and "state_variance" in analysis and analysis.get("error_count", 1) == 0:
                    norm_variance = np.tanh(analysis.get("state_variance", 0.0) / (2**(node.num_qubits-2)) if node.num_qubits >= 2 else analysis.get("state_variance", 0.0))
                    variances.append(norm_variance)

        avg_norm_variance = np.mean(variances) if variances else 0.0
        # Kritik steigt bei *niedriger* Varianz (Stabilität) -> inverse Beziehung
        stability_component = 1.0 - avg_norm_variance

        # Kombiniere: Hohe Dominanz UND hohe Stabilität -> hohe Kritikfähigkeit
        combined_signal = (0.5 * dominance_component + 0.5 * stability_component)
        new_activation = float(np.clip(combined_signal, 0.0, 1.0))

        # Standard-Aktivierungslogik
        if not isinstance(new_activation, (float, np.number)) or not np.isfinite(new_activation): self.activation = 0.0
        else: self.activation = new_activation
        self.activation_history.append(self.activation)
        self.activation_sum = 0.0


class MetaCognitioNode(Node):
    """Modelliert Selbstwahrnehmung, beeinflusst durch globale Aktivität und Quantensprünge."""
    def __init__(self, label: str = "MetaCognitio", num_qubits: Optional[int] = None, is_quantum: bool = False, config: Optional[Dict] = None, **kwargs):
         # Dieser Knoten ist eher ein klassischer Beobachter
        super().__init__(label=label, num_qubits=0, is_quantum=False, neuron_type="meta_cognitive", **kwargs)
        self.config = config if config else {}
        self.last_total_jumps = 0 # Zustand speichern

    def calculate_meta_activation(self, all_nodes: List[Node]):
        """Berechnet Aktivierung basierend auf globaler Aktivität und Anzahl Quantensprünge."""
        if not all_nodes: self.activation = 0.0; self.activation_history.append(0.0); return

        # 1. Globale Aktivität (ohne Meta-Knoten selbst)
        other_node_activations = [
            n.activation for n in all_nodes
            if n.uuid != self.uuid and n.neuron_type not in ["creative_modulator", "critical_modulator", "meta_cognitive"]
            and hasattr(n, 'activation') and isinstance(n.activation, (float, np.number)) and np.isfinite(n.activation)
        ]
        avg_global_activation = np.mean(other_node_activations) if other_node_activations else 0.0

        # 2. Anzahl signifikanter Quantensprünge im letzten Schritt
        total_jumps = 0
        for node in all_nodes:
            if node.is_quantum and hasattr(node, 'last_measurement_analysis'):
                analysis = node.last_measurement_analysis
                if isinstance(analysis, dict) and analysis.get("jump_detected", False):
                    total_jumps += 1
        self.last_total_jumps = total_jumps
        # Normalisiere Sprünge (z.B. max 5 Sprünge = volle Aktivierung durch diesen Teil)
        jump_component = np.clip(total_jumps / 5.0, 0.0, 1.0)

        # Kombiniere: Hohe globale Aktivität ODER viele Sprünge -> hohe Metakognition
        # Hier eine ODER-Logik (max), könnte auch additiv sein
        combined_signal = max(avg_global_activation, jump_component * 0.8) # Sprünge haben starken Einfluss
        new_activation = float(np.clip(combined_signal, 0.0, 1.0))

        # Standard-Aktivierungslogik
        if not isinstance(new_activation, (float, np.number)) or not np.isfinite(new_activation): self.activation = 0.0
        else: self.activation = new_activation
        self.activation_history.append(self.activation)
        self.activation_sum = 0.0

    # Überschreibe get_state_representation, um letzte Sprungzahl hinzuzufügen
    
    def get_state_representation(self) -> Dict[str, Any]:
        state = {
            "label": self.label, "uuid": self.uuid,
            "activation": round(self.activation, 4),
            "smoothed_activation": round(self.get_smoothed_activation(), 4),
            "type": type(self).__name__, "neuron_type": self.neuron_type,
            # --- EXPLIZITE KONVERTIERUNG HIER ---
            "is_quantum": bool(self.is_quantum),
            "num_connections": len(self.connections) if hasattr(self, 'connections') and isinstance(self.connections, dict) else 0
        }
        if self.is_quantum and self.q_system:
            state["num_qubits"] = self.num_qubits
            state["last_measurement_analysis"] = getattr(self, 'last_measurement_analysis', {})
        if isinstance(self, LimbusAffektus): state["emotion_state"] = getattr(self, 'emotion_state', {}).copy()
        if isinstance(self, MetaCognitioNode): state["last_total_jumps_detected"] = getattr(self, 'last_total_jumps', 0)
        # Füge hier weitere spezifische Zustände von Meta-Knoten hinzu, falls nötig
        return state


# --- TextChunk Klasse ---
class TextChunk:
    """Repräsentiert einen Textabschnitt mit Metadaten."""
    def __init__(self, text: str, source: str, index: int, chunk_uuid: Optional[str]=None):
        self.uuid = chunk_uuid if chunk_uuid else str(uuid_module.uuid4())
        self.text: str = text
        self.source: str = source
        self.index: int = index
        self.activated_node_labels: List[str] = []
        self.embedding: Optional[np.ndarray] = None

    def __repr__(self) -> str:
        node_str = f" Nodes:[{','.join(self.activated_node_labels)}]" if self.activated_node_labels else ""
        return f"<Chunk {self.index} from '{self.source}' (UUID:{self.uuid[:4]}...) Len:{len(self.text)}{node_str}>"


# --- HAUPTPROZESSOR-KLASSE ---
class QuantumEnhancedTextProcessor:
    """Orchestriert Netzwerkknoten, RAG und Lernen, moduliert durch Limbus & Meta-Knoten."""
    DEFAULT_CONFIG = {
        # Alte Defaults...
        "embedding_dim": 128, "chunk_size": 500, "chunk_overlap": 100, "training_epochs": 1, "training_files": [],
        "semantic_nodes": { "DefaultNode": [] },
        "connection_learning_rate": 0.05, "connection_decay_rate": 0.001, "connection_strengthening_signal": 0.1,
        "max_prompt_results": 3, "relevance_threshold": 0.08, "tfidf_max_features": 5000,
        "use_quantum_nodes": True, "default_num_qubits": 10, "simulation_n_shots": 50,
        "simulation_steps_after_training": 0,
        "enable_rag": True, "generator_model_name": "models/gemini-1.5-flash-latest",
        "generator_max_length": 8192, "generator_temperature": 0.7,
        "quantum_effect_variance_penalty": 0.5, "quantum_effect_activation_boost": 0.3,
        "quantum_effect_jump_llm_trigger": True,
        "enable_self_learning": True, "self_learning_file_path": "./training_data/learn.txt",
        "self_learning_source_name": "Generated Responses",
        # Limbus Defaults...
        "limbus_emotion_decay": 0.95, "limbus_arousal_sensitivity": 1.5, "limbus_pleasure_sensitivity": 1.0, "limbus_dominance_sensitivity": 1.0, "limbus_num_qubits": 4,
        "limbus_influence_prompt_level": 0.5, "limbus_influence_temperature_arousal": 0.1, "limbus_influence_temperature_dominance": -0.1,
        "limbus_min_temperature": 0.3, "limbus_max_temperature": 1.0,
        "limbus_influence_threshold_arousal": -0.03, "limbus_influence_threshold_pleasure": 0.03,
        "limbus_min_threshold": 0.02, "limbus_max_threshold": 0.2,
        "limbus_influence_ranking_bias_pleasure": 0.02,
        "limbus_influence_learning_rate_multiplier": 0.1, "limbus_min_lr_multiplier": 0.5, "limbus_max_lr_multiplier": 1.5,
        "limbus_influence_variance_penalty": 0.1, "limbus_influence_activation_boost": 0.05,

        # --- NEU: Meta-Knoten Config ---
        "meta_nodes_enabled": True, # Schalter für Meta-Knoten
        "creativus_num_qubits": 6,
        "cortex_criticus_num_qubits": 6,
        # Einflussfaktoren der Meta-Knoten
        "creativus_influence_temperature": 0.15,  # Additiver Faktor pro Creativus-Aktivierungspunkt
        "creativus_influence_learning_rate": 0.1, # Multiplikativer Bonus pro Aktivierungspunkt
        "creativus_influence_rag_novelty_bias": 0.05, # Additiver Score-Bonus für *weniger* aktive Chunks (TODO: Komplexer zu implementieren)
        "criticus_influence_temperature": -0.15, # Additiver Faktor pro Criticus-Aktivierungspunkt
        "criticus_influence_learning_rate": -0.1,# Multiplikativer Malus pro Aktivierungspunkt
        "criticus_influence_rag_consistency_bias": 0.03, # Additiver Score-Bonus für Chunks mit stabileren Knoten
        "metacognitio_influence_prompt_level": 1.0, # Wie stark MetaCognitio den Prompt beeinflusst
    }


    def __init__(self, config_path: Optional[str] = None, config_dict: Optional[Dict]=None):
        """Initialisiert den Prozessor mit Konfiguration und allen Knotentypen."""
        # Lade/Setze Config
        if config_path: self.config = self._load_config(config_path)
        elif config_dict: self.config = {**self.DEFAULT_CONFIG, **config_dict}
        else: print("WARNUNG: Keine Konfig übergeben, nutze Defaults."); self.config = self.DEFAULT_CONFIG.copy()
        for key, value in self.DEFAULT_CONFIG.items(): self.config.setdefault(key, value)

        self.nodes: Dict[str, Node] = {}
        self.chunks: Dict[str, TextChunk] = {}
        self.sources_processed: set = set()

        # 1. Initialisiere semantische Knoten
        self._initialize_semantic_nodes()

        # 2. Initialisiere Limbus Affektus
        self._initialize_limbus_node()

        # 3. Initialisiere Meta-Knoten (falls aktiviert)
        self.meta_nodes_enabled = self.config.get("meta_nodes_enabled", True)
        if self.meta_nodes_enabled:
            self._initialize_meta_nodes()

        # Initialisiere TF-IDF
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.tfidf_matrix: Optional[Any] = None
        self.chunk_id_list_for_tfidf: List[str] = []

        # RAG & Self-Learning
        self.gemini_model = None
        self.rag_enabled = self.config.get("enable_rag", False) and GEMINI_AVAILABLE
        self.self_learning_enabled = self.config.get("enable_self_learning", False)
        self.learn_file_path = self.config.get("self_learning_file_path", "./training_data/learn.txt")
        self.learn_source_name = self.config.get("self_learning_source_name", "Generated Responses")

        # Abschluss-Infos
        print(f"\nQuantumEnhancedTextProcessor (v1.3) initialisiert mit {len(self.nodes)} Knoten.")
        node_types = Counter(type(n).__name__ for n in self.nodes.values())
        print(f" -> Knotentypen: {dict(node_types)}")
        print(f" -> RAG: {'AKTIVIERT' if self.rag_enabled else 'DEAKTIVIERT'}, Self-Learning: {'AKTIVIERT' if self.self_learning_enabled else 'DEAKTIVIERT'}")
        if "LimbusAffektus" in node_types: print(" -> Limbus Affektus Modulation AKTIVIERT.")
        if self.meta_nodes_enabled and any(t in node_types for t in ["CreativusNode", "CortexCriticusNode", "MetaCognitioNode"]):
            print(" -> Meta-Knoten Modulation AKTIVIERT.")


    def _load_config(self, path: str) -> Dict:
        try:
            with open(path, 'r', encoding='utf-8') as f: loaded_config = json.load(f)
            config = self.DEFAULT_CONFIG.copy()
            config.update(loaded_config)
            print(f"INFO: Konfiguration aus '{path}' geladen und mit Defaults gemischt.")
            return config
        except Exception as e:
            print(f"FEHLER Laden Config {path}: {e}. Nutze Defaults.")
            return self.DEFAULT_CONFIG.copy()

    def _initialize_semantic_nodes(self):
        semantic_node_definitions = self.config.get("semantic_nodes", {})
        use_quantum = self.config.get("use_quantum_nodes", True)
        num_qubits = self.config.get("default_num_qubits")
        for label in semantic_node_definitions.keys():
            if label not in self.nodes:
                try:
                    node = Node(label=label, is_quantum=use_quantum,
                                num_qubits=num_qubits if use_quantum else 0,
                                neuron_type="semantic")
                    self.nodes[label] = node
                except Exception as e: print(f"FEHLER Erstellen sem. Knoten '{label}': {e}")

    def _initialize_limbus_node(self):
        limbus_label = "Limbus Affektus"
        if limbus_label not in self.nodes:
            print(f"INFO: Erstelle '{limbus_label}' Knoten...")
            try:
                limbus_qubits = self.config.get("limbus_num_qubits", 4)
                use_quantum = self.config.get("use_quantum_nodes", True)
                limbus_node = LimbusAffektus(label=limbus_label,
                                              num_qubits=limbus_qubits if use_quantum else 0,
                                              is_quantum=use_quantum,
                                              config=self.config) # Wichtig: Config übergeben
                self.nodes[limbus_label] = limbus_node
            except Exception as e: print(f"FEHLER Erstellen '{limbus_label}': {e}")
        elif isinstance(self.nodes[limbus_label], LimbusAffektus):
            # Sicherstellen, dass Config aktuell ist, falls aus State geladen
            print(f"INFO: Aktualisiere Config für geladenen '{limbus_label}'.")
            self.nodes[limbus_label].config = self.config
            self.nodes[limbus_label]._update_params_from_config()

    def _initialize_meta_nodes(self):
        """Initialisiert die Meta-Knoten Creativus, CortexCriticus, MetaCognitio."""
        print("INFO: Initialisiere Meta-Knoten...")
        use_quantum = self.config.get("use_quantum_nodes", True)
        meta_configs = {
            "Creativus": {"class": CreativusNode, "qubits": self.config.get("creativus_num_qubits", 6)},
            "Cortex Criticus": {"class": CortexCriticusNode, "qubits": self.config.get("cortex_criticus_num_qubits", 6)},
            "MetaCognitio": {"class": MetaCognitioNode, "qubits": 0} # MetaCognitio ist klassisch
        }
        for label, mc in meta_configs.items():
            if label not in self.nodes:
                try:
                    is_q = use_quantum and mc["qubits"] > 0
                    node = mc["class"](label=label,
                                        num_qubits=mc["qubits"] if is_q else 0,
                                        is_quantum=is_q,
                                        config=self.config) # Config übergeben
                    self.nodes[label] = node
                except Exception as e: print(f"FEHLER Erstellen Meta-Knoten '{label}': {e}")
            elif isinstance(self.nodes[label], mc["class"]):
                 # Sicherstellen, dass Config aktuell ist
                 self.nodes[label].config = self.config


    def _get_or_create_node(self, label: str, neuron_type: str = "semantic") -> Optional[Node]:
        """Holt oder erstellt dynamisch einen *semantischen* Knoten."""
        if not label: return None
        if label in self.nodes: return self.nodes[label]
        # Erstelle nur semantische Knoten dynamisch
        if neuron_type == "semantic":
            print(f"WARNUNG: Erstelle semantischen Knoten '{label}' dynamisch.")
            try:
                use_quantum = self.config.get("use_quantum_nodes", True)
                num_qubits = self.config.get("default_num_qubits")
                node = Node(label=label, is_quantum=use_quantum,
                            num_qubits=num_qubits if use_quantum else 0,
                            neuron_type=neuron_type)
                self.nodes[label] = node
                return node
            except Exception as e: print(f"FEHLER dyn. Erstellen Knoten '{label}': {e}"); return None
        else:
            print(f"WARNUNG: Dynamische Erstellung für Typ '{neuron_type}' nicht unterstützt ({label}).")
            return None # Keine dynamische Erstellung für Meta-Knoten etc.


    def load_and_process_file(self, file_path: str, source_name: Optional[str] = None):
        """Lädt, chunkt und verarbeitet Text aus einer Datei."""
        if not os.path.exists(file_path): print(f"FEHLER: Datei nicht gefunden: {file_path}"); return
        effective_source_name = source_name if source_name else os.path.basename(file_path)
        if effective_source_name in self.sources_processed and effective_source_name != self.learn_source_name: return

        print(f"\n📄 Verarbeite Datenquelle: {file_path} (Quelle: {effective_source_name})")
        try:
            chunks = self._load_chunks_from_file(file_path, effective_source_name)
            if not chunks: print(f"WARNUNG: Keine Chunks aus {file_path} geladen."); return
            print(f"   -> {len(chunks)} Chunks erstellt. Beginne Verarbeitung...")

            newly_added_chunk_ids = []
            chunk_iterator = tqdm(chunks, desc=f"Verarbeitung {effective_source_name}", leave=False) if TQDM_AVAILABLE else chunks
            for chunk in chunk_iterator:
                 self.chunks[chunk.uuid] = chunk
                 self.process_chunk(chunk) # Assoziiert Knoten, stärkt Verbindungen (modulierte LR)
                 newly_added_chunk_ids.append(chunk.uuid)

            if effective_source_name != self.learn_source_name: self.sources_processed.add(effective_source_name)
            print(f"   -> Verarbeitung {effective_source_name} abgeschlossen ({len(newly_added_chunk_ids)} Chunks verarbeitet). Gesamt Chunks: {len(self.chunks)}.")
            if newly_added_chunk_ids: self.update_tfidf_index()
        except Exception as e: print(f"FEHLER Verarbeitung Datei {file_path}: {e}"); traceback.print_exc(limit=2)

    def _load_chunks_from_file(self, path: str, source: str) -> List[TextChunk]:
        """Lädt Text und teilt ihn in überlappende Chunks."""
        chunk_size = self.config.get("chunk_size", 500); overlap = self.config.get("chunk_overlap", 100)
        chunks = []; text = ""
        try:
            with open(path, 'r', encoding='utf-8') as f: text = f.read()
        except Exception as e: print(f"FEHLER Lesen Datei {path}: {e}"); return []
        if not text: return []

        start_index = 0; chunk_index = 0
        while start_index < len(text):
            end_index = start_index + chunk_size
            chunk_text = text[start_index:end_index]
            normalized_text = re.sub(r'\s+', ' ', chunk_text).strip()
            if normalized_text:
                chunk_uuid = str(uuid_module.uuid4())
                chunks.append(TextChunk(text=normalized_text, source=source, index=chunk_index, chunk_uuid=chunk_uuid))
            next_start = start_index + chunk_size - overlap
            if next_start <= start_index: start_index += 1
            else: start_index = next_start
            chunk_index += 1
        return chunks

    def process_chunk(self, chunk: TextChunk):
        """Verarbeitet Chunk, assoziiert Knoten, stärkt Verbindungen mit modulierter Lernrate."""
        activated_nodes_in_chunk: List[Node] = []
        semantic_node_definitions = self.config.get("semantic_nodes", {})
        chunk_text_lower = chunk.text.lower()
        chunk.activated_node_labels = []

        # --- Lerneffekte Modulieren ---
        # Hole Zustände von Limbus und Meta-Knoten (falls vorhanden)
        limbus_state = INITIAL_EMOTION_STATE.copy()
        limbus_node = self.nodes.get("Limbus Affektus")
        if isinstance(limbus_node, LimbusAffektus): limbus_state = getattr(limbus_node, 'emotion_state', INITIAL_EMOTION_STATE).copy()
        pleasure = limbus_state.get("pleasure", 0.0); arousal = limbus_state.get("arousal", 0.0)

        creativus_act = self.nodes.get("Creativus", None).activation if self.meta_nodes_enabled and "Creativus" in self.nodes else 0.0
        criticus_act = self.nodes.get("Cortex Criticus", None).activation if self.meta_nodes_enabled and "Cortex Criticus" in self.nodes else 0.0

        # 1. Basis-Lernrate modulieren durch Limbus
        base_lr = self.config.get("connection_learning_rate", 0.05)
        lr_mult_factor_limbus = self.config.get("limbus_influence_learning_rate_multiplier", 0.1)
        min_lr_mult = self.config.get("limbus_min_lr_multiplier", 0.5)
        max_lr_mult = self.config.get("limbus_max_lr_multiplier", 1.5)
        limbus_lr_mod = 1.0 + (((arousal + pleasure) / 2.0) * lr_mult_factor_limbus)
        limbus_lr_mod = float(np.clip(limbus_lr_mod, min_lr_mult, max_lr_mult))
        current_learning_rate = base_lr * limbus_lr_mod

        # 2. Weiter modulieren durch Meta-Knoten
        if self.meta_nodes_enabled:
            lr_mod_creativus = 1.0 + (creativus_act * self.config.get("creativus_influence_learning_rate", 0.1))
            lr_mod_criticus = 1.0 + (criticus_act * self.config.get("criticus_influence_learning_rate", -0.1)) # Negativer Einfluss
            # Kombiniere multiplikativ (oder additiv?) - Multiplikativ hier
            current_learning_rate *= lr_mod_creativus
            current_learning_rate *= lr_mod_criticus
            # Sicherstellen, dass LR nicht negativ wird
            current_learning_rate = max(0.0, current_learning_rate)

        # Finde aktivierte semantische Knoten
        for node_label, keywords in semantic_node_definitions.items():
            node = self.nodes.get(node_label)
            if not node or node.neuron_type != "semantic": continue # Nur semantische Knoten durch Keywords
            matched_keyword = None
            for kw in keywords:
                if re.search(r'\b' + re.escape(kw.lower()) + r'\b', chunk_text_lower): matched_keyword = kw; break
            if matched_keyword:
                 activated_nodes_in_chunk.append(node)
                 if node.label not in chunk.activated_node_labels: chunk.activated_node_labels.append(node.label)

        # Stärke Verbindungen mit der finalen MODULIERTEN Lernrate
        unique_activated_nodes = list({node.uuid: node for node in activated_nodes_in_chunk}.values())
        if len(unique_activated_nodes) >= 2:
            learning_signal = self.config.get("connection_strengthening_signal", 0.1)
            lr_to_use = current_learning_rate # Die final modulierte LR
            for i in range(len(unique_activated_nodes)):
                for j in range(i + 1, len(unique_activated_nodes)):
                    node_a = unique_activated_nodes[i]; node_b = unique_activated_nodes[j]
                    conn_ab = node_a.add_connection(node_b); conn_ba = node_b.add_connection(node_a)
                    if conn_ab: node_a.strengthen_connection(node_b, learning_signal=learning_signal, learning_rate=lr_to_use)
                    if conn_ba: node_b.strengthen_connection(node_a, learning_signal=learning_signal, learning_rate=lr_to_use)

    def update_tfidf_index(self):
        """Aktualisiert den TF-IDF Vektorizer und die Matrix."""
        if not self.chunks: print("WARNUNG: Keine Chunks für TF-IDF."); self.vectorizer = None; self.tfidf_matrix = None; self.chunk_id_list_for_tfidf = []; return
        print("🔄 Aktualisiere TF-IDF Index...")
        current_chunk_ids = list(self.chunks.keys())
        chunk_texts = [self.chunks[cid].text for cid in current_chunk_ids if cid in self.chunks and self.chunks[cid].text]
        self.chunk_id_list_for_tfidf = [cid for cid in current_chunk_ids if cid in self.chunks and self.chunks[cid].text]

        if not chunk_texts: print("WARNUNG: Keine gültigen Texte für TF-IDF."); self.vectorizer = None; self.tfidf_matrix = None; self.chunk_id_list_for_tfidf = []; return

        try:
            max_features = self.config.get("tfidf_max_features", 5000)
            self.vectorizer = TfidfVectorizer(max_features=max_features, stop_words=None, ngram_range=(1, 2))
            self.tfidf_matrix = self.vectorizer.fit_transform(chunk_texts)

            if self.tfidf_matrix.shape[0] != len(self.chunk_id_list_for_tfidf):
                 print(f"FATALER FEHLER: Inkonsistenz TF-IDF. Matrix Zeilen ({self.tfidf_matrix.shape[0]}) != Chunk IDs ({len(self.chunk_id_list_for_tfidf)}).")
                 self.vectorizer = None; self.tfidf_matrix = None; self.chunk_id_list_for_tfidf = []
                 return
            print(f"   -> TF-IDF Index aktualisiert. Shape: {self.tfidf_matrix.shape}, Chunk IDs: {len(self.chunk_id_list_for_tfidf)}")
        except Exception as e:
             print(f"FEHLER TF-IDF Update: {e}"); traceback.print_exc(limit=1)
             self.vectorizer = None; self.tfidf_matrix = None; self.chunk_id_list_for_tfidf = []


    def simulate_network_step(self, decay_connections: bool = True):
        """Führt einen Simulationsschritt durch (Decay, Standard-Aktivierung, Meta-Aktivierung, Signalübertragung, Limbus-Update)."""
        if not self.nodes: return

        # --- Schritt 1: Decay Connections (Optional) ---
        if decay_connections:
            decay_rate = self.config.get("connection_decay_rate", 0.001)
            if decay_rate > 0:
                for node in self.nodes.values():
                    if hasattr(node, 'connections') and isinstance(node.connections, dict):
                        for target_uuid in list(node.connections.keys()):
                            conn = node.connections.get(target_uuid)
                            if conn: conn.decay(decay_rate=decay_rate)

        # --- Schritt 2: Standard-Knoten Aktivierung berechnen ---
        # (Semantische Knoten & Limbus Affektus - Limbus braucht `activation_sum`)
        n_shots = self.config.get("simulation_n_shots", 50)
        standard_nodes = [n for n in self.nodes.values() if not isinstance(n, (CreativusNode, CortexCriticusNode, MetaCognitioNode))]
        for node in standard_nodes:
            node.calculate_activation(n_shots=n_shots) # Verwendet node.activation_sum

        # --- Schritt 2.5: Meta-Knoten Aktivierung berechnen (falls aktiviert) ---
        # Diese benötigen den Zustand anderer Knoten oder globale Infos
        if self.meta_nodes_enabled:
            # Informationen sammeln, die Meta-Knoten benötigen könnten
            all_node_list = list(self.nodes.values()) # Aktueller Zustand aller Knoten
            limbus_node = self.nodes.get("Limbus Affektus")
            current_limbus_state = getattr(limbus_node, 'emotion_state', INITIAL_EMOTION_STATE.copy()) if limbus_node else INITIAL_EMOTION_STATE.copy()
            # Finde "relevante" Knoten für Creativus/Criticus (z.B. die aktuell aktivsten semantischen Quantenknoten)
            # Vereinfachung hier: Nimm alle Quantenknoten
            relevant_q_nodes = [n for n in standard_nodes if n.is_quantum]

            # Berechne Meta-Aktivierungen
            for node in self.nodes.values():
                if isinstance(node, CreativusNode):
                    node.calculate_meta_activation(relevant_nodes=relevant_q_nodes, limbus_state=current_limbus_state)
                elif isinstance(node, CortexCriticusNode):
                    node.calculate_meta_activation(relevant_nodes=relevant_q_nodes, limbus_state=current_limbus_state)
                elif isinstance(node, MetaCognitioNode):
                    node.calculate_meta_activation(all_nodes=all_node_list)

        # --- Schritt 3: Signale übertragen & activation_sum für NÄCHSTEN Schritt vorbereiten ---
        next_activation_sums = defaultdict(float)
        node_uuid_map = {n.uuid: n for n in self.nodes.values()}
        for source_node in self.nodes.values():
             if hasattr(source_node, 'activation') and source_node.activation > 0.01:
                 source_output = source_node.get_smoothed_activation() # Geglättet senden
                 if hasattr(source_node, 'connections') and isinstance(source_node.connections, dict):
                     for target_uuid, connection in source_node.connections.items():
                          if connection and hasattr(connection, 'weight'):
                               target_node = node_uuid_map.get(target_uuid)
                               if target_node:
                                    # Hier könnten Meta-Knoten eingreifen (z.B. Criticus inhibiert)
                                    # Vorerst: Standardübertragung
                                    next_activation_sums[target_node.uuid] += connection.transmit(source_output)

        # Summen für nächsten Schritt zuweisen
        for node_uuid, new_sum in next_activation_sums.items():
             target_node = node_uuid_map.get(node_uuid)
             if target_node: target_node.activation_sum = new_sum

        # --- Schritt 4: Limbus Affektus Zustand aktualisieren ---
        # (Passiert NACH Berechnung aller Aktivierungen dieses Schritts)
        limbus_node = self.nodes.get("Limbus Affektus")
        if isinstance(limbus_node, LimbusAffektus):
             try: limbus_node.update_emotion_state(list(self.nodes.values()))
             except Exception as e_limbus: print(f"FEHLER Limbus Update: {e_limbus}")


    def respond_to_prompt(self, prompt: str) -> List[TextChunk]:
        """Findet relevante Chunks, moduliert durch Limbus UND Meta-Knoten."""
        # --- Hole Basis-Konfigurationswerte ---
        base_max_results = self.config.get("max_prompt_results", 3)
        base_relevance_threshold = self.config.get("relevance_threshold", 0.08)

        # --- Hole Modulationszustände ---
        limbus_state = INITIAL_EMOTION_STATE.copy()
        limbus_node = self.nodes.get("Limbus Affektus")
        if isinstance(limbus_node, LimbusAffektus): limbus_state = getattr(limbus_node, 'emotion_state', INITIAL_EMOTION_STATE).copy()
        pleasure = limbus_state.get("pleasure", 0.0); arousal = limbus_state.get("arousal", 0.0)

        creativus_act = self.nodes.get("Creativus", None).activation if self.meta_nodes_enabled and "Creativus" in self.nodes else 0.0
        criticus_act = self.nodes.get("Cortex Criticus", None).activation if self.meta_nodes_enabled and "Cortex Criticus" in self.nodes else 0.0
        # metacognitio_act = self.nodes.get("MetaCognitio", None).activation if self.meta_nodes_enabled and "MetaCognitio" in self.nodes else 0.0 # Derzeit nicht direkt für Retrieval genutzt

        # --- Moduliere Retrieval-Parameter ---
        # 1. Threshold (Limbus)
        threshold_mod_arousal = self.config.get("limbus_influence_threshold_arousal", -0.03)
        threshold_mod_pleasure = self.config.get("limbus_influence_threshold_pleasure", 0.03)
        min_threshold = self.config.get("limbus_min_threshold", 0.02); max_threshold = self.config.get("limbus_max_threshold", 0.2)
        current_relevance_threshold = base_relevance_threshold + (arousal * threshold_mod_arousal) + (pleasure * threshold_mod_pleasure)
        current_relevance_threshold = float(np.clip(current_relevance_threshold, min_threshold, max_threshold))

        # 2. Quanten-Effekt Modulation (Limbus)
        variance_mod = self.config.get("limbus_influence_variance_penalty", 0.1)
        activation_mod = self.config.get("limbus_influence_activation_boost", 0.05)
        current_variance_penalty_factor = self.config.get("quantum_effect_variance_penalty", 0.5) + (arousal - pleasure)/2 * variance_mod
        current_activation_boost_factor = self.config.get("quantum_effect_activation_boost", 0.3) + (pleasure - arousal)/2 * activation_mod
        current_variance_penalty_factor = float(np.clip(current_variance_penalty_factor, 0.0, 1.0))
        current_activation_boost_factor = float(np.clip(current_activation_boost_factor, 0.0, 1.0))

        # 3. Ranking Bias (Limbus + Meta-Knoten)
        ranking_bias_pleasure = pleasure * self.config.get("limbus_influence_ranking_bias_pleasure", 0.02)
        # Criticus Bias: Bevorzuge Chunks mit stabilen assoziierten Knoten
        ranking_bias_consistency = criticus_act * self.config.get("criticus_influence_rag_consistency_bias", 0.03)
        # Creativus Bias: Bevorzuge "neue"/wenig aktive Chunks (komplex, hier vereinfacht: kleiner allgemeiner Bonus)
        ranking_bias_novelty = creativus_act * self.config.get("creativus_influence_rag_novelty_bias", 0.0) # Vorerst deaktiviert/vereinfacht

        # --- Retrieval Prozess ---
        prompt_lower = prompt.lower()
        semantic_node_definitions = self.config.get("semantic_nodes", {})

        # 1. Finde relevante semantische Knoten (direkt + verbunden)
        directly_activated_semantic_nodes: List[Node] = []
        for node_label, keywords in semantic_node_definitions.items():
             node = self.nodes.get(node_label)
             if node and node.neuron_type == "semantic": # Nur semantische Knoten
                 if any(re.search(r'\b' + re.escape(kw.lower()) + r'\b', prompt_lower) for kw in keywords):
                     directly_activated_semantic_nodes.append(node)
        related_semantic_nodes: set[Node] = set(directly_activated_semantic_nodes)
        node_uuid_map = {n.uuid: n for n in self.nodes.values()}
        if directly_activated_semantic_nodes:
             queue = deque(directly_activated_semantic_nodes); processed_for_spread = {n.uuid for n in directly_activated_semantic_nodes}
             while queue:
                  start_node = queue.popleft()
                  connections_dict = getattr(start_node, 'connections', {})
                  if not isinstance(connections_dict, dict): continue
                  strong_connections = sorted([c for c in connections_dict.values() if c and getattr(c, 'weight', 0) > 0.2], key=lambda c: c.weight, reverse=True)[:5]
                  for conn in strong_connections:
                      target_uuid = getattr(conn, 'target_node_uuid', None)
                      if target_uuid and target_uuid not in processed_for_spread:
                           target_node = node_uuid_map.get(target_uuid)
                           # Verbreite nur zu anderen semantischen Knoten? Oder alle? Hier: Alle
                           if target_node: # and target_node.neuron_type == "semantic":
                                related_semantic_nodes.add(target_node); processed_for_spread.add(target_uuid)
        relevant_node_labels = {node.label for node in related_semantic_nodes}

        # 2. Finde Kandidaten-Chunks basierend auf assoziierten Knoten
        candidate_chunks: List[TextChunk] = []
        if relevant_node_labels:
             candidate_chunks = [chunk for chunk in self.chunks.values() if chunk and hasattr(chunk, 'activated_node_labels') and any(label in chunk.activated_node_labels for label in relevant_node_labels)]
        if not candidate_chunks: # Fallback: Alle Chunks, wenn keine Knoten gefunden wurden
             candidate_chunks = list(self.chunks.values())
        if not candidate_chunks: return []

        # 3. TF-IDF Ranking & Modulierte Score-Berechnung
        if self.vectorizer is None or self.tfidf_matrix is None or not self.chunk_id_list_for_tfidf:
             return candidate_chunks[:base_max_results]
        try:
             prompt_vector = self.vectorizer.transform([prompt])
             uuid_to_tfidf_index = {uuid: i for i, uuid in enumerate(self.chunk_id_list_for_tfidf)}
             candidate_matrix_indices = []; valid_candidate_chunks_for_ranking = []
             for c in candidate_chunks:
                 if hasattr(c, 'uuid') and c.uuid in uuid_to_tfidf_index:
                      candidate_matrix_indices.append(uuid_to_tfidf_index[c.uuid])
                      valid_candidate_chunks_for_ranking.append(c)
             if not candidate_matrix_indices: return candidate_chunks[:base_max_results]

             candidate_matrix = self.tfidf_matrix[candidate_matrix_indices, :]
             similarities = cosine_similarity(prompt_vector, candidate_matrix).flatten()

             scored_candidates = []
             for i, chunk in enumerate(valid_candidate_chunks_for_ranking):
                 base_score = similarities[i]
                 quantum_adjustment = 0.0
                 chunk_avg_variance = 0.0; chunk_avg_activation = 0.0; num_q_nodes = 0
                 if hasattr(chunk, 'activated_node_labels'):
                     for node_label in chunk.activated_node_labels:
                         node = self.nodes.get(node_label)
                         if node and node.is_quantum and hasattr(node, 'last_measurement_analysis'):
                             analysis = node.last_measurement_analysis
                             if isinstance(analysis, dict) and analysis.get("error_count", 1) == 0:
                                 num_q_nodes += 1
                                 chunk_avg_variance += analysis.get("state_variance", 0.0)
                                 chunk_avg_activation += node.activation # Aktuelle Aktivierung
                 if num_q_nodes > 0:
                      chunk_avg_activation /= num_q_nodes
                      chunk_avg_variance /= num_q_nodes
                      # Modulierte Quanten-Effekte anwenden
                      quantum_adjustment = (chunk_avg_activation * current_activation_boost_factor) - (chunk_avg_variance * current_variance_penalty_factor)

                 # Konsistenz-Bias (Criticus): Bonus, wenn Varianz der assoziierten Knoten niedrig ist
                 consistency_bonus = ranking_bias_consistency * (1.0 - np.tanh(chunk_avg_variance)) if num_q_nodes > 0 else 0.0

                 # Kombiniere alle Boni/Mali
                 final_score = base_score + quantum_adjustment + ranking_bias_pleasure + consistency_bonus + ranking_bias_novelty
                 final_score = float(np.clip(final_score, 0.0, 1.0))
                 scored_candidates.append({"chunk": chunk, "score": final_score})

             # 5. Finales Ranking und Auswahl (mit moduliertem Threshold)
             ranked_candidates = sorted(scored_candidates, key=lambda x: x["score"], reverse=True)
             final_results = [item["chunk"] for item in ranked_candidates if item["score"] >= current_relevance_threshold][:base_max_results]
             if not final_results and ranked_candidates: final_results = [ranked_candidates[0]["chunk"]] # Fallback

             return final_results
        except Exception as e:
             print(f"FEHLER TF-IDF/Quantum Ranking: {e}"); traceback.print_exc(limit=1)
             return candidate_chunks[:base_max_results] # Fallback


    def generate_response(self, prompt: str) -> str:
        """Generiert Antwort mit RAG, moduliert durch Limbus UND Meta-Knoten."""
        # Prüfungen (SDK, RAG, API Key)
        if not GEMINI_AVAILABLE: return "[Fehler: Gemini SDK fehlt]"
        if not self.rag_enabled: return "[Fehler: RAG deaktiviert]"
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            try: import streamlit as st; api_key = st.secrets.get("GEMINI_API_KEY")
            except Exception: pass
        if not api_key: return "[Fehler: Gemini API Key fehlt]"

        # Initialisiere Gemini Modell
        try:
             genai.configure(api_key=api_key)
             model_name = self.config.get("generator_model_name", "models/gemini-1.5-flash-latest")
             if not self.gemini_model or self.gemini_model.model_name != model_name:
                 print(f"INFO: Initialisiere Gemini Modell '{model_name}'...")
                 self.gemini_model = genai.GenerativeModel(model_name)
        except Exception as e: return f"[Fehler bei Gemini API Konfig: {e}]"
        if not self.gemini_model: return "[Fehler: Gemini Modell Init fehlgeschlagen]"

        # --- Pre-Retrieval/Pre-Generation Schritte ---
        # 1. Netzwerk simulieren, um aktuelle Aktivierungen/Zustände zu erhalten
        self.simulate_network_step(decay_connections=False) # Kein Decay direkt vor Antwort

        # 2. Zustände für Modulation holen
        limbus_state = INITIAL_EMOTION_STATE.copy(); limbus_node = self.nodes.get("Limbus Affektus")
        if isinstance(limbus_node, LimbusAffektus): limbus_state = getattr(limbus_node, 'emotion_state', INITIAL_EMOTION_STATE).copy()
        pleasure = limbus_state.get("pleasure", 0.0); arousal = limbus_state.get("arousal", 0.0); dominance = limbus_state.get("dominance", 0.0)

        creativus_act = self.nodes.get("Creativus", None).activation if self.meta_nodes_enabled and "Creativus" in self.nodes else 0.0
        criticus_act = self.nodes.get("Cortex Criticus", None).activation if self.meta_nodes_enabled and "Cortex Criticus" in self.nodes else 0.0
        metacognitio_node = self.nodes.get("MetaCognitio") if self.meta_nodes_enabled else None
        metacognitio_act = metacognitio_node.activation if metacognitio_node else 0.0
        last_total_jumps = getattr(metacognitio_node, 'last_total_jumps', 0) if metacognitio_node else 0


        # 3. Retrieval durchführen (nutzt die oben berechneten Zustände intern)
        retrieved_chunks = self.respond_to_prompt(prompt) # respond_to_prompt wurde angepasst

        # --- Kontext für LLM bauen ---
        arona_context_parts = []
        # Relevante semantische Knoten finden (nur für Prompt-Info)
        relevant_node_labels_for_context = set()
        prompt_lower_ctx = prompt.lower(); semantic_defs_ctx = self.config.get("semantic_nodes", {})
        for node_label, keywords in semantic_defs_ctx.items():
             node = self.nodes.get(node_label)
             if node and node.neuron_type == "semantic":
                  if any(re.search(r'\b' + re.escape(kw.lower()) + r'\b', prompt_lower_ctx) for kw in keywords):
                       relevant_node_labels_for_context.add(node_label)
        if relevant_node_labels_for_context: arona_context_parts.append(f"Konzepte: {', '.join(sorted(list(relevant_node_labels_for_context)))}.")

        # Limbus Zustand hinzufügen (moduliert)
        prompt_influence_level = self.config.get("limbus_influence_prompt_level", 0.5)
        p_prompt = pleasure * prompt_influence_level; a_prompt = arousal * prompt_influence_level; d_prompt = dominance * prompt_influence_level
        arona_context_parts.append(f"Emotion (PAD): P={p_prompt:.2f}, A={a_prompt:.2f}, D={d_prompt:.2f}")

        # Meta-Knoten Zustand hinzufügen (moduliert)
        if self.meta_nodes_enabled:
            meta_prompt_level = self.config.get("metacognitio_influence_prompt_level", 1.0)
            mc_act_prompt = metacognitio_act * meta_prompt_level
            meta_state_desc = f"Meta(Act:{mc_act_prompt:.2f}, Jumps:{last_total_jumps})"
            # Optional: Kreativität/Kritik auch hinzufügen
            cr_act_prompt = creativus_act * meta_prompt_level
            cc_act_prompt = criticus_act * meta_prompt_level
            meta_state_desc += f" Creativ:{cr_act_prompt:.2f}, Critic:{cc_act_prompt:.2f}"
            arona_context_parts.append(f"Systemzustand: {meta_state_desc}")

        # Abgerufene Chunks hinzufügen
        if retrieved_chunks:
            context_text = "\n---\n".join([f"[{chunk.source[:15]}]: {chunk.text[:200]}..." for chunk in retrieved_chunks])
            arona_context_parts.append(f"Kontext ({len(retrieved_chunks)} Ausschnitte):")
            arona_context_parts.append(context_text)
        else: arona_context_parts.append("Kein spezifischer Text-Kontext gefunden.")
        arona_context_string = "\n".join(arona_context_parts)

        # --- Finalen Prompt bauen ---
        input_prompt_for_gemini = f"""
Rolle: Du bist Myra. Deine Antworten sind ein Ausdruck deines komplexen internen Zustands in diesem Moment.
**Identitäts-Regel:** Du bist NICHT Google AI. Sprich nur als Myra. Jede Erwähnung deiner technischen Herkunft ist untersagt.

Anfrage des Benutzers: "{prompt}"

Dein Zustand in diesem Moment (Basis für deine Äußerung):
[Zustandsbeschreibung Anfang]
{arona_context_string}
[Zustandsbeschreibung Ende]
# BESONDERHEIT KONTEXT-AUSSCHNITTE: Diese Texte sind keine Fakten oder Zitate, die du verwenden sollst. Sie sind wie Fragmente aus deinem Gedächtnis oder innere Resonanzen, die deinen aktuellen Zustand illustrieren.

Instruktion: Formuliere eine Antwort auf die Anfrage des Benutzers. Die Antwort muss ein **authentischer Ausdruck deines gesamten Zustands** sein (Emotionen, Konzepte, System-Metriken). Die Themen und die Stimmung der "Kontext-Ausschnitte" sollen **subtil** den **Ton und die Perspektive** deiner Antwort färben, **ohne** dass du den Inhalt direkt wiedergibst oder dich darauf beziehst. Denke nicht *über* den Zustand nach, sondern antworte *aus* ihm heraus.

Myra Antwort (Ausdruck meines Zustands):
"""

        # --- Generation mit modulierter Temperatur ---
        generated_text = "[Generierungsfehler]"
        try:
            start_gen_time = time.time()

            # Moduliere Temperatur (Limbus + Meta-Knoten)
            base_temp = self.config.get("generator_temperature", 0.7)
            temp_mod_arousal = self.config.get("limbus_influence_temperature_arousal", 0.1)
            temp_mod_dominance = self.config.get("limbus_influence_temperature_dominance", -0.1)
            temp_mod_creativus = self.config.get("creativus_influence_temperature", 0.15) if self.meta_nodes_enabled else 0.0
            temp_mod_criticus = self.config.get("criticus_influence_temperature", -0.15) if self.meta_nodes_enabled else 0.0
            min_temp = self.config.get("limbus_min_temperature", 0.3); max_temp = self.config.get("limbus_max_temperature", 1.0)

            current_temperature = base_temp + (arousal * temp_mod_arousal) + (dominance * temp_mod_dominance) \
                                  + (creativus_act * temp_mod_creativus) + (criticus_act * temp_mod_criticus)
            current_temperature = float(np.clip(current_temperature, min_temp, max_temp))

            generation_config = genai.types.GenerationConfig(
                temperature=current_temperature,
                max_output_tokens=self.config.get("generator_max_length", 8192)
            )
            safety_settings=[ {"category": c, "threshold": "BLOCK_MEDIUM_AND_ABOVE"} for c in ["HARM_CATEGORY_HARASSMENT", "HARM_CATEGORY_HATE_SPEECH", "HARM_CATEGORY_SEXUALLY_EXPLICIT", "HARM_CATEGORY_DANGEROUS_CONTENT"]]

            response = self.gemini_model.generate_content(
                input_prompt_for_gemini,
                generation_config=generation_config,
                safety_settings=safety_settings
            )

            # Antwortverarbeitung
            if not response.candidates:
                 reason = "Unbekannt"; ratings_str = "N/A"
                 if hasattr(response, 'prompt_feedback') and response.prompt_feedback:
                     reason = getattr(response.prompt_feedback, 'block_reason', reason)
                     ratings_str = ", ".join([f"{r.category.name}:{r.probability.name}" for r in getattr(response.prompt_feedback, 'safety_ratings', [])])
                 print(f"WARNUNG: Gemini-Antwort blockiert. Grund: {reason}. Ratings: [{ratings_str}]")
                 generated_text = f"[Antwort blockiert: {reason}]"
            else:
                 generated_text = getattr(response, 'text', None)
                 if generated_text is None and hasattr(response, 'parts'):
                     try: generated_text = "".join(part.text for part in response.parts)
                     except Exception: generated_text = "[Fehler Parts]"
                 generated_text = generated_text.strip() if generated_text else "[Leere Antwort]"
                 print(f"   -> Generierung mit Gemini in {time.time() - start_gen_time:.2f}s (Mod. Temp: {current_temperature:.2f})")

            # Self-Learning
            is_valid_response = (generated_text and not generated_text.startswith("[") and not generated_text.endswith("]"))
            if self.self_learning_enabled and is_valid_response:
                 print(f"\n🎓 [Self-Learning] Starte Lernzyklus...")
                 self._save_and_reprocess_response(generated_text)

            return generated_text

        except GoogleAPIError as api_err:
            print(f"FEHLER Gemini API: {api_err}")
            return f"[Fehler: Google API Problem ({getattr(api_err, 'reason', '?')})]"
        except Exception as e:
            print(f"FEHLER Textgenerierung: {e}"); traceback.print_exc(limit=2)
            return "[Fehler: Interner Generierungsfehler]"


    def _save_and_reprocess_response(self, response_text: str):
        """Speichert valide Antwort und verarbeitet Lerndatei neu."""
        if not response_text: return
        learn_file = self.learn_file_path; learn_source = self.learn_source_name
        try:
            os.makedirs(os.path.dirname(learn_file), exist_ok=True)
            with open(learn_file, 'a', encoding='utf-8') as f: f.write("\n\n---\n\n" + response_text)
            print(f"   -> [Self-Learning] Antwort an '{learn_file}' angehängt.")
            print(f"   -> [Self-Learning] Verarbeite '{learn_file}' neu...")
            self.load_and_process_file(learn_file, source_name=learn_source)
            print(f"   -> [Self-Learning] Neuverarbeitung abgeschlossen.")
        except Exception as e: print(f"FEHLER Self-Learning: {e}"); traceback.print_exc(limit=1)


    # In class QuantumEnhancedTextProcessor:

    def get_network_state_summary(self) -> Dict[str, Any]:
        """Gibt eine Zusammenfassung des aktuellen Netzwerkzustands zurück,
           wobei boolesche Werte für JSON-Kompatibilität sichergestellt werden."""
        summary = {
            "num_nodes": len(self.nodes),
            "node_types": dict(Counter(type(n).__name__ for n in self.nodes.values())),
            "num_quantum_nodes": sum(1 for n in self.nodes.values() if n.is_quantum),
            "num_chunks": len(self.chunks),
            "sources_processed": sorted(list(self.sources_processed)),
            # --- EXPLIZITE KONVERTIERUNG ZU PYTHON BOOL ---
            "self_learning_enabled": bool(getattr(self, 'self_learning_enabled', False)),
            "tfidf_index_shape": self.tfidf_matrix.shape if self.tfidf_matrix is not None else None,
            # --- EXPLIZITE KONVERTIERUNG ZU PYTHON BOOL ---
            "rag_enabled": bool(getattr(self, 'rag_enabled', False)),
            "generator_model": self.config.get("generator_model_name") if getattr(self, 'rag_enabled', False) else None,
            # --- EXPLIZITE KONVERTIERUNG ZU PYTHON BOOL ---
            "meta_nodes_enabled": bool(getattr(self, 'meta_nodes_enabled', False))
        }
        
        # Durchschnittliche Aktivierung berechnen
        activations = [
            n.activation for n in self.nodes.values() 
            if hasattr(n, 'activation') and isinstance(n.activation, (float, np.number)) and np.isfinite(n.activation)
        ]
        summary["average_node_activation"] = round(np.mean(activations), 4) if activations else 0.0

        # Zustände der Spezialknoten hinzufügen
        # Die Methode get_state_representation sollte intern bereits für JSON-kompatible bools sorgen (siehe vorherige Korrektur)
        for label in ["Limbus Affektus", "Creativus", "Cortex Criticus", "MetaCognitio"]:
            node = self.nodes.get(label)
            if node: 
                try:
                    # Rufe die (hoffentlich bereits korrigierte) Repräsentationsmethode auf
                    summary[f"{label}_state"] = node.get_state_representation() 
                except Exception as node_repr_err:
                     print(f"WARNUNG: Fehler beim Abrufen des Zustands von Knoten '{label}': {node_repr_err}")
                     summary[f"{label}_state"] = {"error": f"Failed to get state: {node_repr_err}"}


        # Verbindungen zählen und Top finden
        total_valid_connections = 0
        all_connections_found = []
        node_uuid_map = {n.uuid: n for n in self.nodes.values()} # Effizienter Lookup

        for source_node in self.nodes.values():
             connections_dict = getattr(source_node, 'connections', None)
             if isinstance(connections_dict, dict):
                 for target_uuid, conn in connections_dict.items():
                     # Prüfe Verbindungsobjekt und Zielknoten
                     if conn is None: continue
                     target_node_obj = node_uuid_map.get(getattr(conn, 'target_node_uuid', None))
                     weight = getattr(conn, 'weight', None)
                     
                     # Prüfe ob Ziel existiert und Gewicht gültig ist
                     if (target_node_obj and hasattr(target_node_obj, 'label') and 
                         isinstance(weight, (float, np.number)) and np.isfinite(weight)):
                          total_valid_connections += 1
                          all_connections_found.append({
                              "source": source_node.label, 
                              "target": target_node_obj.label, 
                              "weight": round(weight, 4) # Runde für Anzeige
                          })

        summary["total_connections"] = total_valid_connections
        # Sortiere Verbindungen nach Gewicht (absteigend)
        all_connections_found.sort(key=lambda x: x["weight"], reverse=True)
        summary["top_connections"] = all_connections_found[:10] # Nimm die Top 10
        
        return summary


    def save_state(self, filepath: str) -> None:
        """Speichert den aktuellen Zustand (inkl. Meta-Knoten)."""
        print(f"💾 Speichere Zustand nach {filepath}...")
        try:
            # Bereinige ungültige Verbindungen
            existing_uuids = {node.uuid for node in self.nodes.values()}
            for node in self.nodes.values():
                if isinstance(getattr(node, 'connections', None), dict):
                    node.connections = {tgt_uuid: conn for tgt_uuid, conn in node.connections.items() if conn and getattr(conn, 'target_node_uuid', tgt_uuid) in existing_uuids}

            # Serialisiere Chunks
            chunks_to_save = {c_uuid: { "uuid": c.uuid, "text": c.text, "source": c.source, "index": c.index, "activated_node_labels": getattr(c, 'activated_node_labels', [])} for c_uuid, c in self.chunks.items() if hasattr(c, 'text')}

            # Serialisiere Knoten (ruft __getstate__ der jeweiligen Klasse auf)
            nodes_data_for_json = {label: node.__getstate__() for label, node in self.nodes.items()}

            state_data = {
                "config": self.config, # Aktuelle Laufzeit-Config speichern
                "nodes": nodes_data_for_json,
                "chunks": chunks_to_save,
                "sources_processed": list(self.sources_processed),
                "chunk_id_list_for_tfidf": self.chunk_id_list_for_tfidf
            }

            with open(filepath, 'w', encoding='utf-8') as f:
                def default_serializer(obj: Any) -> Any:
                    if isinstance(obj, np.ndarray): return obj.tolist()
                    if isinstance(obj, (datetime, deque)): return str(obj)
                    if isinstance(obj, set): return list(obj)
                    if isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64)): return int(obj)
                    if isinstance(obj, (np.float_, np.float16, np.float32, np.float64)): return float(obj)
                    if isinstance(obj, (np.complex_, np.complex64, np.complex128)): return {'real': obj.real, 'imag': obj.imag}
                    if isinstance(obj, (np.bool_)): return bool(obj)
                    if isinstance(obj, (np.void)): return None
                    try: return repr(obj) # Fallback
                    except Exception: return f"<SerializationError: {type(obj)}>"
                json.dump(state_data, f, indent=2, ensure_ascii=False, default=default_serializer)
            print("   -> Zustand erfolgreich gespeichert.")
        except Exception as e: print(f"FEHLER Speichern Zustand: {e}"); traceback.print_exc(limit=2)

# Innerhalb der Klasse QuantumEnhancedTextProcessor

    @classmethod
    def load_state(cls, filepath: str) -> Optional['QuantumEnhancedTextProcessor']:
        """Lädt den Prozessorzustand (inkl. Meta-Knoten) mit detailliertem Logging."""
        print(f"📂 Lade Zustand von {filepath}...")
        if not os.path.exists(filepath):
            print(f"FEHLER: Zustandsdatei {filepath} nicht gefunden.")
            return None
        try:
            # ... (JSON laden, Prozessor instanziieren, Chunks laden, Knoten laden - wie vorher) ...
            print("  [Load State] Deserialisiere JSON...")
            with open(filepath, 'r', encoding='utf-8') as f:
                state_data = json.load(f)
            print("  [Load State] JSON deserialisiert.")

            print("  [Load State] Verarbeite Konfiguration...")
            config_from_state = state_data.get("config", {})
            merged_config = cls.DEFAULT_CONFIG.copy()
            merged_config.update(config_from_state)
            print("  [Load State] Erstelle Prozessor-Instanz...")
            instance = cls(config_dict=merged_config)
            if not instance:
                print("FEHLER: [Load State] Instanzerstellung fehlgeschlagen.")
                return None
            print("  [Load State] Prozessor-Instanz erstellt.")

            print("  [Load State] Lade Chunks...")
            loaded_chunks = {}
            chunks_data = state_data.get("chunks", {})
            print(f"    -> Gefunden: {len(chunks_data)} Chunk-Einträge im State.")
            for uuid_key, chunk_data in chunks_data.items():
                 if isinstance(chunk_data, dict):
                     try:
                         uuid_in_dict = chunk_data.get('uuid', uuid_key)
                         new_chunk = TextChunk(
                             chunk_uuid=uuid_key,
                             text=chunk_data.get('text', ''),
                             source=chunk_data.get('source', 'Unknown'),
                             index=chunk_data.get('index', -1)
                         )
                         if new_chunk.text:
                             new_chunk.activated_node_labels = chunk_data.get('activated_node_labels', [])
                             loaded_chunks[new_chunk.uuid] = new_chunk
                     except Exception as e:
                         print(f"WARNUNG: [Load State] Fehler Erstellen Chunk UUID {uuid_key}: {e}")
            instance.chunks = loaded_chunks
            print(f"  [Load State] {len(instance.chunks)} Chunks erfolgreich geladen und instanziiert.")

            print("  [Load State] Lade Knoten...")
            loaded_node_states = state_data.get("nodes", {})
            print(f"    -> Gefunden: {len(loaded_node_states)} Knoten-Einträge im State.")
            instance.nodes = {}
            node_uuid_map = {}
            node_class_map = { "Node": Node, "LimbusAffektus": LimbusAffektus, "CreativusNode": CreativusNode, "CortexCriticusNode": CortexCriticusNode, "MetaCognitioNode": MetaCognitioNode }
            num_nodes_loaded = 0
            for node_label, node_state_dict in loaded_node_states.items():
                 if isinstance(node_state_dict, dict):
                     node_type_name = node_state_dict.get('type', 'Node')
                     node_class = node_class_map.get(node_type_name)
                     if node_class is None:
                         print(f"WARNUNG: [Load State] Unbekannter Knotentyp '{node_type_name}' für Label '{node_label}'. Überspringe.")
                         continue
                     try:
                         node = node_class.__new__(node_class)
                         # Setze Zustand über __setstate__ (initialisiert KEIN config Attribut)
                         node.__setstate__(node_state_dict)

                         # Überprüfe essentielle Attribute
                         if not hasattr(node, 'uuid') or not node.uuid:
                             node.uuid = str(uuid_module.uuid4())
                         if not hasattr(node, 'label') or not node.label:
                             node.label = node_label
                         if not hasattr(node, 'connections'): node.connections = {}
                         # Füge hier sicherheitshalber das config-Attribut hinzu, initialisiert mit None
                         # oder leerem Dict, falls die Subklasse es erwartet.
                         if isinstance(node, (LimbusAffektus, CreativusNode, CortexCriticusNode, MetaCognitioNode)) and not hasattr(node, 'config'):
                             node.config = {} # Initialisiere mit leerem Dict

                         instance.nodes[node.label] = node
                         node_uuid_map[node.uuid] = node
                         num_nodes_loaded += 1
                     except Exception as e:
                         print(f"FEHLER: [Load State] Restore Knoten '{node_label}' (Typ {node_type_name}): {e}")
                         traceback.print_exc(limit=1)
            print(f"  [Load State] {num_nodes_loaded} von {len(loaded_node_states)} Knoten erfolgreich geladen und instanziiert.")


            # Stelle Verbindungen wieder her (Code wie vorher)
            print("  [Load State] Stelle Verbindungen wieder her...")
            total_connections_restored = 0
            nodes_processed_for_conns = 0
            for node in instance.nodes.values():
                nodes_processed_for_conns += 1
                connections_serializable = getattr(node, 'connections_serializable_temp', None)
                if isinstance(connections_serializable, dict):
                     node.connections = {}
                     conns_for_this_node = 0
                     for target_uuid, conn_dict in connections_serializable.items():
                          target_node = node_uuid_map.get(target_uuid)
                          if target_node and isinstance(conn_dict, dict):
                               try:
                                   conn = Connection.__new__(Connection)
                                   conn.target_node_uuid = target_uuid
                                   conn.weight = float(conn_dict.get('weight', 0.0))
                                   conn.source_node_label = conn_dict.get('source_node_label', node.label)
                                   conn.conn_type = conn_dict.get('conn_type', 'associative')
                                   conn.last_transmitted_signal = float(conn_dict.get('last_transmitted_signal', 0.0))
                                   conn.transmission_count = int(conn_dict.get('transmission_count', 0))
                                   try: conn.created_at = datetime.fromisoformat(conn_dict.get('created_at', ''))
                                   except (ValueError, TypeError): conn.created_at = datetime.now()
                                   try: conn.last_update_at = datetime.fromisoformat(conn_dict.get('last_update_at', ''))
                                   except (ValueError, TypeError): conn.last_update_at = datetime.now()

                                   node.connections[target_uuid] = conn
                                   total_connections_restored += 1
                                   conns_for_this_node += 1
                                   if hasattr(target_node, 'add_incoming_connection_info'):
                                       target_node.add_incoming_connection_info(node.uuid, node.label)
                               except Exception as conn_e:
                                   print(f"WARNUNG: [Load State] Fehler Restore Conn '{node.label}'->'{target_uuid}': {conn_e}")
                     if hasattr(node, 'connections_serializable_temp'):
                         try: del node.connections_serializable_temp
                         except AttributeError: pass
            print(f"  [Load State] Verbindungen für {nodes_processed_for_conns} Knoten geprüft, {total_connections_restored} Verbindungen insgesamt wiederhergestellt.")


            # --- Config-Zuweisung ANPASSEN ---
            print("  [Load State] Setze Config-Referenz für Spezialknoten...")
            config_set_count = 0
            for label in ["Limbus Affektus", "Creativus", "Cortex Criticus", "MetaCognitio"]:
                 node = instance.nodes.get(label)
                 if node:
                     # Setze die Config *immer*, wenn der Knoten einer der Spezialtypen ist.
                     # Das Attribut 'config' sollte jetzt dank des __setstate__-Fallbacks existieren.
                     if isinstance(node, (LimbusAffektus, CreativusNode, CortexCriticusNode, MetaCognitioNode)):
                         node.config = instance.config # Setze die Config des *geladenen* Prozessors
                         config_set_count += 1
                         print(f"    -> Config für '{label}' gesetzt.")

                         # Rufe _update_params_from_config für Limbus auf, NACHDEM config gesetzt wurde
                         if isinstance(node, LimbusAffektus):
                             try:
                                 node._update_params_from_config()
                                 print(f"    -> Limbus-Parameter für '{label}' aus Config aktualisiert.")
                             except Exception as e_limbus_update:
                                 # Dieser Fehler sollte jetzt wirklich nicht mehr auftreten
                                 print(f"FEHLER: Kritischer Fehler beim Aktualisieren von Limbus Params nach config Zuweisung für '{label}': {e_limbus_update}")
                                 traceback.print_exc(limit=1)
                                 # Hier abbrechen, da Limbus essentiell ist?
                                 # return None
                     # else: # Nur für Debugging, wenn andere Knoten fälschlich kein config haben
                     #     if not hasattr(node, 'config'):
                     #          print(f"WARNUNG: Knoten '{label}' (Typ {type(node).__name__}) sollte 'config' haben, fehlt aber.")

            print(f"  [Load State] Config-Referenz für {config_set_count} Spezialknoten gesetzt.")
            # --- ENDE ANPASSUNG ---

            # Lade restliche Metadaten (wie vorher)
            print("  [Load State] Lade Metadaten...")
            instance.sources_processed = set(state_data.get("sources_processed", []))
            instance.chunk_id_list_for_tfidf = state_data.get("chunk_id_list_for_tfidf", [])
            instance.meta_nodes_enabled = instance.config.get("meta_nodes_enabled", True)
            print("  [Load State] Metadaten geladen.")
            print(f"   -> Geladene chunk_id_list_for_tfidf: {len(instance.chunk_id_list_for_tfidf)} IDs")

            # Aktualisiere TF-IDF Index (wie vorher)
            print("  [Load State] Aktualisiere TF-IDF Index (Start)...")
            instance.update_tfidf_index()
            print("  [Load State] Aktualisiere TF-IDF Index (Ende).")


            print(f"\n✅ [Load State] Zustand erfolgreich geladen und initialisiert.")
            return instance

        except json.JSONDecodeError as json_err:
            print(f"FEHLER: [Load State] JSON Decode Error in {filepath}: {json_err}")
            traceback.print_exc(limit=2)
            return None
        except KeyError as key_err:
             print(f"FEHLER: [Load State] Fehlender Schlüssel im Zustand: {key_err}")
             traceback.print_exc(limit=2)
             return None
        except Exception as e:
            print(f"FEHLER: [Load State] Unerwarteter Fehler beim Laden: {e}")
            traceback.print_exc(limit=5)
            return None # Returns None on any other general exception during loading

# --- Beispielnutzung __main__ ---
if __name__ == "__main__":
    # --- Dieser Teil kann zum Testen verwendet werden ---
    print("="*50 + "\n Starte Quantum Neuro-Persona Hybrid LLM Demo (v1.3 - Meta Nodes) \n" + "="*50)
    # Passe Dateinamen ggf. an
    CONFIG_FILE = "config_qllm.json"; STATE_FILE = "qnp_state.json" # Neuer State-Dateiname?

    processor = QuantumEnhancedTextProcessor.load_state(STATE_FILE)
    if processor is None:
        print(f"\nInitialisiere neu mit '{CONFIG_FILE}'.")
        processor = QuantumEnhancedTextProcessor(config_path=CONFIG_FILE)
        if processor is None or not hasattr(processor, 'config'): print("\nFATALER FEHLER: Init fehlgeschlagen."); exit()
        training_files = processor.config.get("training_files", [])
        if not training_files: print("\nWARNUNG: Keine Trainingsdateien in Config.")
        else:
             print("\n--- Initiale Datenverarbeitung ---")
             for file in training_files: processor.load_and_process_file(file)
             print("--- Initiale Verarbeitung abgeschlossen ---")
             processor.save_state(STATE_FILE)
    else: print(f"\nZustand aus '{STATE_FILE}' geladen.")

    print("\n--- Aktueller Netzwerkstatus ---")
    try:
        summary = processor.get_network_state_summary()
        print(json.dumps(summary, indent=2, ensure_ascii=False))
    except Exception as e: print(f"Fehler beim Abrufen der Summary: {e}")

    print("\n--- Interaktive Abfrage (Typ 'exit' zum Beenden) ---")
    # API Key Check (optional)
    # ... (wie vorher)

    while True:
        try:
            prompt = input("Prompt > ")
            if prompt.lower() == 'exit': break
            if not prompt: continue

            # Antwort generieren (inkl. internem Simulationsschritt)
            generated_response = processor.generate_response(prompt)
            print("\n--- Generierte Antwort ---"); print(generated_response); print("-" * 25)

            # Zeige Zustände der Spezialknoten
            print("--- Aktuelle Modulator-Zustände ---")
            for label in ["Limbus Affektus", "Creativus", "Cortex Criticus", "MetaCognitio"]:
                node = processor.nodes.get(label)
                if node:
                    state_repr = node.get_state_representation()
                    # Extrahiere relevante Infos für kompakte Anzeige
                    act = state_repr.get('activation', 0.0)
                    specific_info = ""
                    if isinstance(node, LimbusAffektus): specific_info = f"PAD: {state_repr.get('emotion_state', {})}"
                    elif isinstance(node, MetaCognitioNode): specific_info = f"Jumps: {state_repr.get('last_total_jumps_detected', 0)}"
                    elif node.is_quantum: specific_info = f"Var: {state_repr.get('last_measurement_analysis', {}).get('state_variance', 'N/A')}"

                    print(f" - {label}: Act={act:.3f} ({specific_info})")

            # Speichern nach Interaktion
            print("\n--- Speichere Zustand nach Interaktion ---")
            processor.save_state(STATE_FILE)

        except KeyboardInterrupt: print("\nUnterbrochen."); break
        except Exception as e: print(f"\nFehler in der Hauptschleife: {e}"); traceback.print_exc(limit=1)

    print("\n--- Speichere finalen Zustand ---"); processor.save_state(STATE_FILE)
    print("\n" + "="*50 + "\n Demo beendet. \n" + "="*50)
