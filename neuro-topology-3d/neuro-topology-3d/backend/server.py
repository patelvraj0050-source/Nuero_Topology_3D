"""
NeuroTopology 3D - High-Performance Streaming Backend & Topological Inference Server
Serves static 3D web client, REST API endpoints, and streaming token-by-token circuits.
Works out of the box with Python 3.11 standard library with optional FastAPI/WebSockets acceleration.
"""

import sys
import os
import json
import time
import math
from http.server import HTTPServer, SimpleHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
from typing import Any, Dict, List
import threading

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

from sae_engine import SAEDictionary, InferenceCircuitSimulator
from topology import TopologicalEngine

# Global singletons
sae_dict = SAEDictionary(num_features=16384)
simulator = InferenceCircuitSimulator(sae_dict)
topo_engine = TopologicalEngine(entropy_threshold=1.45)

FRONTEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "frontend"))

class NeuroTopologyHandler(SimpleHTTPRequestHandler):
    """Handles static 3D assets, API requests, and streaming telemetry."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=FRONTEND_DIR, **kwargs)

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/api/status":
            self.send_json_response({
                "status": "online",
                "engine": "NeuroTopology-3D-v1.0",
                "backbone": "Gemma-2-2B-IT + Gemma Scope SAE (Layer 12)",
                "dictionary_size": 16384,
                "latent_space": "Hyperbolic Poincare Ball (H^3)",
                "tda_metric": "Persistent Homology (VR-Filtration)",
                "active_devices": ["WebGL / WebGPU Client", "Python 3.11 Inference Node"],
                "fps_target": 60
            })
            return

        elif path == "/api/scenarios":
            self.send_json_response(simulator.scenarios)
            return

        elif path == "/api/features":
            # Return subsampled dictionary for inspection
            query = parse_qs(parsed.query)
            limit = int(query.get("limit", [50])[0])
            offset = int(query.get("offset", [0])[0])
            items = sae_dict.features[offset:offset+limit]
            self.send_json_response({"total": len(sae_dict.features), "features": items})
            return

        # Fallback to serving static files from frontend/
        return super().do_GET()

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path

        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length) if content_length > 0 else b"{}"

        try:
            payload = json.loads(body.decode("utf-8"))
        except Exception:
            payload = {}

        if path == "/api/step":
            scenario_key = payload.get("scenario", "math_proof")
            token_idx = int(payload.get("token_index", 0))
            custom_token = payload.get("token", None)

            scenario = simulator.scenarios.get(scenario_key, simulator.scenarios["math_proof"])
            tokens = scenario["tokens"]

            if custom_token:
                current_token = custom_token
            else:
                current_token = tokens[token_idx % len(tokens)]

            # Generate SAE activations for this token step
            step_data = simulator.generate_step(
                token=current_token,
                token_index=token_idx,
                scenario_key=scenario_key,
                top_k=int(payload.get("top_k", 24))
            )

            # Compute real-time Persistent Homology & TDA metrics
            tda_result = topo_engine.compute_persistence(
                nodes=step_data["nodes"],
                edges=step_data["edges"]
            )

            response_data = {
                "token": step_data["token"],
                "token_index": step_data["token_index"],
                "scenario": step_data["scenario"],
                "is_adversarial": step_data["is_adversarial"],
                "nodes": step_data["nodes"],
                "edges": step_data["edges"],
                "tda": tda_result,
                "timestamp": time.time()
            }

            self.send_json_response(response_data)
            return

        elif path == "/api/benchmark":
            # Run batch benchmark across all scenarios
            benchmark_results = {}
            for s_key, s_data in simulator.scenarios.items():
                entropies = []
                tcc_scores = []
                anomalies = 0
                for idx, t in enumerate(s_data["tokens"]):
                    step = simulator.generate_step(t, idx, scenario_key=s_key)
                    tda = topo_engine.compute_persistence(step["nodes"], step["edges"])
                    entropies.append(tda["entropy"])
                    tcc_scores.append(tda["tcc"])
                    if tda["is_anomalous"]:
                        anomalies += 1

                benchmark_results[s_key] = {
                    "title": s_data["title"],
                    "avg_entropy": round(sum(entropies) / len(entropies), 4),
                    "max_entropy": round(max(entropies), 4),
                    "avg_tcc": round(sum(tcc_scores) / len(tcc_scores), 4),
                    "min_tcc": round(min(tcc_scores), 4),
                    "anomaly_rate": round(anomalies / len(s_data["tokens"]), 4),
                    "is_adversarial": s_data["is_adversarial"]
                }

            self.send_json_response(benchmark_results)
            return

        self.send_error(404, "Endpoint not found")

    def send_json_response(self, data: Any, code: int = 200):
        body = json.dumps(data).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()


# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

def start_server(port: int = 8000):
    server_address = ("127.0.0.1", port)
    httpd = HTTPServer(server_address, NeuroTopologyHandler)
    print("=" * 70, flush=True)
    print(f"[*] NeuroTopology 3D - High-Performance Interpretability Server", flush=True)
    print(f"[*] Real-time 3D Client & API live at: http://127.0.0.1:{port}", flush=True)
    print(f"[*] Gemma-Scope SAE Engine: 16,384 Features mapped to H^3 Poincare Ball", flush=True)
    print(f"[*] Topological Data Analysis: Vietoris-Rips Persistent Homology Active", flush=True)
    print("=" * 70, flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server gracefully...")
        httpd.server_close()

if __name__ == "__main__":
    port = 8000
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
        except ValueError:
            pass
    start_server(port)
