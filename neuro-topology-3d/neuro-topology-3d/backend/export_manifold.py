"""
Exports initial 3D manifold points and scenario definitions to frontend/data/
"""
import json
import os
from sae_engine import SAEDictionary, InferenceCircuitSimulator

def main():
    print("Generating 16,384 SAE Feature Dictionary in 3D Poincare Space...")
    sae_dict = SAEDictionary(num_features=16384)
    simulator = InferenceCircuitSimulator(sae_dict)

    data_dir = os.path.join(os.path.dirname(__file__), "..", "frontend", "data")
    os.makedirs(data_dir, exist_ok=True)

    # Subsample 2,048 primary representative nodes for initial WebGL point cloud
    manifold_points = sae_dict.export_manifold_points(sample_rate=8)
    manifold_path = os.path.join(data_dir, "latent_manifold.json")
    with open(manifold_path, "w", encoding="utf-8") as f:
        json.dump(manifold_points, f, indent=None)
    print(f"Exported {len(manifold_points)} manifold anchor points to {manifold_path}")

    # Export scenario metadata
    scenarios_meta = {
        k: {
            "title": v["title"],
            "description": v["description"],
            "tokens": v["tokens"],
            "primary_domain": v["primary_domain"],
            "is_adversarial": v["is_adversarial"]
        }
        for k, v in simulator.scenarios.items()
    }
    scenarios_path = os.path.join(data_dir, "scenarios.json")
    with open(scenarios_path, "w", encoding="utf-8") as f:
        json.dump(scenarios_meta, f, indent=2)
    print(f"Exported scenario definitions to {scenarios_path}")

if __name__ == "__main__":
    main()
