"""
NeuroTopology 3D - Topological Data Analysis (TDA) & Persistent Homology Engine
Mathematical formulation of Persistent Homology, Betti numbers (beta_0, beta_1),
and Topological Circuit Coherence (TCC) for Sparse Autoencoder feature activations.
"""

import math
import heapq
from typing import List, Dict, Tuple, Any

class UnionFind:
    """Disjoint Set Union (DSU) for 0-dimensional persistent homology (beta_0)."""
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n
        self.birth = [0.0] * n
        self.components = n

    def find(self, i: int) -> int:
        if self.parent[i] == i:
            return i
        self.parent[i] = self.find(self.parent[i])
        return self.parent[i]

    def union(self, i: int, j: int, current_dist: float) -> Tuple[bool, float]:
        root_i = self.find(i)
        root_j = self.find(j)
        if root_i == root_j:
            return False, 0.0  # Already connected -> forms a 1-cycle (beta_1 candidate)

        # Merge younger component into older component (standard persistence rule)
        if self.birth[root_i] > self.birth[root_j]:
            root_i, root_j = root_j, root_i
        elif self.birth[root_i] == self.birth[root_j] and self.rank[root_i] < self.rank[root_j]:
            root_i, root_j = root_j, root_i

        death_time = current_dist
        lifetime = death_time - self.birth[root_j]

        self.parent[root_j] = root_i
        if self.rank[root_i] == self.rank[root_j]:
            self.rank[root_i] += 1
        self.components -= 1
        return True, lifetime


class TopologicalEngine:
    """
    Computes real-time topological descriptors over active Sparse Autoencoder circuits:
    - Betti-0 (\beta_0): Number of disconnected semantic clusters
    - Betti-1 (\beta_1): Number of independent circular reasoning loops
    - Persistent Homology Entropy (H_topo): Measure of structural manifold disorder
    - Topological Circuit Coherence (TCC): Normalized ratio of harmonic loop density
    """

    def __init__(self, entropy_threshold: float = 1.45):
        self.entropy_threshold = entropy_threshold
        self.history: List[Dict[str, float]] = []

    @staticmethod
    def poincare_distance(u: Tuple[float, float, float], v: Tuple[float, float, float]) -> float:
        """Computes Riemannian distance in the 3D Poincaré Ball Model H^3."""
        ux, uy, uz = u
        vx, vy, vz = v
        norm_u_sq = min(ux*ux + uy*uy + uz*uz, 0.999)
        norm_v_sq = min(vx*vx + vy*vy + vz*vz, 0.999)
        diff_sq = (ux - vx)**2 + (uy - vy)**2 + (uz - vz)**2

        arg = 1.0 + 2.0 * diff_sq / ((1.0 - norm_u_sq) * (1.0 - norm_v_sq))
        arg = max(arg, 1.0)
        return math.acosh(arg)

    @staticmethod
    def euclidean_distance(u: Tuple[float, float, float], v: Tuple[float, float, float]) -> float:
        """Euclidean distance fallback."""
        return math.sqrt((u[0] - v[0])**2 + (u[1] - v[1])**2 + (u[2] - v[2])**2)

    def compute_persistence(
        self,
        nodes: List[Dict[str, Any]],
        edges: List[Dict[str, Any]],
        max_filtration_radius: float = 2.5
    ) -> Dict[str, Any]:
        """
        Computes 0D and 1D persistent homology over the active circuit graph.
        nodes: List of {'id': int, 'coords': (x, y, z), 'activation': float}
        edges: List of {'source': int, 'target': int, 'weight': float}
        """
        n = len(nodes)
        if n == 0:
            return {
                "beta_0": 0,
                "beta_1": 0,
                "entropy": 0.0,
                "tcc": 1.0,
                "is_anomalous": False,
                "filtration_lifetimes": []
            }

        if n == 1:
            return {
                "beta_0": 1,
                "beta_1": 0,
                "entropy": 0.0,
                "tcc": 1.0,
                "is_anomalous": False,
                "filtration_lifetimes": [1.0]
            }

        node_map = {node['id']: idx for idx, node in enumerate(nodes)}

        # Build filtration edges sorted by distance / inverse weight
        filtration_edges = []
        for edge in edges:
            src = edge['source']
            tgt = edge['target']
            if src in node_map and tgt in node_map:
                u_idx = node_map[src]
                v_idx = node_map[tgt]
                # Distance inversely proportional to circuit co-activation weight
                weight = max(edge.get('weight', 0.1), 0.001)
                dist = 1.0 / weight
                filtration_edges.append((dist, u_idx, v_idx))

        # Also consider spatial proximity in the Poincaré embedding
        for i in range(n):
            for j in range(i + 1, n):
                p_dist = self.poincare_distance(nodes[i]['coords'], nodes[j]['coords'])
                if p_dist < max_filtration_radius:
                    filtration_edges.append((p_dist, i, j))

        filtration_edges.sort(key=lambda x: x[0])

        dsu = UnionFind(n)
        lifetimes_0d: List[float] = []
        cycle_candidates: List[Tuple[float, int, int]] = []

        # 0-Dimensional Filtration (Vietoris-Rips 0-skeleton)
        for dist, u, v in filtration_edges:
            merged, lifetime = dsu.union(u, v, dist)
            if merged:
                if lifetime > 0.001:
                    lifetimes_0d.append(lifetime)
            else:
                # Potential 1-cycle birth
                cycle_candidates.append((dist, u, v))

        beta_0 = dsu.components
        # 1-Dimensional cycles (loops in reasoning)
        beta_1 = len(cycle_candidates)

        # Persistent Homology Entropy: H_topo = - sum(p_i * log(p_i))
        total_lifetime = sum(lifetimes_0d)
        entropy = 0.0
        if total_lifetime > 1e-6 and len(lifetimes_0d) > 1:
            for l in lifetimes_0d:
                p = l / total_lifetime
                if p > 1e-8:
                    entropy -= p * math.log(p)

        # Topological Circuit Coherence (TCC)
        # Ratio of harmonic cycle density to fragmented components, damped by entropy
        loop_factor = (beta_1 + 1.0) / (beta_0 + 1.0)
        tcc = min(1.0, loop_factor * math.exp(-0.5 * entropy))

        # Anomaly detection: Entropy spike or severe component fragmentation
        is_anomalous = (entropy > self.entropy_threshold) or (beta_0 > 25 and beta_1 == 0)

        result = {
            "beta_0": int(beta_0),
            "beta_1": int(beta_1),
            "entropy": round(float(entropy), 4),
            "tcc": round(float(tcc), 4),
            "is_anomalous": bool(is_anomalous),
            "num_active_features": n,
            "num_synaptic_edges": len(edges)
        }

        self.history.append(result)
        if len(self.history) > 100:
            self.history.pop(0)

        return result
