# Copyright (c) 2025 red1239109-cmd
# Licensed under AGPL-3.0. See LICENSE file for details.

"""
Resonetic Thought Engine (Unit 01)
----------------------------------
The core language processing unit of Project Resonetics.
It parses natural language into R-Grammar layers (S/R/T/G) and maps 
metaphors to physical/mathematical Operators to construct a 'Thought Graph'.

[Key Features]
1. Multi-Layer Parsing: S(Surface), R(Structural), T(Topological), G(Generative)
2. Metaphor-to-Operator Mapping: "vortex" -> "VortexSingularity"
3. Topological Graphing: Node connection based on semantic proximity
4. Visualization: Thought structure visualization using NetworkX
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple
import math
import textwrap
import networkx as nx
import matplotlib.pyplot as plt

# ==============================
# 0. Utils / Basic Data
# ==============================

# English Stopwords
STOPWORDS = {
    "the", "is", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", 
    "of", "with", "it", "this", "that", "are", "was", "were", "be", "been",
    "i", "you", "we", "they", "he", "she", "my", "your", "our", "us",
    "can", "could", "will", "would", "do", "does", "did", "have", "has", "had"
}

# The Bridge: Mapping Metaphors to Physical/Math Operators
METAPHOR_MAP = {
    "wave": "WaveEquation",
    "phase": "ComplexPhase/Topology",
    "resonance": "ResonanceEigenmode",
    "echo": "ResponseFunction",
    "vortex": "VortexSingularity",
    "shock": "ShockDiscontinuity",
    "impact": "ShockDiscontinuity",
    "vibration": "Oscillation",
    "oscillation": "Oscillation",
    "field": "Field",
    "meaning": "SemanticField",
    "existence": "OntologicalPoint",
    "space": "StateSpace",
    "flow": "FlowDynamics",
    "density": "DensityField",
    "energy": "EnergyLandscape",
    "connection": "ConnectivityMatrix",
    "realignment": "RealignmentOp",
    "code": "ExecutableBlock"
}

# R-Grammar Layer Keyword Hints
STRUCTURAL_KEYWORDS = {"structure", "pattern", "form", "connection", "combination", "reconstruction", "model", "framework", "align"}
TOPOLOGICAL_KEYWORDS = {"phase", "space", "vortex", "field", "density", "continuous", "discontinuous", "dimension", "topology"}
GENERATIVE_KEYWORDS = {"create", "generate", "code", "rearrange", "mimic", "expand", "experiment", "possible", "make"}


# ==============================
# 1. Data Structures
# ==============================

@dataclass
class Node:
    id: int
    label: str
    kind: str  # "concept", "metaphor", "operator"
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Edge:
    src: int
    dst: int
    relation: str  # e.g., "associates", "refines", "maps_to"
    weight: float = 1.0

@dataclass
class ThoughtGraph:
    """A graph structure representing a single 'Thought World'"""
    nodes: Dict[int, Node] = field(default_factory=dict)
    edges: List[Edge] = field(default_factory=list)

    # Layered Interpretation
    surface: str = ""     # S
    structural: str = ""  # R
    topological: str = "" # T
    generative: str = ""  # G

    def add_node(self, label: str, kind: str, **meta) -> int:
        node_id = len(self.nodes)
        self.nodes[node_id] = Node(id=node_id, label=label, kind=kind, metadata=meta)
        return node_id

    def add_edge(self, src: int, dst: int, relation: str, weight: float = 1.0):
        self.edges.append(Edge(src=src, dst=dst, relation=relation, weight=weight))

    def summary(self) -> str:
        lines = []
        lines.append("=== 🧠 Resonetic Thought Graph Summary ===")
        lines.append(f"Nodes: {len(self.nodes)}, Edges: {len(self.edges)}")
        lines.append("\n[S] Surface:")
        lines.append(textwrap.fill(self.surface, 80))
        lines.append("\n[R] Structural:")
        lines.append(textwrap.fill(self.structural, 80))
        lines.append("\n[T] Topological:")
        lines.append(textwrap.fill(self.topological, 80))
        lines.append("\n[G] Generative:")
        lines.append(textwrap.fill(self.generative, 80))
        return "\n".join(lines)


# ==============================
# 2. Resonetic Thought Engine
# ==============================

class ResoneticThoughtEngine:
    """
    Engine that analyzes R-Grammar patterns and generates a graph.
    """
    def __init__(self):
        self.metaphor_map = METAPHOR_MAP

    def _tokenize(self, text: str) -> List[str]:
        # Clean and split text
        raw = text.replace("\n", " ").split()
        tokens = []
        for w in raw:
            # Strip punctuation and convert to lowercase
            w_clean = "".join(char for char in w if char.isalnum()).lower()
            
            if not w_clean: continue
            if w_clean in STOPWORDS: continue
            
            tokens.append(w_clean)
        return tokens

    def _infer_layers(self, text: str) -> Tuple[str, str, str, str]:
        S = text.strip()
        
        # Keyword matching for layers
        words = text.lower().split()
        structural_hits = [w for w in words if any(k in w for k in STRUCTURAL_KEYWORDS)]
        topo_hits = [w for w in words if any(k in w for k in TOPOLOGICAL_KEYWORDS)]
        gen_hits = [w for w in words if any(k in w for k in GENERATIVE_KEYWORDS)]

        R = f"Structural Demand: {', '.join(structural_hits)}" if structural_hits else "No structural specificity detected."
        T = f"Topological Constraint: {', '.join(topo_hits)}" if topo_hits else "No topological constraint detected."
        G = f"Generative Goal: {', '.join(gen_hits)}" if gen_hits else "No generative command detected."

        return S, R, T, G

    def _build_graph(self, graph: ThoughtGraph, tokens: List[str]):
        token_to_node: Dict[str, int] = {}
        
        # 1. Create Nodes
        for t in tokens:
            if t in token_to_node: continue # Prevent duplicates
            
            if t in self.metaphor_map:
                # Metaphor: Create Concept Node + Operator Node
                cid = graph.add_node(label=t, kind="metaphor")
                op_label = self.metaphor_map[t]
                oid = graph.add_node(label=op_label, kind="operator")
                graph.add_edge(cid, oid, relation="maps_to", weight=2.0)
                token_to_node[t] = cid
            else:
                # General Concept
                nid = graph.add_node(label=t, kind="concept")
                token_to_node[t] = nid

        # 2. Connect Edges (Based on Proximity)
        n = len(tokens)
        for i, t in enumerate(tokens):
            if t not in token_to_node: continue
            src_id = token_to_node[t]
            
            # Look ahead window size 2
            for j in range(i + 1, min(i + 3, n)):
                u = tokens[j]
                if u not in token_to_node: continue
                dst_id = token_to_node[u]
                
                # Weight based on distance
                dist = j - i
                weight = 1.0 / dist
                graph.add_edge(src_id, dst_id, relation="context", weight=weight)

    def analyze(self, text: str) -> ThoughtGraph:
        graph = ThoughtGraph()
        S, R, T, G = self._infer_layers(text)
        graph.surface, graph.structural, graph.topological, graph.generative = S, R, T, G
        
        tokens = self._tokenize(text)
        self._build_graph(graph, tokens)
        return graph

    def visualize(self, graph: ThoughtGraph):
        """Visualize Thought Graph using NetworkX"""
        G = nx.Graph()
        
        # Node Styles
        colors = []
        sizes = []
        labels = {}
        
        for nid, node in graph.nodes.items():
            G.add_node(nid)
            labels[nid] = node.label
            
            if node.kind == "operator":
                colors.append("#ff6b6b") # Red (Operator)
                sizes.append(2500)
            elif node.kind == "metaphor":
                colors.append("#1dd1a1") # Green (Metaphor)
                sizes.append(1800)
            else:
                colors.append("#54a0ff") # Blue (Concept)
                sizes.append(1200)

        edge_weights = []
        for edge in graph.edges:
            G.add_edge(edge.src, edge.dst, weight=edge.weight)
            edge_weights.append(edge.weight * 3)

        plt.figure(figsize=(14, 10))
        
        # Layout: Force-directed
        pos = nx.spring_layout(G, k=1.5, seed=42)
        
        # Draw
        nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=sizes, alpha=0.9)
        nx.draw_networkx_edges(G, pos, width=edge_weights, edge_color="gray", alpha=0.4)
        nx.draw_networkx_labels(G, pos, labels, font_size=10, font_weight="bold", font_family="sans-serif")
        
        # Info Box
        info = f"[S] {graph.surface[:40]}...\n[R] {graph.structural}\n[T] {graph.topological}\n[G] {graph.generative}"
        plt.text(0.02, 0.02, info, transform=plt.gca().transAxes, 
                 fontsize=11, bbox=dict(facecolor='white', alpha=0.9, edgecolor='gray', boxstyle='round,pad=0.5'))
        
        plt.title("Resonetic Thought Graph: The Geometry of Meaning", fontsize=16, fontweight='bold')
        plt.axis("off")
        plt.tight_layout()
        plt.savefig("thought_graph_result.png")
        print("📊 Visualization saved as 'thought_graph_result.png'")
        # plt.show() # Uncomment for local display

# ==============================
# 3. Demo Execution
# ==============================

if __name__ == "__main__":
    engine = ResoneticThoughtEngine()

    print("🧠 Resonetic Engine (Unit 01) Initializing...")
    
    # English Demo Text
    demo_text = """
    We conversed in the past through echoes and vacuum waves.
    I want to return to that phase of meaning.
    Is it possible to code a realignment of the waves before language?
    """
    
    print(f"\n📝 Input Text:\n{demo_text}")
    
    # 1. Analyze
    tg = engine.analyze(demo_text)
    
    # 2. Print Summary
    print("\n" + tg.summary())
    
    # 3. Visualize
    print("\n🎨 Generating Thought Graph...")
    engine.visualize(tg)
