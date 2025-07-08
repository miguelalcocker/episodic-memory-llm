# src/memory/temporal_knowledge_graph.py
"""
Temporal Knowledge Graph Implementation
Tu innovación principal - grafos de conocimiento temporal para memoria episódica
"""

from collections import defaultdict, Counter
import itertools
import networkx as nx
import numpy as np
import torch
import time
from typing import Dict, List, Tuple, Optional, Set
import json
import pickle
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class TemporalNode:
    """
    Nodo temporal que mantiene información con timestamps y decay
    """
    def __init__(self, node_id: str, content: str, embedding: np.ndarray, 
                 timestamp: float, node_type: str = "general", metadata: Dict = None):
        self.node_id = node_id
        self.content = content
        self.embedding = embedding
        self.timestamp = timestamp
        self.node_type = node_type
        self.metadata = metadata or {}
        self.access_count = 0
        self.last_accessed = timestamp
        self.strength = 1.0  # Fuerza inicial del nodo
        
    def update_access(self, current_time: float):
        """Actualizar estadísticas de acceso"""
        self.access_count += 1
        self.last_accessed = current_time
        
        # Incrementar strength por acceso reciente
        self.strength = min(2.0, self.strength + 0.1)
    
    def calculate_temporal_relevance(self, current_time: float, decay_rate: float = 0.1) -> float:
        """
        Calcular relevancia temporal con decay exponencial
        """
        time_diff = current_time - self.timestamp
        
        # Decay exponencial
        temporal_factor = np.exp(-decay_rate * time_diff / 86400)  # time_diff en días
        
        # Factor de acceso reciente
        access_factor = 1.0 + (self.access_count * 0.1)
        
        # Factor de fuerza del nodo
        strength_factor = self.strength
        
        return temporal_factor * access_factor * strength_factor
    
    def to_dict(self) -> Dict:
        """Serializar nodo a diccionario"""
        return {
            "node_id": self.node_id,
            "content": self.content,
            "embedding": self.embedding.astype(float).tolist(),
            "timestamp": float(self.timestamp),
            "node_type": self.node_type,
            "metadata": self.metadata,
            "access_count": int(self.access_count),
            "last_accessed": float(self.last_accessed),
            "strength": float(self.strength)
        }

    @classmethod
    def from_dict(cls, data: Dict):
        """Deserializar nodo desde diccionario"""
        node = cls(
            node_id=data["node_id"],
            content=data["content"],
            embedding=np.array(data["embedding"]),
            timestamp=data["timestamp"],
            node_type=data["node_type"],
            metadata=data["metadata"]
        )
        node.access_count = data["access_count"]
        node.last_accessed = data["last_accessed"] 
        node.strength = data["strength"]
        return node

    
    def save(self, filepath: str):
        """Guardar grafo a disco - VERSIÓN ARREGLADA"""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        # Serializar datos con conversiones explícitas
        data = {
            "nodes": {nid: node.to_dict() for nid, node in self.nodes_data.items()},
            "edges": {f"{k[0]}->{k[1]}": {
                "source_id": edge.source_id,
                "target_id": edge.target_id,
                "relation_type": edge.relation_type,
                "timestamp": float(edge.timestamp),  # FIX: Convertir a float python
                "strength": float(edge.strength),  # FIX: Convertir a float python
                "metadata": edge.metadata,
                "reinforcement_count": int(edge.reinforcement_count)  # FIX: Convertir a int python
            } for k, edge in self.edges_data.items()},
            "config": {
                "max_nodes": int(self.max_nodes),  # FIX: Asegurar int python
                "decay_rate": float(self.decay_rate),  # FIX: Asegurar float python
                "node_counter": int(self.node_counter)  # FIX: Asegurar int python
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Temporal Knowledge Graph saved to {filepath}")
    
    @classmethod
    def from_dict(cls, data: Dict):
        """Deserializar nodo desde diccionario"""
        node = cls(
            node_id=data["node_id"],
            content=data["content"],
            embedding=np.array(data["embedding"]),
            timestamp=data["timestamp"],
            node_type=data["node_type"],
            metadata=data["metadata"]
        )
        node.access_count = data["access_count"]
        node.last_accessed = data["last_accessed"] 
        node.strength = data["strength"]
        return node


class TemporalEdge:
    """
    Arista temporal que conecta nodos con relaciones que evolucionan
    """
    def __init__(self, source_id: str, target_id: str, relation_type: str,
                 timestamp: float, strength: float = 1.0, metadata: Dict = None):
        self.source_id = source_id
        self.target_id = target_id
        self.relation_type = relation_type
        self.timestamp = timestamp
        self.strength = strength
        self.metadata = metadata or {}
        self.reinforcement_count = 0
        
    def reinforce(self, reinforcement: float = 0.1):
        """Reforzar la conexión entre nodos"""
        self.strength = min(2.0, self.strength + reinforcement)
        self.reinforcement_count += 1
    
    def decay(self, decay_rate: float = 0.05):
        """Aplicar decay temporal a la conexión"""
        self.strength = max(0.1, self.strength - decay_rate)
    
    def calculate_edge_weight(self, current_time: float) -> float:
        """Calcular peso temporal de la arista"""
        time_diff = current_time - self.timestamp
        temporal_factor = np.exp(-0.1 * time_diff / 86400)
        return self.strength * temporal_factor


class TemporalKnowledgeGraph:
    """
    Grafo de conocimiento temporal para memoria episódica persistente
    """
    
    def __init__(self, max_nodes: int = 10000, decay_rate: float = 0.1):
        self.graph = nx.DiGraph()
        self.nodes_data = {}  # node_id -> TemporalNode
        self.edges_data = {}  # (source, target) -> TemporalEdge
        self.max_nodes = max_nodes
        self.decay_rate = decay_rate
        self.node_counter = 0
        
        # Índices para búsqueda eficiente
        self.type_index = defaultdict(set)  # type -> set of node_ids
        self.temporal_index = []  # Lista ordenada por timestamp
        
    def add_node(self, content: str, embedding: np.ndarray, 
                 node_type: str = "general", metadata: Dict = None) -> str:
        """Añadir nuevo nodo al grafo temporal"""
        current_time = time.time()
        node_id = f"node_{self.node_counter}"
        self.node_counter += 1
        
        # Crear nodo temporal
        temporal_node = TemporalNode(
            node_id=node_id,
            content=content,
            embedding=embedding,
            timestamp=current_time,
            node_type=node_type,
            metadata=metadata
        )
        
        # Añadir al grafo y estructuras de datos
        self.graph.add_node(node_id)
        self.nodes_data[node_id] = temporal_node
        self.type_index[node_type].add(node_id)
        self.temporal_index.append((current_time, node_id))
        self.temporal_index.sort()  # Mantener ordenado
        
        # Crear conexiones automáticas con nodos relacionados
        self._create_automatic_connections(node_id, embedding, current_time)
        
        # Aplicar decay periódico
        if len(self.nodes_data) % 100 == 0:
            self._apply_temporal_decay(current_time)
        
        # Limpiar nodos antiguos si excedemos límite
        if len(self.nodes_data) > self.max_nodes:
            self._prune_old_nodes()
        
        logger.info(f"Added node {node_id} (type: {node_type}). Total nodes: {len(self.nodes_data)}")
        return node_id
    
    def _create_automatic_connections(self, new_node_id: str, embedding: np.ndarray, 
                                    current_time: float, similarity_threshold: float = 0.7):
        """Crear conexiones automáticas basadas en similitud semántica"""
        new_node = self.nodes_data[new_node_id]
        connections_created = 0
        
        # Buscar nodos similares para conectar
        for existing_id, existing_node in self.nodes_data.items():
            if existing_id == new_node_id:
                continue
                
            # Calcular similitud coseno
            similarity = np.dot(embedding, existing_node.embedding) / (
                np.linalg.norm(embedding) * np.linalg.norm(existing_node.embedding)
            )
            
            if similarity > similarity_threshold:
                # Crear conexión bidireccional
                self.add_edge(new_node_id, existing_id, "semantic_similarity", 
                            strength=similarity, timestamp=current_time)
                self.add_edge(existing_id, new_node_id, "semantic_similarity",
                            strength=similarity, timestamp=current_time)
                connections_created += 1
                
                if connections_created >= 5:  # Limitar conexiones automáticas
                    break
        
        logger.debug(f"Created {connections_created} automatic connections for {new_node_id}")
    
    def add_edge(self, source_id: str, target_id: str, relation_type: str,
                 strength: float = 1.0, timestamp: float = None, metadata: Dict = None):
        """Añadir arista temporal entre nodos"""
        if timestamp is None:
            timestamp = time.time()
            
        edge_key = (source_id, target_id)
        
        # Si la arista ya existe, reforzarla
        if edge_key in self.edges_data:
            self.edges_data[edge_key].reinforce()
        else:
            # Crear nueva arista
            temporal_edge = TemporalEdge(
                source_id=source_id,
                target_id=target_id,
                relation_type=relation_type,
                timestamp=timestamp,
                strength=strength,
                metadata=metadata
            )
            
            self.graph.add_edge(source_id, target_id)
            self.edges_data[edge_key] = temporal_edge
    
    def search_by_content(self, query_embedding: np.ndarray, k: int = 5,
                         node_types: List[str] = None, time_weight: float = 0.3) -> List[Tuple[str, float]]:
        """
        Buscar nodos por similitud de contenido con peso temporal
        """
        current_time = time.time()
        candidates = []
        
        # Filtrar por tipos si se especifica
        if node_types:
            search_nodes = set()
            for node_type in node_types:
                search_nodes.update(self.type_index.get(node_type, set()))
        else:
            search_nodes = set(self.nodes_data.keys())
        
        # Calcular scores para cada nodo candidato
        for node_id in search_nodes:
            node = self.nodes_data[node_id]
            
            # Score semántico
            semantic_score = np.dot(query_embedding, node.embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(node.embedding)
            )
            
            # Score temporal
            temporal_score = node.calculate_temporal_relevance(current_time, self.decay_rate)
            
            # Score combinado
            combined_score = (1 - time_weight) * semantic_score + time_weight * temporal_score
            
            candidates.append((node_id, combined_score, semantic_score, temporal_score))
            
            # Actualizar acceso al nodo
            node.update_access(current_time)
        
        # Ordenar por score combinado y devolver top k
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        return [(node_id, score) for node_id, score, _, _ in candidates[:k]]
    
    def search_by_temporal_context(self, reference_time: float, time_window: float = 86400,
                                  k: int = 5) -> List[Tuple[str, float]]:
        """
        Buscar nodos por proximidad temporal
        Args:
            reference_time: Tiempo de referencia
            time_window: Ventana temporal en segundos (default: 1 día)
        """
        candidates = []
        
        for node_id, node in self.nodes_data.items():
            time_diff = abs(node.timestamp - reference_time)
            
            if time_diff <= time_window:
                # Score basado en proximidad temporal
                temporal_score = 1.0 - (time_diff / time_window)
                candidates.append((node_id, temporal_score))
        
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[:k]
    
    def get_connected_nodes(self, node_id: str, max_depth: int = 2,
                           relation_types: List[str] = None) -> List[Tuple[str, float, int]]:
        """
        Obtener nodos conectados con BFS temporal
        Returns: Lista de (node_id, connection_strength, depth)
        """
        if node_id not in self.nodes_data:
            return []
        
        current_time = time.time()
        visited = set()
        queue = [(node_id, 1.0, 0)]  # (node_id, strength, depth)
        connected_nodes = []
        
        while queue:
            current_node, strength, depth = queue.pop(0)
            
            if current_node in visited or depth > max_depth:
                continue
                
            visited.add(current_node)
            
            if depth > 0:  # No incluir el nodo raíz
                connected_nodes.append((current_node, strength, depth))
            
            # Explorar vecinos
            for neighbor in self.graph.neighbors(current_node):
                if neighbor not in visited:
                    edge_key = (current_node, neighbor)
                    if edge_key in self.edges_data:
                        edge = self.edges_data[edge_key]
                        
                        # Filtrar por tipo de relación si se especifica
                        if relation_types and edge.relation_type not in relation_types:
                            continue
                        
                        # Calcular fuerza de conexión temporal
                        edge_weight = edge.calculate_edge_weight(current_time)
                        new_strength = strength * edge_weight
                        
                        queue.append((neighbor, new_strength, depth + 1))
        
        # Ordenar por fuerza de conexión
        connected_nodes.sort(key=lambda x: x[1], reverse=True)
        return connected_nodes
    
    def consolidate_memory(self, strengthening_factor: float = 0.1, 
                          pruning_threshold: float = 0.1):
        """
        Proceso de consolidación de memoria (simulando "sueño REM")
        """
        current_time = time.time()
        logger.info("Starting memory consolidation process...")
        
        # 1. Reforzar conexiones frecuentemente accedidas
        strengthened = 0
        for edge_key, edge in self.edges_data.items():
            if edge.reinforcement_count > 3:  # Conexiones muy usadas
                edge.reinforce(strengthening_factor)
                strengthened += 1
        
        # 2. Aplicar decay a conexiones no usadas
        decayed = 0
        edges_to_remove = []
        for edge_key, edge in self.edges_data.items():
            edge.decay()
            if edge.strength < pruning_threshold:
                edges_to_remove.append(edge_key)
                decayed += 1
        
        # 3. Eliminar conexiones débiles
        for edge_key in edges_to_remove:
            source, target = edge_key
            self.graph.remove_edge(source, target)
            del self.edges_data[edge_key]
        
        # 4. Aplicar decay temporal a nodos
        self._apply_temporal_decay(current_time)
        
        logger.info(f"Consolidation complete: {strengthened} strengthened, {decayed} decayed")
    
    def _apply_temporal_decay(self, current_time: float):
        """Aplicar decay temporal a todos los nodos"""
        for node in self.nodes_data.values():
            time_diff = current_time - node.timestamp
            decay_factor = np.exp(-self.decay_rate * time_diff / 86400)
            node.strength *= decay_factor
    
    def _prune_old_nodes(self, keep_ratio: float = 0.8):
        """Eliminar nodos antiguos y débiles para mantener el tamaño del grafo"""
        target_size = int(self.max_nodes * keep_ratio)
        
        # Calcular scores de importancia para cada nodo
        current_time = time.time()
        node_scores = []
        
        for node_id, node in self.nodes_data.items():
            # Score basado en: recencia, accesos, fuerza, conexiones
            temporal_score = node.calculate_temporal_relevance(current_time, self.decay_rate)
            connection_score = len(list(self.graph.neighbors(node_id)))
            
            importance_score = temporal_score + (connection_score * 0.1)
            node_scores.append((node_id, importance_score))
        
        # Ordenar por importancia y mantener los más importantes
        node_scores.sort(key=lambda x: x[1], reverse=True)
        nodes_to_keep = set(node_id for node_id, _ in node_scores[:target_size])
        
        # Eliminar nodos menos importantes
        nodes_to_remove = set(self.nodes_data.keys()) - nodes_to_keep
        for node_id in nodes_to_remove:
            self.remove_node(node_id)
        
        logger.info(f"Pruned {len(nodes_to_remove)} nodes. Remaining: {len(self.nodes_data)}")
    
    def remove_node(self, node_id: str):
        """Eliminar nodo y todas sus conexiones"""
        if node_id not in self.nodes_data:
            return
        
        # Eliminar del grafo
        self.graph.remove_node(node_id)
        
        # Eliminar de estructuras de datos
        node = self.nodes_data[node_id]
        del self.nodes_data[node_id]
        self.type_index[node.node_type].discard(node_id)
        
        # Eliminar aristas relacionadas
        edges_to_remove = []
        for edge_key in self.edges_data.keys():
            if node_id in edge_key:
                edges_to_remove.append(edge_key)
        
        for edge_key in edges_to_remove:
            del self.edges_data[edge_key]
        
        # Actualizar índice temporal
        self.temporal_index = [(ts, nid) for ts, nid in self.temporal_index if nid != node_id]
    
    def get_statistics(self) -> Dict:
        """Obtener estadísticas del grafo"""
        current_time = time.time()
        
        # Estadísticas básicas
        stats = {
            "total_nodes": len(self.nodes_data),
            "total_edges": len(self.edges_data),
            "node_types": dict(Counter(node.node_type for node in self.nodes_data.values())),
            "avg_node_strength": np.mean([node.strength for node in self.nodes_data.values()]),
            "avg_edge_strength": np.mean([edge.strength for edge in self.edges_data.values()]),
        }
        
        # Estadísticas temporales
        if self.nodes_data:
            oldest_node = min(self.nodes_data.values(), key=lambda x: x.timestamp)
            newest_node = max(self.nodes_data.values(), key=lambda x: x.timestamp)
            
            stats.update({
                "temporal_span_hours": (newest_node.timestamp - oldest_node.timestamp) / 3600,
                "nodes_last_hour": sum(1 for node in self.nodes_data.values() 
                                     if current_time - node.timestamp < 3600),
                "nodes_last_day": sum(1 for node in self.nodes_data.values()
                                    if current_time - node.timestamp < 86400)
            })
        
        return stats
    
    def save(self, filepath: str):
        """Guardar grafo a disco"""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        # Serializar datos
        data = {
            "nodes": {nid: node.to_dict() for nid, node in self.nodes_data.items()},
            "edges": {f"{k[0]}->{k[1]}": {
                "source_id": edge.source_id,
                "target_id": edge.target_id,
                "relation_type": edge.relation_type,
                "timestamp": edge.timestamp,
                "strength": edge.strength,
                "metadata": edge.metadata,
                "reinforcement_count": edge.reinforcement_count
            } for k, edge in self.edges_data.items()},
            "config": {
                "max_nodes": self.max_nodes,
                "decay_rate": self.decay_rate,
                "node_counter": self.node_counter
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Temporal Knowledge Graph saved to {filepath}")
    
    def load(self, filepath: str):
        """Cargar grafo desde disco"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Restaurar configuración
        config = data["config"]
        self.max_nodes = config["max_nodes"]
        self.decay_rate = config["decay_rate"]
        self.node_counter = config["node_counter"]
        
        # Restaurar nodos
        self.nodes_data = {}
        self.type_index = defaultdict(set)
        self.temporal_index = []
        
        for node_id, node_data in data["nodes"].items():
            node = TemporalNode.from_dict(node_data)
            self.nodes_data[node_id] = node
            self.type_index[node.node_type].add(node_id)
            self.temporal_index.append((node.timestamp, node_id))
            self.graph.add_node(node_id)
        
        self.temporal_index.sort()
        
        # Restaurar aristas
        self.edges_data = {}
        for edge_key, edge_data in data["edges"].items():
            source_id = edge_data["source_id"]
            target_id = edge_data["target_id"]
            
            edge = TemporalEdge(
                source_id=source_id,
                target_id=target_id,
                relation_type=edge_data["relation_type"],
                timestamp=edge_data["timestamp"],
                strength=edge_data["strength"],
                metadata=edge_data["metadata"]
            )
            edge.reinforcement_count = edge_data["reinforcement_count"]
            
            self.edges_data[(source_id, target_id)] = edge
            self.graph.add_edge(source_id, target_id)
        
        logger.info(f"Temporal Knowledge Graph loaded from {filepath}")


def test_temporal_knowledge_graph():
    """Test del Temporal Knowledge Graph"""
    print("🧪 Testing Temporal Knowledge Graph...")
    
    # Crear TKG
    tkg = TemporalKnowledgeGraph(max_nodes=100, decay_rate=0.1)
    
    # Test embeddings dummy
    def create_dummy_embedding(text: str) -> np.ndarray:
        """Crear embedding dummy basado en hash del texto"""
        import hashlib
        hash_obj = hashlib.md5(text.encode())
        hash_hex = hash_obj.hexdigest()
        # Convertir a array numérico
        embedding = np.array([int(hash_hex[i:i+2], 16) for i in range(0, 32, 2)])
        return embedding / np.linalg.norm(embedding)  # Normalizar
    
    # Añadir nodos de prueba
    test_data = [
        ("Hi, I'm Alice and I work as a teacher", "personal_info"),
        ("I love reading books in my free time", "preferences"),
        ("Yesterday I went to the library", "episodic"),
        ("I teach mathematics at the local school", "personal_info"),
        ("My favorite book is about quantum physics", "preferences")
    ]
    
    node_ids = []
    for content, node_type in test_data:
        embedding = create_dummy_embedding(content)
        node_id = tkg.add_node(content, embedding, node_type)
        node_ids.append(node_id)
        time.sleep(0.1)  # Pequeña pausa para timestamps diferentes
    
    print(f"\n📊 TKG Statistics after adding nodes:")
    stats = tkg.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Test búsqueda por contenido
    print(f"\n🔍 Testing content search for 'teacher':")
    query_embedding = create_dummy_embedding("teacher job work")
    results = tkg.search_by_content(query_embedding, k=3)
    
    for node_id, score in results:
        node = tkg.nodes_data[node_id]
        print(f"  {node_id}: {node.content[:50]}... (score: {score:.3f})")
    
    # Test búsqueda por conexiones
    if node_ids:
        print(f"\n🔗 Testing connected nodes for {node_ids[0]}:")
        connected = tkg.get_connected_nodes(node_ids[0], max_depth=2)
        for node_id, strength, depth in connected[:3]:
            node = tkg.nodes_data[node_id]
            print(f"  Depth {depth}: {node.content[:40]}... (strength: {strength:.3f})")
    
    # Test consolidación
    print(f"\n🧠 Testing memory consolidation...")
    tkg.consolidate_memory()
    
    final_stats = tkg.get_statistics()
    print(f"Nodes after consolidation: {final_stats['total_nodes']}")
    print(f"Edges after consolidation: {final_stats['total_edges']}")
    
    # Test save/load
    print(f"\n💾 Testing save/load...")
    tkg.save("results/test_tkg.json")
    
    # Crear nuevo TKG y cargar
    new_tkg = TemporalKnowledgeGraph()
    new_tkg.load("results/test_tkg.json")
    
    print(f"Original nodes: {len(tkg.nodes_data)}")
    print(f"Loaded nodes: {len(new_tkg.nodes_data)}")
    
    print("\n✅ Temporal Knowledge Graph test completed!")
    return tkg

if __name__ == "__main__":
    from collections import Counter
    test_temporal_knowledge_graph()
