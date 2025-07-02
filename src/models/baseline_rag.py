# src/models/baseline_rag.py
"""
Baseline RAG (Retrieval-Augmented Generation) Implementation
Primera implementación del proyecto - memoria episódica básica
"""

import torch
import torch.nn as nn
from transformers import (
    GPT2LMHeadModel, 
    GPT2Tokenizer, 
    GPT2Config
)
import faiss
import numpy as np
from typing import List, Dict, Tuple, Optional
import json
import pickle
from pathlib import Path
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleMemoryStore:
    """
    Almacén de memoria simple usando FAISS para retrieval rápido
    """
    def __init__(self, embedding_dim: int = 768):
        self.embedding_dim = embedding_dim
        self.index = faiss.IndexFlatIP(embedding_dim)  # Inner Product (cosine similarity)
        self.memories = []  # Lista de textos/conversaciones
        self.metadata = []  # Timestamps, contexto, etc.
        
    def add_memory(self, embedding: np.ndarray, text: str, metadata: Dict = None):
        """Añadir nueva memoria al store"""
        # Normalizar embedding para cosine similarity
        embedding = embedding / np.linalg.norm(embedding)
        
        self.index.add(embedding.reshape(1, -1).astype('float32'))
        self.memories.append(text)
        self.metadata.append(metadata or {})
        
        logger.info(f"Added memory: {text[:50]}... (Total: {len(self.memories)})")
    
    def search_memories(self, query_embedding: np.ndarray, k: int = 5) -> List[Tuple[str, float, Dict]]:
        """Buscar memorias más relevantes"""
        if len(self.memories) == 0:
            return []
            
        # Normalizar query
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        
        # Buscar
        scores, indices = self.index.search(
            query_embedding.reshape(1, -1).astype('float32'), 
            min(k, len(self.memories))
        )
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0:  # FAISS devuelve -1 para indices no válidos
                results.append((
                    self.memories[idx],
                    float(score),
                    self.metadata[idx]
                ))
        
        return results
    
    def save(self, path: str):
        """Guardar memoria a disco"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        faiss.write_index(self.index, f"{path}_index.faiss")
        with open(f"{path}_data.pkl", 'wb') as f:
            pickle.dump({
                'memories': self.memories,
                'metadata': self.metadata,
                'embedding_dim': self.embedding_dim
            }, f)
        logger.info(f"Memory saved to {path}")
    
    def load(self, path: str):
        """Cargar memoria desde disco"""
        self.index = faiss.read_index(f"{path}_index.faiss")
        
        with open(f"{path}_data.pkl", 'rb') as f:
            data = pickle.load(f)
            self.memories = data['memories']
            self.metadata = data['metadata']
            self.embedding_dim = data['embedding_dim']
        
        logger.info(f"Memory loaded from {path}. Total memories: {len(self.memories)}")


class BaselineRAG(nn.Module):
    """
    Baseline RAG Model - Fundación para memoria episódica
    """
    
    def __init__(
        self,
        model_name: str = "gpt2-medium",
        memory_size: int = 1000,
        max_context_length: int = 1024,
        device: str = None
    ):
        super().__init__()
        
        # Device setup
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        
        # Load model and tokenizer
        logger.info(f"Loading {model_name} model...")
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        
        # Add padding token if not exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model.to(self.device)
        self.model.eval()
        
        # Memory components
        self.memory_store = SimpleMemoryStore(embedding_dim=self.model.config.hidden_size)
        self.max_context_length = max_context_length
        self.memory_size = memory_size
        
        # Conversación actual
        self.current_conversation = []
        
        logger.info(f"BaselineRAG initialized on {self.device}")
    
    def get_text_embedding(self, text: str) -> np.ndarray:
        """
        Obtener embedding de un texto usando el modelo base
        """
        inputs = self.tokenizer(
            text, 
            return_tensors="pt", 
            max_length=512, 
            truncation=True, 
            padding=True
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.transformer(**inputs)
            # Usar el último hidden state y hacer mean pooling
            embeddings = outputs.last_hidden_state.mean(dim=1)
        
        return embeddings.cpu().numpy()[0]
    
    def add_to_memory(self, text: str, role: str = "user", metadata: Dict = None):
        """
        Añadir texto a la memoria episódica
        """
        import time
        
        # Crear metadata con timestamp
        mem_metadata = {
            "role": role,
            "timestamp": time.time(),
            "conversation_turn": len(self.current_conversation)
        }
        if metadata:
            mem_metadata.update(metadata)
        
        # Obtener embedding y añadir a memoria
        embedding = self.get_text_embedding(text)
        self.memory_store.add_memory(embedding, text, mem_metadata)
        
        # Añadir a conversación actual
        self.current_conversation.append({"role": role, "content": text})
    
    def retrieve_relevant_memories(self, query: str, k: int = 3) -> List[str]:
        """
        Recuperar memorias relevantes para una query
        """
        query_embedding = self.get_text_embedding(query)
        relevant_memories = self.memory_store.search_memories(query_embedding, k=k)
        
        # Extraer solo los textos, ordenados por relevancia
        memory_texts = [memory[0] for memory in relevant_memories]
        
        if memory_texts:
            logger.info(f"Retrieved {len(memory_texts)} relevant memories for: {query[:50]}...")
        
        return memory_texts
    
    def build_context(self, current_input: str, max_memories: int = 3) -> str:
        """
        Construir contexto combinando memoria relevante y conversación actual
        """
        context_parts = []
        
        # 1. Recuperar memorias relevantes
        relevant_memories = self.retrieve_relevant_memories(current_input, k=max_memories)
        
        if relevant_memories:
            context_parts.append("Previous relevant memories:")
            for i, memory in enumerate(relevant_memories[:max_memories]):
                context_parts.append(f"Memory {i+1}: {memory}")
            context_parts.append("")
        
        # 2. Añadir conversación reciente
        if self.current_conversation:
            context_parts.append("Recent conversation:")
            # Usar últimos N turnos para no exceder max_context_length
            recent_turns = self.current_conversation[-5:]  # Últimos 5 turnos
            for turn in recent_turns:
                context_parts.append(f"{turn['role']}: {turn['content']}")
            context_parts.append("")
        
        # 3. Añadir input actual
        context_parts.append(f"User: {current_input}")
        context_parts.append("Assistant:")
        
        full_context = "\n".join(context_parts)
        
        # Truncar si es demasiado largo
        tokens = self.tokenizer.encode(full_context)
        if len(tokens) > self.max_context_length - 50:  # Dejar espacio para respuesta
            # Truncar desde el principio manteniendo input actual
            truncated_tokens = tokens[-(self.max_context_length - 50):]
            full_context = self.tokenizer.decode(truncated_tokens)
        
        return full_context
    
    def generate_response(self, user_input: str, max_length: int = 150) -> str:
        """
        Generar respuesta usando contexto + memoria
        """
        # Construir contexto con memoria
        context = self.build_context(user_input)
        
        # Tokenizar
        inputs = self.tokenizer(
            context,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_context_length,
            padding=True
        ).to(self.device)
        
        # Generar respuesta
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=inputs['input_ids'].shape[1] + max_length,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # Decodificar solo la parte nueva (respuesta)
        input_length = inputs['input_ids'].shape[1]
        response_tokens = outputs[0][input_length:]
        response = self.tokenizer.decode(response_tokens, skip_special_tokens=True)
        
        # Limpiar respuesta
        response = response.strip()
        if not response:
            response = "I understand. Could you tell me more?"
        
        return response
    
    def chat(self, user_input: str) -> str:
        """
        Interfaz principal para chat con memoria episódica
        """
        logger.info(f"User input: {user_input}")
        
        # Añadir input del usuario a memoria
        self.add_to_memory(user_input, role="user")
        
        # Generar respuesta
        response = self.generate_response(user_input)
        
        # Añadir respuesta a memoria
        self.add_to_memory(response, role="assistant")
        
        logger.info(f"Assistant response: {response}")
        return response
    
    def save_memory(self, path: str):
        """Guardar memoria persistente"""
        self.memory_store.save(path)
        
        # Guardar también conversación actual
        with open(f"{path}_conversation.json", 'w') as f:
            json.dump(self.current_conversation, f, indent=2)
    
    def load_memory(self, path: str):
        """Cargar memoria persistente"""
        self.memory_store.load(path)
        
        # Cargar conversación si existe
        conv_path = f"{path}_conversation.json"
        if Path(conv_path).exists():
            with open(conv_path, 'r') as f:
                self.current_conversation = json.load(f)


# Funciones de utilidad para testing
def test_baseline_rag():
    """
    Test básico del sistema RAG
    """
    print("🚀 Testing Baseline RAG...")
    
    # Inicializar modelo
    rag = BaselineRAG(model_name="gpt2-medium", device="cpu")  # Cambiar a "cuda" si tienes GPU
    
    # Conversación de prueba
    test_inputs = [
        "Hi, my name is Alex and I love astronomy",
        "What's my favorite subject?",
        "I also enjoy playing chess in my free time",
        "What do you know about my hobbies?",
        "Tell me about the planets in our solar system"
    ]
    
    print("\n" + "="*50)
    print("BASELINE RAG CONVERSATION TEST")
    print("="*50)
    
    for i, user_input in enumerate(test_inputs):
        print(f"\n--- Turn {i+1} ---")
        response = rag.chat(user_input)
        print(f"User: {user_input}")
        print(f"Assistant: {response}")
    
    # Mostrar estadísticas de memoria
    print(f"\n📊 Memory Statistics:")
    print(f"Total memories stored: {len(rag.memory_store.memories)}")
    print(f"Current conversation length: {len(rag.current_conversation)}")
    
    # Test de retrieval
    print(f"\n🔍 Testing memory retrieval for 'hobbies':")
    memories = rag.retrieve_relevant_memories("hobbies", k=3)
    for j, memory in enumerate(memories):
        print(f"  {j+1}. {memory}")
    
    print("\n✅ Baseline RAG test completed!")
    return rag

if __name__ == "__main__":
    # Ejecutar test
    model = test_baseline_rag()
    
    # Guardar memoria de ejemplo
    print("\n💾 Saving memory...")
    model.save_memory("results/baseline_test_memory")
    print("Memory saved!")
