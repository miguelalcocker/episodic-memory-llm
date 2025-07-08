# src/models/episodic_memory_llm_v2.py
"""
EpisodicMemoryLLM v2.0 - Integración con sistema de memoria avanzado
Objetivo: Accuracy 70%+ en tareas de memoria episódica
"""

import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import numpy as np
from typing import List, Dict, Tuple, Optional
import time
import logging
import sys
import os

# Imports del sistema base
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from memory.temporal_knowledge_graph import TemporalKnowledgeGraph
from memory.advanced_memory_retrieval import AdvancedMemoryRetrieval

logger = logging.getLogger(__name__)

class EpisodicMemoryLLM_V2(nn.Module):
    """
    LLM con memoria episódica avanzada - Version 2.0
    Mejoras principales: accuracy, structured info extraction, hybrid search
    """
    
    def __init__(
        self,
        model_name: str = "gpt2-medium",
        max_context_length: int = 1024,
        tkg_max_nodes: int = 5000,
        tkg_decay_rate: float = 0.1,
        device: str = None
    ):
        super().__init__()
        
        # Device setup
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        
        # Load base LLM
        logger.info(f"Loading {model_name} model...")
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model.to(self.device)
        self.model.eval()
        
        # Initialize Temporal Knowledge Graph
        self.tkg = TemporalKnowledgeGraph(
            max_nodes=tkg_max_nodes,
            decay_rate=tkg_decay_rate
        )
        
        # Initialize Advanced Memory Retrieval System
        self.memory_system = AdvancedMemoryRetrieval(self.tkg, self.tokenizer)
        
        # IMPORTANTE: Verificar que se inicializó correctamente
        print(f"DEBUG V2: AdvancedMemoryRetrieval initialized: {self.memory_system is not None}")
        print(f"DEBUG V2: TKG connected: {self.memory_system.tkg is not None}")

        self.max_context_length = max_context_length
        self.conversation_history = []
        
        # Generation parameters optimized
        self.generation_config = {
            "temperature": 0.8,
            "top_p": 0.9,
            "do_sample": True,
            "repetition_penalty": 1.1,
            "no_repeat_ngram_size": 3
        }
        
        logger.info(f"EpisodicMemoryLLM_V2 initialized on {device}")
    
    def get_text_embedding(self, text: str) -> np.ndarray:
        """
        Obtener embedding del texto usando el LLM base
        """
        text = text.strip()[:500]
        
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=256,
            truncation=True,
            padding=True
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.transformer(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1)
        
        return embeddings.cpu().numpy()[0]
    
    def classify_content(self, text: str, role: str) -> str:
        """
        Clasificación mejorada y más precisa
        """
        text_lower = text.lower().strip()
        
        if role == "user":
            # Memory queries - MÁS AGRESIVO en detección
            memory_indicators = [
                "what's my", "what is my", "whats my",
                "what do you know about my", "what do you remember about my",
                "tell me about my", "do you remember my",
                "what did i", "where do i", "where did i",
                "remember", "recall", "know about"
            ]
            
            if any(indicator in text_lower for indicator in memory_indicators):
                return "memory_query"
            
            # Contextual queries - MÁS ESPECÍFICO
            context_indicators = [
                "recommend", "suggest", "what should", "can you",
                "advice", "help me", "what would you", "based on"
            ]
            
            if any(indicator in text_lower for indicator in context_indicators):
                return "contextual_query"
            
            # Personal info - EXPANDIDO
            personal_indicators = [
                "my name is", "i'm", "i am", "i work as", "work at",
                "i live in", "from", "my job", "my profession", "my career"
            ]
            
            if any(indicator in text_lower for indicator in personal_indicators):
                return "personal_info"
            
            # Preferences - EXPANDIDO
            preference_indicators = [
                "i love", "i like", "i enjoy", "my favorite", "my hobby",
                "i prefer", "passionate about", "interested in", "i really like"
            ]
            
            if any(indicator in text_lower for indicator in preference_indicators):
                return "preferences"
            
            # Experiences - EXPANDIDO
            experience_indicators = [
                "yesterday", "last week", "ago", "when i", "i went",
                "i visited", "i traveled", "i experienced", "i did", "recently"
            ]
            
            if any(indicator in text_lower for indicator in experience_indicators):
                return "episodic"
            
            return "general"
        
        else:  # assistant
            return "response"
    
    def add_to_memory(self, text: str, role: str = "user", metadata: Dict = None):
        """
        Añadir información al TKG con clasificación mejorada
        """
        content_type = self.classify_content(text, role)
        
        enhanced_metadata = {
            "role": role,
            "content_type": content_type,
            "conversation_turn": len(self.conversation_history),
            "timestamp": time.time()
        }
        if metadata:
            enhanced_metadata.update(metadata)
        
        embedding = self.get_text_embedding(text)
        
        node_id = self.tkg.add_node(
            content=text,
            embedding=embedding,
            node_type=content_type,
            metadata=enhanced_metadata
        )
        
        self.conversation_history.append({
            "role": role,
            "content": text,
            "node_id": node_id,
            "metadata": enhanced_metadata
        })
        
        logger.info(f"Added to TKG: {content_type} - {text[:50]}...")
        return node_id
    
    def generate_advanced_response(self, user_input: str) -> str:
        """
        VERSIÓN CORREGIDA - Usar siempre AdvancedMemoryRetrieval
        """
        # FORZAR uso del sistema avanzado para TODAS las queries
        query_embedding = self.get_text_embedding(user_input)
        
        # DEBUG: Mostrar qué está haciendo
        print(f"DEBUG V2: Processing '{user_input}' with advanced system")
        
        # SIEMPRE usar el sistema avanzado
        response = self.memory_system.generate_smart_response(user_input, query_embedding)
        
        print(f"DEBUG V2: Advanced system returned: '{response}'")
        
        return response
    
    def generate_contextual_response(self, user_input: str, max_length: int = 50) -> str:
        """
        Respuesta contextual mejorada que también puede usar memoria
        """
        input_lower = user_input.lower()
        
        # Primero intentar extraer info de memoria para respuestas contextuales
        try:
            query_embedding = self.get_text_embedding(user_input)
            memories = self.memory_system.hybrid_search(user_input, query_embedding, max_results=3)
            
            if memories:
                all_content = " ".join([mem["content"].lower() for mem in memories])
                
                # Respuestas contextuales informadas por memoria
                if any(greeting in input_lower for greeting in ["hi", "hello", "hey"]):
                    # Verificar si hay nombre en memoria
                    import re
                    for mem in memories:
                        name_match = re.search(r"(?:i'm|my name is) (\w+)", mem["content"].lower())
                        if name_match:
                            name = name_match.group(1).capitalize()
                            return f"Hello {name}! Nice to meet you."
                    return "Hello! I'm glad to chat with you."
                
                # Si menciona trabajo/profesión
                if any(word in input_lower for word in ["work", "job"]):
                    if "teacher" in all_content:
                        return "That's great! Teaching is a wonderful profession."
                    elif "engineer" in all_content:
                        return "Engineering is fascinating work!"
                    
                # Si menciona hobbies/intereses
                if any(word in input_lower for word in ["love", "enjoy", "like"]):
                    if "reading" in all_content:
                        return "Reading is such a enriching hobby!"
                    elif "chess" in all_content:
                        return "Chess is an excellent strategic game!"
                    return "That sounds like a wonderful interest!"
        
        except Exception as e:
            print(f"DEBUG V2: Contextual memory lookup failed: {e}")
        
        # Fallbacks básicos
        if any(greeting in input_lower for greeting in ["hi", "hello", "hey"]):
            return "Hello! I'm glad to chat with you."
        elif any(word in input_lower for word in ["thank", "thanks"]):
            return "You're welcome! I'm here to help."
        elif any(word in input_lower for word in ["love", "enjoy", "like"]):
            return "That sounds interesting! Tell me more about what you enjoy."
        else:
            return "I see. Please continue sharing with me."
    
    def chat(self, user_input: str) -> str:
        """
        Interfaz principal CORREGIDA
        """
        logger.info(f"User input: {user_input}")

        # Añadir input a memoria
        self.add_to_memory(user_input, role="user")

        # Determinar tipo de query
        query_type = self.classify_content(user_input, "user")

        print(f"DEBUG V2: Classified '{user_input}' as '{query_type}'")

        # USAR SISTEMA AVANZADO PARA CASI TODO
        if query_type in ["memory_query", "contextual_query"]:
            print("DEBUG V2: Using advanced response for memory/contextual query")
            response = self.generate_advanced_response(user_input)
        else:
            # Para otros tipos, TAMBIÉN intentar sistema avanzado primero
            print("DEBUG V2: Trying advanced response for other query types")
            try:
                advanced_response = self.generate_advanced_response(user_input)
                
                # Si el sistema avanzado da una respuesta útil, usarla
                if (len(advanced_response) > 20 and 
                    "understand" not in advanced_response.lower() and
                    "tell me more" not in advanced_response.lower()):
                    response = advanced_response
                else:
                    response = self.generate_contextual_response(user_input)
            except:
                response = self.generate_contextual_response(user_input)

        # Añadir respuesta a memoria
        self.add_to_memory(response, role="assistant")

        logger.info(f"Assistant response: {response}")
        return response
    
    def get_memory_statistics(self) -> Dict:
        """
        Obtener estadísticas detalladas de memoria
        """
        tkg_stats = self.tkg.get_statistics()
        
        # Estadísticas adicionales del sistema avanzado
        total_memories = len([h for h in self.conversation_history if h["role"] == "user"])
        memory_queries = len([h for h in self.conversation_history 
                            if h.get("metadata", {}).get("content_type") == "memory_query"])
        
        stats = {
            "conversation_turns": len(self.conversation_history),
            "total_user_inputs": total_memories,
            "memory_queries": memory_queries,
            "tkg_nodes": tkg_stats["total_nodes"],
            "tkg_edges": tkg_stats["total_edges"],
            "node_types": tkg_stats["node_types"],
            "temporal_span_hours": tkg_stats.get("temporal_span_hours", 0),
            "avg_node_strength": tkg_stats["avg_node_strength"],
            "avg_edge_strength": tkg_stats["avg_edge_strength"],
            "memory_efficiency": memory_queries / total_memories if total_memories > 0 else 0
        }
        
        return stats
    
    def consolidate_memory(self):
        """
        Ejecutar consolidación avanzada de memoria
        """
        logger.info("Starting advanced memory consolidation...")
        self.tkg.consolidate_memory()
        logger.info("Advanced memory consolidation completed")
    
    def save_memory(self, filepath: str):
        """Guardar estado completo - sin save por ahora debido a JSON issues"""
        logger.info(f"Memory save requested to {filepath} - skipping due to serialization issues")
        # TODO: Fix JSON serialization and implement save
        pass
    
    def load_memory(self, filepath: str):
        """Cargar estado - placeholder"""
        logger.info(f"Memory load requested from {filepath} - not implemented yet")
        pass


def test_episodic_memory_llm_v2():
    """
    Test completo del modelo v2.0 con sistema avanzado
    """
    print("🧠 Testing EpisodicMemoryLLM v2.0...")
    
    # Inicializar modelo v2
    model = EpisodicMemoryLLM_V2(
        model_name="gpt2-medium",
        device="cpu",
        tkg_max_nodes=1000
    )
    
    # Test scenario más desafiante
    advanced_test_conversation = [
        "Hi, I'm Sarah and I work as a software engineer at Google",
        "I love hiking and outdoor activities", 
        "I also enjoy reading science fiction novels",
        "Yesterday I went to a great sushi restaurant in downtown",
        "What's my name?",
        "What do you know about my hobbies?",
        "Where do I work?",
        "What kind of food did I eat recently?",
        "Can you recommend outdoor activities for me?",
        "What's my profession?"
    ]
    
    print("\n" + "="*60)
    print("EPISODIC MEMORY LLM v2.0 TEST")
    print("="*60)
    
    results = []
    for i, user_input in enumerate(advanced_test_conversation):
        print(f"\n--- Turn {i+1} ---")
        
        start_time = time.time()
        response = model.chat(user_input)
        response_time = time.time() - start_time
        
        print(f"User: {user_input}")
        print(f"Assistant: {response}")
        print(f"Response time: {response_time:.2f}s")
        
        # Guardar para análisis
        results.append({
            "input": user_input,
            "response": response,
            "response_time": response_time
        })
        
        # Mostrar estadísticas cada 5 turnos
        if (i + 1) % 5 == 0:
            stats = model.get_memory_statistics()
            print(f"\n📊 Memory Stats:")
            print(f"  TKG Nodes: {stats['tkg_nodes']}")
            print(f"  TKG Edges: {stats['tkg_edges']}")
            print(f"  Node Types: {stats['node_types']}")
            print(f"  Memory Efficiency: {stats['memory_efficiency']:.2f}")
    
    # Análisis de calidad de respuestas
    print(f"\n🔍 Response Quality Analysis:")
    
    expected_responses = {
        "What's my name?": ["sarah"],
        "What do you know about my hobbies?": ["hiking", "reading", "outdoor"],
        "Where do I work?": ["google", "software engineer"],
        "What kind of food did I eat recently?": ["sushi", "restaurant"],
        "Can you recommend outdoor activities for me?": ["hiking", "outdoor"],
        "What's my profession?": ["software engineer", "engineer"]
    }
    
    accuracy_scores = []
    for result in results:
        user_input = result["input"]
        response = result["response"].lower()
        
        if user_input in expected_responses:
            expected_keywords = expected_responses[user_input]
            found_keywords = sum(1 for keyword in expected_keywords if keyword in response)
            accuracy = found_keywords / len(expected_keywords)
            accuracy_scores.append(accuracy)
            
            print(f"  Q: {user_input}")
            print(f"  A: {result['response']}")
            print(f"  Accuracy: {accuracy:.2f} ({found_keywords}/{len(expected_keywords)})")
    
    overall_accuracy = sum(accuracy_scores) / len(accuracy_scores) if accuracy_scores else 0
    avg_response_time = sum(r["response_time"] for r in results) / len(results)
    
    print(f"\n📈 Overall Performance:")
    print(f"  Overall Accuracy: {overall_accuracy:.2f} ({overall_accuracy*100:.1f}%)")
    print(f"  Average Response Time: {avg_response_time:.2f}s")
    print(f"  Memory Queries Handled: {len(accuracy_scores)}")
    
    # Test de consolidación
    print(f"\n🧠 Testing memory consolidation...")
    model.consolidate_memory()
    
    final_stats = model.get_memory_statistics()
    print(f"Final TKG nodes: {final_stats['tkg_nodes']}")
    print(f"Final TKG edges: {final_stats['tkg_edges']}")
    
    print("\n✅ EpisodicMemoryLLM v2.0 test completed!")
    return model, overall_accuracy, avg_response_time


def quick_comparison_test():
    """
    Test rápido de comparación con el modelo v1
    """
    print("\n🥊 Quick V1 vs V2 Comparison...")
    
    # Test scenario simple pero efectivo
    test_inputs = [
        "Hi, I'm Alice and I work as a teacher",
        "I love reading books",
        "What's my job?",
        "What do you know about my hobbies?"
    ]
    
    # Test v2
    print("\n🧠 Testing V2:")
    model_v2 = EpisodicMemoryLLM_V2(device="cpu")
    
    v2_results = []
    for inp in test_inputs:
        start_time = time.time()
        response = model_v2.chat(inp)
        response_time = time.time() - start_time
        v2_results.append({"input": inp, "response": response, "time": response_time})
        print(f"  {inp} → {response}")
    
    # Calcular accuracy para queries de memoria
    v2_accuracy = 0
    memory_queries = ["What's my job?", "What do you know about my hobbies?"]
    
    for result in v2_results:
        if result["input"] == "What's my job?" and "teacher" in result["response"].lower():
            v2_accuracy += 0.5
        elif result["input"] == "What do you know about my hobbies?" and "reading" in result["response"].lower():
            v2_accuracy += 0.5
    
    v2_avg_time = sum(r["time"] for r in v2_results) / len(v2_results)
    
    print(f"\n📊 V2 Performance:")
    print(f"  Accuracy: {v2_accuracy:.1f}/1.0 ({v2_accuracy*100:.0f}%)")
    print(f"  Avg Response Time: {v2_avg_time:.2f}s")
    
    return model_v2, v2_accuracy


if __name__ == "__main__":
    # Ejecutar test completo
    model, accuracy, avg_time = test_episodic_memory_llm_v2()
    
    # Ejecutar comparación rápida
    # quick_model, quick_accuracy = quick_comparison_test()
    
    print(f"\n🎯 FINAL RESULTS:")
    print(f"  Advanced Model Accuracy: {accuracy*100:.1f}%")
    print(f"  Target Accuracy: 70%")
    print(f"  Status: {'✅ TARGET ACHIEVED' if accuracy >= 0.7 else '🔧 NEEDS IMPROVEMENT'}")
