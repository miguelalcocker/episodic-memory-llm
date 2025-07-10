# src/models/episodic_memory_llm_v3_fixed.py
"""
EpisodicMemoryLLM V3.0 - FIXED VERSION
Import issues resolved - ready for BREAKTHROUGH testing
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

# Import our components with correct path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from memory.temporal_knowledge_graph import TemporalKnowledgeGraph
from memory.advanced_memory_retrieval_v3 import AdvancedMemoryRetrieval_V3

logger = logging.getLogger(__name__)

class EpisodicMemoryLLM_V3(nn.Module):
    """
    BREAKTHROUGH: EpisodicMemoryLLM V3 with Advanced Preference System
    FIXED VERSION - All imports resolved
    """
    
    def __init__(
        self,
        model_name: str = "gpt2-medium",
        max_context_length: int = 1024,
        tkg_max_nodes: int = 5000,
        tkg_decay_rate: float = 0.1,
        device: str = None,
        advanced_mode: bool = True
    ):
        super().__init__()
        
        # Device setup
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        
        # Load base LLM
        logger.info(f"🚀 Loading {model_name} for BREAKTHROUGH V3...")
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model.to(self.device)
        self.model.eval()
        
        # Initialize Enhanced Temporal Knowledge Graph
        self.tkg = TemporalKnowledgeGraph(
            max_nodes=tkg_max_nodes,
            decay_rate=tkg_decay_rate
        )
        
        # BREAKTHROUGH: Initialize V3 Advanced Memory System
        self.memory_system = AdvancedMemoryRetrieval_V3(self.tkg, self.tokenizer)
        
        self.max_context_length = max_context_length
        self.conversation_history = []
        self.advanced_mode = advanced_mode
        
        # Enhanced generation parameters
        self.generation_config = {
            "temperature": 0.7,
            "top_p": 0.9,
            "do_sample": True,
            "repetition_penalty": 1.1,
            "no_repeat_ngram_size": 3
        }
        
        # Performance tracking
        self.performance_metrics = {
            "total_queries": 0,
            "successful_recalls": 0,
            "average_confidence": 0.0,
            "response_times": []
        }
        
        print(f"🧠 V3 BREAKTHROUGH INITIALIZED:")
        print(f"   - AdvancedMemoryRetrieval_V3: {self.memory_system is not None}")
        print(f"   - TKG connected: {self.memory_system.tkg is not None}")
        print(f"   - Preference System: {hasattr(self.memory_system, 'preference_system')}")
        print(f"   - Device: {device}")
        print(f"   - Advanced Mode: {advanced_mode}")
    
    def get_text_embedding(self, text: str) -> np.ndarray:
        """Enhanced embedding generation"""
        text = text.strip()[:800]
        
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding=True
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.transformer(**inputs)
            hidden_states = outputs.last_hidden_state
            mean_embeddings = hidden_states.mean(dim=1)
            max_embeddings = hidden_states.max(dim=1)[0]
            combined_embeddings = torch.cat([mean_embeddings, max_embeddings], dim=1)
        
        return combined_embeddings.cpu().numpy()[0]
    
    def classify_content_v3(self, text: str, role: str) -> str:
        """Enhanced content classification"""
        text_lower = text.lower().strip()
        
        if role == "user":
            # Memory query detection
            memory_indicators = [
                "what's my", "what is my", "whats my", "what are my",
                "what do you know about my", "what do you remember about my",
                "tell me about my", "do you remember my", "remind me of my",
                "what did i", "where do i", "where did i", "when did i",
                "remember", "recall", "know about"
            ]
            
            if any(indicator in text_lower for indicator in memory_indicators):
                return "memory_query"
            
            # Contextual query detection
            context_indicators = [
                "recommend", "suggest", "what should", "can you help",
                "advice", "help me", "what would you", "based on"
            ]
            
            if any(indicator in text_lower for indicator in context_indicators):
                return "contextual_query"
            
            # Personal info detection
            personal_indicators = [
                "my name is", "i'm", "i am", "i work as", "work at",
                "i live in", "from", "my job", "my profession"
            ]
            
            if any(indicator in text_lower for indicator in personal_indicators):
                return "personal_info"
            
            # Preference detection
            preference_indicators = [
                "i love", "i like", "i enjoy", "my favorite", "my hobby",
                "i prefer", "passionate about", "interested in"
            ]
            
            if any(indicator in text_lower for indicator in preference_indicators):
                return "preferences"
            
            # Experience detection
            experience_indicators = [
                "yesterday", "last week", "last month", "ago", "when i",
                "i went", "i visited", "i traveled", "recently"
            ]
            
            if any(indicator in text_lower for indicator in experience_indicators):
                return "episodic"
            
            return "general"
        else:
            return "response"
    
    def add_to_memory_v3(self, text: str, role: str = "user", metadata: Dict = None):
        """Enhanced memory addition"""
        content_type = self.classify_content_v3(text, role)
        
        enhanced_metadata = {
            "role": role,
            "content_type": content_type,
            "conversation_turn": len(self.conversation_history),
            "timestamp": time.time(),
            "v3_enhanced": True
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
            "metadata": enhanced_metadata,
            "embedding": embedding
        })
        
        return node_id
    
    def generate_advanced_response_v3(self, user_input: str) -> Tuple[str, Dict]:
        """V3 Advanced response generation"""
        start_time = time.time()
        
        query_embedding = self.get_text_embedding(user_input)
        
        print(f"🧠 V3 Processing: '{user_input}' with breakthrough system")
        
        response = self.memory_system.generate_smart_response_v3(user_input, query_embedding)
        
        response_time = time.time() - start_time
        
        # Performance tracking
        self.performance_metrics["total_queries"] += 1
        self.performance_metrics["response_times"].append(response_time)
        
        confidence = self._calculate_response_confidence(response, user_input)
        
        if confidence > 0.7:
            self.performance_metrics["successful_recalls"] += 1
        
        self.performance_metrics["average_confidence"] = (
            (self.performance_metrics["average_confidence"] * (self.performance_metrics["total_queries"] - 1) + confidence) /
            self.performance_metrics["total_queries"]
        )
        
        performance_data = {
            "response_time": response_time,
            "confidence": confidence,
            "query_type": self.classify_content_v3(user_input, "user")
        }
        
        print(f"🎯 V3 Response: '{response[:60]}...' (confidence: {confidence:.2f})")
        
        return response, performance_data
    
    def _calculate_response_confidence(self, response: str, query: str) -> float:
        """Calculate confidence score"""
        confidence = 0.5
        
        # Boost for specific information
        if any(indicator in response.lower() for indicator in ["you work", "you enjoy", "your name", "you went"]):
            confidence += 0.3
        
        # Penalty for generic responses
        if any(generic in response.lower() for generic in ["i understand", "tell me more", "i don't have"]):
            confidence -= 0.2
        
        # Boost for detailed responses
        if len(response.split()) > 8:
            confidence += 0.1
        
        # Query-specific boosts
        query_lower = query.lower()
        if "name" in query_lower and "name" in response.lower():
            confidence += 0.2
        elif "job" in query_lower and ("work" in response.lower() or "job" in response.lower()):
            confidence += 0.2
        elif "hobbies" in query_lower and ("enjoy" in response.lower() or "love" in response.lower()):
            confidence += 0.2
        
        return max(0.0, min(1.0, confidence))
    
    def chat_v3(self, user_input: str) -> Dict:
        """Enhanced chat interface"""
        logger.info(f"V3 User input: {user_input}")
        
        self.add_to_memory_v3(user_input, role="user")
        
        response, performance_data = self.generate_advanced_response_v3(user_input)
        
        self.add_to_memory_v3(response, role="assistant")
        
        result = {
            "response": response,
            "performance": performance_data,
            "memory_stats": self.get_memory_statistics_v3(),
            "conversation_turn": len(self.conversation_history) // 2
        }
        
        logger.info(f"V3 Assistant response: {response}")
        return result
    
    def get_memory_statistics_v3(self) -> Dict:
        """Enhanced memory statistics"""
        tkg_stats = self.tkg.get_statistics()
        
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
            "memory_efficiency": memory_queries / total_memories if total_memories > 0 else 0,
            "v3_performance": self.performance_metrics,
            "success_rate": (self.performance_metrics["successful_recalls"] / 
                           max(1, self.performance_metrics["total_queries"])),
            "avg_response_time": (sum(self.performance_metrics["response_times"]) / 
                                max(1, len(self.performance_metrics["response_times"]))),
            "system_version": "V3_BREAKTHROUGH"
        }
        
        return stats
    
    def benchmark_v3(self, test_scenarios: List[Dict]) -> Dict:
        """V3 Benchmarking"""
        print("🚀 Running V3 BREAKTHROUGH Benchmark...")
        
        results = {
            "model_version": "V3_BREAKTHROUGH",
            "scenarios": [],
            "overall_metrics": {
                "total_accuracy": 0.0,
                "category_accuracies": {},
                "response_times": [],
                "confidence_scores": []
            }
        }
        
        total_score = 0
        total_tests = 0
        
        for scenario_idx, scenario in enumerate(test_scenarios):
            print(f"\n🎯 V3 Scenario: {scenario['name']}")
            
            # Reset conversation
            self.conversation_history = []
            self.tkg = TemporalKnowledgeGraph(max_nodes=1000, decay_rate=0.1)
            self.memory_system = AdvancedMemoryRetrieval_V3(self.tkg, self.tokenizer)
            
            scenario_result = {
                "scenario_name": scenario["name"],
                "setup_responses": [],
                "test_results": [],
                "scenario_accuracy": 0.0
            }
            
            # Setup phase
            for setup_input in scenario["setup"]:
                result = self.chat_v3(setup_input)
                scenario_result["setup_responses"].append({
                    "input": setup_input,
                    "response": result["response"],
                    "performance": result["performance"]
                })
            
            # Test phase
            scenario_score = 0
            scenario_tests = 0
            
            for test in scenario["tests"]:
                start_time = time.time()
                result = self.chat_v3(test["query"])
                response_time = time.time() - start_time
                
                score = self._evaluate_response_v3(result["response"], test["expected_keywords"])
                
                test_result = {
                    "query": test["query"],
                    "response": result["response"],
                    "expected_keywords": test["expected_keywords"],
                    "score": score,
                    "response_time": response_time,
                    "type": test["type"],
                    "v3_performance": result["performance"]
                }
                
                scenario_result["test_results"].append(test_result)
                results["overall_metrics"]["response_times"].append(response_time)
                results["overall_metrics"]["confidence_scores"].append(result["performance"]["confidence"])
                
                scenario_score += score
                scenario_tests += 1
                total_score += score
                total_tests += 1
                
                print(f"   🔍 {test['query']}")
                print(f"      Response: {result['response'][:60]}...")
                print(f"      Score: {score:.2f} | Time: {response_time:.2f}s | Confidence: {result['performance']['confidence']:.2f}")
            
            scenario_result["scenario_accuracy"] = scenario_score / max(1, scenario_tests)
            results["scenarios"].append(scenario_result)
        
        # Calculate overall metrics
        results["overall_metrics"]["total_accuracy"] = total_score / max(1, total_tests)
        results["overall_metrics"]["avg_response_time"] = sum(results["overall_metrics"]["response_times"]) / max(1, len(results["overall_metrics"]["response_times"]))
        results["overall_metrics"]["avg_confidence"] = sum(results["overall_metrics"]["confidence_scores"]) / max(1, len(results["overall_metrics"]["confidence_scores"]))
        
        # Category breakdown
        category_scores = {}
        for scenario in results["scenarios"]:
            for test_result in scenario["test_results"]:
                test_type = test_result["type"]
                if test_type not in category_scores:
                    category_scores[test_type] = []
                category_scores[test_type].append(test_result["score"])
        
        for category, scores in category_scores.items():
            results["overall_metrics"]["category_accuracies"][category] = sum(scores) / len(scores)
        
        print(f"\n🏆 V3 BREAKTHROUGH RESULTS:")
        print(f"   Overall Accuracy: {results['overall_metrics']['total_accuracy']:.1%}")
        print(f"   Average Confidence: {results['overall_metrics']['avg_confidence']:.2f}")
        print(f"   Average Response Time: {results['overall_metrics']['avg_response_time']:.2f}s")
        
        return results
    
    def _evaluate_response_v3(self, response: str, expected_keywords: List[str]) -> float:
        """Enhanced response evaluation"""
        response_lower = response.lower()
        
        # Basic keyword matching
        found_keywords = sum(1 for keyword in expected_keywords 
                           if keyword.lower() in response_lower)
        base_score = found_keywords / len(expected_keywords) if expected_keywords else 0
        
        # Semantic matching bonus
        semantic_bonus = 0
        for keyword in expected_keywords:
            if keyword.lower() not in response_lower:
                if self._semantic_match(keyword, response_lower):
                    semantic_bonus += 0.1
        
        # Response quality bonus
        quality_bonus = 0
        if len(response.split()) > 5:
            quality_bonus += 0.1
        
        if any(specific in response_lower for specific in ["you work as", "you enjoy", "your name is"]):
            quality_bonus += 0.15
        
        # Generic response penalty
        generic_penalty = 0
        if any(generic in response_lower for generic in ["i understand", "tell me more", "i don't have"]):
            generic_penalty = 0.2
        
        final_score = min(1.0, base_score + semantic_bonus + quality_bonus - generic_penalty)
        return max(0.0, final_score)
    
    def _semantic_match(self, keyword: str, response: str) -> bool:
        """Check for semantic matches"""
        semantic_mappings = {
            "teacher": ["educator", "instructor", "teaching"],
            "software engineer": ["developer", "programmer", "coding"],
            "reading": ["books", "novels", "literature"],
            "chess": ["board games", "strategy games"],
            "hiking": ["outdoor activities", "nature", "walking"],
            "restaurant": ["dining", "eating", "meal"],
            "google": ["alphabet", "tech company"]
        }
        
        keyword_lower = keyword.lower()
        if keyword_lower in semantic_mappings:
            return any(synonym in response for synonym in semantic_mappings[keyword_lower])
        
        return False


def create_breakthrough_test_scenarios():
    """Create comprehensive test scenarios"""
    return [
        {
            "name": "Personal Info Mastery",
            "setup": [
                "Hi, I'm Sarah and I work as a software engineer at Google",
                "I love reading mystery novels and science fiction books",
                "I also enjoy playing chess and hiking on weekends"
            ],
            "tests": [
                {
                    "query": "What's my name?",
                    "expected_keywords": ["Sarah"],
                    "type": "name_recall"
                },
                {
                    "query": "What's my job?",
                    "expected_keywords": ["software engineer", "engineer"],
                    "type": "job_recall"
                },
                {
                    "query": "Where do I work?",
                    "expected_keywords": ["Google"],
                    "type": "company_recall"
                },
                {
                    "query": "What do you know about my hobbies?",
                    "expected_keywords": ["reading", "chess", "hiking"],
                    "type": "preference_recall"
                },
                {
                    "query": "What kind of books do I like?",
                    "expected_keywords": ["mystery", "science fiction"],
                    "type": "specific_preference_recall"
                }
            ]
        }
    ]