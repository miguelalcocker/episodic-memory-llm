# src/memory/interference_memory_model.py
"""
MEMORY INTERFERENCE MODEL - PSYCHOLOGICAL REALISM
Miguel's Revolutionary Implementation of Human Memory Interference
Target: Model proactive/retroactive interference like human brain
"""

import numpy as np
import time
import math
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
from enum import Enum
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)

class InterferenceType(Enum):
    """Types of memory interference"""
    PROACTIVE = "proactive"      # Old memories interfere with new ones
    RETROACTIVE = "retroactive"  # New memories interfere with old ones
    LATERAL = "lateral"          # Similar memories interfere with each other
    TEMPORAL = "temporal"        # Time-based interference

@dataclass
class InterferenceEffect:
    """Models interference between two memories"""
    source_memory: str
    target_memory: str
    interference_type: InterferenceType
    strength: float
    created_time: float
    semantic_similarity: float
    temporal_proximity: float
    resolution_attempts: int

class MemoryInterferenceModel:
    """
    REVOLUTIONARY: Human-inspired memory interference modeling
    
    Based on psychological research:
    1. Proactive interference (old → new)
    2. Retroactive interference (new → old)  
    3. Lateral interference (similar memories compete)
    4. Temporal interference (time-based conflicts)
    5. Interference resolution mechanisms
    """
    
    def __init__(self, max_interference_strength: float = 0.6):
        self.max_interference_strength = max_interference_strength
        
        # PSYCHOLOGICAL PARAMETERS from research
        self.INTERFERENCE_CONSTANTS = {
            InterferenceType.PROACTIVE: {
                "base_strength": 0.4,
                "similarity_weight": 0.6,
                "temporal_weight": 0.3,
                "decay_rate": 0.05,
                "resolution_difficulty": 0.7
            },
            InterferenceType.RETROACTIVE: {
                "base_strength": 0.5,
                "similarity_weight": 0.8,
                "temporal_weight": 0.4,
                "decay_rate": 0.03,
                "resolution_difficulty": 0.6
            },
            InterferenceType.LATERAL: {
                "base_strength": 0.3,
                "similarity_weight": 0.9,
                "temporal_weight": 0.2,
                "decay_rate": 0.08,
                "resolution_difficulty": 0.5
            },
            InterferenceType.TEMPORAL: {
                "base_strength": 0.2,
                "similarity_weight": 0.4,
                "temporal_weight": 0.8,
                "decay_rate": 0.1,
                "resolution_difficulty": 0.4
            }
        }
        
        # SIMILARITY THRESHOLDS for different interference types
        self.SIMILARITY_THRESHOLDS = {
            InterferenceType.PROACTIVE: 0.4,
            InterferenceType.RETROACTIVE: 0.5,
            InterferenceType.LATERAL: 0.6,
            InterferenceType.TEMPORAL: 0.3
        }
        
        # TEMPORAL WINDOWS for interference (in seconds)
        self.TEMPORAL_WINDOWS = {
            InterferenceType.PROACTIVE: 3600,    # 1 hour
            InterferenceType.RETROACTIVE: 1800,  # 30 minutes
            InterferenceType.LATERAL: 7200,      # 2 hours
            InterferenceType.TEMPORAL: 300       # 5 minutes
        }
        
        # Storage for interference effects
        self.active_interferences: Dict[str, InterferenceEffect] = {}
        self.interference_history: List[InterferenceEffect] = []
        self.resolution_attempts: Dict[str, int] = defaultdict(int)
        
        # Memory categorization for interference
        self.memory_categories = {
            "names": ["name", "called", "i'm", "my name"],
            "jobs": ["work", "job", "profession", "career", "employed"],
            "locations": ["live", "from", "located", "city", "country"],
            "preferences": ["love", "like", "enjoy", "favorite", "prefer"],
            "experiences": ["went", "visited", "experienced", "happened", "did"]
        }
        
        logger.info("🧠 Memory Interference Model initialized")
    
    def detect_interference(self, new_memory_id: str, new_content: str, 
                          existing_memories: Dict[str, Dict]) -> List[InterferenceEffect]:
        """
        Detect potential interference between new memory and existing memories
        """
        detected_interferences = []
        current_time = time.time()
        
        for existing_id, existing_data in existing_memories.items():
            if existing_id == new_memory_id:
                continue
            
            existing_content = existing_data.get("content", "")
            existing_timestamp = existing_data.get("timestamp", current_time)
            
            # Calculate similarities
            semantic_sim = self._calculate_semantic_similarity(new_content, existing_content)
            temporal_proximity = self._calculate_temporal_proximity(current_time, existing_timestamp)
            
            # Check for each type of interference
            for interference_type in InterferenceType:
                if self._should_create_interference(interference_type, semantic_sim, temporal_proximity):
                    
                    # Create interference effect
                    interference = self._create_interference_effect(
                        new_memory_id, existing_id, interference_type,
                        semantic_sim, temporal_proximity, current_time
                    )
                    
                    detected_interferences.append(interference)
                    
                    # Store in active interferences
                    interference_key = f"{interference_type.value}_{new_memory_id}_{existing_id}"
                    self.active_interferences[interference_key] = interference
        
        logger.debug(f"Detected {len(detected_interferences)} interference effects for {new_memory_id}")
        return detected_interferences
    
    def _calculate_semantic_similarity(self, content1: str, content2: str) -> float:
        """
        Calculate semantic similarity between two memory contents
        """
        # Tokenize and normalize
        words1 = set(content1.lower().split())
        words2 = set(content2.lower().split())
        
        # Basic Jaccard similarity
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        jaccard_sim = intersection / union if union > 0 else 0
        
        # Category-based similarity boost
        category_sim = self._calculate_category_similarity(content1, content2)
        
        # Combine similarities
        final_similarity = 0.7 * jaccard_sim + 0.3 * category_sim
        
        return min(final_similarity, 1.0)
    
    def _calculate_category_similarity(self, content1: str, content2: str) -> float:
        """
        Calculate similarity based on memory categories
        """
        content1_lower = content1.lower()
        content2_lower = content2.lower()
        
        # Check if both memories belong to same category
        for category, keywords in self.memory_categories.items():
            in_category1 = any(keyword in content1_lower for keyword in keywords)
            in_category2 = any(keyword in content2_lower for keyword in keywords)
            
            if in_category1 and in_category2:
                return 0.8  # High similarity for same category
        
        return 0.0
    
    def _calculate_temporal_proximity(self, time1: float, time2: float) -> float:
        """
        Calculate temporal proximity between two memories
        """
        time_diff = abs(time1 - time2)
        
        # Normalize to [0, 1] with exponential decay
        max_relevant_time = 86400  # 24 hours
        proximity = math.exp(-time_diff / max_relevant_time)
        
        return proximity
    
    def _should_create_interference(self, interference_type: InterferenceType, 
                                  semantic_sim: float, temporal_proximity: float) -> bool:
        """
        Determine if interference should be created based on thresholds
        """
        constants = self.INTERFERENCE_CONSTANTS[interference_type]
        threshold = self.SIMILARITY_THRESHOLDS[interference_type]
        
        # Weighted score combining semantic and temporal factors
        combined_score = (
            constants["similarity_weight"] * semantic_sim +
            constants["temporal_weight"] * temporal_proximity
        )
        
        return combined_score >= threshold
    
    def _create_interference_effect(self, source_id: str, target_id: str, 
                                  interference_type: InterferenceType,
                                  semantic_sim: float, temporal_proximity: float,
                                  current_time: float) -> InterferenceEffect:
        """
        Create interference effect between two memories
        """
        constants = self.INTERFERENCE_CONSTANTS[interference_type]
        
        # Calculate interference strength
        base_strength = constants["base_strength"]
        similarity_component = constants["similarity_weight"] * semantic_sim
        temporal_component = constants["temporal_weight"] * temporal_proximity
        
        interference_strength = min(
            base_strength + similarity_component + temporal_component,
            self.max_interference_strength
        )
        
        return InterferenceEffect(
            source_memory=source_id,
            target_memory=target_id,
            interference_type=interference_type,
            strength=interference_strength,
            created_time=current_time,
            semantic_similarity=semantic_sim,
            temporal_proximity=temporal_proximity,
            resolution_attempts=0
        )
    
    def apply_interference_effects(self, memory_id: str, base_strength: float) -> float:
        """
        Apply all active interference effects to a memory's strength
        """
        current_time = time.time()
        total_interference = 0.0
        
        # Find all interferences affecting this memory
        relevant_interferences = []
        for interference_key, interference in self.active_interferences.items():
            if interference.target_memory == memory_id:
                relevant_interferences.append(interference)
        
        # Apply each interference effect
        for interference in relevant_interferences:
            # Calculate age-based decay
            age = current_time - interference.created_time
            constants = self.INTERFERENCE_CONSTANTS[interference.interference_type]
            decay_factor = math.exp(-constants["decay_rate"] * age / 3600)  # hourly decay
            
            # Apply decayed interference
            current_interference = interference.strength * decay_factor
            total_interference += current_interference
            
            # Update interference strength
            interference.strength = current_interference
        
        # Calculate final strength with interference
        interference_factor = 1.0 - min(total_interference, 0.9)  # Cap at 90% reduction
        final_strength = base_strength * interference_factor
        
        logger.debug(f"Applied {len(relevant_interferences)} interferences to {memory_id}: "
                    f"{base_strength:.3f} → {final_strength:.3f}")
        
        return final_strength
    
    def attempt_interference_resolution(self, memory_id: str, access_count: int) -> float:
        """
        Attempt to resolve interference through repeated access (like humans do)
        """
        resolution_benefit = 0.0
        
        # Find interferences affecting this memory
        relevant_interferences = []
        for interference_key, interference in self.active_interferences.items():
            if interference.target_memory == memory_id:
                relevant_interferences.append((interference_key, interference))
        
        for interference_key, interference in relevant_interferences:
            # Increase resolution attempts
            interference.resolution_attempts += 1
            
            # Calculate resolution success based on attempts and difficulty
            constants = self.INTERFERENCE_CONSTANTS[interference.interference_type]
            difficulty = constants["resolution_difficulty"]
            
            # Resolution probability increases with attempts
            resolution_prob = min(interference.resolution_attempts * 0.1, 0.8)
            resolution_prob *= (1.0 - difficulty)
            
            # Apply resolution if successful
            if resolution_prob > 0.5:
                # Reduce interference strength
                reduction_factor = 0.2 * interference.resolution_attempts
                interference.strength *= (1.0 - reduction_factor)
                
                resolution_benefit += reduction_factor * 0.1
                
                # Remove very weak interferences
                if interference.strength < 0.1:
                    del self.active_interferences[interference_key]
                    logger.debug(f"Resolved interference {interference_key}")
        
        return resolution_benefit
    
    def simulate_interference_competition(self, competing_memories: List[str]) -> Dict[str, float]:
        """
        Simulate competition between similar memories (lateral interference)
        """
        competition_results = {}
        
        if len(competing_memories) < 2:
            return {mem_id: 1.0 for mem_id in competing_memories}
        
        # Calculate base competition strength for each memory
        base_strengths = {}
        for mem_id in competing_memories:
            # This would be connected to your actual memory strength system
            base_strengths[mem_id] = 1.0  # Placeholder
        
        # Apply lateral interference between competing memories
        for i, mem1 in enumerate(competing_memories):
            competition_factor = 1.0
            
            for j, mem2 in enumerate(competing_memories):
                if i != j:
                    # Find interference between these memories
                    interference_key = f"lateral_{mem1}_{mem2}"
                    if interference_key in self.active_interferences:
                        interference = self.active_interferences[interference_key]
                        competition_factor *= (1.0 - interference.strength * 0.3)
            
            competition_results[mem1] = competition_factor
        
        return competition_results
    
    def get_interference_statistics(self) -> Dict:
        """
        Get detailed statistics about current interference state
        """
        if not self.active_interferences:
            return {"total_interferences": 0}
        
        # Count by type
        type_counts = {}
        for interference in self.active_interferences.values():
            int_type = interference.interference_type.value
            type_counts[int_type] = type_counts.get(int_type, 0) + 1
        
        # Strength distribution
        strengths = [i.strength for i in self.active_interferences.values()]
        
        # Resolution statistics
        resolution_stats = {}
        for interference in self.active_interferences.values():
            attempts = interference.resolution_attempts
            if attempts > 0:
                resolution_stats[attempts] = resolution_stats.get(attempts, 0) + 1
        
        return {
            "total_interferences": len(self.active_interferences),
            "type_distribution": type_counts,
            "average_strength": np.mean(strengths) if strengths else 0,
            "max_strength": max(strengths) if strengths else 0,
            "min_strength": min(strengths) if strengths else 0,
            "strong_interferences": sum(1 for s in strengths if s > 0.4),
            "weak_interferences": sum(1 for s in strengths if s < 0.2),
            "resolution_attempts": resolution_stats,
            "total_resolved": len(self.interference_history)
        }
    
    def prune_expired_interferences(self, max_age_hours: float = 24.0) -> List[str]:
        """
        Remove interference effects that are too old or too weak
        """
        current_time = time.time()
        expired_keys = []
        
        for interference_key, interference in list(self.active_interferences.items()):
            age_hours = (current_time - interference.created_time) / 3600
            
            # Remove if too old or too weak
            if age_hours > max_age_hours or interference.strength < 0.05:
                expired_keys.append(interference_key)
                
                # Move to history
                self.interference_history.append(interference)
                del self.active_interferences[interference_key]
        
        logger.info(f"Pruned {len(expired_keys)} expired interference effects")
        return expired_keys
    
    def analyze_interference_patterns(self) -> Dict:
        """
        Analyze patterns in interference effects for insights
        """
        all_interferences = list(self.active_interferences.values()) + self.interference_history
        
        if not all_interferences:
            return {"pattern_analysis": "No interference data available"}
        
        # Memory pairs that frequently interfere
        memory_pairs = {}
        for interference in all_interferences:
            pair = tuple(sorted([interference.source_memory, interference.target_memory]))
            memory_pairs[pair] = memory_pairs.get(pair, 0) + 1
        
        # Most interfering memories
        source_counts = {}
        target_counts = {}
        for interference in all_interferences:
            source_counts[interference.source_memory] = source_counts.get(interference.source_memory, 0) + 1
            target_counts[interference.target_memory] = target_counts.get(interference.target_memory, 0) + 1
        
        # Type effectiveness
        type_effectiveness = {}
        for interference in all_interferences:
            int_type = interference.interference_type.value
            if int_type not in type_effectiveness:
                type_effectiveness[int_type] = []
            type_effectiveness[int_type].append(interference.strength)
        
        # Calculate averages
        for int_type in type_effectiveness:
            strengths = type_effectiveness[int_type]
            type_effectiveness[int_type] = {
                "count": len(strengths),
                "avg_strength": np.mean(strengths),
                "max_strength": max(strengths)
            }
        
        return {
            "frequent_interference_pairs": dict(sorted(memory_pairs.items(), key=lambda x: x[1], reverse=True)[:5]),
            "most_interfering_sources": dict(sorted(source_counts.items(), key=lambda x: x[1], reverse=True)[:5]),
            "most_interfered_targets": dict(sorted(target_counts.items(), key=lambda x: x[1], reverse=True)[:5]),
            "interference_type_effectiveness": type_effectiveness,
            "total_interference_events": len(all_interferences)
        }


def test_interference_memory_model():
    """
    Test the interference memory model with realistic scenarios
    """
    print("🧠 Testing Memory Interference Model...")
    
    # Initialize model
    interference_model = MemoryInterferenceModel()
    
    # Create test memories that should interfere
    test_memories = {
        "name_1": {
            "content": "My name is Alice",
            "timestamp": time.time() - 3600,  # 1 hour ago
            "type": "personal_info"
        },
        "name_2": {
            "content": "My name is Alicia",  # Similar to Alice - should interfere
            "timestamp": time.time() - 1800,  # 30 minutes ago
            "type": "personal_info"
        },
        "job_1": {
            "content": "I work as a software engineer",
            "timestamp": time.time() - 7200,  # 2 hours ago
            "type": "personal_info"
        },
        "job_2": {
            "content": "I work as a data scientist",  # Conflicting job info
            "timestamp": time.time() - 600,   # 10 minutes ago
            "type": "personal_info"
        },
        "hobby_1": {
            "content": "I love reading science fiction",
            "timestamp": time.time() - 5400,  # 1.5 hours ago
            "type": "preferences"
        },
        "hobby_2": {
            "content": "I love reading fantasy novels",  # Similar hobby
            "timestamp": time.time() - 300,   # 5 minutes ago
            "type": "preferences"
        }
    }
    
    print("\n📝 Adding memories and detecting interference...")
    
    # Add memories one by one and detect interference
    for memory_id, memory_data in test_memories.items():
        existing_memories = {k: v for k, v in test_memories.items() if k != memory_id}
        
        interferences = interference_model.detect_interference(
            memory_id, memory_data["content"], existing_memories
        )
        
        print(f"\nMemory: {memory_data['content']}")
        print(f"Detected {len(interferences)} interference effects:")
        
        for interference in interferences:
            print(f"  - {interference.interference_type.value}: "
                  f"{interference.source_memory} → {interference.target_memory} "
                  f"(strength: {interference.strength:.3f})")
    
    print("\n🎯 Testing interference application...")
    
    # Test interference effects on memory strength
    for memory_id in ["name_1", "name_2", "job_1", "job_2"]:
        base_strength = 1.0
        interfered_strength = interference_model.apply_interference_effects(memory_id, base_strength)
        
        print(f"Memory {memory_id}: {base_strength:.3f} → {interfered_strength:.3f} "
              f"(reduction: {(1-interfered_strength)*100:.1f}%)")
    
    print("\n🔧 Testing interference resolution...")
    
    # Simulate repeated access to resolve interference
    for attempt in range(3):
        print(f"\nResolution attempt {attempt + 1}:")
        
        for memory_id in ["name_1", "name_2"]:
            resolution_benefit = interference_model.attempt_interference_resolution(memory_id, attempt + 1)
            print(f"  {memory_id}: resolution benefit = {resolution_benefit:.3f}")
    
    print("\n🏆 Testing lateral interference competition...")
    
    # Test competition between similar memories
    competing_names = ["name_1", "name_2"]
    competition_results = interference_model.simulate_interference_competition(competing_names)
    
    print("Name competition results:")
    for memory_id, competition_factor in competition_results.items():
        print(f"  {memory_id}: competition factor = {competition_factor:.3f}")
    
    print("\n📊 Interference statistics:")
    stats = interference_model.get_interference_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("\n🔍 Analyzing interference patterns...")
    patterns = interference_model.analyze_interference_patterns()
    for key, value in patterns.items():
        print(f"  {key}: {value}")
    
    print("\n🗑️ Testing interference pruning...")
    
    # Test pruning of expired interferences
    pruned = interference_model.prune_expired_interferences(max_age_hours=0.1)  # Very short for testing
    print(f"Pruned {len(pruned)} expired interferences")
    
    print("\n✅ Memory Interference Model test completed!")
    return interference_model


if __name__ == "__main__":
    test_interference_memory_model()