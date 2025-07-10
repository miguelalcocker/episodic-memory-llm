# src/memory/ebbinghaus_memory_system.py
"""
EBBINGHAUS FORGETTING CURVES - COGNITIVE BREAKTHROUGH
Miguel's Revolutionary Integration of Human Memory Science
Target: Transform 85.7% → 92%+ through psychological realism
"""

import numpy as np
import time
import math
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class MemoryType(Enum):
    """Types of memory with different forgetting characteristics"""
    SEMANTIC = "semantic"        # Facts, concepts (slower forgetting)
    EPISODIC = "episodic"       # Personal experiences (faster forgetting)
    PROCEDURAL = "procedural"   # Skills, habits (very slow forgetting)
    WORKING = "working"         # Temporary information (very fast forgetting)

@dataclass
class MemoryStrength:
    """Memory strength with psychological backing"""
    initial_strength: float
    current_strength: float
    rehearsal_count: int
    last_accessed: float
    creation_time: float
    memory_type: MemoryType
    interference_resistance: float
    consolidation_level: float

class EbbinghausMemorySystem:
    """
    REVOLUTIONARY: Human-inspired memory system based on Ebbinghaus curves
    
    Key innovations:
    1. Psychologically accurate forgetting curves
    2. Memory type-specific decay rates
    3. Rehearsal strengthening effects
    4. Interference modeling
    5. Consolidation simulation
    """
    
    def __init__(self, base_decay_rate: float = 0.1):
        self.base_decay_rate = base_decay_rate
        
        # PSYCHOLOGICAL CONSTANTS from Ebbinghaus research
        self.FORGETTING_CURVE_CONSTANTS = {
            MemoryType.SEMANTIC: {
                "initial_retention": 0.9,    # 90% retention after 1 hour
                "decay_rate": 0.05,          # Slower decay for facts
                "consolidation_boost": 0.3,  # Strong consolidation
                "interference_resistance": 0.8
            },
            MemoryType.EPISODIC: {
                "initial_retention": 0.7,    # 70% retention after 1 hour
                "decay_rate": 0.12,          # Faster decay for experiences
                "consolidation_boost": 0.5,  # Moderate consolidation
                "interference_resistance": 0.6
            },
            MemoryType.PROCEDURAL: {
                "initial_retention": 0.95,   # 95% retention (skills stick)
                "decay_rate": 0.02,          # Very slow decay
                "consolidation_boost": 0.8,  # Very strong consolidation
                "interference_resistance": 0.9
            },
            MemoryType.WORKING: {
                "initial_retention": 0.4,    # 40% retention after 1 hour
                "decay_rate": 0.25,          # Very fast decay
                "consolidation_boost": 0.1,  # Minimal consolidation
                "interference_resistance": 0.3
            }
        }
        
        # REHEARSAL EFFECTS (spaced repetition)
        self.REHEARSAL_MULTIPLIERS = {
            1: 1.0,    # First exposure
            2: 1.3,    # Second exposure
            3: 1.6,    # Third exposure
            4: 1.9,    # Fourth exposure
            5: 2.2,    # Fifth exposure
            6: 2.5     # Sixth+ exposure (plateau)
        }
        
        # INTERFERENCE PARAMETERS
        self.INTERFERENCE_DECAY = 0.02  # How much interference reduces over time
        self.MAX_INTERFERENCE = 0.4     # Maximum interference effect
        
        # CONSOLIDATION PARAMETERS
        self.CONSOLIDATION_THRESHOLD = 86400  # 24 hours for consolidation
        self.CONSOLIDATION_RATE = 0.05        # Rate of consolidation
        
        # Memory storage
        self.memory_strengths: Dict[str, MemoryStrength] = {}
        self.interference_matrix: Dict[Tuple[str, str], float] = {}
        
        logger.info("🧠 Ebbinghaus Memory System initialized with psychological realism")
    
    def classify_memory_type(self, content: str, node_type: str) -> MemoryType:
        """
        Classify memory type based on content and context
        """
        content_lower = content.lower()
        
        # Working memory (temporary, contextual)
        if any(indicator in content_lower for indicator in 
               ["currently", "right now", "at the moment", "just now"]):
            return MemoryType.WORKING
        
        # Procedural memory (skills, habits, processes)
        if any(indicator in content_lower for indicator in
               ["how to", "procedure", "process", "method", "technique"]):
            return MemoryType.PROCEDURAL
        
        # Episodic memory (personal experiences, events)
        if any(indicator in content_lower for indicator in
               ["yesterday", "last week", "when i", "i went", "i experienced", "i remember"]):
            return MemoryType.EPISODIC
        
        # Semantic memory (facts, concepts, knowledge)
        if node_type in ["personal_info", "preferences"] or \
           any(indicator in content_lower for indicator in
               ["my name", "i work", "i am", "i like", "i love", "my favorite"]):
            return MemoryType.SEMANTIC
        
        # Default to semantic for general knowledge
        return MemoryType.SEMANTIC
    
    def add_memory(self, node_id: str, content: str, node_type: str, 
                   initial_strength: float = 1.0) -> None:
        """
        Add new memory with Ebbinghaus-based strength calculation
        """
        current_time = time.time()
        memory_type = self.classify_memory_type(content, node_type)
        
        # Calculate initial strength based on memory type
        type_constants = self.FORGETTING_CURVE_CONSTANTS[memory_type]
        adjusted_strength = initial_strength * type_constants["initial_retention"]
        
        # Create memory strength object
        memory_strength = MemoryStrength(
            initial_strength=adjusted_strength,
            current_strength=adjusted_strength,
            rehearsal_count=1,
            last_accessed=current_time,
            creation_time=current_time,
            memory_type=memory_type,
            interference_resistance=type_constants["interference_resistance"],
            consolidation_level=0.0
        )
        
        self.memory_strengths[node_id] = memory_strength
        
        # Calculate interference with existing memories
        self._calculate_interference(node_id, content)
        
        logger.debug(f"Added {memory_type.value} memory: {node_id} with strength {adjusted_strength:.3f}")
    
    def _calculate_interference(self, new_node_id: str, new_content: str) -> None:
        """
        Calculate interference between new memory and existing memories
        """
        new_content_words = set(new_content.lower().split())
        
        for existing_node_id, existing_strength in self.memory_strengths.items():
            if existing_node_id == new_node_id:
                continue
            
            # Calculate semantic similarity for interference
            # This is simplified - in real implementation, use embeddings
            similarity = self._calculate_content_similarity(new_content, existing_node_id)
            
            if similarity > 0.3:  # Threshold for interference
                interference_strength = similarity * self.MAX_INTERFERENCE
                
                # Reduce interference based on time separation
                time_diff = time.time() - existing_strength.last_accessed
                time_factor = math.exp(-time_diff / 3600)  # Hourly decay
                
                final_interference = interference_strength * time_factor
                
                # Store bidirectional interference
                self.interference_matrix[(new_node_id, existing_node_id)] = final_interference
                self.interference_matrix[(existing_node_id, new_node_id)] = final_interference
    
    def _calculate_content_similarity(self, content: str, node_id: str) -> float:
        """
        Simplified content similarity calculation
        In real implementation, this would use embeddings
        """
        # This is a placeholder - you'd use your existing embedding similarity
        # For now, using simple word overlap
        content_words = set(content.lower().split())
        
        # This would need to be connected to your actual content storage
        # For now, return moderate similarity for demonstration
        return 0.5  # Placeholder
    
    def access_memory(self, node_id: str) -> float:
        """
        Access memory and update strength according to Ebbinghaus curves
        """
        if node_id not in self.memory_strengths:
            return 0.0
        
        current_time = time.time()
        memory = self.memory_strengths[node_id]
        
        # Calculate time since last access
        time_since_access = current_time - memory.last_accessed
        
        # Apply Ebbinghaus forgetting curve
        memory.current_strength = self._apply_forgetting_curve(memory, time_since_access)
        
        # Apply interference effects
        memory.current_strength = self._apply_interference(node_id, memory.current_strength)
        
        # Apply consolidation effects
        memory.current_strength = self._apply_consolidation(memory, current_time)
        
        # Rehearsal effect (accessing strengthens memory)
        memory.rehearsal_count += 1
        rehearsal_multiplier = self.REHEARSAL_MULTIPLIERS.get(
            memory.rehearsal_count, 
            self.REHEARSAL_MULTIPLIERS[6]  # Cap at 6+ rehearsals
        )
        memory.current_strength *= rehearsal_multiplier
        
        # Update access time
        memory.last_accessed = current_time
        
        # Cap strength at reasonable maximum
        memory.current_strength = min(memory.current_strength, 3.0)
        
        logger.debug(f"Accessed memory {node_id}: strength = {memory.current_strength:.3f}")
        
        return memory.current_strength
    
    def _apply_forgetting_curve(self, memory: MemoryStrength, time_elapsed: float) -> float:
        """
        Apply Ebbinghaus forgetting curve based on memory type
        
        Formula: R(t) = R₀ * e^(-t/S)
        Where R(t) = retention at time t, R₀ = initial retention, S = strength factor
        """
        type_constants = self.FORGETTING_CURVE_CONSTANTS[memory.memory_type]
        
        # Convert time to hours for calculation
        hours_elapsed = time_elapsed / 3600
        
        # Calculate decay based on memory type
        decay_rate = type_constants["decay_rate"]
        
        # Apply exponential decay with type-specific parameters
        retention_factor = math.exp(-decay_rate * hours_elapsed)
        
        return memory.current_strength * retention_factor
    
    def _apply_interference(self, node_id: str, current_strength: float) -> float:
        """
        Apply interference effects from competing memories
        """
        total_interference = 0.0
        
        for (source, target), interference in self.interference_matrix.items():
            if target == node_id:
                # Reduce interference over time
                interference_age = time.time() - self.memory_strengths[source].creation_time
                time_decay = math.exp(-self.INTERFERENCE_DECAY * interference_age / 3600)
                active_interference = interference * time_decay
                total_interference += active_interference
        
        # Apply interference resistance based on memory type
        memory = self.memory_strengths[node_id]
        resistance_factor = memory.interference_resistance
        
        effective_interference = total_interference * (1 - resistance_factor)
        
        return current_strength * (1 - effective_interference)
    
    def _apply_consolidation(self, memory: MemoryStrength, current_time: float) -> float:
        """
        Apply memory consolidation effects (strengthening over time)
        """
        time_since_creation = current_time - memory.creation_time
        
        # Consolidation happens over 24 hours
        if time_since_creation >= self.CONSOLIDATION_THRESHOLD:
            consolidation_progress = min(time_since_creation / self.CONSOLIDATION_THRESHOLD, 1.0)
            
            type_constants = self.FORGETTING_CURVE_CONSTANTS[memory.memory_type]
            consolidation_boost = type_constants["consolidation_boost"]
            
            # Update consolidation level
            memory.consolidation_level = consolidation_progress * consolidation_boost
            
            # Apply consolidation boost to strength
            return memory.current_strength * (1 + memory.consolidation_level)
        
        return memory.current_strength
    
    def get_memory_strength(self, node_id: str) -> float:
        """
        Get current memory strength without updating (for queries)
        """
        if node_id not in self.memory_strengths:
            return 0.0
        
        return self.memory_strengths[node_id].current_strength
    
    def consolidate_all_memories(self) -> Dict[str, float]:
        """
        Perform global memory consolidation (simulate sleep)
        """
        logger.info("🌙 Starting global memory consolidation (sleep simulation)")
        
        consolidation_results = {}
        current_time = time.time()
        
        for node_id, memory in self.memory_strengths.items():
            # Apply consolidation to all memories
            old_strength = memory.current_strength
            
            # Consolidation strengthens important memories
            if memory.rehearsal_count >= 3:  # Frequently accessed memories
                consolidation_boost = 0.2 * memory.rehearsal_count
                memory.current_strength *= (1 + consolidation_boost)
            
            # Weak memories get further weakened during consolidation
            if memory.current_strength < 0.3:
                memory.current_strength *= 0.8
            
            # Update consolidation level
            time_since_creation = current_time - memory.creation_time
            if time_since_creation >= self.CONSOLIDATION_THRESHOLD:
                memory.consolidation_level = 1.0
            
            consolidation_results[node_id] = memory.current_strength - old_strength
        
        logger.info(f"🧠 Consolidated {len(consolidation_results)} memories")
        return consolidation_results
    
    def get_memory_statistics(self) -> Dict:
        """
        Get detailed statistics about memory system
        """
        if not self.memory_strengths:
            return {"total_memories": 0}
        
        strengths = [m.current_strength for m in self.memory_strengths.values()]
        
        # Memory type distribution
        type_distribution = {}
        for memory in self.memory_strengths.values():
            mem_type = memory.memory_type.value
            type_distribution[mem_type] = type_distribution.get(mem_type, 0) + 1
        
        # Consolidation statistics
        consolidated_count = sum(1 for m in self.memory_strengths.values() 
                               if m.consolidation_level > 0.5)
        
        return {
            "total_memories": len(self.memory_strengths),
            "average_strength": np.mean(strengths),
            "max_strength": max(strengths),
            "min_strength": min(strengths),
            "strong_memories": sum(1 for s in strengths if s > 1.0),
            "weak_memories": sum(1 for s in strengths if s < 0.3),
            "memory_type_distribution": type_distribution,
            "consolidated_memories": consolidated_count,
            "total_interferences": len(self.interference_matrix),
            "average_rehearsal": np.mean([m.rehearsal_count for m in self.memory_strengths.values()])
        }
    
    def prune_weak_memories(self, threshold: float = 0.1) -> List[str]:
        """
        Remove memories that have become too weak (natural forgetting)
        """
        weak_memories = []
        
        for node_id, memory in list(self.memory_strengths.items()):
            if memory.current_strength < threshold:
                weak_memories.append(node_id)
                del self.memory_strengths[node_id]
                
                # Remove from interference matrix
                keys_to_remove = [k for k in self.interference_matrix.keys() 
                                if k[0] == node_id or k[1] == node_id]
                for key in keys_to_remove:
                    del self.interference_matrix[key]
        
        logger.info(f"🗑️ Pruned {len(weak_memories)} weak memories")
        return weak_memories


def test_ebbinghaus_memory_system():
    """
    Test the Ebbinghaus memory system with realistic scenarios
    """
    print("🧠 Testing Ebbinghaus Memory System...")
    
    # Initialize system
    ebbinghaus = EbbinghausMemorySystem()
    
    # Add different types of memories
    test_memories = [
        ("name_1", "My name is Alice", "personal_info"),
        ("job_1", "I work as a software engineer", "personal_info"),
        ("hobby_1", "I love reading science fiction", "preferences"),
        ("experience_1", "Yesterday I went to a great restaurant", "episodic"),
        ("skill_1", "I know how to program in Python", "procedural"),
        ("temp_1", "I'm currently at the office", "working")
    ]
    
    print("\n📝 Adding memories to system...")
    for node_id, content, node_type in test_memories:
        ebbinghaus.add_memory(node_id, content, node_type)
        print(f"Added: {content} (Type: {ebbinghaus.memory_strengths[node_id].memory_type.value})")
    
    print("\n⏰ Simulating time passage and memory access...")
    
    # Simulate immediate access (should be strong)
    print("\n🔍 Immediate access:")
    for node_id, content, _ in test_memories:
        strength = ebbinghaus.access_memory(node_id)
        print(f"{content[:30]}... → Strength: {strength:.3f}")
    
    # Simulate time passage (1 hour)
    print("\n⏰ After 1 hour passage:")
    for node_id in ebbinghaus.memory_strengths:
        ebbinghaus.memory_strengths[node_id].last_accessed -= 3600  # 1 hour ago
    
    for node_id, content, _ in test_memories:
        strength = ebbinghaus.access_memory(node_id)
        print(f"{content[:30]}... → Strength: {strength:.3f}")
    
    # Test consolidation
    print("\n🌙 Testing memory consolidation...")
    consolidation_results = ebbinghaus.consolidate_all_memories()
    
    for node_id, change in consolidation_results.items():
        if abs(change) > 0.01:
            print(f"Memory {node_id}: {change:+.3f} change")
    
    # Show statistics
    print("\n📊 Memory System Statistics:")
    stats = ebbinghaus.get_memory_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Test weak memory pruning
    print("\n🗑️ Testing weak memory pruning...")
    # Artificially weaken some memories
    ebbinghaus.memory_strengths["temp_1"].current_strength = 0.05
    
    pruned = ebbinghaus.prune_weak_memories()
    print(f"Pruned memories: {pruned}")
    
    print("\n✅ Ebbinghaus Memory System test completed!")
    return ebbinghaus


if __name__ == "__main__":
    test_ebbinghaus_memory_system()