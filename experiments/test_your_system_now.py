# src/benchmarks/legitimate_benchmark_fixed.py
"""
🎯 LEGITIMATE ACADEMIC BENCHMARK SUITE - SYNTAX FIXED
Compare your system against real baselines with statistical rigor
"""

import json
import time
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)

@dataclass
class BenchmarkResult:
    """Structured benchmark result"""
    system_name: str
    dataset_name: str
    query: str
    response: str
    ground_truth: Dict
    scores: Dict
    metadata: Dict
    timestamp: float

class MemorySystem(ABC):
    """Abstract base class for memory systems"""
    
    @abstractmethod
    def add_memory(self, text: str, metadata: Dict = None) -> str:
        """Add information to memory"""
        pass
    
    @abstractmethod
    def query_memory(self, query: str) -> str:
        """Query the memory system"""
        pass
    
    @abstractmethod
    def reset(self):
        """Reset system state"""
        pass

class SimpleRAGBaseline(MemorySystem):
    """Simple RAG baseline using basic vector search"""
    
    def __init__(self):
        self.memories = []
        
    def add_memory(self, text: str, metadata: Dict = None) -> str:
        """Store text and create simple embedding"""
        embedding = self._create_simple_embedding(text)
        
        memory_id = f"mem_{len(self.memories)}"
        self.memories.append({
            'id': memory_id,
            'text': text,
            'metadata': metadata or {},
            'embedding': embedding
        })
        
        return memory_id
    
    def query_memory(self, query: str) -> str:
        """Simple similarity-based retrieval"""
        if not self.memories:
            return "I don't have any information stored yet."
        
        query_embedding = self._create_simple_embedding(query)
        
        # Find most similar memory
        best_similarity = -1
        best_memory = None
        
        for memory in self.memories:
            similarity = self._cosine_similarity(query_embedding, memory['embedding'])
            if similarity > best_similarity:
                best_similarity = similarity
                best_memory = memory
        
        if best_similarity > 0.3:  # Threshold
            return f"Based on what I remember: {best_memory['text']}"
        else:
            return "I don't have specific information about that."
    
    def reset(self):
        """Reset the system"""
        self.memories = []
    
    def _create_simple_embedding(self, text: str) -> np.ndarray:
        """Create simple word-based embedding"""
        words = text.lower().split()
        
        # Simple vocabulary-based vector
        vocab = ['work', 'study', 'like', 'enjoy', 'research', 'university', 
                'professor', 'scientist', 'computer', 'science', 'mit', 'stanford',
                'reading', 'hiking', 'quantum', 'algorithm', 'yesterday', 'month',
                'colleague', 'google', 'restaurant', 'japanese', 'cat', 'adopted']
        
        embedding = np.zeros(len(vocab))
        for i, word in enumerate(vocab):
            if word in words:
                embedding[i] = 1.0
        
        # Normalize
        norm = np.linalg.norm(embedding)
        return embedding / norm if norm > 0 else embedding
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity"""
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

class KeywordMatchingBaseline(MemorySystem):
    """Keyword matching baseline"""
    
    def __init__(self):
        self.memories = []
    
    def add_memory(self, text: str, metadata: Dict = None) -> str:
        """Store text"""
        memory_id = f"mem_{len(self.memories)}"
        self.memories.append({
            'id': memory_id,
            'text': text,
            'metadata': metadata or {},
            'keywords': self._extract_keywords(text)
        })
        return memory_id
    
    def query_memory(self, query: str) -> str:
        """Keyword-based matching"""
        query_keywords = self._extract_keywords(query)
        
        best_score = 0
        best_memory = None
        
        for memory in self.memories:
            # Count keyword overlaps
            overlap = len(set(query_keywords) & set(memory['keywords']))
            score = overlap / len(query_keywords) if query_keywords else 0
            
            if score > best_score:
                best_score = score
                best_memory = memory
        
        if best_score > 0.2:
            return best_memory['text']
        else:
            return "I don't have information about that."
    
    def reset(self):
        self.memories = []
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from text"""
        words = text.lower().split()
        stopwords = {'i', 'am', 'is', 'are', 'the', 'a', 'an', 'and', 'or', 'but'}
        keywords = [word for word in words if word not in stopwords and len(word) > 2]
        return keywords

class RandomResponseBaseline(MemorySystem):
    """Random response baseline - worst case scenario"""
    
    def __init__(self):
        self.memories = []
        self.responses = [
            "I understand.",
            "Tell me more about that.",
            "That's interesting.",
            "I don't have specific information about that.",
            "Could you provide more details?"
        ]
    
    def add_memory(self, text: str, metadata: Dict = None) -> str:
        self.memories.append(text)
        return f"mem_{len(self.memories)}"
    
    def query_memory(self, query: str) -> str:
        import random
        return random.choice(self.responses)
    
    def reset(self):
        self.memories = []

# Simple evaluator for now (will replace with full one later)
class SimpleEvaluator:
    """Temporary simple evaluator"""
    
    def evaluate_memory_recall(self, query: str, response: str, ground_truth: Dict, context_history: List[str] = None) -> Dict:
        """Simple keyword-based evaluation"""
        
        # Count how many expected keywords are in the response
        expected_keywords = ground_truth.get('expected_keywords', [])
        response_lower = response.lower()
        
        matches = 0
        for keyword in expected_keywords:
            if keyword.lower() in response_lower:
                matches += 1
        
        accuracy = matches / len(expected_keywords) if expected_keywords else 0
        
        # Simple grading
        if accuracy >= 0.8:
            grade = "A"
        elif accuracy >= 0.6:
            grade = "B"
        elif accuracy >= 0.4:
            grade = "C"
        else:
            grade = "F"
        
        return {
            'scores': {
                'final_score': accuracy,
                'keyword_matches': matches,
                'total_keywords': len(expected_keywords)
            },
            'grade': grade,
            'feedback': f"Matched {matches}/{len(expected_keywords)} keywords"
        }

class LegitimateAcademicBenchmark:
    """Legitimate benchmark with simple evaluation for now"""
    
    def __init__(self):
        self.evaluator = SimpleEvaluator()
        self.baselines = {
            'simple_rag': SimpleRAGBaseline(),
            'keyword_matching': KeywordMatchingBaseline(),
            'random_response': RandomResponseBaseline()
        }
        
        # Create simple test datasets
        self.test_datasets = self._create_test_datasets()
    
    def _create_test_datasets(self) -> Dict[str, List[Dict]]:
        """Create standardized test datasets"""
        
        datasets = {
            'personal_info': [
                {
                    'context': [
                        "Hello, I'm Dr. Maria Santos, a professor of biology at UC Berkeley.",
                        "I've been studying marine ecosystems for over 10 years.",
                        "I completed my PhD in Marine Biology from Scripps Institution in 2012.",
                        "In my free time, I enjoy scuba diving and underwater photography."
                    ],
                    'queries': [
                        {
                            'query': "What is your name and profession?",
                            'ground_truth': {
                                'expected_keywords': ["maria santos", "professor", "biology", "uc berkeley"]
                            },
                            'difficulty': 'basic'
                        },
                        {
                            'query': "Where did you get your PhD and when?",
                            'ground_truth': {
                                'expected_keywords': ["scripps", "phd", "marine biology", "2012"]
                            },
                            'difficulty': 'basic'
                        },
                        {
                            'query': "What are your research interests and hobbies?",
                            'ground_truth': {
                                'expected_keywords': ["marine ecosystems", "scuba diving", "photography"]
                            },
                            'difficulty': 'intermediate'
                        }
                    ]
                }
            ]
        }
        
        return datasets
    
    def run_comprehensive_benchmark(self, 
                                  test_system: MemorySystem, 
                                  system_name: str = "test_system") -> Dict:
        """Run comprehensive benchmark with statistical analysis"""
        
        print("🎯 RUNNING LEGITIMATE ACADEMIC BENCHMARK")
        print("=" * 60)
        print(f"📝 Testing system: {system_name}")
        print(f"🔬 Baselines: {', '.join(self.baselines.keys())}")
        
        all_results = []
        
        # Test each dataset
        for dataset_name in self.test_datasets.keys():
            print(f"\n📂 Dataset: {dataset_name}")
            dataset_results = self._test_dataset(test_system, system_name, dataset_name)
            all_results.extend(dataset_results)
        
        # Simple analysis
        analysis = self._analyze_results(all_results)
        
        # Generate report
        report = {
            'system_name': system_name,
            'benchmark_info': {
                'total_queries': len(all_results),
                'baselines_compared': list(self.baselines.keys()),
                'timestamp': time.time()
            },
            'raw_results': all_results,
            'analysis': analysis,
            'summary': self._generate_summary(analysis)
        }
        
        # Display results
        self._display_results(report)
        
        return report
    
    def _test_dataset(self, test_system: MemorySystem, system_name: str, dataset_name: str) -> List[BenchmarkResult]:
        """Test system on specific dataset"""
        
        dataset = self.test_datasets[dataset_name]
        results = []
        
        for case_idx, test_case in enumerate(dataset, 1):
            print(f"  📋 Test Case {case_idx}/{len(dataset)}")
            
            context = test_case['context']
            queries = test_case['queries']
            
            # Test all systems
            systems_to_test = {'test_system': test_system, **self.baselines}
            
            for query_data in queries:
                query = query_data['query']
                ground_truth = query_data['ground_truth']
                difficulty = query_data['difficulty']
                
                print(f"    🔍 {difficulty.upper()}: {query}")
                
                # Test all systems
                for system_name_inner, system in systems_to_test.items():
                    system.reset()
                    for ctx in context:
                        system.add_memory(ctx)
                    
                    response = system.query_memory(query)
                    
                    # Evaluate response
                    evaluation = self.evaluator.evaluate_memory_recall(
                        query=query,
                        response=response,
                        ground_truth=ground_truth,
                        context_history=context
                    )
                    
                    result = BenchmarkResult(
                        system_name=system_name_inner,
                        dataset_name=dataset_name,
                        query=query,
                        response=response,
                        ground_truth=ground_truth,
                        scores=evaluation['scores'],
                        metadata={
                            'difficulty': difficulty,
                            'case_id': case_idx,
                            'evaluation': evaluation
                        },
                        timestamp=time.time()
                    )
                    
                    results.append(result)
                    
                    print(f"      📊 {system_name_inner}: {evaluation['scores']['final_score']:.3f}")
        
        return results
    
    def _analyze_results(self, results: List[BenchmarkResult]) -> Dict:
        """Simple analysis of results"""
        
        # Group by system
        system_scores = {}
        for result in results:
            if result.system_name not in system_scores:
                system_scores[result.system_name] = []
            system_scores[result.system_name].append(result.scores['final_score'])
        
        # Calculate means
        system_means = {}
        for system_name, scores in system_scores.items():
            system_means[system_name] = {
                'mean': np.mean(scores),
                'std': np.std(scores),
                'count': len(scores)
            }
        
        return {
            'system_performance': system_means,
            'total_queries': len(results)
        }
    
    def _generate_summary(self, analysis: Dict) -> str:
        """Generate summary"""
        
        if 'test_system' not in analysis['system_performance']:
            return "❌ Test system not found in results"
        
        test_mean = analysis['system_performance']['test_system']['mean']
        
        if test_mean >= 0.80:
            performance_level = "🔥 EXCELLENT"
        elif test_mean >= 0.65:
            performance_level = "✅ GOOD"
        elif test_mean >= 0.50:
            performance_level = "🟡 AVERAGE"
        else:
            performance_level = "❌ POOR"
        
        return f"{performance_level} - Overall accuracy: {test_mean:.1%}"
    
    def _display_results(self, report: Dict):
        """Display results"""
        
        print(f"\n" + "=" * 60)
        print("🏆 BENCHMARK RESULTS")
        print("=" * 60)
        
        analysis = report['analysis']
        
        print(f"📊 SYSTEM PERFORMANCE:")
        for system_name, stats in analysis['system_performance'].items():
            print(f"   {system_name}: {stats['mean']:.3f} ± {stats['std']:.3f}")
        
        print(f"\n💡 SUMMARY: {report['summary']}")

def run_simple_benchmark_test():
    """Simple test of the benchmark"""
    
    benchmark = LegitimateAcademicBenchmark()
    
    # Create a dummy test system
    class DummyTestSystem(MemorySystem):
        def __init__(self):
            self.memories = []
        
        def add_memory(self, text: str, metadata: Dict = None) -> str:
            self.memories.append(text)
            return f"mem_{len(self.memories)}"
        
        def query_memory(self, query: str) -> str:
            # Very simple system that just returns first relevant memory
            for memory in self.memories:
                if any(word in memory.lower() for word in query.lower().split()):
                    return memory
            return "I don't have information about that."
        
        def reset(self):
            self.memories = []
    
    test_system = DummyTestSystem()
    
    # Run benchmark
    results = benchmark.run_comprehensive_benchmark(
        test_system=test_system,
        system_name="dummy_system"
    )
    
    return results

if __name__ == "__main__":
    print("🎯 Testing Simple Benchmark")
    results = run_simple_benchmark_test()