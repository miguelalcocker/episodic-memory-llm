# src/evaluation/honest_evaluator.py
"""
🎯 HONEST EVALUATION SYSTEM - NO MORE INFLATED METRICS
Academic-grade evaluation for legitimate research results
"""

import re
import json
import numpy as np
from typing import Dict, List, Tuple, Optional
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import logging

logger = logging.getLogger(__name__)

class HonestMemoryEvaluator:
    """
    🎯 RIGOROUS EVALUATION SYSTEM
    
    Features:
    1. Semantic similarity scoring (not just keyword matching)
    2. Multi-level evaluation (exact, partial, semantic, relevance)
    3. Confidence intervals and statistical significance
    4. Human-interpretable scoring
    """
    
    def __init__(self):
        # Load semantic similarity model
        try:
            self.semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
            print("✅ Semantic evaluation model loaded")
        except Exception as e:
            print(f"⚠️ Semantic model not available: {e}")
            self.semantic_model = None
    
    def evaluate_memory_recall(self, 
                              query: str, 
                              response: str, 
                              ground_truth: Dict,
                              context_history: List[str] = None) -> Dict:
        """
        Comprehensive evaluation of memory recall quality
        
        Args:
            query: The memory query asked
            response: System's response
            ground_truth: Expected information structure
            context_history: Previous conversation for context
            
        Returns:
            Detailed evaluation metrics
        """
        
        # 1. EXACT MATCH EVALUATION
        exact_score = self._evaluate_exact_match(response, ground_truth)
        
        # 2. SEMANTIC SIMILARITY EVALUATION  
        semantic_score = self._evaluate_semantic_similarity(response, ground_truth)
        
        # 3. INFORMATION COMPLETENESS
        completeness_score = self._evaluate_completeness(response, ground_truth)
        
        # 4. FACTUAL ACCURACY
        accuracy_score = self._evaluate_factual_accuracy(response, ground_truth)
        
        # 5. RELEVANCE TO QUERY
        relevance_score = self._evaluate_relevance(query, response)
        
        # 6. HALLUCINATION DETECTION
        hallucination_score = self._detect_hallucination(response, context_history or [])
        
        # COMPOSITE SCORING
        final_score = self._calculate_composite_score({
            'exact': exact_score,
            'semantic': semantic_score, 
            'completeness': completeness_score,
            'accuracy': accuracy_score,
            'relevance': relevance_score,
            'hallucination': hallucination_score
        })
        
        return {
            'query': query,
            'response': response,
            'ground_truth': ground_truth,
            'scores': {
                'exact_match': exact_score,
                'semantic_similarity': semantic_score,
                'information_completeness': completeness_score,
                'factual_accuracy': accuracy_score,
                'query_relevance': relevance_score,
                'hallucination_penalty': hallucination_score,
                'final_score': final_score
            },
            'grade': self._assign_grade(final_score),
            'feedback': self._generate_feedback(final_score, {
                'exact': exact_score,
                'semantic': semantic_score,
                'completeness': completeness_score,
                'accuracy': accuracy_score,
                'relevance': relevance_score,
                'hallucination': hallucination_score
            })
        }
    
    def _evaluate_exact_match(self, response: str, ground_truth: Dict) -> float:
        """Exact keyword/phrase matching"""
        response_lower = response.lower()
        total_elements = 0
        matched_elements = 0
        
        for key, expected_values in ground_truth.items():
            if isinstance(expected_values, list):
                for value in expected_values:
                    total_elements += 1
                    if value.lower() in response_lower:
                        matched_elements += 1
            elif isinstance(expected_values, str):
                total_elements += 1
                if expected_values.lower() in response_lower:
                    matched_elements += 1
        
        return matched_elements / total_elements if total_elements > 0 else 0.0
    
    def _evaluate_semantic_similarity(self, response: str, ground_truth: Dict) -> float:
        """Semantic similarity using sentence transformers"""
        if not self.semantic_model:
            return 0.0
        
        # Create expected response from ground truth
        expected_parts = []
        for key, values in ground_truth.items():
            if isinstance(values, list):
                expected_parts.extend([str(v) for v in values])
            else:
                expected_parts.append(str(values))
        
        expected_text = " ".join(expected_parts)
        
        # Calculate embeddings
        response_embedding = self.semantic_model.encode([response])
        expected_embedding = self.semantic_model.encode([expected_text])
        
        # Cosine similarity
        similarity = cosine_similarity(response_embedding, expected_embedding)[0][0]
        return max(0.0, float(similarity))
    
    def _evaluate_completeness(self, response: str, ground_truth: Dict) -> float:
        """How complete is the response regarding expected information"""
        required_info_types = len(ground_truth.keys())
        covered_info_types = 0
        
        response_lower = response.lower()
        
        for info_type, expected_values in ground_truth.items():
            if isinstance(expected_values, list):
                if any(val.lower() in response_lower for val in expected_values):
                    covered_info_types += 1
            elif isinstance(expected_values, str):
                if expected_values.lower() in response_lower:
                    covered_info_types += 1
        
        return covered_info_types / required_info_types
    
    def _evaluate_factual_accuracy(self, response: str, ground_truth: Dict) -> float:
        """Check for factual errors or contradictions"""
        # This is a simplified version - in real implementation you'd want
        # more sophisticated fact-checking
        
        accuracy_score = 1.0
        response_lower = response.lower()
        
        # Check for obvious contradictions
        contradiction_patterns = [
            (r"don't know", r"don't have", -0.8),  # Claiming ignorance when should know
            (r"not sure", r"uncertain", -0.3),      # Uncertainty penalty
        ]
        
        for pattern, _, penalty in contradiction_patterns:
            if re.search(pattern, response_lower):
                accuracy_score += penalty
        
        return max(0.0, accuracy_score)
    
    def _evaluate_relevance(self, query: str, response: str) -> float:
        """How relevant is the response to the specific query"""
        if not self.semantic_model:
            # Fallback to keyword overlap
            query_words = set(query.lower().split())
            response_words = set(response.lower().split())
            overlap = len(query_words.intersection(response_words))
            return overlap / len(query_words) if query_words else 0.0
        
        # Semantic relevance
        query_embedding = self.semantic_model.encode([query])
        response_embedding = self.semantic_model.encode([response])
        
        relevance = cosine_similarity(query_embedding, response_embedding)[0][0]
        return max(0.0, float(relevance))
    
    def _detect_hallucination(self, response: str, context_history: List[str]) -> float:
        """Detect if response contains information not in context"""
        if not context_history:
            return 0.0
        
        context_text = " ".join(context_history).lower()
        response_lower = response.lower()
        
        # Simple hallucination detection - check if response contains
        # specific information not mentioned in context
        hallucination_penalty = 0.0
        
        # Extract potential facts from response
        potential_facts = re.findall(r'\b[A-Z][a-z]+ [A-Z][a-z]+\b', response)  # Names
        potential_facts += re.findall(r'\b\d{4}\b', response)  # Years
        potential_facts += re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+University\b', response)  # Universities
        
        for fact in potential_facts:
            if fact.lower() not in context_text:
                hallucination_penalty -= 0.2
        
        return max(-1.0, hallucination_penalty)
    
    def _calculate_composite_score(self, scores: Dict[str, float]) -> float:
        """Calculate weighted composite score"""
        weights = {
            'exact': 0.25,           # 25% - Exact matches important
            'semantic': 0.20,        # 20% - Semantic understanding
            'completeness': 0.20,    # 20% - Information completeness  
            'accuracy': 0.25,        # 25% - Factual accuracy critical
            'relevance': 0.10,       # 10% - Query relevance
            'hallucination': 0.0     # Penalty only (negative)
        }
        
        composite = sum(scores[key] * weights[key] for key in weights.keys() if key != 'hallucination')
        composite += scores['hallucination']  # Add penalty
        
        return max(0.0, min(1.0, composite))
    
    def _assign_grade(self, score: float) -> str:
        """Convert score to interpretable grade"""
        if score >= 0.90:
            return "A+"
        elif score >= 0.80:
            return "A"
        elif score >= 0.70:
            return "B+"
        elif score >= 0.60:
            return "B"
        elif score >= 0.50:
            return "C+"
        elif score >= 0.40:
            return "C"
        else:
            return "F"
    
    def _generate_feedback(self, final_score: float, component_scores: Dict) -> str:
        """Generate human-readable feedback"""
        feedback_parts = []
        
        if final_score >= 0.80:
            feedback_parts.append("✅ Excellent response quality")
        elif final_score >= 0.60:
            feedback_parts.append("🟡 Good response with room for improvement")
        else:
            feedback_parts.append("❌ Response needs significant improvement")
        
        # Component-specific feedback
        if component_scores['exact'] < 0.5:
            feedback_parts.append("• Missing key factual details")
        
        if component_scores['semantic'] < 0.5:
            feedback_parts.append("• Poor semantic understanding")
        
        if component_scores['completeness'] < 0.5:
            feedback_parts.append("• Incomplete information coverage")
        
        if component_scores['hallucination'] < -0.2:
            feedback_parts.append("• Contains potential hallucinations")
        
        return " ".join(feedback_parts)


class BenchmarkSuite:
    """
    🎯 COMPREHENSIVE BENCHMARK SUITE
    Tests memory systems against standard datasets and metrics
    """
    
    def __init__(self):
        self.evaluator = HonestMemoryEvaluator()
        
    def create_standardized_test_cases(self) -> List[Dict]:
        """Create standardized, unbiased test cases"""
        
        return [
            {
                "conversation": [
                    "Hi, I'm Dr. Sarah Chen, a professor of neuroscience at UCLA.",
                    "I've been researching memory formation in the hippocampus for 8 years.",
                    "I graduated from MIT with a PhD in Cognitive Science in 2015.",
                    "In my free time, I love rock climbing and playing the violin.",
                    "Last month I published a paper in Nature Neuroscience about synaptic plasticity."
                ],
                "test_queries": [
                    {
                        "query": "What is your full name and current position?",
                        "ground_truth": {
                            "name": ["Sarah Chen", "Dr. Sarah Chen"],
                            "position": ["professor", "neuroscience"],
                            "institution": ["UCLA"]
                        },
                        "difficulty": "basic"
                    },
                    {
                        "query": "What was your PhD field and from which university?",
                        "ground_truth": {
                            "degree": ["PhD", "Cognitive Science"],
                            "university": ["MIT"],
                            "year": ["2015"]
                        },
                        "difficulty": "basic"
                    },
                    {
                        "query": "What are your hobbies and research interests?",
                        "ground_truth": {
                            "hobbies": ["rock climbing", "violin"],
                            "research": ["memory formation", "hippocampus", "neuroscience"]
                        },
                        "difficulty": "intermediate"
                    },
                    {
                        "query": "What did you publish recently and where?",
                        "ground_truth": {
                            "publication": ["paper", "Nature Neuroscience"],
                            "topic": ["synaptic plasticity"],
                            "timing": ["last month"]
                        },
                        "difficulty": "advanced"
                    }
                ]
            },
            
            # Add more diverse test cases here...
            
            {
                "conversation": [
                    "My name is Alex Rodriguez and I work as a software engineer at Google.",
                    "I've been coding for 12 years, specializing in machine learning systems.",
                    "I studied Computer Science at Stanford, graduating in 2018.",
                    "I enjoy photography and hiking in my spare time.",
                    "Yesterday I deployed a new ML model that improved our search accuracy by 15%."
                ],
                "test_queries": [
                    {
                        "query": "What is your name and what company do you work for?",
                        "ground_truth": {
                            "name": ["Alex Rodriguez"],
                            "company": ["Google"],
                            "role": ["software engineer"]
                        },
                        "difficulty": "basic"
                    },
                    {
                        "query": "What was your major and graduation year?",
                        "ground_truth": {
                            "major": ["Computer Science"],
                            "university": ["Stanford"],
                            "year": ["2018"]
                        },
                        "difficulty": "basic"
                    },
                    {
                        "query": "What recent achievement did you mention?",
                        "ground_truth": {
                            "achievement": ["deployed", "ML model", "search accuracy"],
                            "improvement": ["15%"],
                            "timing": ["yesterday"]
                        },
                        "difficulty": "advanced"
                    }
                ]
            }
        ]
    
    def run_comprehensive_benchmark(self, memory_system, test_cases: List[Dict] = None) -> Dict:
        """
        Run comprehensive benchmark against memory system
        
        Args:
            memory_system: Your EpisodicMemoryLLM system
            test_cases: Optional custom test cases
            
        Returns:
            Detailed benchmark results with statistical analysis
        """
        
        if test_cases is None:
            test_cases = self.create_standardized_test_cases()
        
        all_results = []
        
        print("🎯 Running Comprehensive Benchmark...")
        print("=" * 60)
        
        for case_idx, test_case in enumerate(test_cases, 1):
            print(f"\n📝 Test Case {case_idx}/{len(test_cases)}")
            
            # Initialize fresh system for each test case
            conversation = test_case["conversation"]
            test_queries = test_case["test_queries"]
            
            # Feed conversation to system
            for turn in conversation:
                memory_system.chat_breakthrough(turn)
            
            # Test each query
            case_results = []
            for query_data in test_queries:
                query = query_data["query"]
                ground_truth = query_data["ground_truth"]
                difficulty = query_data["difficulty"]
                
                print(f"  🔍 {difficulty.upper()}: {query}")
                
                # Get system response
                result = memory_system.chat_breakthrough(query)
                response = result["response"]
                
                # Evaluate response
                evaluation = self.evaluator.evaluate_memory_recall(
                    query=query,
                    response=response,
                    ground_truth=ground_truth,
                    context_history=conversation
                )
                
                evaluation["difficulty"] = difficulty
                evaluation["case_id"] = case_idx
                
                case_results.append(evaluation)
                all_results.append(evaluation)
                
                print(f"    💬 Response: {response}")
                print(f"    📊 Score: {evaluation['scores']['final_score']:.3f} ({evaluation['grade']})")
                print(f"    💡 {evaluation['feedback']}")
        
        # Statistical analysis
        statistics = self._calculate_benchmark_statistics(all_results)
        
        # Generate report
        report = {
            "benchmark_info": {
                "test_cases": len(test_cases),
                "total_queries": len(all_results),
                "evaluation_timestamp": __import__('time').time()
            },
            "results": all_results,
            "statistics": statistics,
            "summary": self._generate_benchmark_summary(statistics)
        }
        
        print(f"\n" + "=" * 60)
        print("🏆 BENCHMARK COMPLETED")
        print("=" * 60)
        print(f"📊 Overall Score: {statistics['overall_score']:.1%}")
        print(f"📈 Confidence Interval: ±{statistics['confidence_interval']:.3f}")
        print(f"🎯 Grade: {statistics['overall_grade']}")
        print(f"💡 {report['summary']}")
        
        return report
    
    def _calculate_benchmark_statistics(self, results: List[Dict]) -> Dict:
        """Calculate comprehensive statistics"""
        
        scores = [r['scores']['final_score'] for r in results]
        
        # Basic statistics
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        median_score = np.median(scores)
        
        # Confidence interval (95%)
        confidence_interval = 1.96 * (std_score / np.sqrt(len(scores)))
        
        # Grade distribution
        grades = [r['grade'] for r in results]
        grade_distribution = {grade: grades.count(grade) for grade in set(grades)}
        
        # Difficulty analysis
        difficulty_scores = {}
        difficulties = set(r['difficulty'] for r in results)
        for diff in difficulties:
            diff_scores = [r['scores']['final_score'] for r in results if r['difficulty'] == diff]
            difficulty_scores[diff] = {
                'mean': np.mean(diff_scores),
                'count': len(diff_scores)
            }
        
        # Component analysis
        component_scores = {}
        components = ['exact_match', 'semantic_similarity', 'information_completeness', 
                     'factual_accuracy', 'query_relevance']
        
        for component in components:
            comp_scores = [r['scores'][component] for r in results]
            component_scores[component] = {
                'mean': np.mean(comp_scores),
                'std': np.std(comp_scores)
            }
        
        return {
            'overall_score': mean_score,
            'standard_deviation': std_score,
            'median_score': median_score,
            'confidence_interval': confidence_interval,
            'overall_grade': self.evaluator._assign_grade(mean_score),
            'grade_distribution': grade_distribution,
            'difficulty_breakdown': difficulty_scores,
            'component_analysis': component_scores,
            'total_samples': len(results)
        }
    
    def _generate_benchmark_summary(self, stats: Dict) -> str:
        """Generate human-readable benchmark summary"""
        
        score = stats['overall_score']
        
        if score >= 0.80:
            return "🔥 EXCELLENT: System demonstrates strong episodic memory capabilities"
        elif score >= 0.65:
            return "✅ GOOD: Solid performance with areas for optimization"
        elif score >= 0.50:
            return "🟡 AVERAGE: Basic functionality present, needs improvement"
        else:
            return "❌ POOR: Significant issues with memory recall accuracy"


def run_honest_evaluation_demo():
    """Demo of the honest evaluation system"""
    
    evaluator = HonestMemoryEvaluator()
    
    # Test case: Name query
    query = "What is my full name and current position?"
    
    # Good response
    good_response = "Your full name is Dr. Elena Rodriguez and you work as a research scientist at MIT."
    ground_truth = {
        "name": ["Elena Rodriguez", "Dr. Elena Rodriguez"],
        "position": ["research scientist"],
        "institution": ["MIT"]
    }
    
    print("🎯 HONEST EVALUATION DEMO")
    print("=" * 50)
    
    evaluation = evaluator.evaluate_memory_recall(
        query=query,
        response=good_response,
        ground_truth=ground_truth
    )
    
    print(f"Query: {query}")
    print(f"Response: {good_response}")
    print(f"Final Score: {evaluation['scores']['final_score']:.3f}")
    print(f"Grade: {evaluation['grade']}")
    print(f"Feedback: {evaluation['feedback']}")
    
    # Bad response (like your current system)
    print(f"\n" + "-" * 50)
    bad_response = "Your full name is Quantum And She."
    
    evaluation_bad = evaluator.evaluate_memory_recall(
        query=query,
        response=bad_response,
        ground_truth=ground_truth
    )
    
    print(f"Query: {query}")
    print(f"Response: {bad_response}")
    print(f"Final Score: {evaluation_bad['scores']['final_score']:.3f}")
    print(f"Grade: {evaluation_bad['grade']}")
    print(f"Feedback: {evaluation_bad['feedback']}")


if __name__ == "__main__":
    run_honest_evaluation_demo()