# experiments/benchmark_episodic_vs_baseline.py
"""
Benchmark directo: EpisodicMemoryLLM vs Baseline RAG - VERSION CORREGIDA
"""

import sys
import os
import time
import json
from pathlib import Path

# Añadir src al path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

# FIX: Usar la clase correcta V2
from models.episodic_memory_llm_v2 import EpisodicMemoryLLM_V2
from models.baseline_rag_functional import SimpleFunctionalRAG as BaselineRAG

def create_test_scenarios():
    """
    Crear escenarios específicos para demostrar superioridad
    """
    return [
        {
            "name": "Personal Info Recall",
            "setup": [
                "Hi, I'm Alice and I work as a teacher at Central High School",
                "I love reading mystery novels in my free time", 
                "I also enjoy playing chess on weekends"
            ],
            "tests": [
                {
                    "query": "What's my job?",
                    "expected_keywords": ["teacher", "work", "job"],
                    "type": "factual_recall"
                },
                {
                    "query": "What do you know about my hobbies?", 
                    "expected_keywords": ["reading", "books", "chess"],
                    "type": "preference_recall"
                },
                {
                    "query": "Where do I work?",
                    "expected_keywords": ["Central High", "school"],
                    "type": "specific_recall"
                }
            ]
        },
        {
            "name": "Contextual Integration",
            "setup": [
                "I'm a software engineer at Google",
                "I love hiking and outdoor activities", 
                "I'm looking for weekend activities in San Francisco"
            ],
            "tests": [
                {
                    "query": "Can you recommend weekend activities for me?",
                    "expected_keywords": ["hiking", "outdoor", "San Francisco"],
                    "type": "contextual_integration"
                },
                {
                    "query": "What kind of work do I do?",
                    "expected_keywords": ["software", "engineer", "Google"],
                    "type": "factual_recall"
                }
            ]
        },
        {
            "name": "Multi-Session Memory",
            "setup": [
                "Yesterday I went to a great Italian restaurant",
                "The pasta was incredible, especially the carbonara",
                "I'm planning another dinner this weekend"
            ],
            "tests": [
                {
                    "query": "What restaurant did I visit recently?",
                    "expected_keywords": ["Italian", "restaurant"],
                    "type": "episodic_recall"
                },
                {
                    "query": "What did I think of the food?",
                    "expected_keywords": ["incredible", "great", "carbonara"],
                    "type": "experience_recall"
                }
            ]
        }
    ]

def evaluate_response(response: str, expected_keywords: list) -> float:
    """
    Evaluar qué tan bien una respuesta menciona las keywords esperadas
    """
    response_lower = response.lower()
    found_keywords = sum(1 for keyword in expected_keywords 
                        if keyword.lower() in response_lower)
    
    return found_keywords / len(expected_keywords) if expected_keywords else 0.0

def benchmark_model(model, model_name: str, scenarios: list) -> dict:
    """
    Ejecutar benchmark en un modelo específico - VERSION CORREGIDA
    """
    print(f"\n🔄 Benchmarking {model_name}...")
    
    results = {
        "model_name": model_name,
        "scenarios": [],
        "overall_scores": {
            "factual_recall": [],
            "preference_recall": [],
            "specific_recall": [],
            "contextual_integration": [],
            "episodic_recall": [],
            "experience_recall": []
        },
        "response_times": [],
        "total_tests": 0,
        "successful_tests": 0
    }
    
    for scenario_idx, scenario in enumerate(scenarios):
        print(f"\n  📋 Scenario: {scenario['name']}")
        
        # FIX: Reset más completo del modelo
        if hasattr(model, 'conversation_history'):
            model.conversation_history = []
        if hasattr(model, 'tkg') and model.tkg is not None:
            # Reinicializar TKG completamente
            from memory.temporal_knowledge_graph import TemporalKnowledgeGraph
            model.tkg = TemporalKnowledgeGraph(max_nodes=1000, decay_rate=0.1)
            # Reinicializar memory_system con nuevo TKG
            if hasattr(model, 'memory_system'):
                from memory.advanced_memory_retrieval import AdvancedMemoryRetrieval
                model.memory_system = AdvancedMemoryRetrieval(model.tkg, model.tokenizer)
        if hasattr(model, 'memory_store'):
            if hasattr(model.memory_store, 'memories'):
                model.memory_store.memories = []
                model.memory_store.metadata = []
        
        scenario_result = {
            "scenario_name": scenario["name"],
            "setup_responses": [],
            "test_results": []
        }
        
        # Setup fase
        for setup_input in scenario["setup"]:
            try:
                start_time = time.time()
                response = model.chat(setup_input)
                response_time = time.time() - start_time
                
                scenario_result["setup_responses"].append({
                    "input": setup_input,
                    "response": response,
                    "response_time": response_time
                })
                results["response_times"].append(response_time)
                
            except Exception as e:
                print(f"    ❌ Setup error: {e}")
                scenario_result["setup_responses"].append({
                    "input": setup_input,
                    "response": f"ERROR: {str(e)}",
                    "response_time": 0
                })
        
        # Test fase
        for test in scenario["tests"]:
            try:
                start_time = time.time()
                response = model.chat(test["query"])
                response_time = time.time() - start_time
                
                # Evaluar respuesta
                score = evaluate_response(response, test["expected_keywords"])
                
                test_result = {
                    "query": test["query"],
                    "response": response,
                    "expected_keywords": test["expected_keywords"],
                    "score": score,
                    "response_time": response_time,
                    "type": test["type"]
                }
                
                scenario_result["test_results"].append(test_result)
                results["response_times"].append(response_time)
                results["overall_scores"][test["type"]].append(score)
                results["total_tests"] += 1
                
                if score > 0.5:  # Consideramos exitoso si encuentra >50% keywords
                    results["successful_tests"] += 1
                
                print(f"    🔍 {test['query']}")
                print(f"       Response: {response[:60]}...")
                print(f"       Score: {score:.2f} | Time: {response_time:.2f}s")
                
            except Exception as e:
                print(f"    ❌ Test error: {e}")
                test_result = {
                    "query": test["query"],
                    "response": f"ERROR: {str(e)}",
                    "expected_keywords": test["expected_keywords"],
                    "score": 0.0,
                    "response_time": 0,
                    "type": test["type"]
                }
                scenario_result["test_results"].append(test_result)
                results["total_tests"] += 1
        
        results["scenarios"].append(scenario_result)
    
    return results

def calculate_summary_metrics(results: dict) -> dict:
    """
    Calcular métricas resumen del benchmark
    """
    summary = {
        "model_name": results["model_name"],
        "success_rate": results["successful_tests"] / results["total_tests"] if results["total_tests"] > 0 else 0,
        "avg_response_time": sum(results["response_times"]) / len(results["response_times"]) if results["response_times"] else 0,
        "category_scores": {}
    }
    
    # Calcular scores por categoría
    for category, scores in results["overall_scores"].items():
        if scores:
            summary["category_scores"][category] = {
                "avg_score": sum(scores) / len(scores),
                "max_score": max(scores),
                "min_score": min(scores),
                "count": len(scores)
            }
    
    # Score general
    all_scores = []
    for scores in results["overall_scores"].values():
        all_scores.extend(scores)
    
    summary["overall_score"] = sum(all_scores) / len(all_scores) if all_scores else 0
    
    return summary

def run_comparative_benchmark():
    """
    Ejecutar benchmark comparativo completo - VERSION CORREGIDA
    """
    print("🥊 EPISODIC MEMORY LLM vs BASELINE RAG")
    print("=" * 60)
    
    # Crear escenarios de test
    scenarios = create_test_scenarios()
    
    # Inicializar modelos
    print("\n🔧 Initializing models...")
    
    try:
        # FIX: Usar EpisodicMemoryLLM_V2 explícitamente
        episodic_model = EpisodicMemoryLLM_V2(
            model_name="gpt2-medium",
            device="cpu",
            tkg_max_nodes=1000
        )
        print("✅ EpisodicMemoryLLM_V2 initialized")
    except Exception as e:
        print(f"❌ Failed to initialize EpisodicMemoryLLM_V2: {e}")
        import traceback
        traceback.print_exc()
        return
    
    try:
        baseline_model = BaselineRAG(
            model_name="gpt2-medium",
            device="cpu"
        )
        print("✅ BaselineRAG initialized")
    except Exception as e:
        print(f"❌ Failed to initialize BaselineRAG: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Ejecutar benchmarks
    episodic_results = benchmark_model(episodic_model, "EpisodicMemoryLLM_V2", scenarios)
    baseline_results = benchmark_model(baseline_model, "BaselineRAG", scenarios)
    
    # Calcular métricas resumen
    episodic_summary = calculate_summary_metrics(episodic_results)
    baseline_summary = calculate_summary_metrics(baseline_results)
    
    # Mostrar resultados comparativos
    print("\n" + "=" * 60)
    print("📊 COMPARATIVE RESULTS")
    print("=" * 60)
    
    print(f"\n🧠 EpisodicMemoryLLM_V2:")
    print(f"  Overall Score: {episodic_summary['overall_score']:.3f}")
    print(f"  Success Rate: {episodic_summary['success_rate']:.3f}")
    print(f"  Avg Response Time: {episodic_summary['avg_response_time']:.2f}s")
    
    print(f"\n📚 BaselineRAG:")
    print(f"  Overall Score: {baseline_summary['overall_score']:.3f}")
    print(f"  Success Rate: {baseline_summary['success_rate']:.3f}")
    print(f"  Avg Response Time: {baseline_summary['avg_response_time']:.2f}s")
    
    # Calcular mejoras
    score_improvement = (episodic_summary['overall_score'] - baseline_summary['overall_score']) / baseline_summary['overall_score'] * 100 if baseline_summary['overall_score'] > 0 else 0
    success_improvement = (episodic_summary['success_rate'] - baseline_summary['success_rate']) / baseline_summary['success_rate'] * 100 if baseline_summary['success_rate'] > 0 else 0
    
    print(f"\n🚀 IMPROVEMENTS:")
    print(f"  Score Improvement: {score_improvement:+.1f}%")
    print(f"  Success Rate Improvement: {success_improvement:+.1f}%")
    
    # Detalles por categoría
    print(f"\n📋 CATEGORY BREAKDOWN:")
    for category in episodic_summary["category_scores"]:
        if category in baseline_summary["category_scores"]:
            episodic_score = episodic_summary["category_scores"][category]["avg_score"]
            baseline_score = baseline_summary["category_scores"][category]["avg_score"]
            improvement = (episodic_score - baseline_score) / baseline_score * 100 if baseline_score > 0 else 0
            
            print(f"  {category.replace('_', ' ').title()}:")
            print(f"    Episodic: {episodic_score:.3f} | Baseline: {baseline_score:.3f} | Improvement: {improvement:+.1f}%")
    
    # Guardar resultados
    timestamp = int(time.time())
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    comparison_results = {
        "timestamp": timestamp,
        "episodic_results": episodic_results,
        "baseline_results": baseline_results,
        "episodic_summary": episodic_summary,
        "baseline_summary": baseline_summary,
        "improvements": {
            "score_improvement_percent": score_improvement,
            "success_improvement_percent": success_improvement
        }
    }
    
    results_file = results_dir / f"episodic_vs_baseline_benchmark_FIXED_{timestamp}.json"
    with open(results_file, 'w') as f:
        json.dump(comparison_results, f, indent=2)
    
    print(f"\n💾 Results saved to {results_file}")
    print("\n✅ Comparative benchmark completed!")
    
    return comparison_results

if __name__ == "__main__":
    run_comparative_benchmark()