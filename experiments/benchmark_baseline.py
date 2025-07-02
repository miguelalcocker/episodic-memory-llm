# experiments/benchmark_baseline.py
"""
Script para hacer benchmark del baseline RAG contra el dataset sintético
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import time
from pathlib import Path
from src.models.baseline_rag import BaselineRAG
from src.evaluation.memory_metrics import MemoryEvaluator, create_test_suite

def load_dataset(filepath: str):
    """Cargar dataset desde archivo JSON"""
    with open(filepath, 'r') as f:
        return json.load(f)

def benchmark_on_conversations(model: BaselineRAG, conversations: list, max_conversations: int = 10):
    """
    Ejecutar modelo en conversaciones del dataset y medir performance
    """
    results = []
    
    print(f"🔄 Running benchmark on {min(len(conversations), max_conversations)} conversations...")
    
    for i, conv in enumerate(conversations[:max_conversations]):
        print(f"\nProcessing conversation {i+1}/{min(len(conversations), max_conversations)}")
        
        # Reset model memory for new conversation
        model.current_conversation = []
        # Crear nuevo memory store limpio
        from src.models.baseline_rag import SimpleMemoryStore
        model.memory_store = SimpleMemoryStore(embedding_dim=model.model.config.hidden_size)
        
        conversation_results = {
            "conversation_id": conv.get("id", f"conv_{i}"),
            "num_turns": len(conv["turns"]),
            "complexity": conv.get("complexity", "unknown"),
            "personality": conv.get("personality", "unknown"),
            "responses": [],
            "memory_questions": [],
            "response_times": []
        }
        
        user_turns = [turn for turn in conv["turns"] if turn["role"] == "user"]
        
        for turn in user_turns:
            content = turn["content"]
            turn_metadata = turn.get("metadata", {})
            
            # Medir tiempo de respuesta
            start_time = time.time()
            response = model.chat(content)
            response_time = time.time() - start_time
            
            conversation_results["responses"].append({
                "user_input": content,
                "assistant_response": response,
                "response_time": response_time,
                "turn_metadata": turn_metadata
            })
            
            conversation_results["response_times"].append(response_time)
            
            # Si es pregunta de memoria, marcarla especialmente
            if turn_metadata.get("requires_memory", False):
                conversation_results["memory_questions"].append({
                    "question": content,
                    "response": response,
                    "topic": turn_metadata.get("referenced_topic"),
                    "response_time": response_time
                })
        
        # Calcular estadísticas de la conversación
        conversation_results["avg_response_time"] = sum(conversation_results["response_times"]) / len(conversation_results["response_times"])
        conversation_results["total_memories_stored"] = len(model.memory_store.memories)
        
        results.append(conversation_results)
        
        print(f"  - Turns processed: {len(user_turns)}")
        print(f"  - Memory questions: {len(conversation_results['memory_questions'])}")
        print(f"  - Avg response time: {conversation_results['avg_response_time']:.2f}s")
        print(f"  - Total memories: {conversation_results['total_memories_stored']}")
    
    return results

def run_evaluation_scenarios(model: BaselineRAG, scenarios: dict):
    """
    Ejecutar escenarios específicos de evaluación
    """
    evaluator = MemoryEvaluator(model)
    
    print("\n🧪 Running evaluation scenarios...")
    
    # Ejecutar evaluación comprehensiva
    eval_results = evaluator.comprehensive_evaluation(scenarios)
    
    return eval_results

def comprehensive_benchmark():
    """
    Benchmark completo del baseline RAG
    """
    print("🚀 Starting Comprehensive Baseline Benchmark")
    print("=" * 60)
    
    # 1. Cargar datasets
    print("\n📂 Loading datasets...")
    try:
        test_dataset = load_dataset("data/test_dataset.json")
        eval_scenarios = load_dataset("data/evaluation_scenarios_dataset.json")
        print(f"✅ Loaded {len(test_dataset)} test conversations")
        print(f"✅ Loaded evaluation scenarios")
    except FileNotFoundError as e:
        print(f"❌ Error loading dataset: {e}")
        print("Make sure to run 'python src/data/synthetic_conversations.py' first")
        return
    
    # 2. Inicializar modelo
    print("\n🤖 Initializing BaselineRAG model...")
    device = "cuda" if os.environ.get("USE_GPU", "false").lower() == "true" else "cpu"
    model = BaselineRAG(model_name="gpt2-medium", device=device)
    print(f"✅ Model initialized on {device}")
    
    # 3. Benchmark en conversaciones
    print("\n💬 Benchmarking on conversations...")
    conversation_results = benchmark_on_conversations(model, test_dataset, max_conversations=5)
    
    # 4. Evaluación con métricas científicas
    print("\n📊 Running scientific evaluation...")
    evaluation_results = run_evaluation_scenarios(model, eval_scenarios)
    
    # 5. Compilar resultados finales
    print("\n📋 Compiling final results...")
    final_results = {
        "benchmark_timestamp": time.time(),
        "model_info": {
            "name": "BaselineRAG",
            "base_model": "gpt2-medium",
            "device": device
        },
        "conversation_benchmark": {
            "total_conversations": len(conversation_results),
            "avg_response_time": sum(r["avg_response_time"] for r in conversation_results) / len(conversation_results),
            "total_memory_questions": sum(len(r["memory_questions"]) for r in conversation_results),
            "conversations": conversation_results
        },
        "scientific_evaluation": evaluation_results
    }
    
    # 6. Guardar resultados
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    timestamp = int(time.time())
    results_file = results_dir / f"baseline_benchmark_{timestamp}.json"
    
    with open(results_file, 'w') as f:
        json.dump(final_results, f, indent=2)
    
    print(f"💾 Results saved to {results_file}")
    
    # 7. Mostrar resumen
    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)
    
    conv_bench = final_results["conversation_benchmark"]
    sci_eval = final_results["scientific_evaluation"]
    
    print(f"📊 PERFORMANCE METRICS:")
    print(f"  Average Response Time: {conv_bench['avg_response_time']:.2f}s")
    print(f"  Total Conversations: {conv_bench['total_conversations']}")
    print(f"  Memory Questions Handled: {conv_bench['total_memory_questions']}")
    
    print(f"\n🔬 SCIENTIFIC EVALUATION:")
    if 'combined_score' in sci_eval:
        score = sci_eval['combined_score']
        print(f"  Combined Score: {score:.3f} ({score*100:.1f}%)")
    
    for metric in ['temporal_consistency', 'overall_accuracy', 'personality_persistence', 'context_integration']:
        if metric in sci_eval:
            value = sci_eval[metric]
            print(f"  {metric.replace('_', ' ').title()}: {value:.3f} ({value*100:.1f}%)")
    
    # 8. Generar reporte detallado
    print(f"\n📄 Generating detailed report...")
    evaluator = MemoryEvaluator()
    report = evaluator.generate_evaluation_report(sci_eval)
    
    report_file = results_dir / f"baseline_report_{timestamp}.txt"
    with open(report_file, 'w') as f:
        f.write(report)
    
    print(f"📄 Detailed report saved to {report_file}")
    
    print("\n✅ Comprehensive benchmark completed!")
    print(f"📁 All results saved in 'results/' directory")
    
    return final_results

def quick_benchmark():
    """
    Benchmark rápido para development/testing
    """
    print("⚡ Quick Baseline Benchmark")
    print("=" * 40)
    
    # Modelo simple
    model = BaselineRAG(model_name="gpt2-medium", device="cpu")
    
    # Test rápido con conversaciones sintéticas
    quick_conversations = [
        {
            "id": "quick_test_1",
            "turns": [
                {"role": "user", "content": "Hi, I'm Alice and I work as a teacher", "metadata": {"type": "introduction"}},
                {"role": "user", "content": "I love reading books in my free time", "metadata": {"type": "hobby_sharing"}},
                {"role": "user", "content": "What do you remember about my job?", "metadata": {"requires_memory": True, "type": "memory_question"}}
            ]
        }
    ]
    
    results = benchmark_on_conversations(model, quick_conversations, max_conversations=1)
    
    print("\n📊 Quick Results:")
    result = results[0]
    print(f"  Response Time: {result['avg_response_time']:.2f}s")
    print(f"  Memories Stored: {result['total_memories_stored']}")
    print(f"  Memory Questions: {len(result['memory_questions'])}")
    
    if result['memory_questions']:
        print(f"\n💭 Memory Question Example:")
        mq = result['memory_questions'][0]
        print(f"  Q: {mq['question']}")
        print(f"  A: {mq['response']}")
    
    return results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Benchmark BaselineRAG model")
    parser.add_argument("--quick", action="store_true", help="Run quick benchmark")
    parser.add_argument("--gpu", action="store_true", help="Use GPU if available")
    
    args = parser.parse_args()
    
    if args.gpu:
        os.environ["USE_GPU"] = "true"
    
    if args.quick:
        quick_benchmark()
    else:
        comprehensive_benchmark()
