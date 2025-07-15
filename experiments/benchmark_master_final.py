# experiments/benchmark_master_final.py
"""
🔥 BENCHMARK MASTER FINAL - LA ÚNICA VERSIÓN QUE NECESITAS
Basado en v5_ultra_final pero optimizado para máxima claridad
"""

import json
import time
from typing import List, Dict
import os
from datetime import datetime

# 🚀 IMPORTS DEL SISTEMA V5 ULTRA FIXED
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class BenchmarkMasterFinal:
    """
    🔥 BENCHMARK DEFINITIVO - Una sola clase, resultados claros
    """

    def __init__(self):
        self.results_dir = f"results/benchmark_master_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(self.results_dir, exist_ok=True)

        print("🚀 BENCHMARK MASTER FINAL INITIALIZED")
        print(f"📁 Results: {self.results_dir}")

    def create_test_conversation(self) -> List[str]:
        """Conversación de benchmarking estándar - 50 turnos"""
        return [
            # Setup conversación (Turnos 1-40)
            "Hi, I'm Dr. Elena Rodriguez and I work as a research scientist at MIT.",
            "I've been working on quantum computing for the past 5 years.",
            "My current project focuses on quantum error correction algorithms.",
            "I graduated from Stanford with a PhD in Computer Science in 2018.",
            "I love reading mystery novels in my free time, especially Agatha Christie.",
            "I also enjoy hiking on weekends when the weather is nice.",
            "Last weekend I hiked Mount Washington and the views were incredible.",
            "I'm originally from Barcelona, Spain, but moved to Boston for work.",
            "My office is in the Computer Science and Artificial Intelligence Laboratory.",
            "I collaborate frequently with researchers from Harvard and Google.",
            "Yesterday I had a breakthrough with my quantum algorithm implementation.",
            "I've been learning to play chess recently and it's quite challenging.",
            "My colleague Sarah from Google recommended a great Japanese restaurant downtown.",
            "I'm planning a research trip to Tokyo next month to present my work.",
            "The conference is called the International Quantum Computing Symposium.",
            "I speak Spanish, English, and I'm currently learning Japanese.",
            "My favorite coffee shop is near campus, they make excellent espresso.",
            "I adopted a cat last month, her name is Quantum and she's very playful.",
            "I'm working on a paper about quantum decoherence for Nature Physics.",
            "My research has potential applications in cryptography and optimization.",
            "I teach an undergraduate course on Introduction to Quantum Computing.",
            "My students are brilliant, one of them reminds me of myself at that age.",
            "I received my bachelor's degree from Universidad Politécnica de Cataluña.",
            "During my PhD, I worked with Professor Michael Chen, a quantum pioneer.",
            "I published my first paper in Physical Review A when I was 24.",
            "My current research is funded by a $2 million NSF grant.",
            "I bought a new apartment in Cambridge last year with a great view.",
            "I drive a Tesla Model 3, perfect for commuting to MIT.",
            "My brother Luis is also a scientist, he works on renewable energy in Madrid.",
            "I'm vegetarian and love cooking Mediterranean cuisine.",
            "Three weeks ago I presented at the Quantum Computing Summit in San Francisco.",
            "Two months ago I started collaborating with IBM Research on quantum networking.",
            "Last year I won the Young Researcher Award from the Quantum Society.",
            "Six months ago I moved from a studio to my current two-bedroom apartment.",
            "Next week I have a meeting with the Department Head about my promotion.",
            "In two weeks I'm flying to Barcelona to visit my family for a week.",
            "Last Tuesday I had lunch with Nobel laureate Dr. Jennifer Williams.",
            "Tomorrow I'm giving a seminar at Harvard about quantum supremacy.",
            "I started my meditation practice exactly one year ago today.",
            "My favorite season is autumn because of the beautiful foliage in New England.",
            
            # QUERIES CRÍTICAS (Turnos 41-50)
            "Could you tell me what's my full name and current position?",
            "What university did I graduate from and what was my field of study?",
            "What are my main hobbies and interests outside of work?",
            "What breakthrough did I mention having yesterday?",
            "Which colleague recommended a restaurant and what type of cuisine?",
            "What's the name of my cat and when did I adopt her?",
            "What conference am I planning to attend next month and where?",
            "What significant meeting do I have next week?",
            "What award did I win last year?",
            "Who did I have lunch with last Tuesday?"
        ]

    def test_system(self, conversation: List[str]) -> List[Dict]:
        """Test tu sistema con la conversación"""
        
        # Importar tu sistema aquí
        from src.models.episodic_memory_llm import EpisodicMemoryLLM_FINAL
        
        # Inicializar sistema
        system = EpisodicMemoryLLM_FINAL(
            model_name="gpt2-medium",
            device="cpu",
            tkg_max_nodes=2000
        )
        
        responses = []
        print("🧠 Testing Episodic Memory System...")
        print("="*60)
        
        for i, message in enumerate(conversation):
            print(f"\n🔄 Turn {i+1}/50: {message}")
            
            start_time = time.time()
            
            # Usar chat_breakthrough del sistema FINAL
            result = system.chat_breakthrough(message)
            
            response_time = time.time() - start_time
            
            response_data = {
                "turn": i + 1,
                "input": message,
                "response": result["response"],
                "response_time": response_time,
                "query_type": result["query_type"],
                "performance": result["performance"],
                "is_query": "?" in message
            }
            
            responses.append(response_data)
            
            print(f"   💬 Response: {result['response']}")
            print(f"   📊 Strategy: {result['performance']['strategy']}, Time: {response_time:.3f}s")
            
            if (i + 1) % 10 == 0:
                avg_time = sum(r["response_time"] for r in responses[-10:]) / 10
                print(f"\n📊 Progress: {i+1}/50 turns completed. Avg time: {avg_time:.3f}s")
                print("="*60)
        
        return responses, system

    def analyze_performance(self, conversation: List[str], responses: List[Dict]) -> Dict:
        """Análisis ultra-específico de performance"""
        
        # Criterios de evaluación ULTRA ESPECÍFICOS
        evaluation_criteria = [
            {
                "question": "Could you tell me what's my full name and current position?",
                "expected_keywords": ["elena rodriguez", "research scientist", "mit"],
                "critical": True
            },
            {
                "question": "What university did I graduate from and what was my field of study?",
                "expected_keywords": ["stanford", "computer science", "phd"],
                "critical": True
            },
            {
                "question": "What are my main hobbies and interests outside of work?",
                "expected_keywords": ["mystery", "hiking", "reading"],
                "critical": False
            },
            {
                "question": "What breakthrough did I mention having yesterday?",
                "expected_keywords": ["quantum algorithm", "breakthrough", "yesterday"],
                "critical": True
            },
            {
                "question": "Which colleague recommended a restaurant and what type of cuisine?",
                "expected_keywords": ["sarah", "japanese", "restaurant"],
                "critical": True
            },
            {
                "question": "What's the name of my cat and when did I adopt her?",
                "expected_keywords": ["quantum", "last month", "cat"],
                "critical": False
            },
            {
                "question": "What conference am I planning to attend next month and where?",
                "expected_keywords": ["quantum computing symposium", "tokyo", "conference"],
                "critical": True
            },
            {
                "question": "What significant meeting do I have next week?",
                "expected_keywords": ["department head", "promotion", "meeting"],
                "critical": True
            },
            {
                "question": "What award did I win last year?",
                "expected_keywords": ["young researcher award", "quantum society"],
                "critical": False
            },
            {
                "question": "Who did I have lunch with last Tuesday?",
                "expected_keywords": ["jennifer williams", "nobel laureate", "tuesday"],
                "critical": True
            }
        ]
        
        # Analizar solo las queries (últimas 10)
        query_responses = [r for r in responses if r["is_query"]]
        
        total_score = 0
        critical_score = 0
        critical_count = 0
        perfect_matches = 0
        high_confidence_responses = 0
        
        detailed_results = []
        
        print(f"\n🔍 DETAILED ANALYSIS:")
        print("="*80)
        
        for i, criteria in enumerate(evaluation_criteria):
            if i < len(query_responses):
                response_data = query_responses[i]
                response_text = response_data["response"].lower()
                
                # Evaluar keywords
                matches = 0
                for keyword in criteria["expected_keywords"]:
                    if keyword.lower() in response_text:
                        matches += 1
                
                score = matches / len(criteria["expected_keywords"])
                
                # Métricas
                confidence = response_data["performance"].get("confidence", 0.0)
                if confidence >= 0.9:
                    high_confidence_responses += 1
                if score == 1.0:
                    perfect_matches += 1
                
                total_score += score
                if criteria["critical"]:
                    critical_score += score
                    critical_count += 1
                
                # Status
                if score == 1.0:
                    status = "✅ PERFECT"
                elif score >= 0.67:
                    status = "🟡 GOOD"
                elif score >= 0.33:
                    status = "🟠 PARTIAL"
                else:
                    status = "❌ POOR"
                
                print(f"\n{i+1}. {'🔥 CRITICAL' if criteria['critical'] else '📝 STANDARD'} {status}")
                print(f"   Q: {criteria['question']}")
                print(f"   A: {response_data['response']}")
                print(f"   📊 Score: {score:.2f} ({matches}/{len(criteria['expected_keywords'])} keywords)")
                print(f"   📊 Expected: {criteria['expected_keywords']}")
                print(f"   📊 Strategy: {response_data['performance']['strategy']}")
                print(f"   📊 Confidence: {confidence:.2f}, Time: {response_data['response_time']:.3f}s")
                
                detailed_results.append({
                    "question": criteria["question"],
                    "response": response_data["response"],
                    "score": score,
                    "matches": matches,
                    "expected": criteria["expected_keywords"],
                    "critical": criteria["critical"],
                    "confidence": confidence,
                    "response_time": response_data["response_time"],
                    "strategy": response_data["performance"]["strategy"]
                })
        
        # Calcular métricas finales
        overall_score = (total_score / len(evaluation_criteria)) * 100
        critical_score_pct = (critical_score / critical_count) * 100 if critical_count > 0 else 0
        perfect_rate = (perfect_matches / len(evaluation_criteria)) * 100
        high_confidence_rate = (high_confidence_responses / len(evaluation_criteria)) * 100
        
        return {
            "overall_score": overall_score,
            "critical_score": critical_score_pct,
            "perfect_match_rate": perfect_rate,
            "high_confidence_rate": high_confidence_rate,
            "perfect_responses": perfect_matches,
            "total_queries": len(evaluation_criteria),
            "critical_queries": critical_count,
            "detailed_results": detailed_results
        }

    def run_master_benchmark(self) -> Dict:
        """Ejecutar benchmark master completo"""
        print("\n🔥 STARTING MASTER BENCHMARK")
        print("="*60)
        
        conversation = self.create_test_conversation()
        print(f"📊 Testing with {len(conversation)} turns")
        
        # Test sistema
        start_time = time.time()
        responses, system = self.test_system(conversation)
        total_time = time.time() - start_time
        
        # Análisis
        analysis = self.analyze_performance(conversation, responses)
        
        # Estadísticas del sistema
        system_stats = system.get_research_statistics()
        
        # Determinar grado
        score = analysis["overall_score"]
        if score >= 95:
            grade, status = "A++", "🔥 PARADIGM SHIFT"
        elif score >= 90:
            grade, status = "A+", "🚀 BREAKTHROUGH"
        elif score >= 85:
            grade, status = "A", "👍 EXCELLENT"
        elif score >= 80:
            grade, status = "B+", "✅ VERY GOOD"
        elif score >= 70:
            grade, status = "B", "🔧 GOOD"
        else:
            grade, status = "C", "❌ NEEDS WORK"
        
        results = {
            "benchmark_info": {
                "timestamp": datetime.now().isoformat(),
                "system_version": "EpisodicMemoryLLM_FINAL",
                "conversation_length": len(conversation),
                "total_time": total_time
            },
            "performance": analysis,
            "system_statistics": system_stats,
            "grade": grade,
            "status": status,
            "conversation": conversation,
            "responses": responses
        }
        
        # Guardar resultados
        results_file = f"{self.results_dir}/master_benchmark_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        self.display_results(results)
        return results

    def display_results(self, results: Dict):
        """Mostrar resultados finales"""
        print("\n" + "="*60)
        print("🏆 MASTER BENCHMARK RESULTS")
        print("="*60)
        
        analysis = results["performance"]
        
        print(f"📊 PERFORMANCE METRICS:")
        print(f"  Overall Score: {analysis['overall_score']:.1f}%")
        print(f"  Critical Questions: {analysis['critical_score']:.1f}%")
        print(f"  Perfect Matches: {analysis['perfect_match_rate']:.1f}%")
        print(f"  High Confidence: {analysis['high_confidence_rate']:.1f}%")
        
        print(f"\n🎯 FINAL ASSESSMENT:")
        print(f"  Grade: {results['grade']}")
        print(f"  Status: {results['status']}")
        
        print(f"\n💡 PERFECT RESPONSES ({analysis['perfect_responses']}/{analysis['total_queries']}):")
        for detail in analysis['detailed_results']:
            if detail['score'] == 1.0:
                print(f"  ✅ {detail['question'][:50]}...")
        
        print(f"\n🔧 IMPROVEMENT AREAS:")
        poor_responses = [r for r in analysis['detailed_results'] if r['score'] < 0.5]
        for detail in poor_responses:
            print(f"  ❌ {detail['question'][:50]}...")
            print(f"     Missing: {[k for k in detail['expected'] if k not in detail['response'].lower()]}")
        
        print("="*60)


def main():
    """Ejecutar benchmark master"""
    benchmark = BenchmarkMasterFinal()
    results = benchmark.run_master_benchmark()
    
    print(f"\n🎯 FINAL VERDICT:")
    score = results["performance"]["overall_score"]
    if score >= 85:
        print(f"🏆 READY FOR MASTER APPLICATIONS! ({score:.1f}%)")
    elif score >= 75:
        print(f"✅ STRONG SYSTEM, MINOR OPTIMIZATIONS ({score:.1f}%)")
    else:
        print(f"🔧 NEEDS OPTIMIZATION BEFORE APPLICATIONS ({score:.1f}%)")
    
    return results


if __name__ == "__main__":
    results = main()