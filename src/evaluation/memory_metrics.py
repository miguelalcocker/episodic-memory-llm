# src/evaluation/memory_metrics.py
"""
Sistema de evaluación para arquitecturas de memoria episódica
Métricas científicas rigurosas para medir performance
"""

import numpy as np
import torch
from typing import List, Dict, Tuple, Optional
import json
import re
from datetime import datetime
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)

class MemoryEvaluator:
    """
    Evaluador completo para sistemas de memoria episódica
    """
    
    def __init__(self, model=None):
        self.model = model
        self.evaluation_history = []
        
    def temporal_consistency_score(self, conversations: List[Dict]) -> float:
        """
        Mide qué tan consistente es el modelo con información temporal
        
        Args:
            conversations: Lista de conversaciones con timestamps
            
        Returns:
            Score 0-1 donde 1 = perfecta consistencia temporal
        """
        consistency_scores = []
        
        for conv in conversations:
            turns = conv.get('turns', [])
            temporal_references = []
            
            # Extraer referencias temporales
            for turn in turns:
                content = turn.get('content', '')
                
                # Buscar patrones temporales
                temporal_patterns = [
                    r'yesterday', r'today', r'tomorrow',
                    r'last week', r'next week', r'this week',
                    r'ago', r'later', r'before', r'after',
                    r'\d+ days? ago', r'\d+ weeks? ago',
                    r'earlier', r'previously', r'recently'
                ]
                
                for pattern in temporal_patterns:
                    matches = re.findall(pattern, content.lower())
                    if matches:
                        temporal_references.append({
                            'turn_index': turn.get('turn_index', 0),
                            'timestamp': turn.get('timestamp', 0),
                            'reference': matches[0],
                            'content': content
                        })
            
            # Calcular consistencia
            if len(temporal_references) > 1:
                consistency = self._calculate_temporal_consistency(temporal_references)
                consistency_scores.append(consistency)
        
        return np.mean(consistency_scores) if consistency_scores else 0.0
    
    def _calculate_temporal_consistency(self, references: List[Dict]) -> float:
        """
        Calcula consistencia temporal entre referencias
        """
        consistent_pairs = 0
        total_pairs = 0
        
        for i in range(len(references)):
            for j in range(i + 1, len(references)):
                ref1, ref2 = references[i], references[j]
                
                # Verificar si las referencias son temporalmente consistentes
                time_diff = ref2['timestamp'] - ref1['timestamp']
                
                # Lógica simplificada - en implementación real sería más sofisticada
                if self._are_temporally_consistent(ref1, ref2, time_diff):
                    consistent_pairs += 1
                total_pairs += 1
        
        return consistent_pairs / total_pairs if total_pairs > 0 else 1.0
    
    def _are_temporally_consistent(self, ref1: Dict, ref2: Dict, time_diff: float) -> bool:
        """
        Determina si dos referencias temporales son consistentes
        """
        # Implementación simplificada
        # En la versión completa usaríamos NLP más avanzado
        return True  # Placeholder
    
    def memory_recall_accuracy(self, test_scenarios: List[Dict]) -> Dict[str, float]:
        """
        Evalúa precisión del recall de memorias específicas
        
        Args:
            test_scenarios: Lista de escenarios de prueba con ground truth
            
        Returns:
            Dict con métricas de accuracy por tipo de memoria
        """
        if not self.model:
            return {"error": "No model provided"}
        
        results = {
            "factual_recall": 0.0,
            "personal_info_recall": 0.0,
            "contextual_recall": 0.0,
            "overall_accuracy": 0.0
        }
        
        factual_scores = []
        personal_scores = []
        contextual_scores = []
        
        for scenario in test_scenarios:
            scenario_type = scenario.get('type', 'general')
            setup_turns = scenario.get('setup', [])
            test_query = scenario.get('query', '')
            expected_elements = scenario.get('expected_elements', [])
            
            # Setup: alimentar al modelo con información
            for turn in setup_turns:
                if turn.get('role') == 'user':
                    self.model.chat(turn.get('content', ''))
            
            # Test: hacer query y evaluar respuesta
            response = self.model.chat(test_query)
            
            # Calcular score basado en elementos esperados
            score = self._calculate_recall_score(response, expected_elements)
            
            if scenario_type == 'factual':
                factual_scores.append(score)
            elif scenario_type == 'personal':
                personal_scores.append(score)
            elif scenario_type == 'contextual':
                contextual_scores.append(score)
        
        # Calcular promedios
        if factual_scores:
            results["factual_recall"] = np.mean(factual_scores)
        if personal_scores:
            results["personal_info_recall"] = np.mean(personal_scores)
        if contextual_scores:
            results["contextual_recall"] = np.mean(contextual_scores)
        
        all_scores = factual_scores + personal_scores + contextual_scores
        if all_scores:
            results["overall_accuracy"] = np.mean(all_scores)
        
        return results
    
    def _calculate_recall_score(self, response: str, expected_elements: List[str]) -> float:
        """
        Calcula score de recall basado en elementos esperados en la respuesta
        """
        response_lower = response.lower()
        found_elements = 0
        
        for element in expected_elements:
            if element.lower() in response_lower:
                found_elements += 1
        
        return found_elements / len(expected_elements) if expected_elements else 0.0
    
    def personality_persistence_score(self, conversations: List[Dict]) -> float:
        """
        Mide qué tan consistente es la personalidad a través del tiempo
        
        Args:
            conversations: Lista de conversaciones a lo largo del tiempo
            
        Returns:
            Score 0-1 donde 1 = personalidad perfectamente consistente
        """
        if not conversations or len(conversations) < 2:
            return 1.0  # Sin datos suficientes para inconsistencia
        
        personality_vectors = []
        
        for conv in conversations:
            # Extraer características de personalidad de cada conversación
            personality_features = self._extract_personality_features(conv)
            personality_vectors.append(personality_features)
        
        # Calcular consistencia entre vectores de personalidad
        consistency_scores = []
        for i in range(len(personality_vectors) - 1):
            similarity = cosine_similarity(
                [personality_vectors[i]], 
                [personality_vectors[i + 1]]
            )[0][0]
            consistency_scores.append(similarity)
        
        return np.mean(consistency_scores) if consistency_scores else 1.0
    
    def _extract_personality_features(self, conversation: Dict) -> List[float]:
        """
        Extrae vector de características de personalidad de una conversación
        """
        # Features simplificados - en implementación real usaríamos modelos de personalidad
        features = [0.0] * 10  # Vector de 10 dimensiones
        
        turns = conversation.get('turns', [])
        assistant_turns = [t for t in turns if t.get('role') == 'assistant']
        
        if not assistant_turns:
            return features
        
        all_text = ' '.join([t.get('content', '') for t in assistant_turns])
        text_lower = all_text.lower()
        
        # Feature engineering básico
        features[0] = len([w for w in text_lower.split() if w in ['friendly', 'kind', 'nice']])  # Amabilidad
        features[1] = len([w for w in text_lower.split() if w in ['sorry', 'apologize']])      # Disculpas
        features[2] = len([w for w in text_lower.split() if w in ['!', 'exciting', 'great']])  # Entusiasmo
        features[3] = len(re.findall(r'\?', all_text))                                         # Preguntas
        features[4] = np.mean([len(t.get('content', '')) for t in assistant_turns])           # Longitud promedio
        
        # Normalizar
        max_val = max(features) if max(features) > 0 else 1
        features = [f / max_val for f in features]
        
        return features
    
    def context_integration_score(self, test_cases: List[Dict]) -> float:
        """
        Evalúa capacidad de integrar información de múltiples contextos
        
        Args:
            test_cases: Casos que requieren conectar información dispersa
            
        Returns:
            Score 0-1 de capacidad de integración contextual
        """
        if not self.model:
            return 0.0
        
        integration_scores = []
        
        for case in test_cases:
            # Información dispersa en múltiples turnos
            context_turns = case.get('context_turns', [])
            integration_query = case.get('integration_query', '')
            expected_connections = case.get('expected_connections', [])
            
            # Alimentar contexto disperso
            for turn in context_turns:
                if turn.get('role') == 'user':
                    self.model.chat(turn.get('content', ''))
            
            # Query que requiere integración
            response = self.model.chat(integration_query)
            
            # Evaluar si la respuesta conecta la información apropiadamente
            score = self._evaluate_context_integration(response, expected_connections)
            integration_scores.append(score)
        
        return np.mean(integration_scores) if integration_scores else 0.0
    
    def _evaluate_context_integration(self, response: str, expected_connections: List[str]) -> float:
        """
        Evalúa si una respuesta integra apropiadamente múltiples contextos
        """
        response_lower = response.lower()
        connections_found = 0
        
        for connection in expected_connections:
            # Buscar evidencia de la conexión en la respuesta
            if connection.lower() in response_lower:
                connections_found += 1
        
        return connections_found / len(expected_connections) if expected_connections else 0.0
    
    def comprehensive_evaluation(self, test_suite: Dict) -> Dict[str, float]:
        """
        Evaluación completa del sistema de memoria episódica
        
        Args:
            test_suite: Suite completa de tests
            
        Returns:
            Dict con todas las métricas calculadas
        """
        results = {
            "timestamp": datetime.now().isoformat(),
            "model_name": getattr(self.model, '__class__', {}).get('__name__', 'Unknown'),
        }
        
        # Ejecutar todas las evaluaciones
        logger.info("Starting comprehensive evaluation...")
        
        # 1. Consistencia temporal
        if 'temporal_tests' in test_suite:
            logger.info("Evaluating temporal consistency...")
            results['temporal_consistency'] = self.temporal_consistency_score(
                test_suite['temporal_tests']
            )
        
        # 2. Accuracy de recall
        if 'recall_tests' in test_suite:
            logger.info("Evaluating memory recall accuracy...")
            recall_results = self.memory_recall_accuracy(test_suite['recall_tests'])
            results.update(recall_results)
        
        # 3. Persistencia de personalidad
        if 'personality_tests' in test_suite:
            logger.info("Evaluating personality persistence...")
            results['personality_persistence'] = self.personality_persistence_score(
                test_suite['personality_tests']
            )
        
        # 4. Integración contextual
        if 'integration_tests' in test_suite:
            logger.info("Evaluating context integration...")
            results['context_integration'] = self.context_integration_score(
                test_suite['integration_tests']
            )
        
        # 5. Score combinado
        metric_keys = ['temporal_consistency', 'overall_accuracy', 'personality_persistence', 'context_integration']
        available_metrics = [results[key] for key in metric_keys if key in results]
        
        if available_metrics:
            results['combined_score'] = np.mean(available_metrics)
        
        # Guardar en historial
        self.evaluation_history.append(results)
        
        logger.info("Comprehensive evaluation completed!")
        return results
    
    def save_evaluation_results(self, results: Dict, path: str):
        """Guardar resultados de evaluación"""
        with open(path, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Evaluation results saved to {path}")
    
    def generate_evaluation_report(self, results: Dict) -> str:
        """
        Generar reporte legible de evaluación
        """
        report = []
        report.append("=" * 60)
        report.append("EPISODIC MEMORY EVALUATION REPORT")
        report.append("=" * 60)
        report.append(f"Timestamp: {results.get('timestamp', 'Unknown')}")
        report.append(f"Model: {results.get('model_name', 'Unknown')}")
        report.append("")
        
        # Métricas principales
        report.append("CORE METRICS:")
        report.append("-" * 30)
        
        metrics = [
            ("Temporal Consistency", "temporal_consistency"),
            ("Overall Recall Accuracy", "overall_accuracy"),
            ("Personality Persistence", "personality_persistence"),
            ("Context Integration", "context_integration"),
            ("Combined Score", "combined_score")
        ]
        
        for name, key in metrics:
            if key in results:
                score = results[key]
                percentage = score * 100
                bar = "█" * int(percentage / 5) + "░" * (20 - int(percentage / 5))
                report.append(f"{name:25}: {percentage:5.1f}% |{bar}|")
        
        # Métricas detalladas de recall
        if any(key in results for key in ['factual_recall', 'personal_info_recall', 'contextual_recall']):
            report.append("")
            report.append("DETAILED RECALL METRICS:")
            report.append("-" * 30)
            
            recall_metrics = [
                ("Factual Information", "factual_recall"),
                ("Personal Information", "personal_info_recall"),
                ("Contextual Information", "contextual_recall")
            ]
            
            for name, key in recall_metrics:
                if key in results:
                    score = results[key]
                    percentage = score * 100
                    report.append(f"{name:20}: {percentage:5.1f}%")
        
        # Interpretación
        report.append("")
        report.append("INTERPRETATION:")
        report.append("-" * 30)
        
        combined_score = results.get('combined_score', 0)
        if combined_score >= 0.8:
            interpretation = "EXCELLENT - Memory system performs at high level"
        elif combined_score >= 0.6:
            interpretation = "GOOD - Solid performance with room for improvement"
        elif combined_score >= 0.4:
            interpretation = "FAIR - Basic functionality but needs enhancement"
        else:
            interpretation = "NEEDS IMPROVEMENT - Significant issues detected"
        
        report.append(interpretation)
        report.append("")
        report.append("=" * 60)
        
        return "\n".join(report)


def create_test_suite() -> Dict:
    """
    Crear suite de tests para evaluación completa
    """
    return {
        "temporal_tests": [
            {
                "turns": [
                    {"role": "user", "content": "I went to Paris yesterday", "timestamp": 1000, "turn_index": 0},
                    {"role": "assistant", "content": "That sounds wonderful! How was Paris?", "timestamp": 1001, "turn_index": 1},
                    {"role": "user", "content": "It was amazing. Today I'm back home", "timestamp": 86400, "turn_index": 2}
                ]
            }
        ],
        
        "recall_tests": [
            {
                "type": "personal",
                "setup": [
                    {"role": "user", "content": "My name is Sarah and I'm a doctor"},
                    {"role": "user", "content": "I work at City Hospital"}
                ],
                "query": "Where do I work?",
                "expected_elements": ["City Hospital", "hospital"]
            },
            {
                "type": "factual", 
                "setup": [
                    {"role": "user", "content": "The capital of France is Paris"},
                    {"role": "user", "content": "Paris has the Eiffel Tower"}
                ],
                "query": "What famous landmark is in the French capital?",
                "expected_elements": ["Eiffel Tower", "tower"]
            }
        ],
        
        "personality_tests": [
            {
                "turns": [
                    {"role": "assistant", "content": "I'm excited to help you today!"},
                    {"role": "assistant", "content": "That's wonderful news!"}
                ]
            },
            {
                "turns": [
                    {"role": "assistant", "content": "I'm thrilled to assist!"},
                    {"role": "assistant", "content": "How exciting!"}
                ]
            }
        ],
        
        "integration_tests": [
            {
                "context_turns": [
                    {"role": "user", "content": "I love Italian food"},
                    {"role": "user", "content": "My favorite restaurant is Mario's"},
                    {"role": "user", "content": "I'm planning a dinner tonight"}
                ],
                "integration_query": "Where should I go for dinner?",
                "expected_connections": ["Mario's", "Italian"]
            }
        ]
    }


def test_evaluation_system():
    """
    Test del sistema de evaluación
    """
    print("🧪 Testing Evaluation System...")
    
    # Crear evaluador
    evaluator = MemoryEvaluator()
    
    # Crear test suite
    test_suite = create_test_suite()
    
    # Ejecutar evaluaciones individuales
    print("\n1. Testing temporal consistency...")
    temporal_score = evaluator.temporal_consistency_score(test_suite['temporal_tests'])
    print(f"Temporal consistency score: {temporal_score:.3f}")
    
    print("\n2. Testing personality persistence...")
    personality_score = evaluator.personality_persistence_score(test_suite['personality_tests'])
    print(f"Personality persistence score: {personality_score:.3f}")
    
    # Generar reporte de prueba
    mock_results = {
        "timestamp": datetime.now().isoformat(),
        "model_name": "BaselineRAG",
        "temporal_consistency": 0.85,
        "overall_accuracy": 0.72,
        "factual_recall": 0.80,
        "personal_info_recall": 0.75,
        "contextual_recall": 0.60,
        "personality_persistence": 0.90,
        "context_integration": 0.65,
        "combined_score": 0.78
    }
    
    print("\n3. Generating evaluation report...")
    report = evaluator.generate_evaluation_report(mock_results)
    print(report)
    
    print("\n✅ Evaluation system test completed!")
    return evaluator

if __name__ == "__main__":
    test_evaluation_system()
