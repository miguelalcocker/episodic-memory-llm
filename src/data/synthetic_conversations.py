# src/data/synthetic_conversations.py
"""
Generador de datasets sintéticos para entrenar y evaluar memoria episódica
Crea conversaciones realistas con referencias temporales y contextuales
"""

import random
import json
import time
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import itertools
from pathlib import Path

class ConversationGenerator:
    """
    Generador de conversaciones sintéticas con memoria episódica
    """
    
    def __init__(self):
        # Templates de personalidad
        self.personalities = {
            "friendly": {
                "traits": ["enthusiastic", "helpful", "warm"],
                "phrases": ["That's wonderful!", "I'm excited to help!", "How lovely!"],
                "question_style": "open-ended"
            },
            "professional": {
                "traits": ["formal", "precise", "courteous"],
                "phrases": ["I understand.", "Certainly.", "Allow me to assist you."],
                "question_style": "direct"
            },
            "casual": {
                "traits": ["relaxed", "informal", "conversational"],
                "phrases": ["Cool!", "No worries!", "Sounds good!"],
                "question_style": "casual"
            }
        }
        
        # Templates de información personal
        self.personal_info_templates = {
            "names": ["Alex", "Sarah", "Michael", "Emma", "David", "Lisa", "John", "Maria"],
            "occupations": ["doctor", "teacher", "engineer", "artist", "chef", "lawyer", "nurse", "programmer"],
            "hobbies": ["reading", "painting", "hiking", "cooking", "photography", "music", "dancing", "gaming"],
            "locations": ["New York", "London", "Tokyo", "Paris", "Sydney", "Toronto", "Berlin", "Madrid"],
            "ages": list(range(25, 65)),
            "pets": ["dog", "cat", "bird", "fish", "rabbit"]
        }
        
        # Templates de eventos y experiencias
        self.event_templates = {
            "travel": [
                "I went to {location} {time_ref}",
                "My trip to {location} was {time_ref}",
                "I visited {location} {time_ref}"
            ],
            "work": [
                "I started my job as a {occupation} {time_ref}",
                "I had an important meeting {time_ref}",
                "I finished a big project {time_ref}"
            ],
            "personal": [
                "I adopted a {pet} {time_ref}",
                "I moved to {location} {time_ref}",
                "I started learning {hobby} {time_ref}"
            ],
            "relationships": [
                "I met my best friend {time_ref}",
                "I went on a date {time_ref}",
                "I attended a wedding {time_ref}"
            ]
        }
        
        # Referencias temporales
        self.time_references = {
            "recent": ["yesterday", "last week", "a few days ago", "recently"],
            "medium": ["last month", "a couple of months ago", "this year"],
            "distant": ["last year", "two years ago", "a long time ago", "when I was younger"]
        }
        
        # Templates de preguntas que requieren memoria
        self.memory_question_templates = [
            "What did I tell you about my {topic}?",
            "Do you remember when I mentioned {topic}?",
            "Can you recall what I said about {topic}?",
            "What do you know about my {topic}?",
            "Tell me what you remember about {topic}."
        ]
        
        # Topics para preguntas de memoria
        self.memory_topics = [
            "job", "hobbies", "travel", "family", "pets", "favorite food",
            "hometown", "education", "interests", "goals", "experiences"
        ]
    
    def generate_personal_introduction(self) -> Dict:
        """Generar introducción personal realista"""
        name = random.choice(self.personal_info_templates["names"])
        age = random.choice(self.personal_info_templates["ages"])
        occupation = random.choice(self.personal_info_templates["occupations"])
        location = random.choice(self.personal_info_templates["locations"])
        hobbies = random.sample(self.personal_info_templates["hobbies"], k=random.randint(1, 3))
        
        intro_templates = [
            f"Hi, I'm {name}. I'm {age} years old and I work as a {occupation} in {location}.",
            f"Hello! My name is {name}, I'm a {occupation} living in {location}.",
            f"Nice to meet you! I'm {name}, {age} years old. I work in {location} as a {occupation}."
        ]
        
        intro = random.choice(intro_templates)
        
        if random.random() < 0.7:  # 70% chance de mencionar hobbies
            hobby_text = f" I love {', '.join(hobbies[:-1])} and {hobbies[-1]}." if len(hobbies) > 1 else f" I enjoy {hobbies[0]}."
            intro += hobby_text
        
        return {
            "content": intro,
            "metadata": {
                "type": "personal_introduction",
                "extracted_info": {
                    "name": name,
                    "age": age,
                    "occupation": occupation,
                    "location": location,
                    "hobbies": hobbies
                }
            }
        }
    
    def generate_event_sharing(self, personal_info: Dict) -> Dict:
        """Generar compartir de eventos/experiencias"""
        event_type = random.choice(list(self.event_templates.keys()))
        template = random.choice(self.event_templates[event_type])
        time_category = random.choice(list(self.time_references.keys()))
        time_ref = random.choice(self.time_references[time_category])
        
        # Rellenar template con información personal
        content = template.format(
            location=random.choice(self.personal_info_templates["locations"]),
            occupation=personal_info.get("occupation", random.choice(self.personal_info_templates["occupations"])),
            pet=random.choice(self.personal_info_templates["pets"]),
            hobby=random.choice(personal_info.get("hobbies", self.personal_info_templates["hobbies"])),
            time_ref=time_ref
        )
        
        return {
            "content": content,
            "metadata": {
                "type": "event_sharing",
                "event_type": event_type,
                "time_category": time_category,
                "time_reference": time_ref
            }
        }
    
    def generate_memory_question(self, conversation_history: List[Dict]) -> Dict:
        """Generar pregunta que requiere memoria episódica"""
        if not conversation_history:
            return None
        
        # Buscar información previa que se puede referenciar
        available_topics = []
        for turn in conversation_history:
            metadata = turn.get("metadata", {})
            if metadata.get("type") == "personal_introduction":
                extracted_info = metadata.get("extracted_info", {})
                available_topics.extend(["job", "hobbies", "age", "location"])
            elif metadata.get("type") == "event_sharing":
                available_topics.append(metadata.get("event_type", "experience"))
        
        if not available_topics:
            topic = random.choice(self.memory_topics)
        else:
            topic = random.choice(available_topics)
        
        template = random.choice(self.memory_question_templates)
        content = template.format(topic=topic)
        
        return {
            "content": content,
            "metadata": {
                "type": "memory_question",
                "referenced_topic": topic,
                "requires_memory": True
            }
        }
    
    def generate_contextual_question(self, conversation_history: List[Dict]) -> Dict:
        """Generar pregunta que requiere integrar múltiples contextos"""
        if len(conversation_history) < 2:
            return None
        
        # Templates que requieren conectar información
        contextual_templates = [
            "Based on what I've told you, what would you recommend for me?",
            "Given my interests and background, what do you think I should try?",
            "Considering everything I've shared, what advice would you give me?",
            "What connections do you see between the things I've mentioned?",
            "How do you think my experiences relate to each other?"
        ]
        
        content = random.choice(contextual_templates)
        
        return {
            "content": content,
            "metadata": {
                "type": "contextual_integration",
                "requires_multiple_contexts": True
            }
        }
    
    def generate_conversation(self, 
                            num_turns: int = 10, 
                            personality: str = None,
                            complexity: str = "medium") -> Dict:
        """
        Generar conversación completa con memoria episódica
        
        Args:
            num_turns: Número de turnos en la conversación
            personality: Tipo de personalidad ("friendly", "professional", "casual")
            complexity: Complejidad ("simple", "medium", "complex")
        """
        if personality is None:
            personality = random.choice(list(self.personalities.keys()))
        
        conversation = {
            "id": f"conv_{int(time.time())}_{random.randint(1000, 9999)}",
            "personality": personality,
            "complexity": complexity,
            "turns": [],
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "num_turns": num_turns,
                "memory_requirements": []
            }
        }
        
        # Primer turno: introducción personal
        intro = self.generate_personal_introduction()
        conversation["turns"].append({
            "turn_index": 0,
            "role": "user",
            "content": intro["content"],
            "timestamp": time.time(),
            "metadata": intro["metadata"]
        })
        
        personal_info = intro["metadata"]["extracted_info"]
        
        # Turnos siguientes
        for i in range(1, num_turns):
            if i % 2 == 1:  # Turno del assistant
                # Respuesta simple para mantener flujo
                responses = [
                    "That's interesting! Tell me more.",
                    "I see. What else would you like to share?",
                    "Thank you for sharing that with me.",
                    "That sounds wonderful!",
                    "I appreciate you telling me about that."
                ]
                
                content = random.choice(responses)
                conversation["turns"].append({
                    "turn_index": i,
                    "role": "assistant", 
                    "content": content,
                    "timestamp": time.time() + i,
                    "metadata": {"type": "response"}
                })
                
            else:  # Turno del user
                # Decidir tipo de turno basado en complejidad y progreso
                turn_type = self._decide_turn_type(i, num_turns, complexity, conversation["turns"])
                
                if turn_type == "event_sharing":
                    turn_data = self.generate_event_sharing(personal_info)
                elif turn_type == "memory_question":
                    turn_data = self.generate_memory_question(conversation["turns"])
                elif turn_type == "contextual_question":
                    turn_data = self.generate_contextual_question(conversation["turns"])
                else:  # random_sharing
                    turn_data = self.generate_random_sharing(personal_info)
                
                if turn_data:
                    conversation["turns"].append({
                        "turn_index": i,
                        "role": "user",
                        "content": turn_data["content"],
                        "timestamp": time.time() + i,
                        "metadata": turn_data["metadata"]
                    })
                    
                    # Registrar requerimientos de memoria
                    if turn_data["metadata"].get("requires_memory"):
                        conversation["metadata"]["memory_requirements"].append({
                            "turn_index": i,
                            "type": turn_data["metadata"]["type"],
                            "topic": turn_data["metadata"].get("referenced_topic")
                        })
        
        return conversation
    
    def _decide_turn_type(self, turn_index: int, total_turns: int, complexity: str, history: List[Dict]) -> str:
        """Decidir qué tipo de turno generar basado en contexto"""
        progress = turn_index / total_turns
        
        # Probabilidades basadas en complejidad y progreso
        if complexity == "simple":
            if progress < 0.7:
                return "event_sharing" if random.random() < 0.8 else "random_sharing"
            else:
                return "memory_question" if random.random() < 0.6 else "event_sharing"
                
        elif complexity == "medium":
            if progress < 0.5:
                return "event_sharing" if random.random() < 0.7 else "random_sharing"
            elif progress < 0.8:
                return "memory_question" if random.random() < 0.5 else "event_sharing"
            else:
                return "contextual_question" if random.random() < 0.4 else "memory_question"
                
        else:  # complex
            if progress < 0.3:
                return "event_sharing" if random.random() < 0.6 else "random_sharing"
            elif progress < 0.6:
                return "memory_question" if random.random() < 0.4 else "event_sharing"
            else:
                return "contextual_question" if random.random() < 0.6 else "memory_question"
    
    def generate_random_sharing(self, personal_info: Dict) -> Dict:
        """Generar compartir de información aleatoria"""
        templates = [
            "I really enjoy {hobby} on weekends.",
            "My favorite place in {location} is the park.",
            "I've been thinking about changing careers.",
            "I love trying new restaurants.",
            "I'm planning to take a vacation soon.",
            "I've been reading this amazing book lately.",
            "I'm learning a new skill in my free time."
        ]
        
        template = random.choice(templates)
        content = template.format(
            hobby=random.choice(personal_info.get("hobbies", ["reading"])),
            location=personal_info.get("location", "the city")
        )
        
        return {
            "content": content,
            "metadata": {
                "type": "random_sharing",
                "casual_info": True
            }
        }
    
    def generate_dataset(self, 
                        num_conversations: int = 100,
                        conversation_lengths: List[int] = None,
                        complexity_distribution: Dict[str, float] = None) -> List[Dict]:
        """
        Generar dataset completo de conversaciones
        
        Args:
            num_conversations: Número de conversaciones a generar
            conversation_lengths: Lista de longitudes posibles
            complexity_distribution: Distribución de complejidades
        """
        if conversation_lengths is None:
            conversation_lengths = [6, 8, 10, 12, 15, 20]
        
        if complexity_distribution is None:
            complexity_distribution = {
                "simple": 0.3,
                "medium": 0.5, 
                "complex": 0.2
            }
        
        dataset = []
        
        print(f"Generating {num_conversations} conversations...")
        
        for i in range(num_conversations):
            # Seleccionar parámetros
            length = random.choice(conversation_lengths)
            complexity = random.choices(
                list(complexity_distribution.keys()),
                weights=list(complexity_distribution.values())
            )[0]
            personality = random.choice(list(self.personalities.keys()))
            
            # Generar conversación
            conversation = self.generate_conversation(
                num_turns=length,
                personality=personality,
                complexity=complexity
            )
            
            dataset.append(conversation)
            
            if (i + 1) % 20 == 0:
                print(f"Generated {i + 1}/{num_conversations} conversations...")
        
        return dataset
    
    def save_dataset(self, dataset: List[Dict], filepath: str):
        """Guardar dataset a archivo JSON"""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(dataset, f, indent=2)
        
        print(f"Dataset saved to {filepath}")
        print(f"Total conversations: {len(dataset)}")
        print(f"Total turns: {sum(len(conv['turns']) for conv in dataset)}")
    
    def create_evaluation_scenarios(self) -> Dict:
        """Crear escenarios específicos para evaluación"""
        scenarios = {
            "temporal_consistency": [],
            "memory_recall": [],
            "personality_persistence": [],
            "context_integration": []
        }
        
        # Escenarios de consistencia temporal
        for i in range(10):
            conv = self.generate_conversation(num_turns=8, complexity="medium")
            scenarios["temporal_consistency"].append(conv)
        
        # Escenarios de recall específicos
        recall_scenarios = [
            {
                "type": "personal",
                "setup": [
                    {"role": "user", "content": "Hi, I'm Emma and I work as a doctor at City Hospital"},
                    {"role": "user", "content": "I specialize in pediatrics and love working with children"}
                ],
                "query": "What's my job and where do I work?",
                "expected_elements": ["doctor", "City Hospital", "pediatrics"]
            },
            {
                "type": "factual",
                "setup": [
                    {"role": "user", "content": "I went to Paris last month and visited the Louvre"},
                    {"role": "user", "content": "The Mona Lisa was incredible to see in person"}
                ],
                "query": "Where did I see the Mona Lisa?",
                "expected_elements": ["Paris", "Louvre", "museum"]
            },
            {
                "type": "contextual",
                "setup": [
                    {"role": "user", "content": "I love Italian food, especially pasta"},
                    {"role": "user", "content": "My favorite restaurant is Tony's on Main Street"},
                    {"role": "user", "content": "I'm planning a special dinner for my anniversary"}
                ],
                "query": "Where should I go for my anniversary dinner?",
                "expected_elements": ["Tony's", "Main Street", "Italian"]
            }
        ]
        
        scenarios["memory_recall"] = recall_scenarios
        
        # Escenarios de persistencia de personalidad
        personality_scenarios = []
        for personality in self.personalities.keys():
            conv1 = self.generate_conversation(num_turns=6, personality=personality)
            conv2 = self.generate_conversation(num_turns=6, personality=personality)
            personality_scenarios.extend([conv1, conv2])
        
        scenarios["personality_persistence"] = personality_scenarios
        
        # Escenarios de integración contextual
        integration_scenarios = [
            {
                "context_turns": [
                    {"role": "user", "content": "I'm a software engineer"},
                    {"role": "user", "content": "I love hiking on weekends"},
                    {"role": "user", "content": "I'm looking for a new hobby"}
                ],
                "integration_query": "What hobby would you recommend for me?",
                "expected_connections": ["outdoor", "technical", "hiking", "programming"]
            },
            {
                "context_turns": [
                    {"role": "user", "content": "I moved to New York last year"},
                    {"role": "user", "content": "I work in finance downtown"},
                    {"role": "user", "content": "I'm looking for a good coffee shop near work"}
                ],
                "integration_query": "Any coffee shop recommendations?",
                "expected_connections": ["New York", "downtown", "finance district"]
            }
        ]
        
        scenarios["context_integration"] = integration_scenarios
        
        return scenarios


def create_comprehensive_dataset():
    """
    Crear dataset comprensivo para el proyecto
    """
    print("🏗️ Creating Comprehensive Dataset...")
    
    generator = ConversationGenerator()
    
    # Generar diferentes tipos de datasets
    datasets = {}
    
    # 1. Training dataset - conversaciones variadas
    print("\n1. Generating training dataset...")
    training_data = generator.generate_dataset(
        num_conversations=200,
        conversation_lengths=[8, 10, 12, 15],
        complexity_distribution={"simple": 0.2, "medium": 0.6, "complex": 0.2}
    )
    datasets["training"] = training_data
    
    # 2. Validation dataset - más pequeño
    print("\n2. Generating validation dataset...")
    validation_data = generator.generate_dataset(
        num_conversations=50,
        conversation_lengths=[10, 12, 15],
        complexity_distribution={"simple": 0.3, "medium": 0.5, "complex": 0.2}
    )
    datasets["validation"] = validation_data
    
    # 3. Test dataset - evaluación rigurosa
    print("\n3. Generating test dataset...")
    test_data = generator.generate_dataset(
        num_conversations=30,
        conversation_lengths=[15, 20],
        complexity_distribution={"simple": 0.1, "medium": 0.4, "complex": 0.5}
    )
    datasets["test"] = test_data
    
    # 4. Evaluation scenarios específicos
    print("\n4. Creating evaluation scenarios...")
    eval_scenarios = generator.create_evaluation_scenarios()
    datasets["evaluation_scenarios"] = eval_scenarios
    
    # Guardar todos los datasets
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    for name, data in datasets.items():
        filepath = data_dir / f"{name}_dataset.json"
        if name == "evaluation_scenarios":
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
            print(f"Evaluation scenarios saved to {filepath}")
        else:
            generator.save_dataset(data, filepath)
    
    # Estadísticas finales
    print("\n📊 Dataset Statistics:")
    print("=" * 40)
    total_conversations = sum(len(data) for name, data in datasets.items() if name != "evaluation_scenarios")
    total_turns = sum(sum(len(conv['turns']) for conv in data) 
                     for name, data in datasets.items() if name != "evaluation_scenarios")
    
    print(f"Total conversations: {total_conversations}")
    print(f"Total turns: {total_turns}")
    print(f"Training conversations: {len(datasets['training'])}")
    print(f"Validation conversations: {len(datasets['validation'])}")  
    print(f"Test conversations: {len(datasets['test'])}")
    print(f"Evaluation scenarios: 4 categories")
    
    print("\n✅ Comprehensive dataset creation completed!")
    return datasets


def test_conversation_generator():
    """
    Test del generador de conversaciones
    """
    print("🧪 Testing Conversation Generator...")
    
    generator = ConversationGenerator()
    
    # Test 1: Generar una conversación simple
    print("\n1. Generating simple conversation...")
    simple_conv = generator.generate_conversation(num_turns=6, complexity="simple")
    
    print(f"Conversation ID: {simple_conv['id']}")
    print(f"Personality: {simple_conv['personality']}")
    print(f"Number of turns: {len(simple_conv['turns'])}")
    
    for turn in simple_conv['turns'][:4]:  # Mostrar primeros 4 turnos
        print(f"  {turn['role']}: {turn['content']}")
    
    # Test 2: Generar conversación compleja
    print("\n2. Generating complex conversation...")
    complex_conv = generator.generate_conversation(num_turns=10, complexity="complex")
    
    memory_requirements = complex_conv['metadata']['memory_requirements']
    print(f"Memory requirements: {len(memory_requirements)}")
    for req in memory_requirements:
        print(f"  Turn {req['turn_index']}: {req['type']} - {req.get('topic', 'N/A')}")
    
    # Test 3: Crear mini dataset
    print("\n3. Creating mini dataset...")
    mini_dataset = generator.generate_dataset(num_conversations=5)
    
    complexities = [conv['complexity'] for conv in mini_dataset]
    personalities = [conv['personality'] for conv in mini_dataset]
    
    print(f"Complexities: {complexities}")
    print(f"Personalities: {personalities}")
    
    print("\n✅ Conversation generator test completed!")
    return generator


if __name__ == "__main__":
    # Ejecutar tests
    generator = test_conversation_generator()
    
    # Crear dataset completo
    datasets = create_comprehensive_dataset()
