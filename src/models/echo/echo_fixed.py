# 🔧 FIX ECHO MODEL - CREAR ESTE ARCHIVO: src/models/echo/echo_fixed.py

"""
🔧 Echo Architecture FIXED - Versión corregida que SÍ recuerda
Corrige problemas de memoria y retrieval
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Model, GPT2Config, GPT2LMHeadModel
from datetime import datetime, timedelta
import numpy as np
from typing import List, Dict, Tuple, Optional
from sentence_transformers import SentenceTransformer
import re

class ImprovedEpisodicMemory:
    """Memoria episódica mejorada que SÍ recuerda información específica"""
    
    def __init__(self, max_episodes: int = 100):
        self.max_episodes = max_episodes
        
        # Almacenamiento estructurado
        self.episodes = []  # Lista de diccionarios con info completa
        self.facts_db = {}  # Base de datos de hechos extraídos
        
        # Encoder semántico para mejor retrieval
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        
    def extract_facts(self, text: str) -> Dict[str, str]:
        """Extraer hechos específicos del texto"""
        facts = {}
        text_lower = text.lower()
        
        # Extraer nombre
        name_patterns = [
            r"my name is (\w+)",
            r"i'm (\w+)",
            r"call me (\w+)"
        ]
        for pattern in name_patterns:
            match = re.search(pattern, text_lower)
            if match:
                facts['name'] = match.group(1).capitalize()
        
        # Extraer edad
        age_patterns = [
            r"i'm (\d+) years old",
            r"i am (\d+) years old",
            r"(\d+) years old"
        ]
        for pattern in age_patterns:
            match = re.search(pattern, text_lower)
            if match:
                facts['age'] = match.group(1)
        
        # Extraer trabajo
        job_patterns = [
            r"i work as (?:a |an )?(.+?)(?:\s+in|\s+at|$)",
            r"i'm (?:a |an )?(.+?)(?:\s+in|\s+at|$)",
            r"my job is (.+?)(?:\s+in|\s+at|$)"
        ]
        for pattern in job_patterns:
            match = re.search(pattern, text_lower)
            if match:
                job = match.group(1).strip()
                if job and len(job) < 50:  # Filtrar respuestas muy largas
                    facts['job'] = job
        
        # Extraer ubicación
        location_patterns = [
            r"in (\w+)(?:\s|$)",
            r"at (\w+)(?:\s|$)",
            r"from (\w+)(?:\s|$)"
        ]
        for pattern in location_patterns:
            match = re.search(pattern, text_lower)
            if match:
                location = match.group(1).capitalize()
                if location and len(location) < 20:
                    facts['location'] = location
        
        # Extraer hobbies
        hobby_patterns = [
            r"i love (.+?)(?:\s+and|\s*$)",
            r"i like (.+?)(?:\s+and|\s*$)",
            r"i enjoy (.+?)(?:\s+and|\s*$)"
        ]
        for pattern in hobby_patterns:
            matches = re.findall(pattern, text_lower)
            if matches:
                hobbies = []
                for match in matches:
                    # Limpiar y dividir hobbies
                    hobby_list = re.split(r'\s+and\s+', match)
                    hobbies.extend([h.strip() for h in hobby_list if h.strip()])
                if hobbies:
                    facts['hobbies'] = hobbies
        
        return facts
    
    def add_episode(self, user_input: str, bot_response: str, timestamp: datetime):
        """Añadir episodio con extracción de hechos"""
        
        # Extraer hechos del input del usuario
        extracted_facts = self.extract_facts(user_input)
        
        # Actualizar base de datos de hechos
        for key, value in extracted_facts.items():
            if key == 'hobbies':
                if 'hobbies' not in self.facts_db:
                    self.facts_db['hobbies'] = []
                if isinstance(value, list):
                    self.facts_db['hobbies'].extend(value)
                else:
                    self.facts_db['hobbies'].append(value)
            else:
                self.facts_db[key] = value
        
        # Crear embedding semántico
        embedding = self.encoder.encode(user_input)
        
        # Almacenar episodio completo
        episode = {
            'user_input': user_input,
            'bot_response': bot_response,
            'timestamp': timestamp,
            'facts': extracted_facts,
            'embedding': embedding
        }
        
        self.episodes.append(episode)
        
        # Limitar tamaño
        if len(self.episodes) > self.max_episodes:
            self.episodes.pop(0)
    
    def answer_question(self, question: str) -> str:
        """Responder pregunta basada en hechos almacenados"""
        question_lower = question.lower()
        
        # Preguntas sobre nombre
        if any(word in question_lower for word in ['name', 'called']):
            if 'name' in self.facts_db:
                return f"Your name is {self.facts_db['name']}."
            else:
                return "I don't know your name yet."
        
        # Preguntas sobre edad
        if any(word in question_lower for word in ['age', 'old']):
            if 'age' in self.facts_db:
                return f"You are {self.facts_db['age']} years old."
            else:
                return "I don't know your age yet."
        
        # Preguntas sobre trabajo
        if any(word in question_lower for word in ['work', 'job', 'profession']):
            if 'job' in self.facts_db:
                location_part = f" in {self.facts_db['location']}" if 'location' in self.facts_db else ""
                return f"You work as a {self.facts_db['job']}{location_part}."
            else:
                return "I don't know what you do for work yet."
        
        # Preguntas sobre hobbies
        if any(word in question_lower for word in ['hobby', 'hobbies', 'like', 'enjoy', 'love']):
            if 'hobbies' in self.facts_db and self.facts_db['hobbies']:
                hobbies_list = list(set(self.facts_db['hobbies']))  # Remove duplicates
                if len(hobbies_list) == 1:
                    return f"You like {hobbies_list[0]}."
                elif len(hobbies_list) == 2:
                    return f"You like {hobbies_list[0]} and {hobbies_list[1]}."
                else:
                    return f"You like {', '.join(hobbies_list[:-1])}, and {hobbies_list[-1]}."
            else:
                return "I don't know your hobbies yet."
        
        # Preguntas sobre ubicación
        if any(word in question_lower for word in ['where', 'location', 'city']):
            if 'location' in self.facts_db:
                return f"You are in {self.facts_db['location']}."
            else:
                return "I don't know where you are located."
        
        # Búsqueda semántica para otras preguntas
        if len(self.episodes) > 0:
            question_embedding = self.encoder.encode(question)
            
            # Calcular similaridades
            similarities = []
            for episode in self.episodes:
                similarity = np.dot(question_embedding, episode['embedding']) / (
                    np.linalg.norm(question_embedding) * np.linalg.norm(episode['embedding'])
                )
                similarities.append(similarity)
            
            # Encontrar episodio más similar
            best_idx = np.argmax(similarities)
            if similarities[best_idx] > 0.5:  # Threshold de similaridad
                return f"Based on our conversation: {self.episodes[best_idx]['user_input']}"
        
        return "I'm not sure about that. Could you tell me more?"
    
    def get_context_for_generation(self, current_input: str, max_context: int = 3) -> str:
        """Generar contexto relevante para la generación"""
        context_parts = []
        
        # Añadir hechos relevantes
        if self.facts_db:
            facts_context = "What I know about you: "
            fact_strings = []
            
            if 'name' in self.facts_db:
                fact_strings.append(f"your name is {self.facts_db['name']}")
            if 'age' in self.facts_db:
                fact_strings.append(f"you are {self.facts_db['age']} years old")
            if 'job' in self.facts_db:
                job_location = self.facts_db['job']
                if 'location' in self.facts_db:
                    job_location += f" in {self.facts_db['location']}"
                fact_strings.append(f"you work as a {job_location}")
            if 'hobbies' in self.facts_db and self.facts_db['hobbies']:
                hobbies = list(set(self.facts_db['hobbies']))
                fact_strings.append(f"you like {', '.join(hobbies)}")
            
            if fact_strings:
                facts_context += ", ".join(fact_strings) + ". "
                context_parts.append(facts_context)
        
        # Añadir episodios recientes relevantes
        if len(self.episodes) > 0:
            recent_episodes = self.episodes[-max_context:]
            for episode in recent_episodes[-2:]:  # Solo los 2 más recientes
                context_parts.append(f"User said: {episode['user_input']}")
        
        return "\n".join(context_parts)

class EchoModelFixed:
    """Modelo Echo mejorado que usa memoria estructurada"""
    
    def __init__(self, base_model_name: str = "gpt2"):
        # Modelo base para generación
        self.tokenizer = None  # Se asigna desde fuera
        self.base_model = GPT2LMHeadModel.from_pretrained(base_model_name)
        
        # Memoria episódica mejorada
        self.memory = ImprovedEpisodicMemory()
        
        # Mover a GPU si disponible
        if torch.cuda.is_available():
            self.base_model = self.base_model.cuda()
    
    def generate_response(self, user_input: str, tokenizer, max_length: int = 100) -> str:
        """Generar respuesta usando memoria episódica"""
        current_time = datetime.now()
        
        # Verificar si es una pregunta que podemos responder directamente
        direct_answer = self.memory.answer_question(user_input)
        if not direct_answer.startswith("I'm not sure"):
            # Añadir a memoria y devolver respuesta directa
            self.memory.add_episode(user_input, direct_answer, current_time)
            return direct_answer
        
        # Para otras respuestas, generar con contexto
        context = self.memory.get_context_for_generation(user_input)
        
        # Crear prompt con contexto
        if context:
            prompt = f"{context}\nUser: {user_input}\nAssistant:"
        else:
            prompt = f"User: {user_input}\nAssistant:"
        
        # Tokenizar con attention mask
        inputs = tokenizer(
            prompt, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=512,
            add_special_tokens=True
        )
        
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        # Generar respuesta
        with torch.no_grad():
            outputs = self.base_model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],  # Usar attention mask
                max_length=inputs['input_ids'].shape[1] + 50,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                no_repeat_ngram_size=2  # Evitar repeticiones
            )
        
        # Decodificar respuesta
        full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = full_response[len(prompt):].strip()
        
        # Limpiar respuesta
        if "\n" in response:
            response = response.split("\n")[0]
        
        # Limitar longitud
        if len(response) > 200:
            response = response[:200] + "..."
        
        # Añadir a memoria
        self.memory.add_episode(user_input, response, current_time)
        
        return response
    
    def get_memory_stats(self) -> Dict:
        """Obtener estadísticas de memoria"""
        return {
            'total_episodes': len(self.memory.episodes),
            'facts_stored': len(self.memory.facts_db),
            'facts': dict(self.memory.facts_db)
        }

# Test rápido
if __name__ == "__main__":
    print("🔧 Testing Fixed Echo Model...")
    
    from transformers import GPT2Tokenizer
    
    # Crear modelo
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    model = EchoModelFixed()
    
    # Test scenario
    test_inputs = [
        "Hi, my name is Alex and I'm 25 years old",
        "I work as a data scientist in Madrid", 
        "I love playing guitar and reading books",
        "What's my name?",
        "How old am I?",
        "Where do I work?",
        "What are my hobbies?"
    ]
    
    print("\n🧠 TESTING FIXED MEMORY:")
    print("=" * 50)
    
    for i, user_input in enumerate(test_inputs, 1):
        print(f"\nTurn {i}:")
        print(f"User: {user_input}")
        
        response = model.generate_response(user_input, tokenizer)
        print(f"Fixed Echo: {response}")
        
        # Mostrar hechos almacenados
        stats = model.get_memory_stats()
        print(f"Facts: {stats['facts']}")
    
    print("\n✅ Fixed model test completed!")
