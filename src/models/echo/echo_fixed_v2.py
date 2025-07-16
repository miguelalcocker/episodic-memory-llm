"""
🔧 Echo Architecture FIXED v2 - Regex corregidos
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
    """Memoria episódica mejorada con regex corregidos"""
    
    def __init__(self, max_episodes: int = 100):
        self.max_episodes = max_episodes
        self.episodes = []
        self.facts_db = {}
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        
    def extract_facts(self, text: str) -> Dict[str, str]:
        """Extraer hechos específicos del texto - REGEX CORREGIDOS"""
        facts = {}
        text_lower = text.lower().strip()
        
        print(f"🔍 Analizando: '{text_lower}'")  # Debug
        
        # REGEX CORREGIDO - Extraer nombre ANTES que edad
        name_patterns = [
            r"my name is ([a-zA-Z]+)",
            r"i'm ([a-zA-Z]+)(?:\s+and)",  # "I'm Alex and..."
            r"call me ([a-zA-Z]+)"
        ]
        for pattern in name_patterns:
            match = re.search(pattern, text_lower)
            if match:
                name = match.group(1).capitalize()
                if name and len(name) < 20 and name.isalpha():  # Solo letras
                    facts['name'] = name
                    print(f"✅ Nombre encontrado: {name}")
                    break
        
        # REGEX CORREGIDO - Extraer edad
        age_patterns = [
            r"(\d+) years old",
            r"i'm (\d+)",
            r"i am (\d+)"
        ]
        for pattern in age_patterns:
            match = re.search(pattern, text_lower)
            if match:
                age = match.group(1)
                if age and 10 <= int(age) <= 100:  # Edad razonable
                    facts['age'] = age
                    print(f"✅ Edad encontrada: {age}")
                    break
        
        # REGEX CORREGIDO - Extraer trabajo
        job_patterns = [
            r"i work as (?:a |an )?([^0-9\n\.]+?)(?:\s+in|\s+at|$)",
            r"i'm (?:a |an )?([^0-9\n\.]+?)(?:\s+in|\s+at|$)",
            r"my job is ([^0-9\n\.]+?)(?:\s+in|\s+at|$)"
        ]
        for pattern in job_patterns:
            match = re.search(pattern, text_lower)
            if match:
                job = match.group(1).strip()
                # Filtrar trabajos válidos
                if (job and 
                    len(job) < 50 and 
                    not job.isdigit() and 
                    'years' not in job and
                    job != 'a' and job != 'an'):
                    facts['job'] = job
                    print(f"✅ Trabajo encontrado: {job}")
                    break
        
        # REGEX CORREGIDO - Extraer ubicación
        location_patterns = [
            r"in ([A-Za-z][a-zA-Z\s]{2,20})(?:\s|$)",
            r"at ([A-Za-z][a-zA-Z\s]{2,20})(?:\s|$)"
        ]
        for pattern in location_patterns:
            match = re.search(pattern, text_lower)
            if match:
                location = match.group(1).strip().title()
                if location and len(location) < 30 and not location.isdigit():
                    facts['location'] = location
                    print(f"✅ Ubicación encontrada: {location}")
                    break
        
        # REGEX CORREGIDO - Extraer hobbies
        hobby_patterns = [
            r"i love (.+?)(?:\s+and\s+(.+?))?(?:\s*$)",
            r"i like (.+?)(?:\s+and\s+(.+?))?(?:\s*$)",
            r"i enjoy (.+?)(?:\s+and\s+(.+?))?(?:\s*$)"
        ]
        
        for pattern in hobby_patterns:
            match = re.search(pattern, text_lower)
            if match:
                hobbies = []
                for group in match.groups():
                    if group:
                        # Dividir por "and"
                        hobby_parts = re.split(r'\s+and\s+', group)
                        for hobby in hobby_parts:
                            hobby = hobby.strip()
                            if (hobby and 
                                len(hobby) < 50 and 
                                not hobby.isdigit() and
                                'years' not in hobby):
                                hobbies.append(hobby)
                
                if hobbies:
                    facts['hobbies'] = hobbies
                    print(f"✅ Hobbies encontrados: {hobbies}")
                break
        
        print(f"📊 Hechos extraídos: {facts}")
        return facts
    
    def add_episode(self, user_input: str, bot_response: str, timestamp: datetime):
        """Añadir episodio con extracción de hechos"""
        
        # Extraer hechos del input del usuario
        extracted_facts = self.extract_facts(user_input)
        
        # Actualizar base de datos de hechos (SIN SOBRESCRIBIR)
        for key, value in extracted_facts.items():
            if key == 'hobbies':
                if 'hobbies' not in self.facts_db:
                    self.facts_db['hobbies'] = []
                if isinstance(value, list):
                    # Añadir solo hobbies nuevos
                    for hobby in value:
                        if hobby not in self.facts_db['hobbies']:
                            self.facts_db['hobbies'].append(hobby)
                else:
                    if value not in self.facts_db['hobbies']:
                        self.facts_db['hobbies'].append(value)
            else:
                # Solo actualizar si no existe o si el nuevo valor es mejor
                if key not in self.facts_db or len(str(value)) > len(str(self.facts_db[key])):
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
        
        return "I'm not sure about that. Could you tell me more?"

class EchoModelFixed:
    """Modelo Echo mejorado que usa memoria estructurada"""
    
    def __init__(self, base_model_name: str = "gpt2"):
        self.base_model = GPT2LMHeadModel.from_pretrained(base_model_name)
        self.memory = ImprovedEpisodicMemory()
        
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
        
        # Para statements informativos, simplemente responder y almacenar
        statements = ['my name is', 'i work as', 'i love', 'i like', 'i am', "i'm"]
        if any(stmt in user_input.lower() for stmt in statements):
            response = "Got it, I'll remember that!"
            self.memory.add_episode(user_input, response, current_time)
            return response
        
        # Para otras respuestas, generar respuesta simple
        response = "That's interesting. Tell me more!"
        self.memory.add_episode(user_input, response, current_time)
        return response
    
    def get_memory_stats(self) -> Dict:
        """Obtener estadísticas de memoria"""
        return {
            'total_episodes': len(self.memory.episodes),
            'facts_stored': len(self.memory.facts_db),
            'facts': dict(self.memory.facts_db)
        }

# Test
if __name__ == "__main__":
    print("🔧 Testing Fixed Echo Model v2...")
    
    from transformers import GPT2Tokenizer
    
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    model = EchoModelFixed()
    
    test_inputs = [
        "Hi, my name is Alex and I'm 25 years old",
        "I work as a data scientist in Madrid", 
        "I love playing guitar and reading books",
        "What's my name?",
        "How old am I?",
        "Where do I work?",
        "What are my hobbies?"
    ]
    
    print("\n🧠 TESTING FIXED MEMORY v2:")
    print("=" * 50)
    
    for i, user_input in enumerate(test_inputs, 1):
        print(f"\nTurn {i}:")
        print(f"User: {user_input}")
        
        response = model.generate_response(user_input, tokenizer)
        print(f"Fixed Echo v2: {response}")
        
        stats = model.get_memory_stats()
        print(f"Facts: {stats['facts']}")
    
    print("\n✅ Fixed model v2 test completed!")
