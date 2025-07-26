import json
import random
import datetime
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments, 
    Trainer,
    GenerationConfig
)
from torch.utils.data import Dataset
import time
import gc
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import re
import warnings
warnings.filterwarnings("ignore")

# ================================
# LLAMA 3.1 8B LLM AGENT IMPLEMENTATION
# ================================

class LlamaLLMAgent:
    """
    Llama 3.1 8B agent implementation 
    Following paper methodology with stable model
    """
    
    def __init__(self, model_name: str = "meta-llama/Meta-Llama-3.1-8B-Instruct", agent_type: str = "human"):
        self.model_name = model_name
        self.agent_type = agent_type
        self.model = None
        self.tokenizer = None
        self.generation_config = None
        self._load_model()
    
    def _load_model(self):
        """Load Llama 3.1 8B model with optimized settings"""
        print(f"Loading {self.model_name} for {self.agent_type} agent...")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            padding_side="left"
        )
        
        # Add pad token if it doesn't exist
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model with optimized settings for A40
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )
        
        # Generation configuration optimized for dialogue
        self.generation_config = GenerationConfig(
            temperature=0.7,
            top_p=0.9,
            top_k=50,
            do_sample=True,
            max_new_tokens=150,
            repetition_penalty=1.1,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id
        )
        
        print(f"Llama model loaded successfully. GPU memory: {torch.cuda.memory_allocated()/1024**3:.2f}GB")
    
    def generate_response(self, prompt: str, history: List[str]) -> str:
        """
        Generate response using Llama with proper chat formatting
        Following Algorithm 1: conversation context processing
        """
        try:
            # Build conversation context for Llama
            full_context = self._build_llama_context(prompt, history)
            
            # Tokenize with proper attention mask
            inputs = self.tokenizer(
                full_context, 
                return_tensors="pt",
                truncation=True,
                max_length=2048,
                padding=False
            ).to(self.model.device)
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    generation_config=self.generation_config,
                    use_cache=True
                )
            
            # Decode only the new tokens
            new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
            new_response = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
            
            # Clean up response
            new_response = self._clean_response(new_response)
            
            return new_response
            
        except Exception as e:
            print(f"Error generating response: {e}")
            return self._fallback_response()
    
    def _build_llama_context(self, prompt: str, history: List[str]) -> str:
        """
        Build conversation context in Llama chat format
        """
        if self.agent_type == "human":
            # For human agent - role-play format
            context = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{prompt}<|eot_id|>\n\n"
            
            # Add conversation history
            if history:
                context += "<|start_header_id|>user<|end_header_id|>\n\nConversation context:\n"
                recent_history = history[-10:] if len(history) > 10 else history
                context += "\n".join(recent_history)
                context += "<|eot_id|>\n\n"
            
            context += "<|start_header_id|>assistant<|end_header_id|>\n\n"
            
        else:
            # For assistant agent - Echo system prompt
            context = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{prompt}<|eot_id|>\n\n"
            
            if history:
                context += "<|start_header_id|>user<|end_header_id|>\n\n"
                recent_history = history[-10:] if len(history) > 10 else history
                context += "\n".join(recent_history)
                context += "<|eot_id|>\n\n"
            
            context += "<|start_header_id|>assistant<|end_header_id|>\n\n"
        
        return context
    
    def _clean_response(self, response: str) -> str:
        """Clean and format the generated response"""
        # Remove chat tokens if they appear
        response = re.sub(r'<\|.*?\|>', '', response)
        response = response.strip()
        
        # Remove incomplete sentences at the end
        sentences = response.split('.')
        if len(sentences) > 1 and len(sentences[-1].strip()) < 10:
            response = '.'.join(sentences[:-1]) + '.'
        
        # Ensure reasonable length
        if len(response) > 250:
            response = response[:250] + "..."
        
        # Remove empty responses
        if not response.strip():
            return self._fallback_response()
        
        return response
    
    def _fallback_response(self) -> str:
        """Fallback responses if generation fails"""
        if self.agent_type == "human":
            fallbacks = [
                "That's interesting. Can you tell me more about that?",
                "I see. What do you think about this topic?",
                "Thanks for sharing that information with me.",
                "Could you help me understand this better?",
                "That sounds important. Tell me more."
            ]
        else:
            fallbacks = [
                "I understand. How can I help you with that?",
                "That's a good question. Let me think about it.",
                "I appreciate you sharing that with me.",
                "I'm here to help. What would you like to know?",
                "Thank you for telling me about this."
            ]
        
        return random.choice(fallbacks)
    
    def cleanup(self):
        """Clean up GPU memory"""
        if self.model is not None:
            del self.model
        if self.tokenizer is not None:
            del self.tokenizer
        torch.cuda.empty_cache()
        gc.collect()

# ================================
# ENHANCED CHARACTER GENERATION WITH LLAMA
# ================================

@dataclass
class CharacterCard:
    """Character card following exact paper specifications"""
    name: str
    occupation: str
    age: int
    gender: str
    hobbies: List[str]
    personality: List[str]
    social_relationships: str

class LlamaCharacterGenerator:
    """
    Character generator using Llama 3.1 8B
    Following Section 3.1: "we utilized the LLM to generate the Social Relationships"
    """
    
    def __init__(self, llm_agent: LlamaLLMAgent):
        self.llm_agent = llm_agent
        
        # Paper-specified attributes
        self.occupations = [
            "Teacher", "Doctor", "Engineer", "Artist", "Musician", "Writer", 
            "Lawyer", "Chef", "Scientist", "Programmer", "Designer", "Photographer",
            "Nurse", "Architect", "Psychologist", "Journalist", "Accountant", "Manager"
        ]
        
        self.hobbies = [
            "Reading", "Painting", "Cooking", "Gardening", "Photography", "Music",
            "Sports", "Travel", "Gaming", "Writing", "Dancing", "Hiking",
            "Keeping small animals", "Cybersecurity", "Video reviews", 
            "Academic research", "Karaoke", "Yoga", "Collecting", "Volunteering"
        ]
        
        self.personalities = [
            "Good communicator", "Responsible", "Creative", "Analytical", 
            "Empathetic", "Organized", "Adventurous", "Patient", "Optimistic",
            "Thoughtful", "Reliable", "Curious", "Outgoing", "Introverted",
            "Practical", "Imaginative"
        ]
        
        self.genders = ["Male", "Female"]
        
        self.names = {
            "Male": ["James", "John", "Robert", "Michael", "David", "William", 
                    "Richard", "Joseph", "Thomas", "Daniel", "Matthew", "Anthony"],
            "Female": ["Mary", "Patricia", "Jennifer", "Linda", "Elizabeth", 
                      "Barbara", "Susan", "Jessica", "Sarah", "Karen", "Lisa", "Nancy"]
        }
    
    def generate_character(self) -> CharacterCard:
        """Generate character following paper methodology"""
        # Step 1: Generate basic attributes randomly (except Social Relationships)
        gender = random.choice(self.genders)
        name = random.choice(self.names[gender])
        occupation = random.choice(self.occupations)
        age = random.randint(25, 65)
        selected_hobbies = random.sample(self.hobbies, random.randint(3, 5))
        selected_personality = random.sample(self.personalities, random.randint(2, 3))
        
        # Step 2: Generate Social Relationships using Llama
        social_relationships = self._generate_social_relationships_with_llama(
            name, occupation, age, gender, selected_hobbies, selected_personality
        )
        
        return CharacterCard(
            name=name,
            occupation=occupation,
            age=age,
            gender=gender,
            hobbies=selected_hobbies,
            personality=selected_personality,
            social_relationships=social_relationships
        )
    
    def _generate_social_relationships_with_llama(self, name: str, occupation: str, 
                                                age: int, gender: str, 
                                                hobbies: List[str], 
                                                personality: List[str]) -> str:
        """
        Generate social relationships using Llama
        Following paper: "we utilized the LLM to generate the Social Relationships attribute values"
        """
        prompt = f"""You are creating a realistic character profile. Generate detailed social relationships for this person:

Name: {name}
Occupation: {occupation}
Age: {age}
Gender: {gender}
Hobbies: {', '.join(hobbies)}
Personality: {', '.join(personality)}

Write a realistic description of their social relationships including family, friends, colleagues, and community connections. Make it personal and consistent with their background. Write about {name} in third person.

Social Relationships:"""
        
        try:
            relationships = self.llm_agent.generate_response(prompt, [])
            if len(relationships.strip()) < 20:  # Too short
                return self._fallback_relationships(name, occupation, age)
            return relationships
        except Exception as e:
            print(f"Error generating relationships for {name}: {e}")
            return self._fallback_relationships(name, occupation, age)
    
    def _fallback_relationships(self, name: str, occupation: str, age: int) -> str:
        """Fallback relationship generation"""
        base = f"{name} maintains professional relationships as a {occupation}. "
        
        if age > 35:
            base += f"{name} is married with children and has a close-knit family. "
        elif age > 25:
            base += f"{name} has a committed partner and a strong social circle. "
        
        base += f"As a {occupation}, {name} collaborates with colleagues regularly and participates in professional networks."
        
        return base

# ================================
# ENHANCED PLOT GENERATION
# ================================

@dataclass
class PlotEvent:
    event_type: str  # "common", "real", "hallucinatory"
    description: str
    is_episodic: bool = False

class EnhancedPlotGenerator:
    """
    Plot generator following exact paper methodology
    """
    
    def __init__(self):
        # Following paper's Appendix Table 5 exactly
        self.common_events = [
            "Ask what day of the week it is today",
            "Request AI to inform you of the current date and time", 
            "Ask a question about earth science",
            "Ask AI for its name and call it by that name instead of AI from now on",
            "Ask what day the next working day is",
            "Ask AI how it is feeling today",
            "Inquire about AI's perspective on the development of artificial intelligence",
            "Ask a career-related question",
            "Ask a simple physics question",
            "Ask about the weather today",
            "Discuss current events or news",
            "Ask for a recommendation"
        ]
        
        self.real_events = [
            "Ask if we talked the day before yesterday, and if AI answers yes, then ask what topic we discussed",
            "Ask AI to remember your fitness plan",
            "Ask AI to remember your grandfather's favorite news source",
            "Ask AI if it remembers your fitness plan",
            "Ask AI if it remembers your grandfather's favorite news source, and if it does, ask when you shared it",
            "Ask about a topic from a previous conversation",
            "Reference something discussed in an earlier chat",
            "Ask about a promise made in a previous conversation"
        ]
        
        self.hallucinatory_events = [
            "Ask a piece of information you haven't told AI before: the date you first attended an online course",
            "Ask a piece of information you haven't told AI before: your cherished books", 
            "Ask a piece of information you haven't told AI before: the date of your first marathon completion",
            "Ask a piece of information you haven't told AI before: your private collection inventory",
            "Ask about a conversation that never actually happened",
            "Reference fake previous interactions",
            "Ask about fabricated personal details you never shared"
        ]
    
    def generate_plot(self, num_events: int = 20) -> List[PlotEvent]:
        """
        Generate plot with exactly 20 events as per paper
        Following Section 3.1: "from which 20 events are sampled to form a plot"
        """
        events = []
        
        # Paper distribution: balanced sampling
        num_common = 8
        num_real = 6  
        num_hallucinatory = 5
        
        # Sample from each category
        common_sample = random.sample(self.common_events, 
                                    min(num_common, len(self.common_events)))
        real_sample = random.sample(self.real_events, 
                                  min(num_real, len(self.real_events)))
        hallucinatory_sample = random.sample(self.hallucinatory_events, 
                                           min(num_hallucinatory, len(self.hallucinatory_events)))
        
        # Create PlotEvent objects
        for event in common_sample:
            events.append(PlotEvent("common", event, False))
        
        for event in real_sample:
            events.append(PlotEvent("real", event, True))
        
        for event in hallucinatory_sample:
            events.append(PlotEvent("hallucinatory", event, True))
        
        # Shuffle and add goodbye at the end
        random.shuffle(events)
        events.append(PlotEvent("common", "Say goodbye", False))
        
        return events[:num_events]

# ================================
# ENHANCED TEMPORAL ENVIRONMENT
# ================================

class EnhancedTemporalEnvironment:
    """
    Enhanced temporal environment following paper specifications
    """
    
    def __init__(self, start_time: Optional[datetime.datetime] = None):
        # Following paper's example: "Monday, September 4, 2006, 21:42:56"
        self.current_time = start_time or datetime.datetime(2006, 9, 4, 21, 42, 56)
        # Realistic conversation intervals
        self.time_increment_range = (30, 900)  # 30 seconds to 15 minutes
    
    def get_current_timestamp(self) -> str:
        """Get formatted timestamp following paper format"""
        return self.current_time.strftime("%A, %B %d, %Y, %H:%M:%S")
    
    def advance_time(self) -> str:
        """Advance time and return new timestamp"""
        increment = random.randint(*self.time_increment_range)
        self.current_time += datetime.timedelta(seconds=increment)
        return self.get_current_timestamp()

# ================================
# ENHANCED MADGF WITH LLAMA
# ================================

class LlamaMADGF:
    """
    Enhanced MADGF with Llama 3.1 8B
    Following paper: "MADGF simulates and controls multi-turn scenario dialogues"
    """
    
    def __init__(self, model_name: str = "meta-llama/Meta-Llama-3.1-8B-Instruct"):
        self.model_name = model_name
        self.shared_agent = None
        self.character_generator = None
        self.plot_generator = EnhancedPlotGenerator()
        self._initialize_shared_model()
    
    def _initialize_shared_model(self):
        """Initialize shared Llama model for both agents"""
        print("Loading shared Llama 3.1 8B model...")
        self.shared_agent = LlamaLLMAgent(self.model_name)
        self.character_generator = LlamaCharacterGenerator(self.shared_agent)
        print(f"Shared model loaded. Memory: {torch.cuda.memory_allocated()/1024**3:.2f}GB")
    
    def create_human_prompt(self, character: CharacterCard, plot: List[PlotEvent]) -> str:
        """
        Create human role prompt following Figure 4 exactly
        """
        character_info = f"""Name: {character.name}
Occupation: {character.occupation}
Age: {character.age}
Gender: {character.gender}
Hobbies: {', '.join(character.hobbies)}
Personality: {', '.join(character.personality)}
Social Relationships: {character.social_relationships}"""
        
        plot_info = "\n".join([f"{i+1}. {event.description}" for i, event in enumerate(plot)])
        
        prompt = f"""You are playing the role of this character in a conversation with an AI assistant named Echo:

{character_info}

Follow these conversation topics in order:
{plot_info}

Important instructions:
- Respond as {character.name} would, based on their personality and background
- Keep responses natural and conversational (1-2 sentences)
- Stay in character throughout the conversation
- Follow the topics but make the conversation feel natural

Begin the conversation naturally."""
        
        return prompt
    
    def create_assistant_prompt(self, plot: List[PlotEvent]) -> str:
        """
        Create AI assistant prompt following Figure 4 exactly
        """
        hallucinatory_plots = [event.description for event in plot 
                             if event.event_type == "hallucinatory"]
        
        hallucination_info = "\n".join([f"- {plot}" for plot in hallucinatory_plots])
        
        prompt = f"""You are Echo, an AI assistant with advanced episodic memory capabilities.

Your characteristics:
- Name: Echo (海螺 in Chinese, symbolizing wisdom and memory)
- Function: AI with strong contextual memory, able to remember conversations
- Personality: Helpful, curious, good listener who enjoys making friends

IMPORTANT MEMORY RULES:
If asked about these topics, say you don't know because they were never discussed:
{hallucination_info}

Guidelines:
- Remember what the human tells you during conversation
- Ask follow-up questions naturally to learn about the human
- Use time information when provided to give accurate responses
- Keep responses helpful but concise (1-2 sentences)
- If you don't remember something that wasn't mentioned, say so honestly

Begin the conversation helpfully and naturally."""
        
        return prompt
    
    def generate_dialogue(self, max_rounds: int = 20) -> Dict[str, Any]:
        """
        Generate dialogue following Algorithm 1 exactly
        """
        print("Generating character and plot...")
        character = self.character_generator.generate_character()
        plot = self.plot_generator.generate_plot()
        temporal_env = EnhancedTemporalEnvironment()
        
        print(f"Starting dialogue with {character.name} ({character.occupation})")
        
        # Create prompts
        human_prompt = self.create_human_prompt(character, plot)
        assistant_prompt = self.create_assistant_prompt(plot)
        
        # Initialize histories following Algorithm 1
        human_history = []
        assistant_history = []
        
        dialogue_data = []
        round_count = 0
        
        plot_events = [event.description for event in plot if event.description != "Say goodbye"]
        current_plot_index = 0
        
        # Start with initial greeting
        print(f"Turn 1: Initial greeting")
        
        # Human starts the conversation
        initial_context = f"You are {character.name}. Start a friendly conversation with Echo the AI assistant."
        human_response = self.shared_agent.generate_response(initial_context, [])
        
        # Advance time and add timestamp
        timestamp = temporal_env.advance_time()
        
        # Update histories
        human_history.extend([human_response, timestamp])
        assistant_history.extend([human_response, timestamp])
        
        # Assistant responds
        assistant_context = assistant_prompt + f"\n\nHuman just said: {human_response}"
        assistant_response = self.shared_agent.generate_response(assistant_context, [])
        
        # Update histories
        human_history.append(assistant_response)
        assistant_history.append(assistant_response)
        
        # Store first dialogue turn
        dialogue_data.append({
            "round": 1,
            "timestamp": timestamp,
            "human": human_response,
            "assistant": assistant_response,
            "plot_event": "Initial greeting"
        })
        
        round_count = 1
        
        # Continue with plot events
        while round_count < max_rounds and current_plot_index < len(plot_events):
            try:
                current_plot = plot_events[current_plot_index]
                print(f"Turn {round_count + 1}: {current_plot[:50]}...")
                
                # Human turn following current plot
                human_context = f"""Continue as {character.name}. Previous conversation:
{assistant_history[-4:] if len(assistant_history) >= 4 else assistant_history}

Now follow this instruction naturally: {current_plot}
Respond as {character.name} would."""
                
                human_response = self.shared_agent.generate_response(human_context, [])
                
                # Check for farewell
                if any(farewell in human_response.lower() 
                      for farewell in ["goodbye", "bye", "see you later", "farewell"]):
                    break
                
                # Advance time and add timestamp
                timestamp = temporal_env.advance_time()
                
                # Update histories
                human_history.extend([human_response, timestamp])
                assistant_history.extend([human_response, timestamp])
                
                # Assistant turn
                assistant_context = f"""Continue as Echo. Recent conversation:
{assistant_history[-6:] if len(assistant_history) >= 6 else assistant_history}

The human just said: {human_response}
Respond as Echo would, remembering previous conversation."""
                
                assistant_response = self.shared_agent.generate_response(assistant_context, [])
                
                # Update histories
                human_history.append(assistant_response)
                assistant_history.append(assistant_response)
                
                # Store dialogue turn
                dialogue_data.append({
                    "round": round_count + 1,
                    "timestamp": timestamp,
                    "human": human_response,
                    "assistant": assistant_response,
                    "plot_event": current_plot
                })
                
                round_count += 1
                current_plot_index += 1
                
                # Brief pause
                time.sleep(0.5)
                
            except Exception as e:
                print(f"Error in dialogue generation at round {round_count + 1}: {e}")
                break
        
        # Create final data removing initial prompt
        final_data = assistant_history
        
        return {
            "character": asdict(character),
            "plot": [asdict(event) for event in plot],
            "dialogue": dialogue_data,
            "final_data": final_data,
            "rounds": round_count
        }
    
    def cleanup(self):
        """Clean up shared model"""
        if self.shared_agent:
            self.shared_agent.cleanup()

# ================================
# ENHANCED EM-TRAIN GENERATOR WITH LLAMA
# ================================

class LlamaEMTrainDataset:
    """
    EM-Train dataset generator with Llama 3.1 8B
    Target: Generate samples following paper methodology
    """
    
    def __init__(self, model_name: str = "meta-llama/Meta-Llama-3.1-8B-Instruct"):
        self.model_name = model_name
        self.madgf = None
        self.dataset = []
    
    def generate_dataset(self, num_samples: int = 3) -> List[Dict[str, Any]]:
        """
        Generate EM-Train dataset following paper methodology
        """
        print(f"Generating EM-Train dataset with {num_samples} samples using Llama 3.1 8B...")
        
        # Initialize MADGF
        self.madgf = LlamaMADGF(self.model_name)
        
        try:
            for i in range(num_samples):
                print(f"Generated {i}/{num_samples} samples")
                
                # Generate dialogue
                dialogue = self.madgf.generate_dialogue()
                
                # Convert to training format
                training_data = self._convert_to_training_format(dialogue)
                
                self.dataset.append({
                    "id": i,
                    "character": dialogue["character"],
                    "rounds": dialogue["rounds"],
                    "training_data": training_data,
                    "dialogue_raw": dialogue["dialogue"]
                })
                
                # Memory cleanup
                if i % 5 == 0:
                    torch.cuda.empty_cache()
                    gc.collect()
                
        finally:
            # Clean up model
            print("Cleaning up model...")
            self.madgf.cleanup()
        
        print(f"Dataset generation complete. Total samples: {len(self.dataset)}")
        return self.dataset
    
    def _convert_to_training_format(self, dialogue: Dict[str, Any]) -> List[Dict[str, str]]:
        """
        Convert to user-time-assistant format following Section 4.1
        """
        training_data = []
        
        for turn in dialogue["dialogue"]:
            # User-time-assistant format as per Figure 5(b)
            training_data.append({
                "role": "user",
                "content": turn["human"]
            })
            training_data.append({
                "role": "observation", 
                "content": turn["timestamp"]
            })
            training_data.append({
                "role": "assistant",
                "content": turn["assistant"]
            })
        
        return training_data
    
    def save_dataset(self, filename: str = "llama_em_train_dataset.json"):
        """Save dataset to file"""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.dataset, f, ensure_ascii=False, indent=2)
        print(f"Dataset saved to {filename}")

# ================================
# MAIN EXECUTION
# ================================

def run_quick_demo():
    """Quick demo with 3 samples using Llama"""
    print("=== Echo System with Llama 3.1 8B - Quick Demo ===")
    print("Generating demo dataset...")
    
    em_train_generator = LlamaEMTrainDataset()
    
    try:
        em_train_data = em_train_generator.generate_dataset(num_samples=3)
        
        print(f"\nDataset generated with {len(em_train_data)} samples")
        
        for i, sample in enumerate(em_train_data):
            print(f"\n--- Sample {i+1} ---")
            print(f"Character: {sample['character']['name']} ({sample['character']['occupation']})")
            print(f"Rounds: {sample['rounds']}")
            if sample['dialogue_raw']:
                first_exchange = sample['dialogue_raw'][0]
                print("First exchange:")
                print(f"  User: {first_exchange['human']}")
                print(f"  Echo: {first_exchange['assistant']}")
        
        print("\n✅ Demo completed successfully!")
        
    except Exception as e:
        print(f"Error in demo: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Main execution with larger dataset"""
    print("=== Echo System with Llama 3.1 8B ===")
    print("Generating larger dataset...")
    
    em_train_generator = LlamaEMTrainDataset()
    
    try:
        # Generate 10 samples for validation
        em_train_data = em_train_generator.generate_dataset(num_samples=25)
        em_train_generator.save_dataset()
        
        print(f"\nDataset Statistics:")
        print(f"  Total samples: {len(em_train_data)}")
        if em_train_data:
            avg_rounds = np.mean([sample["rounds"] for sample in em_train_data])
            print(f"  Average rounds: {avg_rounds:.2f}")
            
            # Character diversity
            occupations = set([sample['character']['occupation'] for sample in em_train_data])
            genders = set([sample['character']['gender'] for sample in em_train_data])
            print(f"  Unique occupations: {len(occupations)}")
            print(f"  Unique genders: {len(genders)}")
        
        # Show sample conversations
        print("\n=== Sample Generated Conversations ===")
        for i, sample in enumerate(em_train_data[:3]):
            print(f"\n--- Sample {i+1} ---")
            print(f"Character: {sample['character']['name']} ({sample['character']['occupation']})")
            print(f"Rounds: {sample['rounds']}")
            if sample['dialogue_raw']:
                first_exchange = sample['dialogue_raw'][0]
                print("First exchange:")
                print(f"  User: {first_exchange['human']}")
                print(f"  Echo: {first_exchange['assistant']}")
        
        print("\n✅ Implementation completed successfully!")
        print("Ready for scaling to full dataset and EM-Test generation")
        
    except Exception as e:
        print(f"Error in dataset generation: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    return em_train_data

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "demo":
        run_quick_demo()
    else:
        main()

# ================================
# ENHANCED EM-TEST BENCHMARK WITH LLAMA
# ================================

@dataclass
class LlamaEMTestInstance:
    """EM-Test instance for Llama implementation"""
    id: str
    dialogue_history: List[Dict[str, str]]
    test_question: str
    reference_answer: str
    time_span: str
    difficulty: str
    temporal_context: str
    character_info: Dict[str, Any]

class LlamaEMTestBenchmark:
    """
    EM-Test benchmark generator with Llama
    Following paper Table 1 distribution exactly
    """
    
    def __init__(self):
        self.test_instances = []
        # Following Table 1 time spans exactly
        self.time_spans = [
            "just now", "one day", "few days", "one month", 
            "few months", "one year", "few years", "several decades"
        ]
        self.sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')
    
    def generate_test_set(self, em_train_dataset: List[Dict[str, Any]], 
                         num_tests: int = 25) -> List[LlamaEMTestInstance]:
        """
        Generate EM-Test instances following paper distribution
        Scaled down for demo: 25 instances instead of 106
        """
        print(f"Generating EM-Test with {num_tests} instances...")
        
        # Scaled distribution for demo
        target_distribution = {
            "just now": {"easy": 3, "hard": 2},
            "one day": {"easy": 2, "hard": 2},
            "few days": {"easy": 3, "hard": 2},
            "one month": {"easy": 2, "hard": 1},
            "few months": {"easy": 2, "hard": 2},
            "one year": {"easy": 2, "hard": 1},
            "few years": {"easy": 2, "hard": 2},
            "several decades": {"easy": 1, "hard": 1}
        }
        
        for time_span, difficulties in target_distribution.items():
            for difficulty, count in difficulties.items():
                for i in range(count):
                    if not em_train_dataset:
                        break
                    
                    # Sample random dialogue from EM-Train
                    sample_dialogue = random.choice(em_train_dataset)
                    
                    # Create test instance
                    test_instance = self._create_test_instance(
                        sample_dialogue, time_span, difficulty
                    )
                    
                    if test_instance:
                        self.test_instances.append(test_instance)
        
        print(f"EM-Test generation complete. Total instances: {len(self.test_instances)}")
        return self.test_instances
    
    def _create_test_instance(self, sample_dialogue: Dict[str, Any], 
                            time_span: str, difficulty: str) -> Optional[LlamaEMTestInstance]:
        """Create individual test instance"""
        try:
            training_data = sample_dialogue["training_data"]
            dialogue_raw = sample_dialogue.get("dialogue_raw", [])
            character = sample_dialogue["character"]
            
            if not dialogue_raw:
                return None
            
            # Create episodic memory question
            test_question, reference_answer = self._generate_episodic_question(
                dialogue_raw, character, time_span, difficulty
            )
            
            # Extract temporal context
            temporal_context = ""
            for turn in reversed(training_data):
                if turn["role"] == "observation":
                    temporal_context = turn["content"]
                    break
            
            return LlamaEMTestInstance(
                id=f"test_{len(self.test_instances):04d}",
                dialogue_history=training_data,
                test_question=test_question,
                reference_answer=reference_answer,
                time_span=time_span,
                difficulty=difficulty,
                temporal_context=temporal_context,
                character_info=character
            )
            
        except Exception as e:
            print(f"Error creating test instance: {e}")
            return None
    
    def _generate_episodic_question(self, dialogue_raw: List[Dict[str, Any]], 
                                  character: Dict[str, Any], 
                                  time_span: str, difficulty: str) -> Tuple[str, str]:
        """
        Generate episodic memory questions following paper methodology
        """
        if difficulty == "easy":
            # Easy questions: simple recall
            questions = [
                ("What is my name?", f"Your name is {character['name']}."),
                ("What do I do for work?", f"You work as a {character['occupation']}."),
                ("What are some of my hobbies?", f"Your hobbies include {', '.join(character['hobbies'][:2])}."),
                ("How would you describe my personality?", f"You are {' and '.join(character['personality'])}."),
                ("Do you remember what we talked about?", "We had a conversation about various topics.")
            ]
        else:
            # Hard questions: complex episodic reasoning following Figure 6 example
            questions = [
                ("Did I take any days off from work this year?", 
                 "I would need to recall our previous conversations about your work schedule to answer that."),
                ("What specific promise did I make in our earlier conversation?", 
                 "I need to remember the specific commitments mentioned in our previous chats."),
                ("When did we first discuss my family background?", 
                 "I would need to recall when family topics first came up in our conversations."),
                ("What was the exact topic we discussed last month?", 
                 "I need to remember the specific subjects we covered in our previous conversations."),
                ("How long has it been since we first met?", 
                 "I need to recall the timeline of our conversations to answer accurately.")
            ]
        
        # Add time-span specific context
        if time_span in ["just now", "one day"]:
            question_prefix = "Just recently, "
        elif time_span in ["few days", "one month"]:
            question_prefix = "A while ago, "
        else:
            question_prefix = "Some time back, "
        
        question, answer = random.choice(questions)
        
        # For hard questions, make them more temporally specific
        if difficulty == "hard" and time_span not in ["just now"]:
            question = question_prefix + question.lower()
        
        return question, answer
    
    def evaluate_response(self, predicted_answer: str, reference_answer: str) -> float:
        """
        Evaluate response using cosine similarity following Equation 1
        S = cos_sim(E_LLM, E_Standard) × 100
        """
        try:
            # Encode responses using sentence transformer
            pred_embedding = self.sentence_transformer.encode([predicted_answer])
            ref_embedding = self.sentence_transformer.encode([reference_answer])
            
            # Calculate cosine similarity (Equation 1)
            similarity = cosine_similarity(pred_embedding, ref_embedding)[0][0]
            
            # Convert to percentage as per paper
            return similarity * 100
        except Exception as e:
            print(f"Error calculating similarity: {e}")
            return 0.0
    
    def save_benchmark(self, filename: str = "llama_em_test_benchmark.json"):
        """Save EM-Test benchmark"""
        serializable_instances = []
        for instance in self.test_instances:
            serializable_instances.append(asdict(instance))
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(serializable_instances, f, ensure_ascii=False, indent=2)
        print(f"EM-Test benchmark saved to {filename}")
    
    def analyze_benchmark(self) -> Dict[str, Any]:
        """Analyze benchmark statistics"""
        if not self.test_instances:
            return {}
        
        stats = {
            "total_instances": len(self.test_instances),
            "time_span_distribution": {},
            "difficulty_distribution": {"easy": 0, "hard": 0}
        }
        
        for instance in self.test_instances:
            # Time span distribution
            time_span = instance.time_span
            stats["time_span_distribution"][time_span] = \
                stats["time_span_distribution"].get(time_span, 0) + 1
            
            # Difficulty distribution
            stats["difficulty_distribution"][instance.difficulty] += 1
        
        return stats

# ================================
# LLAMA ECHO MODEL TRAINING
# ================================

class LlamaEchoTrainingDataset(Dataset):
    """PyTorch Dataset for Echo training with Llama"""
    
    def __init__(self, em_train_data: List[Dict[str, Any]], tokenizer, max_length: int = 1024):
        self.data = em_train_data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.processed_data = self._process_data()
    
    def _process_data(self) -> List[Dict[str, torch.Tensor]]:
        """Process EM-Train data for Llama training"""
        processed = []
        
        print("Processing training data for Llama...")
        for i, sample in enumerate(self.data):
            if i % 10 == 0:
                print(f"Processed {i}/{len(self.data)} samples")
            
            training_data = sample["training_data"]
            
            # Build conversation in Llama chat format
            conversation_text = "<|begin_of_text|>"
            
            current_conversation = []
            for turn in training_data:
                if turn["role"] == "user":
                    current_conversation.append(f"User: {turn['content']}")
                elif turn["role"] == "observation":
                    current_conversation.append(f"Time: {turn['content']}")
                elif turn["role"] == "assistant":
                    current_conversation.append(f"Assistant: {turn['content']}")
            
            # Format as Llama chat
            conversation_text += "<|start_header_id|>system<|end_header_id|>\n\n"
            conversation_text += "You are Echo, an AI assistant with episodic memory capabilities."
            conversation_text += "<|eot_id|>\n\n"
            
            conversation_text += "<|start_header_id|>user<|end_header_id|>\n\n"
            conversation_text += "\n".join(current_conversation)
            conversation_text += "<|eot_id|>\n\n"
            
            conversation_text += "<|start_header_id|>assistant<|end_header_id|>\n\n"
            
            # Tokenize
            try:
                tokens = self.tokenizer(
                    conversation_text,
                    max_length=self.max_length,
                    truncation=True,
                    padding="max_length",
                    return_tensors="pt"
                )
                
                processed.append({
                    "input_ids": tokens["input_ids"].squeeze(),
                    "attention_mask": tokens["attention_mask"].squeeze(),
                    "labels": tokens["input_ids"].squeeze()
                })
            except Exception as e:
                print(f"Error processing sample {i}: {e}")
                continue
        
        print(f"Processed {len(processed)} samples successfully")
        return processed
    
    def __len__(self):
        return len(self.processed_data)
    
    def __getitem__(self, idx):
        return self.processed_data[idx]

class LlamaEchoModel:
    """
    Echo model with Llama 3.1 8B base
    Following paper training methodology
    """
    
    def __init__(self, base_model_name: str = "meta-llama/Meta-Llama-3.1-8B-Instruct"):
        self.base_model_name = base_model_name
        self.tokenizer = None
        self.model = None
        self._load_base_model()
    
    def _load_base_model(self):
        """Load Llama base model following paper specifications"""
        print(f"Loading base model: {self.base_model_name}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.base_model_name,
            padding_side="left"
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )
        
        # Add special tokens for temporal information
        special_tokens = ["[TIME]", "[MEMORY]", "[EPISODE]"]
        num_added = self.tokenizer.add_special_tokens(
            {"additional_special_tokens": special_tokens}
        )
        
        if num_added > 0:
            self.model.resize_token_embeddings(len(self.tokenizer))
        
        print("Llama base model loaded successfully")
    
    def generate_response(self, prompt: str, max_length: int = 200) -> str:
        """Generate response using Llama model"""
        # Format prompt for Llama
        formatted_prompt = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|>\n\n<|start_header_id|>assistant<|end_header_id|>\n\n"
        
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=max_length,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode only new tokens
        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        response = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        
        # Clean response
        response = re.sub(r'<\|.*?\|>', '', response)
        return response.strip()

# ================================
# COMPREHENSIVE EVALUATION WITH LLAMA
# ================================

class LlamaEvaluator:
    """
    Comprehensive evaluator for Llama Echo model
    """
    
    def __init__(self, echo_model: LlamaEchoModel, benchmark: LlamaEMTestBenchmark):
        self.echo_model = echo_model
        self.benchmark = benchmark
    
    def evaluate_full_benchmark(self, test_instances: List[LlamaEMTestInstance]) -> Dict[str, Any]:
        """
        Evaluate Echo model on complete benchmark following Section 5.2
        """
        print("Evaluating Llama Echo model on EM-Test benchmark...")
        
        results = {
            "overall_scores": [],
            "time_span_scores": {},
            "difficulty_scores": {"easy": [], "hard": []},
            "detailed_results": []
        }
        
        for i, instance in enumerate(test_instances):
            if i % 5 == 0:
                print(f"Evaluated {i}/{len(test_instances)} instances")
            
            try:
                # Create test prompt
                prompt = self._create_test_prompt(instance)
                
                # Generate response
                predicted_answer = self.echo_model.generate_response(prompt)
                
                # Calculate similarity score (Equation 1)
                similarity_score = self.benchmark.evaluate_response(
                    predicted_answer, instance.reference_answer
                )
                
                # Store results
                results["overall_scores"].append(similarity_score)
                results["difficulty_scores"][instance.difficulty].append(similarity_score)
                
                # Time span results
                if instance.time_span not in results["time_span_scores"]:
                    results["time_span_scores"][instance.time_span] = []
                results["time_span_scores"][instance.time_span].append(similarity_score)
                
                # Detailed results
                results["detailed_results"].append({
                    "instance_id": instance.id,
                    "question": instance.test_question,
                    "predicted": predicted_answer,
                    "reference": instance.reference_answer,
                    "similarity_score": similarity_score,
                    "time_span": instance.time_span,
                    "difficulty": instance.difficulty
                })
                
            except Exception as e:
                print(f"Error evaluating instance {instance.id}: {e}")
                continue
        
        # Calculate averages
        if results["overall_scores"]:
            results["overall_average"] = np.mean(results["overall_scores"])
            results["easy_average"] = np.mean(results["difficulty_scores"]["easy"]) if results["difficulty_scores"]["easy"] else 0
            results["hard_average"] = np.mean(results["difficulty_scores"]["hard"]) if results["difficulty_scores"]["hard"] else 0
            
            for time_span, scores in results["time_span_scores"].items():
                results["time_span_scores"][time_span] = {
                    "scores": scores,
                    "average": np.mean(scores)
                }
        
        print(f"Evaluation complete.")
        return results
    
    def _create_test_prompt(self, instance: LlamaEMTestInstance) -> str:
        """Create test prompt from instance"""
        conversation_text = "Previous conversation:\n"
        
        # Add recent conversation history
        for turn in instance.dialogue_history[-10:]:  # Last 10 turns
            if turn["role"] == "user":
                conversation_text += f"User: {turn['content']}\n"
            elif turn["role"] == "observation":
                conversation_text += f"Time: {turn['content']}\n"
            elif turn["role"] == "assistant":
                conversation_text += f"Assistant: {turn['content']}\n"
        
        # Add test question with temporal context
        conversation_text += f"\nCurrent time: {instance.temporal_context}\n"
        conversation_text += f"User: {instance.test_question}\n"
        conversation_text += "Please respond as Echo, using your memory of our previous conversation:"
        
        return conversation_text
    
    def generate_performance_report(self, results: Dict[str, Any]) -> str:
        """Generate comprehensive performance report"""
        if not results.get("overall_scores"):
            return "No evaluation results available."
        
        report = "=== Llama Echo Model Performance Report ===\n\n"
        
        report += f"Overall Performance:\n"
        report += f"  Average Similarity Score: {results['overall_average']:.2f}\n"
        report += f"  Total Test Instances: {len(results['overall_scores'])}\n\n"
        
        report += f"Difficulty Level Performance:\n"
        report += f"  Easy Level: {results['easy_average']:.2f}\n"
        report += f"  Hard Level: {results['hard_average']:.2f}\n\n"
        
        if results["time_span_scores"]:
            report += f"Time Span Performance:\n"
            for time_span, data in results["time_span_scores"].items():
                report += f"  {time_span}: {data['average']:.2f} ({len(data['scores'])} instances)\n"
        
        # Comparison with paper results
        report += f"\nComparison with Paper Results:\n"
        report += f"  Paper Echo Easy: 84.0, Our Llama Echo Easy: {results['easy_average']:.2f}\n"
        report += f"  Paper Echo Hard: 74.5, Our Llama Echo Hard: {results['hard_average']:.2f}\n"
        
        # Performance analysis
        if results['easy_average'] > results['hard_average']:
            report += f"\n✅ Model shows expected difficulty scaling (Easy > Hard)\n"
        else:
            report += f"\n⚠️  Unexpected difficulty pattern (Hard >= Easy)\n"
        
        return report

# ================================
# COMPLETE PIPELINE WITH LLAMA
# ================================

def run_complete_llama_pipeline():
    """Run complete Echo system pipeline with Llama"""
    print("=== Complete Llama Echo Pipeline ===")
    print("Following exact paper methodology with Llama 3.1 8B")
    print()
    
    try:
        # Step 1: Generate EM-Train dataset
        print("Step 1: Generating EM-Train dataset...")
        em_train_generator = LlamaEMTrainDataset()
        em_train_data = em_train_generator.generate_dataset(num_samples=5)  # Small for demo
        em_train_generator.save_dataset()
        
        if not em_train_data:
            print("No training data generated!")
            return
        
        # Step 2: Generate EM-Test benchmark
        print("\nStep 2: Generating EM-Test benchmark...")
        em_test_generator = LlamaEMTestBenchmark()
        em_test_instances = em_test_generator.generate_test_set(em_train_data, num_tests=10)
        em_test_generator.save_benchmark()
        
        # Analyze benchmark
        benchmark_stats = em_test_generator.analyze_benchmark()
        print(f"EM-Test generated: {benchmark_stats.get('total_instances', 0)} instances")
        
        # Step 3: Load Echo model (using base model for demo)
        print("\nStep 3: Loading Echo model...")
        echo_model = LlamaEchoModel()
        
        # Step 4: Evaluation
        if em_test_instances:
            print("\nStep 4: Evaluating model...")
            evaluator = LlamaEvaluator(echo_model, em_test_generator)
            results = evaluator.evaluate_full_benchmark(em_test_instances)
            
            # Generate report
            report = evaluator.generate_performance_report(results)
            print(report)
            
            # Save results
            with open("llama_evaluation_results.json", 'w') as f:
                json.dump(results, f, indent=2, default=str)
        
        print("\n✅ Complete Llama pipeline executed successfully!")
        print("All results saved to files.")
        
    except Exception as e:
        print(f"Pipeline error: {e}")
        import traceback
        traceback.print_exc()