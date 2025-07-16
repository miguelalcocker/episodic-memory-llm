"""
🤖 Echo Architecture - Nuestra implementación basada en el paper
Memoria episódica temporal para LLMs

Basado en: "Echo: A Large Language Model with Temporal Episodic Memory"
Implementación propia mejorada
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Model, GPT2Config
from datetime import datetime, timedelta
import numpy as np
from typing import List, Dict, Tuple, Optional

class TemporalEmbedding(nn.Module):
    """Embeddings temporales para capturar información de tiempo"""
    
    def __init__(self, hidden_size: int = 768, max_time_delta: int = 365):
        super().__init__()
        self.hidden_size = hidden_size
        self.max_time_delta = max_time_delta
        
        # Embeddings para diferentes escalas temporales
        self.hour_embedding = nn.Embedding(24, hidden_size // 4)
        self.day_embedding = nn.Embedding(7, hidden_size // 4)
        self.week_embedding = nn.Embedding(52, hidden_size // 4)
        self.month_embedding = nn.Embedding(12, hidden_size // 4)
        
        # Projection layer
        self.temporal_projection = nn.Linear(hidden_size, hidden_size)
        
    def forward(self, timestamps: List[datetime]) -> torch.Tensor:
        """Convertir timestamps a embeddings temporales"""
        batch_size = len(timestamps)
        
        # Extraer componentes temporales
        hours = torch.tensor([ts.hour for ts in timestamps], dtype=torch.long)
        days = torch.tensor([ts.weekday() for ts in timestamps], dtype=torch.long)
        weeks = torch.tensor([ts.isocalendar()[1] - 1 for ts in timestamps], dtype=torch.long)
        months = torch.tensor([ts.month - 1 for ts in timestamps], dtype=torch.long)
        
        if torch.cuda.is_available():
            hours = hours.cuda()
            days = days.cuda()
            weeks = weeks.cuda()
            months = months.cuda()
        
        # Generar embeddings
        hour_emb = self.hour_embedding(hours)
        day_emb = self.day_embedding(days)
        week_emb = self.week_embedding(weeks)
        month_emb = self.month_embedding(months)
        
        # Concatenar
        temporal_emb = torch.cat([hour_emb, day_emb, week_emb, month_emb], dim=-1)
        
        # Proyectar
        temporal_emb = self.temporal_projection(temporal_emb)
        
        return temporal_emb

class EpisodicMemoryBank(nn.Module):
    """Banco de memoria episódica con atención temporal"""
    
    def __init__(self, hidden_size: int = 768, max_episodes: int = 1000):
        super().__init__()
        self.hidden_size = hidden_size
        self.max_episodes = max_episodes
        
        # Almacenamiento de episodios
        self.episode_embeddings = []
        self.episode_timestamps = []
        self.episode_texts = []
        
        # Atención temporal
        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=8,
            dropout=0.1
        )
        
        # Mecanismo de consolidación (simula sueño REM)
        self.consolidation_network = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size * 2, hidden_size)
        )
        
    def add_episode(self, embedding: torch.Tensor, timestamp: datetime, text: str):
        """Añadir nuevo episodio a la memoria"""
        self.episode_embeddings.append(embedding.detach().cpu())
        self.episode_timestamps.append(timestamp)
        self.episode_texts.append(text)
        
        # Limitar tamaño de memoria
        if len(self.episode_embeddings) > self.max_episodes:
            self.episode_embeddings.pop(0)
            self.episode_timestamps.pop(0)
            self.episode_texts.pop(0)
    
    def retrieve_relevant_episodes(self, 
                                 query_embedding: torch.Tensor, 
                                 current_time: datetime,
                                 top_k: int = 5) -> List[Dict]:
        """Recuperar episodios relevantes usando atención temporal"""
        if len(self.episode_embeddings) == 0:
            return []
        
        # Convertir episodios a tensor
        episode_tensor = torch.stack(self.episode_embeddings)
        if torch.cuda.is_available():
            episode_tensor = episode_tensor.cuda()
        
        # Calcular pesos temporales (más reciente = más peso)
        temporal_weights = []
        for ts in self.episode_timestamps:
            time_diff = (current_time - ts).total_seconds() / 3600  # horas
            # Decay exponencial
            weight = np.exp(-time_diff / 24)  # decay en 24 horas
            temporal_weights.append(weight)
        
        temporal_weights = torch.tensor(temporal_weights, dtype=torch.float32)
        if torch.cuda.is_available():
            temporal_weights = temporal_weights.cuda()
        
        # Atención entre query y episodios
        query_expanded = query_embedding.unsqueeze(0).unsqueeze(0)  # [1, 1, hidden]
        episode_keys = episode_tensor.unsqueeze(1)  # [num_episodes, 1, hidden]
        
        # Calcular similaridades
        similarities = F.cosine_similarity(
            query_expanded, 
            episode_keys, 
            dim=-1
        ).squeeze()
        
        # Combinar con pesos temporales
        combined_scores = similarities * temporal_weights
        
        # Top-k episodios
        top_indices = torch.topk(combined_scores, min(top_k, len(self.episode_embeddings))).indices
        
        relevant_episodes = []
        for idx in top_indices:
            relevant_episodes.append({
                'text': self.episode_texts[idx],
                'timestamp': self.episode_timestamps[idx],
                'embedding': self.episode_embeddings[idx],
                'score': combined_scores[idx].item()
            })
        
        return relevant_episodes
    
    def consolidate_memory(self):
        """Consolidación de memoria (simula sueño REM)"""
        if len(self.episode_embeddings) < 2:
            return
        
        # Aplicar red de consolidación a episodios recientes
        recent_episodes = self.episode_embeddings[-10:]  # últimos 10
        consolidated = []
        
        for episode_emb in recent_episodes:
            if torch.cuda.is_available():
                episode_emb = episode_emb.cuda()
            
            # Aplicar consolidación
            consolidated_emb = self.consolidation_network(episode_emb)
            consolidated.append(consolidated_emb.cpu())
        
        # Reemplazar episodios recientes con versiones consolidadas
        self.episode_embeddings[-len(consolidated):] = consolidated

class EchoModel(nn.Module):
    """Modelo Echo completo con memoria episódica temporal"""
    
    def __init__(self, 
                 base_model_name: str = "gpt2",
                 hidden_size: int = 768,
                 max_episodes: int = 1000):
        super().__init__()
        
        # Modelo base
        self.base_model = GPT2Model.from_pretrained(base_model_name)
        self.hidden_size = hidden_size
        
        # Componentes de memoria episódica
        self.temporal_embedding = TemporalEmbedding(hidden_size)
        self.episodic_memory = EpisodicMemoryBank(hidden_size, max_episodes)
        
        # Capa de fusión para combinar memoria con input actual
        self.memory_fusion = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size)
        )
        
        # Output layer
        self.output_projection = nn.Linear(hidden_size, self.base_model.config.vocab_size)
        
    def forward(self, 
                input_ids: torch.Tensor,
                current_time: datetime,
                retrieve_memory: bool = True) -> Dict:
        """Forward pass con memoria episódica"""
        
        # Encoding del input actual
        base_outputs = self.base_model(input_ids)
        current_embedding = base_outputs.last_hidden_state.mean(dim=1)  # [batch, hidden]
        
        if retrieve_memory and len(self.episodic_memory.episode_embeddings) > 0:
            # Recuperar episodios relevantes
            relevant_episodes = self.episodic_memory.retrieve_relevant_episodes(
                current_embedding[0], current_time, top_k=3
            )
            
            if relevant_episodes:
                # Combinar embeddings de memoria
                memory_embeddings = torch.stack([
                    ep['embedding'] for ep in relevant_episodes
                ]).mean(dim=0)  # Promedio de episodios relevantes
                
                if torch.cuda.is_available():
                    memory_embeddings = memory_embeddings.cuda()
                
                # Fusionar memoria con input actual
                fused_embedding = torch.cat([current_embedding[0], memory_embeddings], dim=-1)
                enhanced_embedding = self.memory_fusion(fused_embedding).unsqueeze(0)
            else:
                enhanced_embedding = current_embedding
        else:
            enhanced_embedding = current_embedding
        
        # Generar output
        logits = self.output_projection(enhanced_embedding)
        
        return {
            'logits': logits,
            'current_embedding': current_embedding,
            'enhanced_embedding': enhanced_embedding
        }
    
    def add_to_memory(self, input_text: str, embedding: torch.Tensor, timestamp: datetime):
        """Añadir interacción a la memoria episódica"""
        self.episodic_memory.add_episode(embedding, timestamp, input_text)

# Test básico
if __name__ == "__main__":
    print("🧠 Testing Echo Architecture...")
    
    model = EchoModel()
    if torch.cuda.is_available():
        model = model.cuda()
    
    # Test input
    from transformers import GPT2Tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    test_text = "Hello, my name is Alex"
    inputs = tokenizer(test_text, return_tensors="pt", padding=True)
    
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}
    
    # Forward pass
    with torch.no_grad():
        outputs = model(inputs['input_ids'], datetime.now())
    
    print(f"✅ Output shape: {outputs['logits'].shape}")
    print(f"✅ Current embedding shape: {outputs['current_embedding'].shape}")
    print("🎉 Echo architecture working!")
