# src/benchmarks/system_adapter.py
class EpisodicMemoryAdapter:
    def __init__(self, system):
        self.system = system
        self.conversation_history = []
    
    def add_memory(self, text: str, metadata: Dict = None) -> str:
        result = self.system.chat_breakthrough(text)
        self.conversation_history.append(text)
        return f"mem_{len(self.conversation_history)}"
    
    def query_memory(self, query: str) -> str:
        result = self.system.chat_breakthrough(query)
        return result["response"]
    
    def reset(self):
        # Reinicializar tu sistema
        self.system = EpisodicMemoryLLM_FINAL()
        self.conversation_history = []
