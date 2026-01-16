class AgentMemory:
    def __init__(self, capacity=5):
        self.capacity = capacity
        self.short_term = []
        self.long_term = {} # Structured key-value storage
        
    def add_event(self, event):
        self.short_term.append(event)
        if len(self.short_term) > self.capacity:
            # "Archive" to long term or just purge
            self.short_term.pop(0)
            
    def get_context(self):
        return " | ".join(self.short_term)

if __name__ == "__main__":
    mem = AgentMemory(3)
    mem.add_event("Observed user input")
    mem.add_event("Generated plan A")
    mem.add_event("Executed step 1")
    mem.add_event("Observed error")
    
    print(f"Current Context: {mem.get_context()}")
