import numpy as np

class SimpleAgent:
    def __init__(self, id):
        self.id = id
        self.position = np.random.randint(0, 10)
        
    def move(self, other_pos):
        # Cooperative logic: try to move closer to the other agent
        if self.position < other_pos:
            self.position += 1
        elif self.position > other_pos:
            self.position -= 1
            
if __name__ == "__main__":
    a1 = SimpleAgent(1)
    a2 = SimpleAgent(2)
    
    print(f"Initial positions: Agent 1 @ {a1.position}, Agent 2 @ {a2.position}")
    for _ in range(5):
        pos1 = a1.position
        pos2 = a2.position
        a1.move(pos2)
        a2.move(pos1)
        
    print(f"Final positions: Agent 1 @ {a1.position}, Agent 2 @ {a2.position}")
