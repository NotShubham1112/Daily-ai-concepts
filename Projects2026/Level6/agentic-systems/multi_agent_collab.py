class SpecializedAgent:
    def __init__(self, role):
        self.role = role
        
    def receive_message(self, msg, sender_role):
        print(f"[{self.role}] Received from {sender_role}: '{msg}'")
        return f"ACK from {self.role}"

if __name__ == "__main__":
    manager = SpecializedAgent("Manager")
    worker = SpecializedAgent("Worker")
    
    reply = worker.receive_message("Finish the report by EOD", manager.role)
    print(reply)
