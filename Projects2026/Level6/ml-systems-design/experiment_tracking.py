import json
import os

class ExperimentTracker:
    def __init__(self, log_dir="logs"):
        self.log_dir = log_dir
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
            
    def log_run(self, run_name, params, metrics):
        run_data = {
            "name": run_name,
            "params": params,
            "metrics": metrics,
            "timestamp": os.path.getmtime(self.log_dir) # Sample time
        }
        filepath = os.path.join(self.log_dir, f"{run_name}.json")
        with open(filepath, 'w') as f:
            json.dump(run_data, f, indent=4)
        print(f"Logged run to {filepath}")

if __name__ == "__main__":
    tracker = ExperimentTracker()
    tracker.log_run("resnet_v1", {"lr": 0.01, "batch": 32}, {"accuracy": 0.94})
