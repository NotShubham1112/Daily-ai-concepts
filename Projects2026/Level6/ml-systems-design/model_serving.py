import json
import time

class ModelServer:
    def __init__(self, model):
        self.model = model
    
    def predict_endpoint(self, request_json):
        """
        Mock REST API endpoint
        """
        try:
            data = json.loads(request_json)
            features = data['features']
            
            start_time = time.time()
            prediction = self.model(features)
            latency = time.time() - start_time
            
            return json.dumps({
                "status": "success",
                "prediction": prediction.tolist() if hasattr(prediction, 'tolist') else prediction,
                "latency_ms": latency * 1000
            })
        except Exception as e:
            return json.dumps({"status": "error", "message": str(e)})

if __name__ == "__main__":
    def dummy_model(x): return sum(x)
    server = ModelServer(dummy_model)
    resp = server.predict_endpoint('{"features": [1, 2, 3]}')
    print(resp)
