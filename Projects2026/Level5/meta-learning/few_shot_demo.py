import numpy as np

def prototypical_inference(query, support_set, support_labels):
    """
    Prototypical Networks (Few-shot) logic:
    1. Compute mean embedding (prototype) per class in support set.
    2. Classify query based on distance to prototypes.
    """
    unique_labels = np.unique(support_labels)
    prototypes = {}
    
    for label in unique_labels:
        mask = (support_labels == label)
        prototypes[label] = np.mean(support_set[mask], axis=0)
        
    # Query classification (Euclidean distance)
    best_dist = float('inf')
    pred_label = None
    
    for label, proto in prototypes.items():
        dist = np.linalg.norm(query - proto)
        if dist < best_dist:
            best_dist = dist
            pred_label = label
            
    return pred_label

if __name__ == "__main__":
    # Support set: 2 classes (A, B), 1-shot (1 example each)
    support = np.array([[1, 2], [10, 20]])
    labels = np.array(['A', 'B'])
    
    # Query point
    q = np.array([1.5, 2.1])
    result = prototypical_inference(q, support, labels)
    print(f"Query {q} categorized as: {result}")
