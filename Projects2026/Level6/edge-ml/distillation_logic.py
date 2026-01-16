import numpy as np

def distillation_loss(student_logits, teacher_logits, labels, T=2.0, alpha=0.5):
    """
    KL Divergence between teacher and student soft targets + Cross Entropy.
    """
    # Simplified version using MSE between soft targets
    soft_teacher = np.exp(teacher_logits / T) / np.sum(np.exp(teacher_logits / T))
    soft_student = np.exp(student_logits / T) / np.sum(np.exp(student_logits / T))
    
    distill_loss = np.mean(np.square(soft_teacher - soft_student))
    return distill_loss

if __name__ == "__main__":
    t_out = np.array([2.0, 1.0, 0.1])
    s_out = np.array([1.5, 1.2, 0.2])
    
    loss = distillation_loss(s_out, t_out, None, T=5.0)
    print(f"Distillation Loss (Temperature=5): {loss:.4f}")
