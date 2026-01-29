import joblib
import numpy as np

def verify(path):
    data = joblib.load(path)
    print(f"--- Verification for {path} ---")
    
    for layer_idx, centroids in data.items():
        # centroids shape should be [num_experts, hidden_dim] (e.g., [4, 2048])
        print(f"\nLayer {layer_idx}:")
        print(f"  Shape: {centroids.shape}")
        
        # 1. Check for NaNs or Infs
        if np.isnan(centroids).any():
            print("  ALERT: NaNs detected in centroids!")
        
        # 2. Check for "Dead" Experts (all zeros)
        norms = np.linalg.norm(centroids, axis=1)
        print(f"  Expert Norms: {norms}")
        if (norms < 1e-6).any():
            print("  ALERT: One or more experts have near-zero weights!")

        # 3. Check for Diversity (are experts different from each other?)
        # Calculate cosine similarity between expert 0 and expert 1
        c1, c2 = centroids[0], centroids[1]
        sim = np.dot(c1, c2) / (np.linalg.norm(c1) * np.linalg.norm(c2))
        print(f"  Cosine Sim (Exp 0 vs 1): {sim:.4f}")
        
        if sim > 0.99:
            print("  WARNING: Experts are nearly identical. Clusters might not have converged.")

if __name__ == "__main__":
    verify("kmeans_trial/kmeans_trial/teacher_centroids_5000_large_BS.pkl")