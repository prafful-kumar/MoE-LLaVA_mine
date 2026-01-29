import torch
import sys

def inspect(path):
    print(f"📂 Loading {path}...")
    data = torch.load(path)
    
    # Get the first sample key
    first_key = list(data.keys())[0]
    sample = data[first_key]
    
    print(f"\n✅ Found {len(data)} samples.")
    print(f"🔑 Sample Key: {first_key}")
    print("-" * 30)
    
    # Check Output IDs (This is likely where the bug is)
    out_ids = sample['output_ids']
    print(f"📝 output_ids type: {type(out_ids)}")
    if isinstance(out_ids, torch.Tensor):
        print(f"📝 output_ids shape: {out_ids.shape}")
        # Print first 20 tokens to see if we see -200 or image tokens
        flat_ids = out_ids.flatten().tolist()
        print(f"📝 First 20 tokens: {flat_ids[:20]}")
        
        if -200 in flat_ids:
            print("✅ FOUND -200 token in output_ids!")
            print(f"   Index of -200: {flat_ids.index(-200)}")
        else:
            print("❌ NO -200 token found. This is the problem.")
            # Check for other common image tokens like 32000 or similar if needed

    # Check Logits
    logits = sample['gating_logit']
    print(f"\n🧠 Gating Logits type: {type(logits)}")
    if isinstance(logits, list):
        print(f"🧠 Number of layers captured: {len(logits)}")
        if len(logits) > 0:
            print(f"🧠 Layer 0 shape: {logits[0].shape}")
            
if __name__ == "__main__":
    # Use the path from your error log
    path = "diagnostic_dataset_claude/diagnostic_routing.pt"
    if len(sys.argv) > 1:
        path = sys.argv[1]
    inspect(path)