import os
import json
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm

# Configuration
OUTPUT_DIR = "diagnostic_dataset"
IMG_DIR = os.path.join(OUTPUT_DIR, "images")
JSON_PATH = os.path.join(OUTPUT_DIR, "diagnostic_data.json")
SAMPLES_PER_CLASS = 50

os.makedirs(IMG_DIR, exist_ok=True)

# Define the sources (Dataset Name, Split, Class Label)
SOURCES = {
    "animal": {"path": "timm/oxford_iiit_pet", "split": "train", "img_key": "image"},
    "food":   {"path": "food101", "split": "train", "img_key": "image"},
    "code":   {"path": "HuggingFaceM4/WebSight", "split": "train", "img_key": "image"},
    "chart":  {"path": "HuggingFaceM4/ChartQA", "name": "chartqa", "split": "val", "img_key": "image"}
}

final_data = []

print(f"🚀 Building Diagnostic Dataset ({SAMPLES_PER_CLASS} samples per category)...")
print(f"📁 Output directory: {OUTPUT_DIR}")
print(f"💾 This will download ~200 images (approx. 50MB total)\n")

for category, config in SOURCES.items():
    print(f"📥 Downloading {category} from {config['path']}...")
    
    try:
        # Load dataset with proper configuration
        name_arg = config.get("name", None)
        
        # Load in streaming mode to avoid downloading entire dataset
        ds = load_dataset(
            config['path'], 
            name_arg, 
            split=config['split'], 
            streaming=True,
            trust_remote_code=True  # Some datasets require this
        )
        ds_iter = iter(ds)
        
        count = 0
        skipped = 0
        pbar = tqdm(total=SAMPLES_PER_CLASS, desc=f"  {category.capitalize()}", leave=True)
        
        # Keep trying until we get enough valid samples
        max_attempts = SAMPLES_PER_CLASS * 3  # Allow some failures
        attempts = 0
        
        while count < SAMPLES_PER_CLASS and attempts < max_attempts:
            attempts += 1
            try:
                sample = next(ds_iter)
                image = sample[config['img_key']]
                
                # Handle None or invalid images
                if image is None:
                    skipped += 1
                    continue
                
                # Convert to RGB to avoid mode errors
                if image.mode != "RGB":
                    image = image.convert("RGB")
                
                # Validate image size (skip very small images)
                if image.width < 50 or image.height < 50:
                    skipped += 1
                    continue
                
                # Save image with proper naming
                img_filename = f"{category}_{count:03d}.jpg"
                img_path = os.path.join(IMG_DIR, img_filename)
                image.save(img_path, quality=95)
                
                # Create metadata entry
                entry = {
                    "id": f"{category}_{count}",
                    "image": img_path,
                    "category": category,
                    "conversations": [
                        {"from": "human", "value": "<image>\nDescribe this image."},
                        {"from": "gpt", "value": f"This is an image representing {category}."} 
                    ]
                }
                final_data.append(entry)
                
                count += 1
                pbar.update(1)
                
            except StopIteration:
                print(f"  ⚠️  Reached end of {category} dataset at {count} samples")
                break
            except Exception as e:
                skipped += 1
                continue
        
        pbar.close()
        
        if count < SAMPLES_PER_CLASS:
            print(f"  ⚠️  Only got {count}/{SAMPLES_PER_CLASS} samples for {category}")
        if skipped > 0:
            print(f"  ℹ️  Skipped {skipped} invalid images for {category}")
        
    except Exception as e:
        print(f"  ❌ Failed to load {category}: {str(e)}")
        print(f"     Try checking if the dataset exists or requires authentication")

print(f"\n{'='*60}")
print(f"✅ Done! Saved {len(final_data)} samples to:")
print(f"   {JSON_PATH}")
print(f"\n📊 Breakdown by category:")

# Show statistics
category_counts = {}
for entry in final_data:
    cat = entry['category']
    category_counts[cat] = category_counts.get(cat, 0) + 1

for cat, count in sorted(category_counts.items()):
    print(f"   {cat.capitalize()}: {count} images")

# Save JSON with nice formatting
with open(JSON_PATH, "w") as f:
    json.dump(final_data, f, indent=2)

print(f"\n💡 Next steps:")
print(f"   1. Point your visualization script to: {JSON_PATH}")
print(f"   2. Run your t-SNE analysis to see if categories cluster")
print(f"   3. If categories mix, your router needs improvement")
print(f"\n{'='*60}")