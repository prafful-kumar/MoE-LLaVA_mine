import os
import json
import random
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Fix PyArrow core dump issue
os.environ['PYARROW_IGNORE_TIMEZONE'] = '1'

# Configuration
OUTPUT_DIR = "diagnostic_dataset_vtab"
IMG_DIR = os.path.join(OUTPUT_DIR, "images")
JSON_PATH = os.path.join(OUTPUT_DIR, "diagnostic_data.json")
SAMPLES_PER_CLASS = 50

os.makedirs(IMG_DIR, exist_ok=True)

# Diverse conversation templates
CONVERSATION_TEMPLATES = {
    "natural": {
        "questions": [
            "<image>\nWhat is the main subject in this image?",
            "<image>\nDescribe what you see in this natural scene.",
            "<image>\nIdentify the objects in this photograph.",
            "<image>\nWhat is depicted in this image?"
        ],
        "answers": [
            "This image shows a natural scene with real-world objects and environments.",
            "The photograph depicts natural subjects captured in realistic settings.",
            "This is a natural image containing recognizable real-world elements.",
            "The scene presents natural objects in their typical contexts."
        ]
    },
    "specialized": {
        "questions": [
            "<image>\nWhat type of specialized image is this?",
            "<image>\nDescribe the technical or medical content.",
            "<image>\nWhat domain does this image belong to?",
            "<image>\nIdentify the specialized subject matter."
        ],
        "answers": [
            "This is a specialized image from a technical or scientific domain.",
            "The image contains domain-specific content requiring expert knowledge.",
            "This depicts specialized subject matter from a particular field.",
            "The content shows technical or scientific imagery."
        ]
    },
    "structured": {
        "questions": [
            "<image>\nWhat structured visual pattern is shown?",
            "<image>\nDescribe the geometric or abstract elements.",
            "<image>\nWhat type of structured image is this?",
            "<image>\nIdentify the visual structure."
        ],
        "answers": [
            "This image contains structured visual patterns and geometric elements.",
            "The visual content shows organized patterns and abstract structures.",
            "This depicts structured visual information with clear patterns.",
            "The image presents geometric or abstract structured content."
        ]
    }
}

# VTAB Dataset Configuration
# VTAB has 3 groups: Natural, Specialized, Structured
# We'll select representative tasks from each group
VTAB_SOURCES = {
    # NATURAL IMAGES (Real-world photographs)
    "natural_cifar": {
        "path": "cifar100",
        "split": "test",
        "img_key": "img",
        "category": "natural",
        "description": "Natural objects and scenes"
    },
    "natural_caltech": {
        "path": "tanganke/caltech101",
        "split": "test", 
        "img_key": "image",
        "category": "natural",
        "description": "Natural objects (Caltech101)"
    },
    "natural_dtd": {
        "path": "tanganke/dtd",
        "split": "test",
        "img_key": "image", 
        "category": "natural",
        "description": "Texture patterns (DTD)"
    },
    "natural_pets": {
        "path": "timm/oxford-iiit-pet",
        "split": "test",
        "img_key": "image",
        "category": "natural", 
        "description": "Pet animals (Oxford-IIIT)"
    },
    
    # SPECIALIZED IMAGES (Medical, satellite, etc.)
    "specialized_patch": {
        "path": "tanganke/patch_camelyon",
        "split": "test",
        "img_key": "image",
        "category": "specialized",
        "description": "Medical pathology images"
    },
    "specialized_eurosat": {
        "path": "tanganke/eurosat",
        "split": "test",
        "img_key": "image",
        "category": "specialized",
        "description": "Satellite imagery (EuroSAT)"
    },
    "specialized_resisc": {
        "path": "timm/resisc45",
        "split": "test",
        "img_key": "image",
        "category": "specialized",
        "description": "Remote sensing images"
    },
    
    # STRUCTURED IMAGES (Abstract patterns, synthetic)
    "structured_clevr": {
        "path": "jxie/clevr",
        "split": "test",
        "img_key": "image",
        "category": "structured",
        "description": "3D rendered objects (CLEVR)"
    },
    "structured_dsprites": {
        "path": "lhao499/dsprites",
        "split": "test",
        "img_key": "image",
        "category": "structured",
        "description": "2D shape sprites"
    },
    "structured_dmlab": {
        "path": "cloneofsimo/dmlab",
        "split": "test",
        "img_key": "image",
        "category": "structured",
        "description": "3D game environments"
    }
}

def get_random_conversation(category):
    """Generate a diverse conversation based on category."""
    templates = CONVERSATION_TEMPLATES[category]
    question = random.choice(templates["questions"])
    answer = random.choice(templates["answers"])
    
    return [
        {"from": "human", "value": question},
        {"from": "gpt", "value": answer}
    ]

final_data = []
failed_datasets = []

print(f"🚀 Building VTAB Diagnostic Dataset ({SAMPLES_PER_CLASS} samples per task)...")
print(f"📁 Output directory: {OUTPUT_DIR}")
print(f"💾 Using VTAB (Visual Task Adaptation Benchmark)")
print(f"\n📊 VTAB Groups:")
print(f"   • Natural: Real-world photographs (CIFAR, Caltech, DTD, Pets)")
print(f"   • Specialized: Domain-specific (Medical, Satellite, Remote sensing)")
print(f"   • Structured: Synthetic/Abstract (CLEVR, dSprites, DMLab)")
print()

for task_name, config in VTAB_SOURCES.items():
    print(f"📥 Downloading {task_name} ({config['description']})...")
    
    try:
        # Load dataset
        ds = load_dataset(
            config['path'],
            split=config['split'],
            streaming=True,
            trust_remote_code=True
        )
        ds_iter = iter(ds)
        
        count = 0
        skipped = 0
        pbar = tqdm(total=SAMPLES_PER_CLASS, desc=f"  {task_name}", leave=True)
        
        max_attempts = SAMPLES_PER_CLASS * 10
        attempts = 0
        
        while count < SAMPLES_PER_CLASS and attempts < max_attempts:
            attempts += 1
            try:
                sample = next(ds_iter)
                
                # Get image
                if config['img_key'] not in sample:
                    skipped += 1
                    continue
                
                image = sample[config['img_key']]
                
                if image is None:
                    skipped += 1
                    continue
                
                # Convert to PIL if needed
                if not isinstance(image, Image.Image):
                    skipped += 1
                    continue
                
                # Convert to RGB
                if image.mode != "RGB":
                    image = image.convert("RGB")
                
                # Validate size
                if image.width < 32 or image.height < 32:
                    skipped += 1
                    continue
                
                # Resize if too large
                max_size = 1024
                if image.width > max_size or image.height > max_size:
                    ratio = min(max_size / image.width, max_size / image.height)
                    new_size = (int(image.width * ratio), int(image.height * ratio))
                    image = image.resize(new_size, Image.LANCZOS)
                
                # Save image
                img_filename = f"{task_name}_{count:03d}.jpg"
                img_path = os.path.join(IMG_DIR, img_filename)
                image.save(img_path, quality=90)
                
                # Generate conversation
                conversation = get_random_conversation(config['category'])
                
                # Create entry
                entry = {
                    "id": f"{task_name}_{count}",
                    "image": img_path,
                    "category": config['category'],
                    "task": task_name,
                    "task_description": config['description'],
                    "conversations": conversation
                }
                final_data.append(entry)
                
                count += 1
                pbar.update(1)
                
            except StopIteration:
                print(f"  ⚠️  Reached end of {task_name} dataset at {count} samples")
                break
            except Exception as e:
                skipped += 1
                continue
        
        pbar.close()
        
        if count < SAMPLES_PER_CLASS:
            print(f"  ⚠️  Only got {count}/{SAMPLES_PER_CLASS} samples for {task_name}")
        if count == 0:
            failed_datasets.append(task_name)
        if skipped > 0:
            print(f"  ℹ️  Skipped {skipped} invalid images for {task_name}")
        
    except Exception as e:
        print(f"  ❌ Failed to load {task_name}: {str(e)}")
        failed_datasets.append(task_name)

print(f"\n{'='*70}")
print(f"✅ Done! Saved {len(final_data)} samples to:")
print(f"   {JSON_PATH}")

# Show statistics
print(f"\n📊 Breakdown by VTAB Group:")
group_counts = {}
for entry in final_data:
    cat = entry['category']
    group_counts[cat] = group_counts.get(cat, 0) + 1

for cat in ['natural', 'specialized', 'structured']:
    if cat in group_counts:
        print(f"   {cat.capitalize():<12} : {group_counts[cat]} images")

print(f"\n📊 Breakdown by Task:")
task_counts = {}
for entry in final_data:
    task = entry['task']
    task_counts[task] = task_counts.get(task, 0) + 1

for task in sorted(task_counts.keys()):
    count = task_counts[task]
    desc = next(c['description'] for n, c in VTAB_SOURCES.items() if n == task)
    print(f"   {task:<22} : {count:3d} images  ({desc})")

if failed_datasets:
    print(f"\n⚠️  Failed datasets: {', '.join(failed_datasets)}")
    print(f"   Continuing with {len(task_counts)} successful tasks")

# Save JSON
with open(JSON_PATH, "w") as f:
    json.dump(final_data, f, indent=2)

print(f"\n💡 VTAB Advantages:")
print(f"   ✓ Standardized benchmark datasets")
print(f"   ✓ 3 distinct visual domains (Natural, Specialized, Structured)")
print(f"   ✓ Tests semantic understanding across diverse visual tasks")
print(f"   ✓ Already validated for transfer learning evaluation")

print(f"\n💡 Expected Routing Behavior:")
print(f"   Good Router:")
print(f"   • Natural images     → One set of experts (e.g., E0, E1)")
print(f"   • Specialized images → Different experts (e.g., E2)")
print(f"   • Structured images  → Different experts (e.g., E3)")
print(f"\n   Bad Router:")
print(f"   • All groups → Same expert distribution (no specialization)")

print(f"\n{'='*70}")

# Create a summary file
summary = {
    "total_samples": len(final_data),
    "samples_per_task": SAMPLES_PER_CLASS,
    "groups": {cat: count for cat, count in group_counts.items()},
    "tasks": {task: count for task, count in task_counts.items()},
    "failed_datasets": failed_datasets
}

summary_path = os.path.join(OUTPUT_DIR, "dataset_summary.json")
with open(summary_path, "w") as f:
    json.dump(summary, f, indent=2)

print(f"📄 Summary saved to: {summary_path}")