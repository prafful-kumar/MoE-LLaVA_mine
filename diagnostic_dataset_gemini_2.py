import os
import json
import random
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm

# =========================================================
# ⚙️ CONFIGURATION
# =========================================================
OUTPUT_DIR = "diagnostic_dataset_varied"
IMG_DIR = os.path.join(OUTPUT_DIR, "images")
JSON_PATH = os.path.join(OUTPUT_DIR, "diagnostic_data.json")
SAMPLES_PER_CLASS = 50

# Ensure directories exist
os.makedirs(IMG_DIR, exist_ok=True)

# =========================================================
# 🍎 DATA MAPS (Fixing "Class 6" -> "Beignets")
# =========================================================
# Food101 has 101 classes. We map index to name to avoid "food (class 6)".
FOOD101_CLASSES = [
    "apple_pie", "baby_back_ribs", "baklava", "beef_carpaccio", "beef_tartare",
    "beet_salad", "beignets", "bibimbap", "bread_pudding", "breakfast_burrito",
    "bruschetta", "caesar_salad", "cannoli", "caprese_salad", "carrot_cake",
    "ceviche", "cheesecake", "cheese_plate", "chicken_curry", "chicken_quesadilla",
    "chicken_wings", "chocolate_cake", "chocolate_mousse", "churros", "clam_chowder",
    "club_sandwich", "crab_cakes", "creme_brulee", "croque_madame", "cup_cakes",
    "deviled_eggs", "donuts", "dumplings", "edamame", "eggs_benedict",
    "escargots", "falafel", "filet_mignon", "fish_and_chips", "foie_gras",
    "french_fries", "french_onion_soup", "french_toast", "fried_calamari", "fried_rice",
    "frozen_yogurt", "garlic_bread", "gnocchi", "greek_salad", "grilled_cheese_sandwich",
    "grilled_salmon", "guacamole", "gyoza", "hamburger", "hot_and_sour_soup",
    "hot_dog", "huevos_rancheros", "hummus", "ice_cream", "lasagna",
    "lobster_bisque", "lobster_roll_sandwich", "macaroni_and_cheese", "macarons", "miso_soup",
    "mussels", "nachos", "omelette", "onion_rings", "oysters",
    "pad_thai", "paella", "pancakes", "panna_cotta", "peking_duck",
    "pho", "pizza", "pork_chop", "poutine", "prime_rib",
    "pulled_pork_sandwich", "ramen", "ravioli", "red_velvet_cake", "risotto",
    "samosa", "sashimi", "scallops", "seaweed_salad", "shrimp_and_grits",
    "spaghetti_bolognese", "spaghetti_carbonara", "spring_rolls", "steak", "strawberry_shortcake",
    "sushi", "tacos", "takoyaki", "tiramisu", "tuna_tartare", "waffles"
]

# =========================================================
# 💬 CONVERSATION TEMPLATES (Context-Aware)
# =========================================================
CONVERSATION_TEMPLATES = {
    "scene": {
        "identification": {
            "questions": ["<image>\nWhat is the main subject?", "<image>\nDescribe this scene.", "<image>\nWhat is happening?"],
            "answers": ["This scene shows {label}.", "The image captures {label}.", "You are looking at {label}."]
        },
        "spatial": {
            "questions": ["<image>\nDescribe the spatial layout.", "<image>\nHow are elements positioned?"],
            "answers": ["In this image of {label}, elements are arranged naturally.", "The {label} scene shows depth and perspective."]
        }
    },
    "food": {
        "identification": {
            "questions": ["<image>\nWhat dish is this?", "<image>\nIdentify this food.", "<image>\nName this cuisine."],
            "answers": ["This is {label}, a delicious dish.", "The image shows {label}.", "This appears to be {label}."]
        },
        "ingredients": {
            "questions": ["<image>\nWhat are the main ingredients?", "<image>\nDescribe the components."],
            "answers": ["{label} typically contains fresh ingredients.", "This {label} is made with distinct components."]
        },
        "texture": {
            "questions": ["<image>\nDescribe the texture.", "<image>\nHow does this food look?"],
            "answers": ["The {label} has a rich texture.", "This {label} appears to have a distinct surface quality."]
        }
    },
    "document": {
        "identification": {
            "questions": ["<image>\nWhat type of document is this?", "<image>\nClassify this form."],
            "answers": ["This is an {label}.", "The image displays an {label}."]
        },
        "structure": {
            "questions": ["<image>\nHow is the text organized?", "<image>\nDescribe the layout."],
            "answers": ["The {label} follows a standard structured layout.", "This {label} contains organized fields and headers."]
        }
    },
    "chart": {
        "identification": {
            "questions": ["<image>\nWhat kind of chart is this?", "<image>\nIdentify the visualization."],
            "answers": ["This is a {label}.", "The image shows a {label}."]
        },
        "trend": {
            "questions": ["<image>\nWhat trend does the data show?", "<image>\nAnalyze the pattern."],
            "answers": ["The {label} reveals data trends.", "This {label} displays quantitative patterns."]
        }
    },
    "code": {
        "identification": {
            "questions": ["<image>\nWhat language is this?", "<image>\nIdentify the code."],
            "answers": ["This is {label}.", "The image displays {label}."]
        },
        "purpose": {
            "questions": ["<image>\nWhat does this code do?", "<image>\nExplain the functionality."],
            "answers": ["This {label} implements specific logic.", "The {label} defines structural elements."]
        }
    },
    "text": {
        "identification": {
            "questions": ["<image>\nWhat text is visible?", "<image>\nRead the content."],
            "answers": ["The image contains {label}.", "I can see {label}."]
        }
    }
}

# =========================================================
# 📚 DATA SOURCES (FIXED)
# =========================================================
SOURCES = {
    "scene": {
        # Official COCO dataset
        "path": "HuggingFaceM4/COCO", 
        "name": "2014_captions", # Specific config for captions
        "split": "validation", 
        "img_key": "image",
        "label_strategy": "caption", 
        "label_key": "sentences" # In this dataset, captions are under 'sentences' -> 'raw'
    },
    "food": {
        "path": "food101", 
        "split": "train", 
        "img_key": "image",
        "label_strategy": "class_name", 
        "label_key": "label"
    },
    "document": {
        # REPLACEMENT: DocVQA (Much faster/smaller than IDL-WDS)
        "path": "nielsr/docvqa_1200_examples", 
        "split": "test", 
        "img_key": "image",
        "label_strategy": "static", 
        "static_label": "official document"
    },
    "chart": {
        "path": "HuggingFaceM4/ChartQA", 
        "name": "val", # Needed for ChartQA
        "split": "val", 
        "img_key": "image",
        "label_strategy": "static", 
        "static_label": "data chart"
    },
    "code": {
        "path": "HuggingFaceM4/WebSight", 
        "split": "train", 
        "img_key": "image",
        "label_strategy": "static", 
        "static_label": "HTML/CSS code"
    },
    "text": {
        "path": "wendlerc/RenderedText", 
        "split": "train", 
        "img_key": "image",
        "label_strategy": "static", 
        "static_label": "rendered text"
    }
}
# =========================================================
# 🛠️ HELPER FUNCTIONS
# =========================================================
def extract_label(sample, config, category):
    """Extracts a human-readable label based on strategy."""
    strategy = config.get("label_strategy", "static")
    
    if strategy == "static":
        return config.get("static_label", category)
    
    elif strategy == "caption":
        # Get first 8 words of caption
        val = sample.get(config["label_key"], "")
        if isinstance(val, list): val = val[0]
        return " ".join(str(val).split()[:8]) if val else category
    
    elif strategy == "class_name":
        idx = sample.get(config["label_key"], 0)
        # Map Food101 index to real name
        if category == "food" and 0 <= idx < len(FOOD101_CLASSES):
            return FOOD101_CLASSES[idx].replace("_", " ")
        return f"{category} (class {idx})"
    
    return category

def get_random_conversation(category, label, q_type_idx):
    """Generates a random conversation from the appropriate template."""
    q_types = list(CONVERSATION_TEMPLATES[category].keys())
    # Rotate through question types to ensure variety (ID -> Texture -> Ingredients...)
    current_type = q_types[q_type_idx % len(q_types)]
    
    templates = CONVERSATION_TEMPLATES[category][current_type]
    
    q_text = random.choice(templates["questions"])
    a_text = random.choice(templates["answers"]).format(label=label)
    
    return [{"from": "human", "value": q_text}, {"from": "gpt", "value": a_text}]

# =========================================================
# 🚀 MAIN GENERATOR LOOP
# =========================================================
final_data = []

print(f"🚀 Building Diagnostic Dataset ({SAMPLES_PER_CLASS} samples/class)...")
print(f"🔄 Shuffling enabled (buffer=10k) to prevent 'Beignet' clusters.")

for category, config in SOURCES.items():
    print(f"📥 Processing {category.upper()}...")
    
    try:
        # Load Streaming + SHUFFLE (Critical Fix)
        ds = load_dataset(config['path'], split=config['split'], streaming=True)
        ds = ds.shuffle(seed=42, buffer_size=10_000)
        ds_iter = iter(ds)
        
        count = 0
        pbar = tqdm(total=SAMPLES_PER_CLASS, desc=f"  {category}", leave=True)
        
        while count < SAMPLES_PER_CLASS:
            try:
                sample = next(ds_iter)
                image = sample.get(config['img_key'])
                
                # Validation Checks
                if image is None: continue
                if hasattr(image, 'mode') and image.mode != "RGB": image = image.convert("RGB")
                if image.width < 50 or image.height < 50: continue # Skip thumbnails
                
                # Resize if massive (saves disk space)
                if image.width > 1024:
                    scale = 1024 / image.width
                    image = image.resize((1024, int(image.height * scale)), Image.LANCZOS)

                # Generate Data
                label = extract_label(sample, config, category)
                conversation = get_random_conversation(category, label, count)
                
                # Save Image
                img_filename = f"{category}_{count:03d}.jpg"
                img_path = os.path.join(IMG_DIR, img_filename)
                image.save(img_path, quality=85)
                
                # Add to JSON
                final_data.append({
                    "id": f"{category}_{count}",
                    "image": img_path,
                    "category": category,
                    "conversations": conversation
                })
                
                count += 1
                pbar.update(1)
                
            except StopIteration:
                break
            except Exception:
                continue # Skip corrupted images
                
        pbar.close()
        
    except Exception as e:
        print(f"❌ Failed {category}: {e}")

# Save JSON
with open(JSON_PATH, "w") as f:
    json.dump(final_data, f, indent=2)

print(f"\n✅ SUCCESS: Saved {len(final_data)} samples to {JSON_PATH}")