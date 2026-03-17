"""
Build diagnostic_data_v2.json

Strategy:
- Keep only 10 images per category (50 total) — small and analyzable
- Pair each image with 5 prompt types: describe, count, yesno, attribute, reason
- Total: 250 samples (same size as v1)
- Each sample gets a 'prompt_type' field in addition to 'category'
- This lets vis_dual_routing.py analyze routing by BOTH category AND prompt type

Why prompt types matter for routing analysis:
- Same image, different question type = different text token sequence
- If routing changes, it means the router responds to the question, not just the image
- If routing stays the same, the image content dominates routing decisions
"""

import json
import os

# ---------------------------------------------------------------------------
# Prompt templates per (category, prompt_type)
# Each entry is a list; we rotate through them as images vary.
# ---------------------------------------------------------------------------

PROMPTS = {
    "animal": {
        "describe": [
            "Describe the animal you see in this image.",
            "What does this animal look like?",
            "Tell me about the animal in this photo.",
        ],
        "count": [
            "How many animals are in this image?",
            "Count the number of animals visible.",
            "How many creatures can you see?",
        ],
        "yesno": [
            "Is this animal outdoors?",
            "Does this animal appear to be a mammal?",
            "Is the animal facing the camera?",
        ],
        "attribute": [
            "What color is this animal?",
            "What is the approximate size of this animal?",
            "What distinguishing physical features does this animal have?",
        ],
        "reason": [
            "Why might this animal behave the way it does in this image?",
            "What habitat does this animal likely live in based on the image?",
            "What can you infer about this animal's diet from its appearance?",
        ],
    },

    "food": {
        "describe": [
            "Describe the food in this image.",
            "What does this dish look like?",
            "Tell me about the food shown here.",
        ],
        "count": [
            "How many distinct food items are on the plate?",
            "Count the number of ingredients you can identify.",
            "How many servings does this appear to be?",
        ],
        "yesno": [
            "Does this food appear to be cooked?",
            "Is this a dessert?",
            "Does the dish contain vegetables?",
        ],
        "attribute": [
            "What is the primary color of this food?",
            "How would you describe the texture of this food?",
            "What is the approximate portion size shown?",
        ],
        "reason": [
            "What cuisine does this dish likely belong to?",
            "What cooking method was likely used to prepare this food?",
            "Why might this food be considered a healthy or unhealthy choice?",
        ],
    },

    "chart": {
        "describe": [
            "Describe the chart shown in this image.",
            "What information does this graph convey?",
            "Explain the visualization in this image.",
        ],
        "count": [
            "How many data series are shown in this chart?",
            "Count the number of bars or data points visible.",
            "How many categories are represented in this chart?",
        ],
        "yesno": [
            "Does this chart show an increasing trend?",
            "Is there a legend in this chart?",
            "Does this chart use a logarithmic scale?",
        ],
        "attribute": [
            "What type of chart is this (bar, line, pie, etc.)?",
            "What are the axis labels in this chart?",
            "What color scheme is used in this chart?",
        ],
        "reason": [
            "What conclusion can you draw from this chart?",
            "What might explain the trend shown in this data?",
            "What domain or field does this chart most likely belong to?",
        ],
    },

    "scene": {
        "describe": [
            "Describe the scene in this image.",
            "What is happening in this photo?",
            "Tell me about this scene.",
        ],
        "count": [
            "How many people are visible in this scene?",
            "Count the number of objects in the foreground.",
            "How many distinct groups of people can you identify?",
        ],
        "yesno": [
            "Is this scene taking place outdoors?",
            "Are there people interacting with each other in this scene?",
            "Does this scene appear to be in a crowded place?",
        ],
        "attribute": [
            "What time of day does this scene appear to be?",
            "What is the setting or location of this scene?",
            "What is the dominant color in this scene?",
        ],
        "reason": [
            "What event or activity is taking place in this scene?",
            "Why might the people in this scene be gathered here?",
            "What can you infer about the mood or atmosphere of this scene?",
        ],
    },

    "screenshot": {
        "describe": [
            "Describe what is shown in this screenshot.",
            "What does this interface look like?",
            "Tell me about the content of this screenshot.",
        ],
        "count": [
            "How many buttons or interactive elements are visible?",
            "Count the number of menu items shown.",
            "How many distinct sections does this interface have?",
        ],
        "yesno": [
            "Is this a mobile application interface?",
            "Does this screenshot show a settings menu?",
            "Is there a navigation bar visible in this screenshot?",
        ],
        "attribute": [
            "What is the primary color scheme of this interface?",
            "What type of application does this screenshot show?",
            "What operating system does this appear to be from?",
        ],
        "reason": [
            "What is the purpose of this application based on the screenshot?",
            "What task is the user likely trying to accomplish in this screenshot?",
            "Why might this interface be designed the way it is?",
        ],
    },
}

PROMPT_TYPES = ["describe", "count", "yesno", "attribute", "reason"]
IMAGES_PER_CATEGORY = 10
CATEGORIES = ["animal", "food", "chart", "scene", "screenshot"]

# ---------------------------------------------------------------------------
# Build dataset
# ---------------------------------------------------------------------------

samples = []
idx = 0

for cat in CATEGORIES:
    for img_num in range(IMAGES_PER_CATEGORY):
        img_path = f"diagnostic_dataset/images/{cat}_{img_num:03d}.jpg"

        for pt in PROMPT_TYPES:
            # Rotate through the 3 template variants using img_num
            prompt_text = PROMPTS[cat][pt][img_num % len(PROMPTS[cat][pt])]

            sample = {
                "id": f"{cat}_{img_num}_{pt}",
                "image": img_path,
                "category": cat,
                "prompt_type": pt,
                "conversations": [
                    {
                        "from": "human",
                        "value": f"<image>\n{prompt_text}"
                    },
                    {
                        "from": "gpt",
                        "value": "Answer based on the image."
                    }
                ]
            }
            samples.append(sample)
            idx += 1

print(f"Total samples: {len(samples)}")
from collections import Counter
print("Per category:", Counter(s['category'] for s in samples))
print("Per prompt type:", Counter(s['prompt_type'] for s in samples))
print("Per (category, prompt_type):", Counter(f"{s['category']}/{s['prompt_type']}" for s in samples))

# Check all images exist
missing = [s for s in samples if not os.path.exists(s['image'])]
if missing:
    print(f"\nWARNING: {len(missing)} images not found:")
    for s in missing[:5]:
        print(f"  {s['image']}")
else:
    print(f"\nAll {len(set(s['image'] for s in samples))} images found.")

out_path = "diagnostic_dataset/diagnostic_data_v2.json"
with open(out_path, "w") as f:
    json.dump(samples, f, indent=2)
print(f"Saved to {out_path}")
