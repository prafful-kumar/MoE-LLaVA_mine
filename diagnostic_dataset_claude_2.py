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
OUTPUT_DIR = "diagnostic_dataset"
IMG_DIR = os.path.join(OUTPUT_DIR, "images")
JSON_PATH = os.path.join(OUTPUT_DIR, "diagnostic_data.json")
SAMPLES_PER_CLASS = 50

os.makedirs(IMG_DIR, exist_ok=True)

# Diverse conversation templates with randomized phrasing
CONVERSATION_TEMPLATES = {
    "animal": {
        "identification": {
            "questions": [
                "<image>\nWhat animal is shown in this image?",
                "<image>\nIdentify the animal in this photo.",
                "<image>\nWhat type of creature is this?",
                "<image>\nCan you name this animal?"
            ],
            "answers": [
                "This image shows an animal. The animal appears to be a domesticated pet with distinct features.",
                "The main subject is an animal. I can see characteristic features typical of this species.",
                "This depicts an animal with recognizable physical characteristics.",
                "The image presents an animal, displaying typical morphological traits."
            ]
        },
        "counting": {
            "questions": [
                "<image>\nHow many animals are visible?",
                "<image>\nCount the animals in this image.",
                "<image>\nHow many creatures can you see?",
                "<image>\nWhat is the number of animals present?"
            ],
            "answers": [
                "There appears to be one animal prominently featured in this image.",
                "I can count one animal as the main subject of this photograph.",
                "The image contains one animal positioned centrally.",
                "A single animal is visible in this frame."
            ]
        },
        "description": {
            "questions": [
                "<image>\nDescribe the animal's appearance.",
                "<image>\nWhat are the physical features of this animal?",
                "<image>\nTell me about this animal's characteristics.",
                "<image>\nHow would you describe this creature?"
            ],
            "answers": [
                "The animal displays characteristic features including fur, eyes, and body structure typical of its species.",
                "This animal has distinct physical attributes with visible features that help identify its type.",
                "The creature shows recognizable traits including coloration and body proportions.",
                "Physical characteristics include typical morphology with clear species-specific features."
            ]
        }
    },
    "food": {
        "identification": {
            "questions": [
                "<image>\nWhat food is shown here?",
                "<image>\nIdentify the dish in this image.",
                "<image>\nWhat type of food is this?",
                "<image>\nCan you name this food item?"
            ],
            "answers": [
                "This image shows a prepared food dish with distinctive appearance and ingredients.",
                "The food item appears to be a specific dish with characteristic presentation.",
                "This depicts a food preparation with recognizable culinary elements.",
                "The image presents a dish with typical food presentation style."
            ]
        },
        "ingredients": {
            "questions": [
                "<image>\nWhat ingredients are in this dish?",
                "<image>\nCan you identify the components?",
                "<image>\nWhat is this food made of?",
                "<image>\nDescribe the ingredients you see."
            ],
            "answers": [
                "This dish contains various ingredients combined to create the final preparation.",
                "The food includes multiple components that contribute to its flavor and appearance.",
                "Several ingredients are visible, working together to form this dish.",
                "The preparation uses a combination of ingredients typical for this type of food."
            ]
        },
        "appearance": {
            "questions": [
                "<image>\nDescribe how this food looks.",
                "<image>\nWhat is the visual presentation?",
                "<image>\nHow would you describe the appearance?",
                "<image>\nTell me about the food's presentation."
            ],
            "answers": [
                "The food displays appealing colors and textures with careful plating and presentation.",
                "Visual presentation includes various colors and textural elements arranged attractively.",
                "The dish shows characteristic appearance with distinct visual qualities.",
                "Presentation features multiple visual elements contributing to overall appeal."
            ]
        }
    },
    "document": {
        "identification": {
            "questions": [
                "<image>\nWhat type of document is this?",
                "<image>\nIdentify this document.",
                "<image>\nWhat kind of form is shown?",
                "<image>\nCan you classify this document?"
            ],
            "answers": [
                "This appears to be a structured document with organized text and information fields.",
                "The document shows typical formatting with sections and labeled areas.",
                "This is a formal document featuring standard layout and text organization.",
                "The image presents a document with characteristic structure and formatting."
            ]
        },
        "content": {
            "questions": [
                "<image>\nWhat information does this contain?",
                "<image>\nDescribe the document's content.",
                "<image>\nWhat data is recorded here?",
                "<image>\nWhat details are in this document?"
            ],
            "answers": [
                "The document contains various types of information organized in structured fields.",
                "Information is presented in organized sections with clear labeling and formatting.",
                "The content includes multiple data points arranged systematically.",
                "Various details are recorded across different sections of the document."
            ]
        },
        "structure": {
            "questions": [
                "<image>\nDescribe the document's layout.",
                "<image>\nHow is this document organized?",
                "<image>\nWhat is the structure?",
                "<image>\nExplain the document format."
            ],
            "answers": [
                "The document follows a structured layout with clearly defined sections and fields.",
                "Organization includes headers, body sections, and designated information areas.",
                "The structure demonstrates systematic arrangement with logical information flow.",
                "Layout features hierarchical organization with clear sectional divisions."
            ]
        }
    },
    "chart": {
        "identification": {
            "questions": [
                "<image>\nWhat type of chart is this?",
                "<image>\nIdentify this visualization.",
                "<image>\nWhat kind of graph is shown?",
                "<image>\nCan you classify this chart?"
            ],
            "answers": [
                "This is a data visualization chart presenting quantitative information graphically.",
                "The chart displays data using visual elements like bars, lines, or other markers.",
                "This visualization represents numerical data in graphical format.",
                "The image shows a chart designed to communicate data patterns visually."
            ]
        },
        "trend": {
            "questions": [
                "<image>\nWhat trend does this show?",
                "<image>\nDescribe the pattern in the data.",
                "<image>\nWhat does this chart indicate?",
                "<image>\nWhat trend is visible?"
            ],
            "answers": [
                "The chart reveals patterns in the data suggesting trends across the measured variables.",
                "Visual patterns indicate directional movement or relationships in the dataset.",
                "The data demonstrates trends that provide insight into the measured phenomenon.",
                "Clear patterns emerge showing how the variables change or relate to each other."
            ]
        },
        "elements": {
            "questions": [
                "<image>\nDescribe the chart elements.",
                "<image>\nWhat components are in this chart?",
                "<image>\nIdentify the chart features.",
                "<image>\nWhat visual elements do you see?"
            ],
            "answers": [
                "The chart includes axes, labels, and data representations with clear visual markers.",
                "Key elements include axis labels, data series, and visual indicators for measurements.",
                "Components feature standard charting elements like scales, markers, and legends.",
                "Visual elements comprise axes, data points, and supporting information for interpretation."
            ]
        }
    },
    "screenshot": {
        "identification": {
            "questions": [
                "<image>\nWhat is shown in this screenshot?",
                "<image>\nDescribe this interface.",
                "<image>\nWhat type of screen is this?",
                "<image>\nIdentify what's displayed."
            ],
            "answers": [
                "This screenshot shows a digital interface with various UI elements and content.",
                "The image displays a screen capture of a user interface or application.",
                "This depicts a digital display with interface components and information.",
                "The screenshot presents a user interface with organized visual elements."
            ]
        },
        "elements": {
            "questions": [
                "<image>\nWhat UI elements are visible?",
                "<image>\nDescribe the interface components.",
                "<image>\nWhat features can you see?",
                "<image>\nIdentify the interface elements."
            ],
            "answers": [
                "The interface includes various UI components like buttons, text, and visual elements.",
                "Multiple interface elements are visible including navigation and content areas.",
                "The screen shows standard UI components organized for user interaction.",
                "Various interface elements contribute to the overall functionality and design."
            ]
        },
        "purpose": {
            "questions": [
                "<image>\nWhat is the purpose of this interface?",
                "<image>\nWhat does this screen do?",
                "<image>\nDescribe the functionality.",
                "<image>\nWhat is this interface for?"
            ],
            "answers": [
                "This interface is designed to present information and enable user interaction.",
                "The screen serves to display content and provide interactive functionality.",
                "This interface facilitates user tasks through organized visual presentation.",
                "The purpose is to enable user interaction with digital content and features."
            ]
        }
    },
    "scene": {
        "identification": {
            "questions": [
                "<image>\nWhat scene is depicted here?",
                "<image>\nDescribe this scene.",
                "<image>\nWhat is happening in this image?",
                "<image>\nIdentify this scene."
            ],
            "answers": [
                "This scene shows an outdoor or indoor setting with various elements and activities.",
                "The image depicts a scene with people, objects, or environmental elements.",
                "This scene captures a moment with multiple visual components and context.",
                "The setting presents a scene with recognizable environmental and contextual features."
            ]
        },
        "counting": {
            "questions": [
                "<image>\nHow many people are in this scene?",
                "<image>\nCount the people visible.",
                "<image>\nHow many individuals can you see?",
                "<image>\nWhat is the number of people present?"
            ],
            "answers": [
                "There are several people visible in this scene at various positions.",
                "Multiple individuals can be counted throughout the image.",
                "The scene contains a number of people engaged in different activities.",
                "Several people are present in this scene at different locations."
            ]
        },
        "activity": {
            "questions": [
                "<image>\nWhat activity is taking place?",
                "<image>\nDescribe what people are doing.",
                "<image>\nWhat's happening in this scene?",
                "<image>\nIdentify the main activity."
            ],
            "answers": [
                "People in the scene appear to be engaged in various activities and interactions.",
                "The scene shows activities typical of this setting with people participating.",
                "Various actions and interactions are taking place among the people present.",
                "The main activity involves people interacting within this environmental context."
            ]
        }
    }
}

# UPDATED SOURCES - All verified working datasets
SOURCES = {
    "animal": {
        "path": "mrm8488/ImageNet1K-val",  # Contains diverse animal images
        "split": "train",
        "img_key": "image",
        "label_strategy": "static",
        "static_label": "animal",
        "filter_fn": lambda sample: True  # Take all
    },
    "food": {
        "path": "food101",
        "split": "train",
        "img_key": "image",
        "label_strategy": "static",
        "static_label": "food"
    },
    "document": {
        "path": "pixparse/pdfa-eng-wds",  # Document/PDF images - WORKS
        "split": "train",
        "img_key": "jpg",
        "label_strategy": "static",
        "static_label": "document"
    },
    "chart": {
        "path": "lmms-lab/ChartQA",  # Charts - Already working!
        "split": "test",
        "img_key": "image",
        "label_strategy": "static",
        "static_label": "chart"
    },
    "screenshot": {
        "path": "HuggingFaceM4/WebSight",  # UI screenshots - WORKS
        "split": "train",
        "img_key": "image",
        "label_strategy": "static",
        "static_label": "screenshot"
    },
    "scene": {
        "path": "detection-datasets/coco",  # Natural scenes - WORKS (no script)
        "name": "default",
        "split": "train",
        "img_key": "image",
        "label_strategy": "static",
        "static_label": "scene"
    }
}

def get_random_conversation(category, sample_idx):
    """Generate a diverse conversation with randomized phrasing."""
    templates = CONVERSATION_TEMPLATES[category]
    question_types = list(templates.keys())
    
    # Cycle through question types
    question_type = question_types[sample_idx % len(question_types)]
    
    question = random.choice(templates[question_type]["questions"])
    answer = random.choice(templates[question_type]["answers"])
    
    return [
        {"from": "human", "value": question},
        {"from": "gpt", "value": answer}
    ]

final_data = []

print(f"🚀 Building Diagnostic Dataset ({SAMPLES_PER_CLASS} samples per category)...")
print(f"📁 Output directory: {OUTPUT_DIR}")
print(f"💾 Downloading images from verified sources")
print(f"💬 Using diverse conversation templates\n")

for category, config in SOURCES.items():
    print(f"📥 Downloading {category} from {config['path']}...")
    
    try:
        # Load in streaming mode
        dataset_config = config.get('name', None)
        ds = load_dataset(
            config['path'],
            dataset_config,
            split=config['split'],
            streaming=True
        )
        ds_iter = iter(ds)
        
        count = 0
        skipped = 0
        pbar = tqdm(total=SAMPLES_PER_CLASS, desc=f"  {category.capitalize()}", leave=True)
        
        max_attempts = SAMPLES_PER_CLASS * 10  # More attempts for challenging datasets
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
                if image.width < 100 or image.height < 100:
                    skipped += 1
                    continue
                
                # Resize if too large
                max_size = 1024
                if image.width > max_size or image.height > max_size:
                    ratio = min(max_size / image.width, max_size / image.height)
                    new_size = (int(image.width * ratio), int(image.height * ratio))
                    image = image.resize(new_size, Image.LANCZOS)
                
                # Save image
                img_filename = f"{category}_{count:03d}.jpg"
                img_path = os.path.join(IMG_DIR, img_filename)
                image.save(img_path, quality=90)
                
                # Generate conversation
                conversation = get_random_conversation(category, count)
                
                # Create entry
                entry = {
                    "id": f"{category}_{count}",
                    "image": img_path,
                    "category": category,
                    "conversations": conversation
                }
                final_data.append(entry)
                
                count += 1
                pbar.update(1)
                
            except StopIteration:
                print(f"  ⚠️  Reached end of {category} dataset at {count} samples")
                break
            except Exception as e:
                skipped += 1
                if skipped % 50 == 0:
                    print(f"  ⚠️  Skipped {skipped} samples (errors during processing)")
                continue
        
        pbar.close()
        
        if count < SAMPLES_PER_CLASS:
            print(f"  ⚠️  Only got {count}/{SAMPLES_PER_CLASS} samples for {category}")
        if skipped > 0:
            print(f"  ℹ️  Skipped {skipped} invalid images for {category}")
        
    except Exception as e:
        print(f"  ❌ Failed to load {category}: {str(e)}")

print(f"\n{'='*70}")
print(f"✅ Done! Saved {len(final_data)} samples to:")
print(f"   {JSON_PATH}")
print(f"\n📊 Breakdown by category:")

# Show statistics
category_counts = {}
for entry in final_data:
    cat = entry['category']
    category_counts[cat] = category_counts.get(cat, 0) + 1

for cat, count in sorted(category_counts.items()):
    print(f"   {cat.capitalize():<12} : {count} images")

# Save JSON
with open(JSON_PATH, "w") as f:
    json.dump(final_data, f, indent=2)

print(f"\n💡 Dataset Categories:")
print(f"   • Animal: Diverse animal images")
print(f"   • Food: Culinary dishes and food items")
print(f"   • Document: Text-heavy documents and forms")
print(f"   • Chart: Data visualizations and graphs")
print(f"   • Screenshot: Web/app interface captures")
print(f"   • Scene: Real-world scenes with people and objects")

print(f"\n💡 Conversation Diversity:")
print(f"   ✓ 3 question types per category")
print(f"   ✓ 4 phrasing variations per question")
print(f"   ✓ 12 total conversation templates per category")

print(f"\n{'='*70}")