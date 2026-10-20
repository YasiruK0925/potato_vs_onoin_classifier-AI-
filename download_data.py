import os
from datasets import load_dataset

print("Downloading dataset info from Hugging Face...")
# Load the dataset from Hugging Face for the train split
dataset = load_dataset("Project-AgML/vegetable_classification_bangladesh", split="train")

# Retrieve the class names corresponding to the dataset labels
labels = dataset.features["label"].names
print(f"Available classes: {labels}")

# We only want to filter and download Potato and Onion images
target_classes = ["Potato", "Onion"]

# Create local directories for saving the filtered images if they don't exist
for cls in target_classes:
    os.makedirs(f"dataset/{cls.lower()}", exist_ok=True)

print("Extracting and saving images...")
count = {"Potato": 0, "Onion": 0}

# Iterate through the dataset items and extract target classes
for idx, item in enumerate(dataset):
    label_idx = item["label"]
    class_name = labels[label_idx]
    
    if class_name in target_classes:
        image = item["image"]  # The image is returned as a PIL Image object
        # Save the image into its respective folder with a unique filename
        image.save(f"dataset/{class_name.lower()}/{class_name}_{count[class_name]}.jpg")
        count[class_name] += 1

print(f"Successfully saved! Potatoes: {count['Potato']}, Onions: {count['Onion']}")