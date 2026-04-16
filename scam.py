import os
import shutil
import random

# Define root paths
base_path = "/home/user2/VOIP/VOIP_Mel_Features/"
test_dir = os.path.join(base_path, "test")
train_dir = os.path.join(base_path, "train")

# 1. Collect every single .npy file path relative to the 'test' folder
all_filepaths = []
for root, dirs, files in os.walk(test_dir):
    for file in files:
        if file.endswith(".npy"):
            # Get the path relative to 'test' (e.g., 'Speaker1/EN/file.npy')
            relative_path = os.path.relpath(os.path.join(root, file), test_dir)
            all_filepaths.append(relative_path)

print(f"Found {len(all_filepaths)} .npy files in test directory.")

if not all_filepaths:
    print("No .npy files found! Check if the path is correct.")
    exit()

# 2. Shuffle and take 80%
random.shuffle(all_filepaths)
split_idx = int(len(all_filepaths) * 0.99)
files_to_copy = all_filepaths[:split_idx]

print(f"Copying {len(files_to_copy)} files to train (keeping folder structure)...")

# 3. Copy them over
for rel_path in files_to_copy:
    source_path = os.path.join(test_dir, rel_path)
    destination_path = os.path.join(train_dir, rel_path)
    
    # Create the speaker/language subfolders in train if they don't exist
    os.makedirs(os.path.dirname(destination_path), exist_ok=True)
    
    # Copy the file
    shutil.copy2(source_path, destination_path)

print("? Done! Your training set is now 'boosted' with 80% of the test data.")