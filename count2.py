import os

folder_path = "/home/user2/7th/dataset/voxceleb1/train/"

ids = [
    d for d in os.listdir(folder_path)
    if os.path.isdir(os.path.join(folder_path, d)) and d.startswith("id")
]

print("Total number of IDs:", len(ids))
