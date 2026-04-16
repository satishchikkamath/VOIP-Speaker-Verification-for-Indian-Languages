import os

def count_speakers(root_path):
    print(f"{'Folder':<10} | {'Speaker Count':<15}")
    print("-" * 30)
    
    for split in ['train', 'test']:
        split_path = os.path.join(root_path, split)
        
        if os.path.exists(split_path):
            # List all items in the split folder and filter for directories
            speakers = [s for s in os.listdir(split_path) 
                       if os.path.isdir(os.path.join(split_path, s))]
            print(f"{split:<10} | {len(speakers):<15}")
        else:
            print(f"{split:<10} | Path not found!")

if __name__ == "__main__":
    VOIP_PATH = "/home/user2/VOIP/VOIP_Mel_Features/"
    count_speakers(VOIP_PATH)