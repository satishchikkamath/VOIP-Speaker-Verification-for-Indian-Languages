import os
from collections import defaultdict

# Path to check
TRAIN_DIR = "/home/user2/VOIP/VOIP_Mel_Features/train"

def check_folder_uniqueness(directory):
    if not os.path.exists(directory):
        print(f"Error: Directory not found: {directory}")
        return

    # Get only directories (ignore files)
    folder_names = [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]
    
    print(f"Scanning {len(folder_names)} folders in: {directory}")
    print("-" * 40)

    # 1. Check for Case-Insensitive Duplicates (e.g., 'User1' vs 'user1')
    lower_map = defaultdict(list)
    for name in folder_names:
        lower_map[name.lower()].append(name)

    case_issues = {k: v for k, v in lower_map.items() if len(v) > 1}

    if case_issues:
        print(f"[CRITICAL] Found {len(case_issues)} Case-Sensitivity collisions!")
        print("These folders are treated as different by Linux but will confuse your model:")
        for k, v in case_issues.items():
            print(f"  - Group '{k}': {v}")
    else:
        print("[OK] No case-sensitivity duplicates found.")

    # 2. Check for Whitespace Duplicates (e.g., 'User1 ' vs 'User1')
    stripped_map = defaultdict(list)
    for name in folder_names:
        stripped_map[name.strip()].append(name)

    space_issues = {k: v for k, v in stripped_map.items() if len(v) > 1}

    if space_issues:
        print(f"\n[CRITICAL] Found {len(space_issues)} Trailing Whitespace collisions!")
        print("These look identical but one has hidden spaces:")
        for k, v in space_issues.items():
            print(f"  - Group '{k}': {v}") # The list will show the quotes so you can see the space
    else:
        print("[OK] No whitespace duplicates found.")

    print("-" * 40)
    if not case_issues and not space_issues:
        print("PASSED: All folder names are unique and clean.")
    else:
        print("FAILED: Please fix the issues above.")

if __name__ == "__main__":
    check_folder_uniqueness(TRAIN_DIR)