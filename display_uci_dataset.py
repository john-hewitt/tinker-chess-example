#!/usr/bin/env python3
"""
Load the saved move prefix dataset.
"""

from datasets import load_from_disk

# Load the dataset from disk
dataset_path = 'lichess_move_prefixes'  # Change this if you used a different output path
dataset_dict = load_from_disk(dataset_path)

# Access the 'train' split
dataset = dataset_dict['train']

print(f"Dataset loaded: {len(dataset)} move prefixes")
print(f"Dataset splits: {list(dataset_dict.keys())}")
print(f"\nDataset features: {dataset.features}")
print(f"\nSample entries:")
for i in range(min(10, len(dataset))):
    print(f"  {i+1}. {dataset[i]['uci_prefix']}")

# Example: Access a specific prefix
print(f"\nExample - First prefix: {dataset[0]['uci_prefix']}")

# Example: Iterate through prefixes
print(f"\nFirst 5 prefixes:")
for i, example in enumerate(dataset.select(range(5))):
    print(f"  {i+1}. {example['uci_prefix']}")

