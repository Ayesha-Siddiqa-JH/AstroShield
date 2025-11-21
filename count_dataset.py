import os
import yaml
from collections import defaultdict

# Paths
data_dir = os.path.join('data', 'raw')
train_labels_dir = os.path.join(data_dir, 'train', 'labels')
train_images_dir = os.path.join(data_dir, 'train', 'images')
test_labels_dir = os.path.join(data_dir, 'test', 'test3', 'labels')
test_images_dir = os.path.join(data_dir, 'test', 'test3', 'images')

# Class names from data.yaml
class_names = [
    'OxygenTank',
    'NitrogenTank',
    'FirstAidBox',
    'FireAlarm',
    'SafetySwitchPanel',
    'EmergencyPhone',
    'FireExtinguisher'
]

def count_objects_in_file(label_file):
    """Count objects in a single label file."""
    counts = defaultdict(int)
    try:
        with open(label_file, 'r') as f:
            for line in f:
                if line.strip():
                    class_id = int(line.strip().split()[0])
                    counts[class_id] += 1
    except Exception as e:
        print(f"Error reading {label_file}: {e}")
    return counts

def count_dataset(base_dir):
    """Count objects in all label files in a directory."""
    total_counts = defaultdict(int)
    image_count = 0
    
    if not os.path.exists(base_dir):
        print(f"Directory not found: {base_dir}")
        return total_counts, image_count
    
    for root, _, files in os.walk(base_dir):
        for file in files:
            if file.endswith('.txt') and not file.endswith('classes.txt'):
                label_file = os.path.join(root, file)
                counts = count_objects_in_file(label_file)
                for class_id, count in counts.items():
                    total_counts[class_id] += count
                if counts:  # If we found any objects, count this as a valid image
                    image_count += 1
    
    return total_counts, image_count

def get_image_count(image_dir):
    """Count number of image files in a directory."""
    if not os.path.exists(image_dir):
        return 0
    return len([f for f in os.listdir(image_dir) 
               if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))])

def main():
    # Count objects in training set
    print("Counting objects in training set...")
    train_counts, train_images_with_objects = count_dataset(train_labels_dir)
    total_train_images = get_image_count(train_images_dir) or train_images_with_objects
    
    # Count objects in test set
    print("\nCounting objects in test set...")
    test_counts, test_images_with_objects = count_dataset(test_labels_dir)
    total_test_images = get_image_count(test_images_dir) or test_images_with_objects
    
    # Calculate totals
    total_counts = defaultdict(int)
    for class_id in range(len(class_names)):
        total_counts[class_id] = train_counts.get(class_id, 0) + test_counts.get(class_id, 0)
    
    # Print results
    print("\n=== Dataset Statistics ===")
    print(f"\nTotal Training Images: {total_train_images}")
    print(f"Training Images with Objects: {train_images_with_objects}")
    print("\nTraining Set Object Counts:")
    for class_id, count in sorted(train_counts.items()):
        if class_id < len(class_names):
            print(f"  {class_names[class_id]}: {count}")
        else:
            print(f"  Class {class_id}: {count} (Warning: Class ID out of range!)")
    
    print(f"\nTotal Test Images: {total_test_images}")
    print(f"Test Images with Objects: {test_images_with_objects}")
    print("\nTest Set Object Counts:")
    for class_id, count in sorted(test_counts.items()):
        if class_id < len(class_names):
            print(f"  {class_names[class_id]}: {count}")
        else:
            print(f"  Class {class_id}: {count} (Warning: Class ID out of range!)")
    
    print("\nTotal Dataset Object Counts:")
    for class_id in range(len(class_names)):
        print(f"  {class_names[class_id]}: {total_counts[class_id]}")
    
    # Save to a JSON file for the frontend
    stats = {
        'total_train_images': total_train_images,
        'train_images_with_objects': train_images_with_objects,
        'total_test_images': total_test_images,
        'test_images_with_objects': test_images_with_objects,
        'class_counts': {class_names[i]: total_counts.get(i, 0) for i in range(len(class_names))}
    }
    
    with open('dataset_statistics.json', 'w') as f:
        import json
        json.dump(stats, f, indent=2)
    
    print("\nStatistics saved to dataset_statistics.json")

if __name__ == "__main__":
    main()
