#!/bin/bash

# Check if the correct number of arguments is provided
if [ "$#" -ne 3 ]; then
  echo "Usage: $0 <train_dir> <test_dir> <merged_dir>"
  exit 1
fi

# Get the train, test, and merged directories from the command-line arguments
train_dir="$1"
test_dir="$2"
merged_dir="$3"

# Create the destination directory if it doesn't exist
mkdir -p "$merged_dir"

# Loop over each class folder (0-9)
for i in {0..9}; do
  # Create the class folder in the destination directory
  mkdir -p "$merged_dir/$i"
  
  # Copy images from the train directory to the merged directory
  cp "$train_dir/$i"/* "$merged_dir/$i/"
  
  # Copy images from the test directory to the merged directory
  cp "$test_dir/$i"/* "$merged_dir/$i/"
done

echo "Merge completed successfully!"
