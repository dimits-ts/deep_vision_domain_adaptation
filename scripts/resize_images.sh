#!/bin/bash

# Check if ImageMagick is installed
if ! command -v convert &> /dev/null
then
    echo "ImageMagick (convert) could not be found. Please install it and try again."
    exit 1
fi

# Check if GNU Parallel is installed
if ! command -v parallel &> /dev/null
then
    echo "GNU Parallel could not be found. Please install it and try again."
    exit 1
fi

# Check if the root directory is provided as a parameter
if [ -z "$1" ]
then
    echo "Usage: $0 /path/to/root/directory"
    exit 1
fi

ROOT_DIR="$1"

# Function to resize images
resize_image() {
    local file="$1"
    local width=$(identify -format "%w" "$file")
    local height=$(identify -format "%h" "$file")
    
    if [[ $width -gt 1500 || $height -gt 1500 ]]; then
        echo "Resizing $file ($width x $height) to fit within 1500x1500 pixels."
        convert "$file" -resize 1500x1500\> "$file"
    else
        echo "$file is already within the desired dimensions."
    fi
}

export -f resize_image

# Find all image files in the specified directory and process them in parallel
find "$ROOT_DIR" -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" -o -iname "*.gif" \) | parallel resize_image {}

echo "Done processing images."
