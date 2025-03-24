#!/bin/bash
# Improved setup script to dynamically extract files

# Create base directories
mkdir -p video_generator/models
mkdir -p video_generator/pipeline
mkdir -p video_generator/templates
mkdir -p data/inputs
mkdir -p data/history
mkdir -p output/videos
mkdir -p output/metadata
mkdir -p output/temp
mkdir -p logs

# Create __init__.py files to make packages
touch video_generator/__init__.py
touch video_generator/models/__init__.py
touch video_generator/pipeline/__init__.py

# Process the consolidated file to extract individual files
CONSOLIDATED_FILE="complete-video-pipeline.py"
CURRENT_FILE=""
CAPTURE=false

# Read the consolidated file line by line
while IFS= read -r line; do
    # Check if this line starts a new file
    if [[ $line == "# FILE:"* ]]; then
        # Close previous file if we were capturing
        if [ "$CAPTURE" = true ] && [ -n "$CURRENT_FILE" ]; then
            echo "Created $CURRENT_FILE"
            CAPTURE=false
        fi
        
        # Extract new filename
        CURRENT_FILE=$(echo "$line" | sed 's/# FILE://')
        
        # Create directory for file if needed
        mkdir -p "$(dirname "$CURRENT_FILE")"
        
        # Start capturing for new file
        CAPTURE=true
        continue
    fi
    
    # Skip file markers and empty lines after file markers
    if [[ $line == "###START_FILE###" ]] || [[ $line == "###END_FILE###" ]]; then
        continue
    fi
    
    # If we're capturing and have a current file, write the line
    if [ "$CAPTURE" = true ] && [ -n "$CURRENT_FILE" ]; then
        echo "$line" >> "$CURRENT_FILE"
    fi
done < "$CONSOLIDATED_FILE"

# Handle case where the last file in the document needs to be closed
if [ "$CAPTURE" = true ] && [ -n "$CURRENT_FILE" ]; then
    echo "Created $CURRENT_FILE"
fi

echo "Setup complete!"
