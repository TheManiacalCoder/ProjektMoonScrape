#!/bin/bash

# Set the new repository URL
NEW_REPO="https://github.com/TheManiacalCoder/MoonScrapeVerBert.git"

# Initialize Git repository if not already initialized
if [ ! -d ".git" ]; then
    echo "Initializing new Git repository..."
    git init
fi

# Add all files to staging
echo "Adding files to staging..."
git add .

# Commit changes
echo "Committing changes..."
git commit -m "Initial commit with BERT implementation"

# Set the new remote repository
echo "Setting new remote repository..."
git remote add origin $NEW_REPO

# Verify remote
echo "Verifying remote..."
git remote -v

# Push to the new repository
echo "Pushing to new repository..."
git push -u origin main

# Check if push was successful
if [ $? -eq 0 ]; then
    echo "Successfully pushed to new repository!"
else
    echo "Push failed. Please check for errors."
fi 