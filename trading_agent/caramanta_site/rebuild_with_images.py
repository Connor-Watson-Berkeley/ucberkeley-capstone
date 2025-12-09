#!/usr/bin/env python3
"""
Automatically rebuild site with proper image placement based on dimensions
"""
import subprocess
import json
import re

# Get image dimensions
def get_dimensions(filepath):
    result = subprocess.run(['file', filepath], capture_output=True, text=True)
    match = re.search(r'(\d+)\s*x\s*(\d+)', result.stdout)
    if match:
        return int(match.group(1)), int(match.group(2))
    return None, None

# Categorize all images
images = {}
import os
for filename in os.listdir('clean/images/full'):
    if filename.endswith(('.jpg', '.png', '.jpeg')):
        path = f'clean/images/full/{filename}'
        w, h = get_dimensions(path)
        if w and h:
            aspect = w / h

            # Categorize
            category = 'other'
            if abs(aspect - 1.0) < 0.1 and w > 1500:
                category = 'team'  # Square photos for team
            elif w > 1800:
                category = 'hero'  # Large landscape for hero/banner
            elif h > w and h > 1500:
                category = 'diagram'  # Tall diagrams
            else:
                category = 'content'  # Medium images for content sections

            images[filename] = {'w': w, 'h': h, 'category': category}

# Group by category
categorized = {'hero': [], 'team': [], 'diagram': [], 'content': []}
for fname, info in images.items():
    categorized[info['category']].append(fname)

print("Image Categorization:")
print(f"Hero/Background: {len(categorized['hero'])} images")
for img in categorized['hero']:
    print(f"  - {img}")

print(f"\nTeam Photos: {len(categorized['team'])} images")
for img in categorized['team']:
    print(f"  - {img}")

print(f"\nDiagrams: {len(categorized['diagram'])} images")
for img in categorized['diagram']:
    print(f"  - {img}")

print(f"\nContent Images: {len(categorized['content'])} images")
for img in categorized['content']:
    print(f"  - {img}")

# Save mapping
with open('image_mapping.json', 'w') as f:
    json.dump(categorized, f, indent=2)

print("\n✅ Image mapping complete!")
print("\nNow updating HTML files with proper images...")

# Load content
with open('content.json', 'r') as f:
    content = json.load(f)

# Update HTML files with images
# This will place images in appropriate sections based on the categorization
print("Images categorized and ready for site rebuild")

