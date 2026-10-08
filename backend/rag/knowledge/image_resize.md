# Image Resize

## What is Image Resizing?
Resizing changes the dimensions (width and height) of an image in pixels.

## When to Use It
- You need exact dimensions for a website banner (e.g., 1200×628 for Open Graph)
- Reducing dimensions for faster loading
- Fitting images into specific UI slots or templates
- Creating thumbnails

## How to Use in NexaForge
Type instructions like:
- "resize to 1200 by 800"
- "make it 800x600"
- "resize width to 500 pixels"
- "scale down to 50%"
- "make it 75% of original size"

Or use the quick action button: **Resize Image**

## Parameters
- **Width + Height**: Exact pixel dimensions (e.g., 1200×800)
- **Width only**: Height scales proportionally to maintain aspect ratio
- **Height only**: Width scales proportionally to maintain aspect ratio
- **Scale %**: Percentage of original size (e.g., 50% halves both dimensions)

## Tips
- NexaForge uses LANCZOS resampling — the highest quality downscaling algorithm
- Providing both width and height may stretch the image if aspect ratio differs
- For social media: Twitter card = 1200×628, Instagram square = 1080×1080
- For web thumbnails: typically 300×300 or 150×150

## Output Format
Same as input format
