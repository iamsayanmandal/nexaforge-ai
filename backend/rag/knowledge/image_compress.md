# Image Compression

## What is Image Compression?
Image compression reduces the file size of an image while maintaining acceptable visual quality.

## When to Use It
- Your image is too large to upload to a website or email
- You need to reduce image size under a specific limit (e.g., under 100KB for profile photos)
- You want to speed up web page loading times
- Social media or form uploads reject your file for being too large

## How to Use in NexaForge
Type instructions like:
- "compress this image under 100KB"
- "make this image less than 200KB"
- "reduce file size to 50KB"

Or use the quick action button: **Compress Image**

## How It Works
NexaForge uses a binary search algorithm on JPEG quality (1–95) to find the highest quality that fits within your target size. This ensures maximum quality at the target file size.

## Tips
- JPEG format is best for compression (smallest file size)
- For logos and graphics with transparency, use PNG instead
- Very small targets (< 20KB) may result in visible quality loss
- Maximum supported target: any size in KB

## Supported Input Formats
JPG, JPEG, PNG, WEBP, BMP, TIFF

## Output Format
Always JPEG (best compression ratio)
