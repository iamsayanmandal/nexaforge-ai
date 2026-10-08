# PDF Compression

## What is PDF Compression?
PDF compression reduces the file size of a PDF document, making it easier to email, upload, or store.

## When to Use It
- Email attachment size limit exceeded
- Website upload limit (e.g., resume portals limit to 2MB or 5MB)
- Slow PDF loading due to large embedded images
- Storage optimization

## How to Use in NexaForge
Type instructions like:
- "compress this PDF under 300KB"
- "make this PDF smaller than 1MB"
- "reduce PDF size to 500KB"

Or use the quick action button: **Compress PDF**

## How It Works
NexaForge uses a two-step strategy:
1. **Lossless compression**: Deflate compression + garbage collection removes unused objects
2. **Image downsampling**: If still too large, reduces quality of embedded images progressively (60% → 40% → 20%)

## Tips
- PDFs with many high-resolution images compress much more than text-only PDFs
- Text-heavy PDFs may already be near their minimum size
- Very aggressive targets (< 100KB for a 10-page image-heavy PDF) may reduce image quality
- For resumes: target 200-500KB (most job portals accept up to 5MB)

## Common Targets
| Use Case | Recommended Target |
|----------|--------------------|
| Email attachment | < 5MB |
| Job portal upload | < 2MB |
| Web embedding | < 500KB |
| Archiving | No limit needed |
