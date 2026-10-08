# PDF Split and Extract Pages

## What is PDF Splitting?
Splitting divides a PDF into a smaller PDF containing a specific range of pages.
Extracting lets you pick individual pages (not necessarily consecutive).

## When to Use Splitting
- Extract chapter 2 (pages 15–40) from a large book PDF
- Share only relevant sections of a report
- Remove unwanted pages from a document

## When to Use Extract Pages
- Get page 1, 5, and 10 from a 50-page document
- Extract non-consecutive pages

## How to Use in NexaForge

### Split (page range)
Type instructions like:
- "split pages 1 to 10"
- "extract pages 5 through 20"
- "get pages 3 to 7"

### Extract Specific Pages
Type instructions like:
- "extract pages 1, 3, 5"
- "get page 2 and page 8"
- "extract pages 1 2 4 6"

Or use the quick action button: **Split / Extract Pages**

## Tips
- Pages are 1-indexed (first page = page 1, not page 0)
- Split preserves all page formatting and embedded images
- To split into multiple chunks, process the PDF multiple times with different ranges
- For extract, you can specify up to 20 individual page numbers

## Common Use Cases
| Task | Operation | Example |
|------|-----------|---------|
| Extract first 5 pages | Split | pages 1 to 5 |
| Remove first page | Split | pages 2 to end |
| Extract cover + summary | Extract | pages 1 and 3 |
| Get appendix only | Split | pages 45 to 60 |
