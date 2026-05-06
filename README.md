# Suspicious Antwerp Scraper

A full scraper for the Suspicious Antwerp fashion store with embeddings.

## Setup

```bash
cd scraper-suspiciousantwerp
pip install -r requirements.txt
```

## Run

```bash
python scraper.py
```

Or test mode:
```bash
python scraper.py --test 5
```

## Files

- `scraper.py` - Main scraper
- `quick_import.py` - Incremental import for remaining products  
- `scraper_full.py` - Full smart incremental
- `scraper_incremental.py` - Legacy batch runner

## Database Results

- Total products: 518
- With image embeddings: 504+
- With info embeddings: 510+
- Source: `scraper-suspiciousantwerp`
- Brand: `Suspicious Antwerp`

## Fields

- `id`: `scraper-suspiciousantwerp-{product-slug}`
- `source`: `scraper-suspiciousantwerp`
- `brand`: `Suspicious Antwerp`
- `title`: Product name
- `price`: Price (CZK format)
- `image_url`: Main product image
- `additional_images`: Comma-separated additional images
- `image_embedding`: SigLIP image embedding
- `info_embedding`: SigLIP text embedding
- `gender`: unisex
- `category`: Product category
- `size`: Available sizes
- `second_hand`: false
- `country`: BE

Note: This scraper uses google/siglip-base-patch16-384 model. Embeddings are generated from image features but may vary in dimension.