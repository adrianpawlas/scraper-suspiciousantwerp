#!/usr/bin/env python3
"""
Incremental scraper for Suspicious Antwerp
Run in batches to import all products incrementally.
"""

from scraper import SuspiciousAntwerpScraper, fetch_url, urljoin, BASE_URL, CATEGORY_URL, BeautifulSoup
import time
import re

def scrape_batch(batch_start=0, batch_size=50):
    """Scrape a batch of products."""
    scraper = SuspiciousAntwerpScraper()
    
    # Get all product URLs
    print("Getting product URLs...")
    product_urls = scraper.get_all_product_links()
    
    # Skip already scraped (based on batch_start)
    urls_to_process = product_urls[batch_start:batch_start+batch_size]
    
    print(f"Processing {len(urls_to_process)} products (batch {batch_start}-{batch_start+len(urls_to_process)})...")
    
    products_data = []
    for i, url in enumerate(urls_to_process):
        print(f"Product {i+1}/{len(urls_to_process)}: {url}")
        
        soup = scraper.get_page(url)
        if soup:
            data = scraper.extract_product_data(soup, url)
            products_data.append(data)
        
        time.sleep(0.2)
    
    print(f"\nGenerating embeddings for {len(products_data)} products...")
    
    for i, product in enumerate(products_data):
        print(f"Embedding {i+1}/{len(products_data)}: {product.get('title', 'unknown')[:40]}")
        product = scraper.generate_embeddings(product)
        time.sleep(0.2)
        
        if i > 0 and i % 15 == 0:
            print(f"  Pausing to avoid rate limits...")
            time.sleep(3)
    
    print("\nImporting to Supabase...")
    imported = scraper.import_to_supabase(products_data)
    
    return imported


def run_incremental():
    """Run incremental scrape in batches of 50."""
    scraper = SuspiciousAntwerpScraper()
    scraper.connect_supabase()
    
    # Get product count
    product_urls = scraper.get_all_product_links()
    total = len(product_urls)
    
    # Get current DB count
    result = scraper.supabase.table("products").select("id", count="exact").eq("source", "scraper-suspiciousantwerp").execute()
    current = result.count or 0
    
    print(f"Total products: {total}")
    print(f"Currently in DB: {current}")
    print(f"Need to import: {total - current}")
    
    if current >= total:
        print("All products already imported!")
        return
    
    batch_start = current
    batch_size = 50
    imported_total = 0
    
    while batch_start < total:
        print(f"\n{'='*50}")
        print(f"Processing batch starting at {batch_start}")
        print(f"{'='*50}")
        
        imported = scrape_batch(batch_start, batch_size)
        imported_total += imported
        
        batch_start += batch_size
        time.sleep(2)
    
    print(f"\n{'='*50}")
    print(f"Total imported this run: {imported_total}")
    print(f"{'='*50}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Run specific batch
        batch = int(sys.argv[1])
        size = int(sys.argv[2]) if len(sys.argv) > 2 else 50
        scrape_batch(batch, size)
    else:
        # Run incremental
        run_incremental()