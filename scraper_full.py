#!/usr/bin/env python3
"""
Incremental scraper for Suspicious Antwerp
Properly skips already imported products.
"""

from scraper import SuspiciousAntwerpScraper
import time

def run_full():
    """Run full scrape in batches of 50."""
    scraper = SuspiciousAntwerpScraper()
    scraper.connect_supabase()
    
    # Get all product URLs from website
    print("Getting all product URLs from website...")
    all_urls = scraper.get_all_product_links()
    total = len(all_urls)
    print(f"Total on website: {total}")
    
    # Get existing URLs from DB
    print("Getting existing product URLs from DB...")
    result = scraper.supabase.table("products").select("product_url").eq("source", "scraper-suspiciousantwerp").execute()
    existing_urls = set(p.get("product_url") for p in result.data if p.get("product_url"))
    print(f"Already in DB: {len(existing_urls)}")
    
    # Filter to only new URLs
    new_urls = [url for url in all_urls if url not in existing_urls]
    print(f"Need to import: {len(new_urls)}")
    
    if not new_urls:
        print("All products already imported!")
        return
    
    # Process in batches
    batch_size = 50
    for batch_start in range(0, len(new_urls), batch_size):
        batch_urls = new_urls[batch_start:batch_start+batch_size]
        print(f"\nBatch {batch_start//batch_size + 1}: {len(batch_urls)} products")
        
        products_data = []
        for i, url in enumerate(batch_urls):
            print(f"Processing {i+1}/{len(batch_urls)}: {url}")
            soup = scraper.get_page(url)
            if soup:
                data = scraper.extract_product_data(soup, url)
                products_data.append(data)
            time.sleep(0.2)
        
        print(f"Generating embeddings...")
        for i, product in enumerate(products_data):
            print(f"Embedding {i+1}/{len(products_data)}...")
            product = scraper.generate_embeddings(product)
            time.sleep(0.2)
            if i > 0 and i % 15 == 0:
                time.sleep(3)
        
        print("Importing to Supabase...")
        imported = scraper.import_to_supabase(products_data)
        print(f"Batch {batch_start//batch_size + 1} complete: {imported} imported")
        
        time.sleep(2)
    
    # Final count
    result = scraper.supabase.table("products").select("id", count="exact").eq("source", "scraper-suspiciousantwerp").execute()
    print(f"\nTotal in DB: {result.count}")


if __name__ == "__main__":
    run_full()