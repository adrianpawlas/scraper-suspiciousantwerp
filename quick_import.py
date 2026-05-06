#!/usr/bin/env python3
"""
Quick batch runner to import remaining products.
"""

from scraper import SuspiciousAntwerpScraper
import time

scraper = SuspiciousAntwerpScraper()
scraper.connect_supabase()

print("Getting all product URLs...")
all_urls = scraper.get_all_product_links()
total = len(all_urls)

result = scraper.supabase.table("products").select("product_url").eq("source", "scraper-suspiciousantwerp").execute()
existing_urls = set(p.get("product_url") for p in result.data if p.get("product_url"))

new_urls = [url for url in all_urls if url not in existing_urls]
print(f"Need to import: {len(new_urls)} / {total}")

if not new_urls:
    print("All done!")
    exit()

batch_size = 30
imported = 0

for i in range(0, len(new_urls), batch_size):
    batch = new_urls[i:i+batch_size]
    print(f"\nBatch {i//batch_size + 1}: {len(batch)} products")
    
    products_data = []
    for j, url in enumerate(batch):
        print(f"  {j+1}/{len(batch)}")
        soup = scraper.get_page(url)
        if soup:
            data = scraper.extract_product_data(soup, url)
            products_data.append(data)
        time.sleep(0.15)
    
    print("  Embedding...")
    for j, product in enumerate(products_data):
        product = scraper.generate_embeddings(product)
        time.sleep(0.15)
        if j > 0 and j % 10 == 0:
            time.sleep(2)
    
    print("  Importing...")
    imp = scraper.import_to_supabase(products_data)
    imported += imp
    print(f"  Imported: {imp}")
    
    time.sleep(2)

print(f"\nTotal imported this run: {imported}")
result = scraper.supabase.table("products").select("id", count="exact").eq("source", "scraper-suspiciousantwerp").execute()
print(f"Total in DB: {result.count}")