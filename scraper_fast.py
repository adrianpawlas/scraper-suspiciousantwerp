#!/usr/bin/env python3
"""Suspicious Antwerp Scraper - Fast version with smart batch processing"""

import os
import re
import json
import time
import logging
from datetime import datetime
from typing import Optional, Set
from urllib.parse import urljoin

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from bs4 import BeautifulSoup
import torch
from PIL import Image
import io
from transformers import AutoModel, AutoProcessor
from supabase import create_client

SUPABASE_URL = os.environ.get("SUPABASE_URL", "https://yqawmzggcgpeyaaynrjk.supabase.co")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InlxYXdtemdnY2dwZXlhYXlucmprIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc1NTAxMDkyNiwiZXhwIjoyMDcwNTg2OTI2fQ.XtLpxausFriraFJeX27ZzsdQsFv3uQKXBBggoz6P4D4")
SOURCE = "scraper-suspiciousantwerp"
BRAND = "Suspicious Antwerp"

CATEGORY_URL = "https://www.suspiciousantwerp.com/collections/current-availabilities"

SESSION = requests.Session()
retries = Retry(total=3, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
SESSION.mount('http://', HTTPAdapter(max_retries=retries))
SESSION.mount('https://', HTTPAdapter(max_retries=retries))
SESSION.headers.update({
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
    "Accept": "text/html,application/xhtml+xml",
    "Accept-Language": "en-US,en;q=0.9",
})

BATCH_SIZE = 30
EMBED_DELAY = 0.2
MAX_RETRIES = 3

logging.basicConfig(filename="scraper.log", level=logging.INFO)


class SigLIPEmbedder:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading SigLIP on {self.device}")
        self.model = AutoModel.from_pretrained("google/siglip-base-patch16-384", trust_remote_code=True).to(self.device)
        self.processor = AutoProcessor.from_pretrained("google/siglip-base-patch16-384")
        self.model.eval()
    
    def _normalize(self, emb):
        return emb / (torch.sqrt((emb ** 2).sum(-1, keepdim=True)) + 1e-8)
    
    def embed_image(self, url: str) -> Optional[list]:
        for attempt in range(MAX_RETRIES):
            try:
                r = SESSION.get(url, timeout=20)
                r.raise_for_status()
                img = Image.open(io.BytesIO(r.content)).convert("RGB")
                inp = self.processor(images=img, return_tensors="pt")
                inp = {k: v.to(self.device) for k, v in inp.items()}
                with torch.no_grad():
                    out = self.model.get_image_features(**inp)
                    emb = self._normalize(out.pooler_output)
                return emb.squeeze().cpu().numpy().tolist()
            except Exception as e:
                if attempt == MAX_RETRIES - 1:
                    print(f"Img error: {e}")
                    return None
                time.sleep(1)
                continue
    
    def embed_text(self, text: str) -> Optional[list]:
        try:
            text = text[:500]
            inp = self.processor(text=text, return_tensors="pt")
            inp = {k: v.to(self.device) for k, v in inp.items()}
            with torch.no_grad():
                out = self.model.get_text_features(**inp)
                emb = self._normalize(out.pooler_output)
            return emb.squeeze().cpu().numpy().tolist()
        except Exception as e:
            print(f"Txt error: {e}")
            return None


def get_urls() -> list:
    urls = []
    page = 1
    
    while page <= 25:
        url = CATEGORY_URL if page == 1 else f"{CATEGORY_URL}?page={page}"
        
        try:
            r = SESSION.get(url, timeout=20)
            if r.status_code != 200:
                break
        except Exception as e:
            print(f"  Failed page {page}: {e}")
            break
        
        soup = BeautifulSoup(r.text, "html.parser")
        
        product_links = set()
        for a in soup.find_all("a", href=True):
            href = a.get("href", "")
            if "/products/" in href and "gift-card" not in href.lower():
                full_url = urljoin("https://www.suspiciousantwerp.com", href)
                product_path = full_url.split("/products/")[-1]
                if product_path and "?" not in product_path:
                    product_links.add(full_url)
        
        product_links = list(product_links)
        
        if not product_links:
            break
        
        if page > 1 and len(product_links) < 5:
            break
        
        urls.extend(product_links)
        print(f"  Page {page}: {len(product_links)} URLs (total: {len(urls)})")
        page += 1
        time.sleep(0.3)
    
    return list(set(urls))


def extract(url: str) -> Optional[dict]:
    for attempt in range(MAX_RETRIES):
        try:
            r = SESSION.get(url, timeout=20)
            if r.status_code != 200:
                return None
            break
        except Exception as e:
            if attempt == MAX_RETRIES - 1:
                print(f"  Failed to fetch: {url}")
                return None
            time.sleep(2)
            continue
    
    html = r.text
    soup = BeautifulSoup(html, "html.parser")
    
    title = ""
    h1 = soup.find("h1")
    if h1:
        title = h1.get_text(strip=True)
    
    if not title and soup.title:
        title_str = soup.title.string or ""
        if title_str:
            title = title_str.split(" – ")[0].split("-")[0].strip()
    
    price_str = ""
    czk_match = re.search(r"(\d[\d\s]*),\d{2}\s*Kč", html)
    if czk_match:
        czk_val = re.sub(r"\s", "", czk_match.group(1))
        price_str = f"{czk_val}CZK"
    
    image_urls = []
    for img in soup.find_all("img"):
        src = img.get("src") or img.get("data-src")
        if src and "cdn/shop/files" in src:
            if any(ex.lower() in src.lower() for ex in ["Slanted", "logo", "brand", "icon"]):
                continue
            src = re.sub(r'\?.*', '', src)
            if src.startswith("//"):
                src = "https:" + src
            elif src.startswith("/"):
                src = "https://www.suspiciousantwerp.com" + src
            if src not in image_urls and len(src) > 50:
                image_urls.append(src)
    
    main_image = image_urls[0] if image_urls else None
    additional_images = ", ".join(image_urls[1:7]) if len(image_urls) > 1 else ""
    
    size_patterns = soup.find_all(["span", "div", "button"])
    sizes_found = set()
    for elem in size_patterns:
        text = elem.get_text(strip=True)
        if re.match(r"^(XXS|XS|S|M|L|XL|XXL|One size|S\/M|M\/L|36|37|38|39|40|41|42|43|44|45|46)$", text):
            sizes_found.add(text)
    sizes = ", ".join(sorted(sizes_found)) if sizes_found else ""
    
    category_links = []
    for a in soup.find_all("a", href=True):
        href = a.get("href", "")
        text = a.get_text(strip=True)
        if "/collections/" in href and "current-availabilities" not in href and text and len(text) < 30:
            category_links.append(text)
    category_links = list(dict.fromkeys(category_links))
    category = ", ".join(category_links[:3]) if category_links else ""
    
    description = ""
    meta_desc = soup.find("meta", attrs={"name": "description"})
    if meta_desc:
        description = meta_desc.get("content", "")[:500] or ""
    
    fit_match = re.search(r"(Oversized Fit|Boxy Fit|Regular Fit|Baggy Fit|One Size)", html, re.I)
    fit = fit_match.group(1) if fit_match else ""
    
    handle = url.split("/products/")[-1]
    
    metadata = json.dumps({
        'title': title, 'description': description, 'vendor': BRAND,
        'category': category, 'sizes': sizes, 'fit': fit,
    })
    
    return {
        'id': f"{SOURCE}-{handle}",
        'source': SOURCE,
        'product_url': url,
        'image_url': main_image,
        'brand': BRAND,
        'title': title,
        'description': description,
        'category': category,
        'gender': 'unisex',
        'metadata': metadata,
        'size': sizes,
        'price': price_str,
        'additional_images': additional_images,
        'second_hand': False,
        'country': None,
    }


def get_existing_products(supabase) -> dict:
    existing = {}
    try:
        result = supabase.table('products').select('id, product_url, title, image_url, price, created_at').eq('source', SOURCE).execute()
        for p in result.data:
            existing[p['product_url']] = p
    except Exception as e:
        print(f"Error fetching existing: {e}")
    return existing


def batch_upsert(supabase, products: list) -> dict:
    results = {'success': 0, 'failed': 0}
    
    for i in range(0, len(products), BATCH_SIZE):
        batch = products[i:i + BATCH_SIZE]
        retry_count = 0
        
        while retry_count < MAX_RETRIES:
            try:
                data = []
                for p in batch:
                    record = {
                        'id': p['id'],
                        'source': p['source'],
                        'product_url': p['product_url'],
                        'image_url': p['image_url'],
                        'brand': p['brand'],
                        'title': p['title'],
                        'description': p['description'],
                        'category': p['category'],
                        'gender': p['gender'],
                        'metadata': p['metadata'],
                        'size': p['size'],
                        'second_hand': p['second_hand'],
                        'image_embedding': p.get('image_embedding'),
                        'country': p['country'],
                        'additional_images': p.get('additional_images'),
                        'price': p['price'],
                        'info_embedding': p.get('info_embedding'),
                    }
                    data.append(record)
                
                supabase.table('products').upsert(data, on_conflict='id').execute()
                results['success'] += len(batch)
                break
                
            except Exception as e:
                retry_count += 1
                if retry_count >= MAX_RETRIES:
                    print(f"Batch failed: {e}")
                    results['failed'] += len(batch)
                else:
                    time.sleep(1)
    
    return results


def delete_stale_products(supabase, seen_urls: Set[str]) -> int:
    deleted = 0
    try:
        result = supabase.table('products').select('id, product_url').eq('source', SOURCE).execute()
        
        for p in result.data:
            if p['product_url'] not in seen_urls:
                supabase.table('products').delete().eq('id', p['id']).execute()
                deleted += 1
                print(f"Deleted stale: {p['id']}")
        
    except Exception as e:
        print(f"Error deleting stale: {e}")
    
    return deleted


def check_changed(existing_product: dict, new_product: dict) -> bool:
    if not existing_product:
        return True
    
    if existing_product.get('title') != new_product.get('title'):
        return True
    if existing_product.get('image_url') != new_product.get('image_url'):
        return True
    if existing_product.get('price') != new_product.get('price'):
        return True
    
    return False


def main():
    embedder = SigLIPEmbedder()
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    
    print("\n=== Fetching existing products ===")
    existing = get_existing_products(supabase)
    print(f"Found {len(existing)} existing products")
    
    print("\n=== Getting product URLs ===")
    urls = get_urls()
    print(f"Found {len(urls)} URLs")
    
    all_products = []
    seen_urls = set()
    
    print("\n=== Extracting products ===")
    for i, url in enumerate(urls):
        print(f"  {i+1}/{len(urls)}: {url}")
        
        p = extract(url)
        if p:
            all_products.append(p)
            seen_urls.add(url)
        
        time.sleep(0.2)
    
    print(f"\nTotal extracted: {len(all_products)}")
    
    new_count = 0
    updated_count = 0
    unchanged_count = 0
    
    products_to_insert = []
    
    print("\n=== Generating embeddings ===")
    for i, p in enumerate(all_products):
        existing_p = existing.get(p['product_url'])
        
        if not existing_p:
            print(f"  {i+1}: NEW - {p['title'][:40]}")
            new_count += 1
            generate_embeddings = True
        elif check_changed(existing_p, p):
            print(f"  {i+1}: CHANGED - {p['title'][:40]}")
            updated_count += 1
            generate_embeddings = True
        else:
            print(f"  {i+1}: UNCHANGED - {p['title'][:40]}")
            unchanged_count += 1
            generate_embeddings = False
            p['image_embedding'] = None
            p['info_embedding'] = None
        
        if generate_embeddings:
            if p.get('image_url'):
                p['image_embedding'] = embedder.embed_image(p['image_url'])
                time.sleep(EMBED_DELAY)
            
            txt = f"{p.get('title', '')} {p.get('category', '')} {p.get('description', '')} {p.get('price', '')} {p.get('gender', '')}"
            p['info_embedding'] = embedder.embed_text(txt)
            time.sleep(EMBED_DELAY)
        
        products_to_insert.append(p)
        
        if len(products_to_insert) >= BATCH_SIZE:
            print(f"\n  Inserting batch of {len(products_to_insert)}...")
            result = batch_upsert(supabase, products_to_insert)
            products_to_insert = []
    
    if products_to_insert:
        print(f"\n  Inserting final batch of {len(products_to_insert)}...")
        result = batch_upsert(supabase, products_to_insert)
    
    print("\n=== Deleting stale products ===")
    stale_deleted = delete_stale_products(supabase, seen_urls)
    
    print("\n" + "="*50)
    print("RUN SUMMARY")
    print("="*50)
    print(f"New products added:   {new_count}")
    print(f"Products updated:     {updated_count}")
    print(f"Unchanged:           {unchanged_count}")
    print(f"Stale deleted:       {stale_deleted}")
    print("="*50)


if __name__ == "__main__":
    main()