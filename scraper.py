#!/usr/bin/env python3
"""
Suspicious Antwerp Scraper
Scrapes products from Suspicious Antwerp, generates embeddings, and imports to Supabase.
"""

import json
import os
import re
import time
from datetime import datetime
from typing import Optional
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from PIL import Image
from supabase import create_client, Client
import torch
from transformers import AutoProcessor, AutoModel
import io
import httpx
import warnings
warnings.filterwarnings("ignore")

load_dotenv()

BASE_URL = "https://www.suspiciousantwerp.com"
CATEGORY_URL = "https://www.suspiciousantwerp.com/collections/current-availabilities"

SUPABASE_URL = os.getenv("SUPABASE_URL", "https://yqawmzggcgpeyaaynrjk.supabase.co")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InlxYXdtemdnY2dwZXlhYXlucmprIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc1NTAxMDkyNiwiZXhwIjoyMDcwNTg2OTI2fQ.XtLpxausFriraFJeX27ZzsdQsFv3uQKXBBggoz6P4D4")

MODEL_NAME = "google/siglip-base-patch16-384"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Connection": "keep-alive",
    "Accept-Encoding": "identity",
}

def fetch_url(url: str, retries: int = 3) -> Optional[BeautifulSoup]:
    """Fetch and parse a URL, disabling content encoding."""
    for attempt in range(retries):
        try:
            response = requests.get(url, timeout=30, headers=HEADERS)
            response.raise_for_status()
            return BeautifulSoup(response.text, "html.parser")
        except Exception as e:
            print(f"Error fetching {url} (attempt {attempt + 1}/{retries}): {e}")
            time.sleep(2 ** attempt)
    return None


class EmbeddingModel:
    """SigLIP embedding model for image and text embeddings."""
    
    def __init__(self, model_name: str = MODEL_NAME):
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading embedding model: {model_name} on {self.device}")
        self.processor = AutoProcessor.from_pretrained(model_name, use_fast=False)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        print(f"Model loaded successfully")
    
    def get_image_embedding(self, image_url: str) -> Optional[list]:
        """Generate 768-dim embedding from an image URL."""
        try:
            response = httpx.get(image_url, timeout=30, follow_redirects=True)
            response.raise_for_status()
            image = Image.open(io.BytesIO(response.content)).convert("RGB")
            
            inputs = self.processor(images=image, return_tensors="pt")
            inputs["input_ids"] = torch.zeros(1, dtype=torch.long)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                embedding = outputs.image_embeds.float().squeeze().cpu().numpy()
            
            if embedding.ndim == 0:
                embedding = embedding.reshape(1)
            
            return embedding.tolist()
        except Exception as e:
            print(f"Error generating image embedding for {image_url}: {e}")
            return None
    
    def get_text_embedding(self, text: str) -> Optional[list]:
        """Generate text embedding from text description."""
        try:
            inputs = self.processor(text=[text], return_tensors="pt")
            
            with torch.no_grad():
                outputs = self.model.get_text_features(**inputs)
                embedding = outputs.pooler_output.squeeze().cpu().numpy()
            
            if embedding.ndim == 0:
                embedding = embedding.reshape(1)
            
            return embedding.tolist()
        except Exception as e:
            print(f"Error generating text embedding: {e}")
            return None


class SuspiciousAntwerpScraper:
    """Main scraper for Suspicious Antwerp store."""
    
    def __init__(self):
        self.embedding_model = EmbeddingModel()
        self.supabase: Optional[Client] = None
        self.stats = {"pages_scraped": 0, "products_found": 0, "products_imported": 0, "errors": 0}
    
    def connect_supabase(self):
        """Initialize Supabase connection."""
        print(f"Connecting to Supabase: {SUPABASE_URL}")
        self.supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        print("Connected to Supabase")
    
    def get_page(self, url: str, retries: int = 3) -> Optional[BeautifulSoup]:
        """Fetch and parse a page."""
        return fetch_url(url, retries)
    
    def get_all_product_links(self) -> list[str]:
        """Scrape all product URLs from all pagination pages."""
        all_products = []
        page = 1
        max_pages = 30
        max_products_per_page = 25
        min_products_threshold = 5
        
        while page <= max_pages:
            if page == 1:
                url = CATEGORY_URL
            else:
                url = f"{CATEGORY_URL}?page={page}"
            
            print(f"Scraping page {page}: {url}")
            soup = self.get_page(url)
            
            if not soup:
                print(f"Failed to fetch page {page}")
                break
            
            product_links = set()
            
            for a in soup.find_all("a", href=True):
                href = a.get("href", "")
                if "/products/" in href and "gift-card" not in href.lower():
                    full_url = urljoin(BASE_URL, href)
                    product_path = full_url.split("/products/")[-1]
                    if product_path and "?" not in product_path:
                        product_links.add(full_url)
            
            product_links = list(product_links)
            
            if not product_links:
                print(f"No products found on page {page}, stopping")
                break
            
            if page > 1 and len(product_links) < min_products_threshold:
                print(f"Only {len(product_links)} products (below threshold), stopping")
                break
            
            all_products.extend(product_links)
            print(f"Found {len(product_links)} products on page {page}")
            
            self.stats["pages_scraped"] += 1
            page += 1
            time.sleep(0.5)
        
        unique_products = list(set(all_products))
        self.stats["products_found"] = len(unique_products)
        return unique_products
    
    def parse_price(self, price_text: str) -> str:
        """Parse and format price with currency."""
        if not price_text:
            return ""
        
        price_text = price_text.strip()
        
        currencies = []
        
        czk_match = re.search(r"(\d[\d\s]*),\d{2}\s*Kč", price_text)
        if czk_match:
            czk_val = re.sub(r"\s", "", czk_match.group(1)).replace(",", ".")
            if czk_val:
                czk_clean = czk_val.split('.')[0]
                currencies.append(f"{czk_clean}CZK")
        
        usd_match = re.search(r"\$(\d+[\d,]*)", price_text)
        if usd_match:
            usd_val = usd_match.group(1).replace(",", "")
            if usd_val:
                currencies.append(f"{usd_val}USD")
        
        eur_match = re.search(r"€\s*(\d+[\d,]*)", price_text)
        if eur_match:
            eur_val = eur_match.group(1).replace(",", "")
            if eur_val:
                currencies.append(f"{eur_val}EUR")
        
        gbppound_match = re.search(r"£\s*(\d+[\d,]*)", price_text)
        if gbppound_match:
            gbp_val = gbppound_match.group(1).replace(",", "")
            if gbp_val:
                currencies.append(f"{gbp_val}GBP")
        
        pln_match = re.search(r"(\d+)\s*PLN", price_text, re.IGNORECASE)
        if pln_match:
            pln_val = pln_match.group(1)
            if pln_val:
                currencies.append(f"{pln_val}PLN")
        
        if not currencies:
            all_nums = re.findall(r"\d+[\d,]*\.?\d*", price_text)
            if all_nums:
                num = all_nums[0].replace(",", "").split('.')[0]
                if num:
                    currencies.append(f"{num}CZK")
        
        return ", ".join(currencies) if currencies else ""
    
    def extract_product_data(self, soup: BeautifulSoup, product_url: str) -> dict:
        """Extract detailed product information from product page."""
        product_id = product_url.split("/products/")[-1].split("?")[0]
        
        data = {
            "id": f"scraper-suspiciousantwerp-{product_id}",
            "source": "scraper-suspiciousantwerp",
            "product_url": product_url,
            "affiliate_url": None,
            "brand": "Suspicious Antwerp",
            "title": "",
            "description": "",
            "category": "",
            "gender": "unisex",
            "image_url": "",
            "additional_images": "",
            "price": "",
            "sale": None,
            "second_hand": False,
            "metadata": {},
            "size": "",
            "country": None,
            "tags": [],
            "created_at": datetime.utcnow().isoformat(),
        }
        
        try:
            h1 = soup.find("h1")
            if h1:
                data["title"] = h1.get_text(strip=True)
            
            if not data["title"] and soup.title:
                title_str = soup.title.string or ""
                if title_str:
                    data["title"] = title_str.split(" – ")[0].split("-")[0].strip()
            
            script_tag = soup.find("script", type="application/ld+json")
            if script_tag:
                try:
                    json_ld = json.loads(script_tag.string)
                    if isinstance(json_ld, dict):
                        if json_ld.get("@type") == "Product":
                            data["description"] = json_ld.get("description", "")
                            offer = json_ld.get("offers", {})
                            if offer:
                                price = offer.get("price", "")
                                currency = offer.get("priceCurrency", "")
                                if price:
                                    if currency == "CZK":
                                        data["price"] = f"{price}CZK"
                                    elif currency == "USD":
                                        data["price"] = f"{price}USD"
                                    elif currency == "EUR":
                                        data["price"] = f"{price}EUR"
                except:
                    pass
            
            html_text = str(soup)
            
            price_patterns = [
                r'(\d{1,3}(?:\s?\d{3})*,?\d{2})\s*Kč',
                r'(\d+[\d,]*)\s*€',
                r'€\s*(\d+[\d,]*)',
                r'\$(\d+[\d,]*)',
            ]
            for pattern in price_patterns:
                match = re.search(pattern, html_text)
                if match:
                    raw_price = match.group(1).replace(" ", "").replace(",", ".")
                    data["price"] = self.parse_price(raw_price + " CZK")
                    break
            
            if not data["price"]:
                price_match = re.search(r'([\d,\s]+)\s*Kč', html_text)
                if price_match:
                    data["price"] = self.parse_price(price_match.group(1) + " CZK")
            
            img_tags = soup.find_all("img")
            image_urls = []
            brand_exclude = ["Slanted", "logo", "brand", "icon", "favicon"]
            
            for img in img_tags:
                src = img.get("src") or img.get("data-src")
                if src and "cdn/shop/files" in src:
                    is_brand = any(ex.lower() in src.lower() for ex in brand_exclude)
                    if is_brand:
                        continue
                    
                    src = re.sub(r'\?.*', '', src)
                    if src.startswith("//"):
                        src = "https:" + src
                    elif src.startswith("/"):
                        src = BASE_URL + src
                    
                    if src not in image_urls and len(src) > 50:
                        image_urls.append(src)
            
            for meta in soup.find_all("meta", property=re.compile(r"og:image")):
                src = meta.get("content")
                if src and "Slanted" not in src and "logo" not in src.lower():
                    if src.startswith("//"):
                        src = "https:" + src
                    elif src.startswith("/"):
                        src = BASE_URL + src
                    if src not in image_urls:
                        image_urls.append(src)
            
            if image_urls:
                data["image_url"] = image_urls[0]
                data["additional_images"] = ", ".join(image_urls[1:7]) if len(image_urls) > 1 else ""
            
            size_patterns = soup.find_all(["span", "div", "button"])
            sizes_found = set()
            for elem in size_patterns:
                text = elem.get_text(strip=True)
                if re.match(r"^(XXS|XS|S|M|L|XL|XXL|One size|S\/M|M\/L|36|37|38|39|40|41|42|43|44|45|46)$", text):
                    sizes_found.add(text)
            if sizes_found:
                data["size"] = ", ".join(sorted(sizes_found, key=lambda x: {"XXS": 0, "XS": 1, "S": 2, "M": 3, "M/L": 4, "L": 5, "XL": 6, "XXL": 7}.get(x, 99)))
            
            category_links = []
            for a in soup.find_all("a", href=True):
                href = a.get("href", "")
                text = a.get_text(strip=True)
                if "/collections/" in href and "current-availabilities" not in href and text and len(text) < 30:
                    category_links.append(text)
            category_links = list(dict.fromkeys(category_links))
            if category_links:
                data["category"] = ", ".join(category_links[:3])
            
            meta_desc = soup.find("meta", attrs={"name": "description"})
            if meta_desc and not data["description"]:
                data["description"] = meta_desc.get("content", "")[:500] or ""
            
            product_info = {}
            
            fit_match = re.search(r"(Oversized Fit|Boxy Fit|Regular Fit|Baggy Fit|One Size)", html_text, re.I)
            if fit_match:
                product_info["fit"] = fit_match.group(1)
            
            material_patterns = [
                (r'(\d+%)\s*(Cotton|Polyester|Organic|Viscose|Linen)', 'material'),
                (r'(\d+g/?m²)', 'weight'),
            ]
            for pattern, key in material_patterns:
                match = re.search(pattern, html_text, re.IGNORECASE)
                if match:
                    product_info[key] = match.group(1)
            
            data["metadata"] = json.dumps(product_info, ensure_ascii=False) if product_info else "{}"
            
        except Exception as e:
            print(f"Error extracting product data: {e}")
            self.stats["errors"] += 1
        
        return data
    
    def generate_embeddings(self, product_data: dict) -> dict:
        """Generate image and text embeddings for product."""
        image_embedding = None
        info_embedding = None
        
        if product_data.get("image_url"):
            image_embedding = self.embedding_model.get_image_embedding(product_data["image_url"])
        
        if not image_embedding:
            print(f"  Warning: No image embedding for {product_data.get('title', 'unknown')[:30]}")
        
        info_components = [
            product_data.get("title", ""),
            product_data.get("category", ""),
            product_data.get("description", ""),
            product_data.get("metadata", ""),
            product_data.get("price", ""),
        ]
        if product_data.get("gender"):
            info_components.append(product_data.get("gender", ""))
        
        info_text = " ".join(filter(None, info_components))
        
        if info_text.strip():
            info_embedding = self.embedding_model.get_text_embedding(info_text.strip())
        
        product_data["image_embedding"] = image_embedding
        product_data["info_embedding"] = info_embedding
        
        return product_data
    
    def import_to_supabase(self, products: list[dict]) -> int:
        """Import products to Supabase database."""
        if not self.supabase:
            self.connect_supabase()
        
        imported = 0
        failed = 0
        
        for i, product in enumerate(products):
            try:
                record = {
                    "id": product["id"],
                    "source": product["source"],
                    "product_url": product["product_url"],
                    "affiliate_url": product.get("affiliate_url"),
                    "image_url": product["image_url"],
                    "brand": product["brand"],
                    "title": product["title"],
                    "description": product.get("description"),
                    "category": product.get("category"),
                    "gender": product.get("gender"),
                    "created_at": product.get("created_at"),
                    "metadata": product.get("metadata"),
                    "size": product.get("size"),
                    "second_hand": product.get("second_hand", False),
                    "country": product.get("country"),
                    "additional_images": product.get("additional_images"),
                    "tags": product.get("tags"),
                    "price": product.get("price"),
                    "sale": product.get("sale"),
                    "image_embedding": product.get("image_embedding"),
                    "info_embedding": product.get("info_embedding"),
                }
                
                record = {k: v for k, v in record.items() if v not in [None, "", [], {}]}
                
                response = self.supabase.table("products").upsert(
                    record,
                    on_conflict="source,product_url"
                ).execute()
                
                imported += 1
                if imported % 10 == 0:
                    print(f"Imported {imported} products...")
                
            except Exception as e:
                print(f"Error importing product {product.get('title', 'unknown')[:30]}: {e}")
                failed += 1
                self.stats["errors"] += 1
        
        print(f"Imported {imported} products, {failed} failed")
        return imported
    
    def run(self, test_mode: bool = False, max_products: int = 20):
        """Main execution function.
        
        Args:
            test_mode: If True, only process max_products items
            max_products: Maximum number of products to process in test mode
        """
        print("=" * 60)
        print("Suspicious Antwerp Scraper Starting")
        print("=" * 60)
        
        print("\n1. Connecting to Supabase...")
        self.connect_supabase()
        
        print("\n2. Scraping all product URLs from listing...")
        product_urls = self.get_all_product_links()
        
        print(f"\nTotal unique products found: {len(product_urls)}")
        
        if not product_urls:
            print("No products found, exiting")
            return
        
        if test_mode:
            product_urls = product_urls[:max_products]
            print(f"Test mode: limited to {max_products} products")
        
        print("\n3. Scraping individual product details...")
        products_data = []
        
        for i, url in enumerate(product_urls):
            print(f"Processing product {i+1}/{len(product_urls)}: {url}")
            
            soup = self.get_page(url)
            if soup:
                data = self.extract_product_data(soup, url)
                products_data.append(data)
            
            time.sleep(0.3)
        
        print(f"\n4. Generating embeddings for {len(products_data)} products...")
        
        for i, product in enumerate(products_data):
            print(f"Embedding {i+1}/{len(products_data)}: {product.get('title', 'unknown')[:40]}")
            product = self.generate_embeddings(product)
            time.sleep(0.2)
            
            if i > 0 and i % 15 == 0:
                print(f"  Pausing to avoid rate limits...")
                time.sleep(3)
        
        print("\n5. Importing to Supabase...")
        imported = self.import_to_supabase(products_data)
        
        self.stats["products_imported"] = imported
        
        print("\n" + "=" * 60)
        print("Scraping Complete!")
        print("=" * 60)
        print(f"Pages scraped: {self.stats['pages_scraped']}")
        print(f"Products found: {self.stats['products_found']}")
        print(f"Products imported: {self.stats['products_imported']}")
        print(f"Errors: {self.stats['errors']}")
        print("=" * 60)
        
        return self.stats


def main():
    """Entry point."""
    scraper = SuspiciousAntwerpScraper()
    scraper.run()


if __name__ == "__main__":
    main()