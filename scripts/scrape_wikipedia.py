import os
import re
import time
import requests
import tqdm
from pathlib import Path

# Adjust python path to be able to import src
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# pyrefly: ignore [missing-import]
from src.data.string_generator import parse_with_lark

API_URL = "https://en.wikipedia.org/w/api.php"

def get_math_articles(session: requests.Session) -> list[str]:
    print("Searching Wikipedia for math articles...")
    keywords = [
        "theorem", "algebra", "calculus", "geometry", "topology",
        "matrix mathematics", "polynomial", "integral", "derivative",
        "probability", "statistics", "differential equation", "tensor",
        "vector space", "manifold", "graph theory", "number theory",
        "combinatorics", "complex analysis", "real analysis", "linear algebra"
    ]
    
    titles = set()
    for kw in tqdm.tqdm(keywords, desc="Querying topics"):
        params = {
            "action": "query",
            "format": "json",
            "list": "search",
            "srsearch": kw,
            "srlimit": 500,  # max allowed for normal users
            "srnamespace": 0 # only articles
        }
        try:
            resp = session.get(API_URL, params=params, timeout=10).json()
            if 'query' in resp and 'search' in resp['query']:
                for item in resp['query']['search']:
                    titles.add(item['title'])
        except Exception as e:
            print(f"Error searching for {kw}: {e}")
        time.sleep(0.5) # polite delay
        
    titles_list = list(titles)
    print(f"Found {len(titles_list)} unique math-related articles.")
    return titles_list

def extract_math_tags(wikitext: str) -> list[str]:
    # Wikipedia uses <math>...</math> tags for LaTeX rendering.
    # Sometimes it has attributes like <math display="block">
    matches = re.findall(r'<math[^>]*>(.*?)</math>', wikitext, re.DOTALL | re.IGNORECASE)
    return [m.strip() for m in matches if m.strip()]

def main():
    out_dir = Path("data/scraped")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("ScanTeX Wikipedia Corpus Scraper")
    print("=" * 60)
    
    session = requests.Session()
    session.headers.update({
        "User-Agent": "ScanTex_Dataset_Builder/1.0 (mailto:college_project@example.com)"
    })
    
    articles = get_math_articles(session)
    
    target_count = 20000
    all_equations = set()
    
    pbar = tqdm.tqdm(total=target_count, desc="Extracting equations", unit="eq")
    
    # We fetch articles in batches of 50 (max allowed by API for page queries)
    batch_size = 50
    for i in range(0, len(articles), batch_size):
        if len(all_equations) >= target_count:
            break
            
        batch_titles = articles[i:i+batch_size]
        titles_str = "|".join(batch_titles)
        
        params = {
            "action": "query",
            "format": "json",
            "prop": "revisions",
            "rvprop": "content",
            "rvslots": "main",
            "titles": titles_str
        }
        
        try:
            resp = session.post(API_URL, data=params, timeout=15).json()
            if 'query' in resp and 'pages' in resp['query']:
                for page_id, page_data in resp['query']['pages'].items():
                    if 'revisions' in page_data:
                        content = page_data['revisions'][0]['slots']['main']['*']
                        eqs = extract_math_tags(content)
                        
                        before = len(all_equations)
                        
                        for raw_eq in eqs:
                            if raw_eq not in all_equations:
                                # Strip standard formatting tags that Lark shouldn't care about
                                eq = re.sub(r'\\begin\{[^}]+\}', '', raw_eq)
                                eq = re.sub(r'\\end\{[^}]+\}', '', eq)
                                eq = re.sub(r'\\label\{[^}]+\}', '', eq)
                                eq = re.sub(r'\\text\{[^}]+\}', '', eq) # Wikipedia sometimes uses text which can break strict parsing
                                eq = re.sub(r'\\displaystyle', '', eq)
                                eq = re.sub(r'\\quad', '', eq)
                                eq = re.sub(r'\\qquad', '', eq)
                                eq = re.sub(r'\\,', '', eq)
                                eq = re.sub(r'\\;', '', eq)
                                eq = re.sub(r'\\!', '', eq)
                                
                                # Validate with Lark
                                tree = parse_with_lark(eq)
                                if tree is not None:
                                    all_equations.add(eq)
                                    if len(all_equations) >= target_count:
                                        break
                                        
                        added = len(all_equations) - before
                        if added > 0:
                            current = min(len(all_equations), target_count)
                            pbar.update(current - pbar.n)
                            
        except Exception as e:
            # Silently catch network errors or parsing glitches and move to next batch
            time.sleep(1)
            continue
            
        # polite delay between batch requests
        time.sleep(0.5)
        
    pbar.close()
    
    # Save the dataset (80-20 split)
    import random
    eq_list = list(all_equations)
    random.shuffle(eq_list)
    
    split_idx = int(len(eq_list) * 0.8)
    train_eqs = eq_list[:split_idx]
    val_eqs = eq_list[split_idx:]
    
    train_file = out_dir / "train_equations.txt"
    val_file = out_dir / "val_equations.txt"
    
    with open(train_file, "w", encoding="utf-8") as f:
        for eq in train_eqs:
            f.write(eq.replace("\n", " ") + "\n")
            
    with open(val_file, "w", encoding="utf-8") as f:
        for eq in val_eqs:
            f.write(eq.replace("\n", " ") + "\n")
            
    print(f"\nScraping complete!")
    print(f"Saved {len(train_eqs)} equations to {train_file}")
    print(f"Saved {len(val_eqs)} equations to {val_file}")

if __name__ == "__main__":
    main()
