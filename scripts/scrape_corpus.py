import os
import re
import time
import tarfile
import tempfile
import arxiv
import requests
import tqdm
from pathlib import Path

def extract_math_blocks(tex_content: str) -> list[str]:
    # remove comments (everything after % on a line)
    no_comments = re.sub(r'(?<!\\)%.*', '', tex_content)
    
    equations = []
    
    # match $$...$$
    equations.extend(re.findall(r'\$\$(.*?)\$\$', no_comments, re.DOTALL))
    
    # match \[...\]
    equations.extend(re.findall(r'\\\[(.*?)\\\]', no_comments, re.DOTALL))
    
    # match environments
    for env in ['equation', 'align', 'equation*', 'align*', 'eqnarray']:
        pattern = r'\\begin\{' + env.replace('*', r'\*') + r'\}(.*?)\\end\{' + env.replace('*', r'\*') + r'\}'
        equations.extend(re.findall(pattern, no_comments, re.DOTALL))
        
    # clean up whitespace and replace newlines with spaces for single-line format
    cleaned = []
    for eq in equations:
        eq = eq.strip()
        # require minimum length and at least one latex command
        if len(eq) > 5 and '\\' in eq:
            cleaned.append(re.sub(r'\s+', ' ', eq))
            
    return cleaned

# pyrefly: ignore [missing-import]
from src.data.string_generator import parse_with_lark

def main():
    out_dir = Path("data/scraped")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("ScanTeX ArXiv Corpus Scraper (math.AG)")
    print("=" * 60)
    
    
    
    client = arxiv.Client()
    search = arxiv.Search(
        query="cat:math.AG",
        max_results=3000, 
        sort_by=arxiv.SortCriterion.SubmittedDate
    )
    
    results_generator = client.results(search)
    print("searching math.AG for recent papers... (this streams dynamically)")
    
    all_equations = set()
    target_count = 20000
    
    headers = {
        "User-Agent": "ScanTex_Dataset_Builder/1.0 (mailto:college_project@example.com)"
    }
    
    pbar = tqdm.tqdm(total=target_count, desc="Extracting equations", unit="eq")
    
    for i, paper in enumerate(results_generator):
        if len(all_equations) >= target_count:
            break
            
        arxiv_id = paper.entry_id.split("/")[-1]
        source_url = f"https://arxiv.org/e-print/{arxiv_id}"
        
        try:
            r = requests.get(source_url, headers=headers, timeout=10)
            if r.status_code == 200:
                with tempfile.TemporaryDirectory() as tmpdir:
                    tar_path = os.path.join(tmpdir, "source.tar.gz")
                    with open(tar_path, "wb") as f:
                        f.write(r.content)
                        
                    try:
                        with tarfile.open(tar_path, "r:gz") as tar:
                            for member in tar.getmembers():
                                if member.name.endswith('.tex'):
                                    f = tar.extractfile(member)
                                    if f is not None:
                                        content = f.read().decode('utf-8', errors='ignore')
                                        eqs = extract_math_blocks(content)
                                        
                                        # update progress bar
                                        before = len(all_equations)
                                        
                                        # ONLY keep equations that our grammar can parse perfectly
                                        for raw_eq in eqs:
                                            if raw_eq not in all_equations:
                                                # Strip structural/formatting tags that our parser doesn't need
                                                eq = re.sub(r'\\begin\{[^}]+\}', '', raw_eq)
                                                eq = re.sub(r'\\end\{[^}]+\}', '', eq)
                                                eq = re.sub(r'\\label\{[^}]+\}', '', eq)
                                                eq = re.sub(r'\\ref\{[^}]+\}', '', eq)
                                                eq = re.sub(r'\\eqref\{[^}]+\}', '', eq)
                                                eq = re.sub(r'\\cite\{[^}]+\}', '', eq)
                                                eq = re.sub(r'\\hspace\{[^}]+\}', '', eq)
                                                eq = re.sub(r'\\vspace\{[^}]+\}', '', eq)
                                                
                                                tree = parse_with_lark(eq)
                                                if tree is not None:
                                                    # We save the cleaned equation so the analysis phase has 0 formatting noise
                                                    all_equations.add(eq)
                                                    if len(all_equations) >= target_count:
                                                        break
                                        
                                        added = len(all_equations) - before
                                        if added > 0:
                                            # make sure we don't exceed the target count on the progress bar visually
                                            current = min(len(all_equations), target_count)
                                            pbar.update(current - pbar.n)
                                            
                                        if len(all_equations) >= target_count:
                                            break
                    except tarfile.ReadError:
                        pass
        except Exception:
            pass
            
        if len(all_equations) >= target_count:
            break
            
        # mandatory safety sleep
        time.sleep(4)

    pbar.close()
    print(f"\nfinished! collected {len(all_equations)} distinct equations.")
    
    out_file = out_dir / "equations.txt"
    with open(out_file, "w", encoding="utf-8") as f:
        for eq in all_equations:
            f.write(f"{eq}\n")
            
    print(f"saved equations to {out_file}")

if __name__ == "__main__":
    main()
