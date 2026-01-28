import os
import sys
import json
from pathlib import Path
from sentence_transformers import SentenceTransformer

# Avoid GPU probing delays on Windows
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Get the backend_py directory (parent of pipeline_scripts)
BACKEND_DIR = Path(__file__).parent.parent


def embed_journals(input_file="pipeline_cache/journals_with_domain_text.json", output_file="pipeline_cache/journals_embedded.json"):
    """
    Generate sentence embeddings for all journals.
    
    Args:
        input_file: Path to JSON file with journals (must have domain_text field)
        output_file: Path to output JSON file with embeddings added
    """
    try:
        # Resolve paths relative to backend_py directory
        input_path = BACKEND_DIR / input_file
        output_path = BACKEND_DIR / output_file
        
        # Load model
        model = SentenceTransformer("all-MiniLM-L6-v2")
        
        # Load journals
        with open(input_path, "r", encoding="utf-8") as f:
            journals = json.load(f)
        
        print(f"Loaded {len(journals)} journals from {input_file}")
        
        # Extract domain_text from journals
        texts = []
        skipped = 0
        journals_to_embed = []
        
        for i, j in enumerate(journals):
            domain_text = j.get("domain_text")
            
            if not domain_text or not domain_text.strip():
                skipped += 1
                print(f"  [SKIP] Journal #{i+1}: missing or empty domain_text")
                continue
            
            texts.append(domain_text)
            journals_to_embed.append(j)
        
        if not texts:
            print("[ERROR] No journals with domain_text found")
            sys.exit(1)
        
        print(f"[OK] Generating embeddings for {len(texts)} journals")
        if skipped > 0:
            print(f"  [WARN] Skipped {skipped} journals due to missing domain_text")
        
        # Generate embeddings with batching for large datasets
        embeddings = model.encode(
            texts,
            show_progress_bar=True,
            convert_to_tensor=False,
            batch_size=32  # Batch processing to manage memory
        )
        
        # Attach embeddings
        for j, emb in zip(journals_to_embed, embeddings):
            j["embedding"] = emb.tolist()
        
        # Save output
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(journals_to_embed, f, indent=2, ensure_ascii=False)
        
        print(f"[OK] Embeddings generated for {len(journals_to_embed)} journals")
        print(f"[OK] Saved to {output_file}")
        
        return journals_to_embed
    
    except FileNotFoundError as e:
        print(f"[ERROR] File not found: {e}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"[ERROR] Invalid JSON in {input_file}: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    # Default behavior: use journals_with_domain_text.json as input
    # For backward compatibility, also accept command-line arguments
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
        output_file = sys.argv[2] if len(sys.argv) > 2 else "pipeline_cache/journals_embedded.json"
    else:
        input_file = "pipeline_cache/journals_with_domain_text.json"
        output_file = "pipeline_cache/journals_embedded.json"
    
    embed_journals(input_file, output_file)
