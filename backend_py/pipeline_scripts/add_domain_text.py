"""
add_domain_text.py

Purpose:
- Automatically generate a `domain_text` field for EACH journal
- This text is natural-language, embedding-friendly, and deterministic
- Uses only local data: title, categories, type, areas

Input: journals_filtered.json
Output: journals_with_domain_text.json
"""

import json
import sys
from pathlib import Path

# Get the backend_py directory (parent of pipeline_scripts)
BACKEND_DIR = Path(__file__).parent.parent


def build_domain_text(journal):
    """
    Construct domain_text from journal metadata.
    
    Args:
        journal: Dictionary with title, type, categories, areas
        
    Returns:
        String: Natural language text describing the journal's domain
    """
    title = journal.get("title", "").strip()
    journal_type = journal.get("type", "").strip()
    categories = journal.get("categories", "").strip()
    areas = journal.get("areas", "").strip()
    
    # Safety checks
    if not title:
        return None
    
    # Build domain text deterministically
    parts = []
    
    # Primary description
    parts.append(f"This {journal_type} publishes research in {categories}")
    
    # Areas/subjects
    if areas:
        parts.append(f"with focus on {areas}")
    
    # Combine into natural language
    domain_text = " ".join(parts).rstrip(".") + "."
    
    return domain_text


def main(input_file="pipeline_cache/journals_filtered.json", output_file="pipeline_cache/journals_with_domain_text.json"):
    """
    Main pipeline: add domain_text to each journal.
    
    Args:
        input_file: Path to input JSON
        output_file: Path to output JSON
    """
    try:
        # Resolve paths relative to backend_py directory
        input_path = BACKEND_DIR / input_file
        output_path = BACKEND_DIR / output_file
        
        # Load filtered journals
        with open(input_path, "r", encoding="utf-8") as f:
            journals = json.load(f)
        
        print(f"Loaded {len(journals)} journals from {input_file}")
        
        # Add domain_text to each journal
        journals_with_text = []
        skipped = 0
        
        for i, journal in enumerate(journals):
            domain_text = build_domain_text(journal)
            
            if not domain_text:
                skipped += 1
                print(f"  [SKIP] Journal #{i+1}: missing title")
                continue
            
            journal["domain_text"] = domain_text
            journals_with_text.append(journal)
        
        # Save output
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(journals_with_text, f, indent=2, ensure_ascii=False)
        
        print(f"[OK] Added domain_text to {len(journals_with_text)} journals")
        if skipped > 0:
            print(f"  [WARN] Skipped {skipped} journals due to missing data")
        print(f"[OK] Saved to {output_file}")
        
        return journals_with_text
    
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
    main()
