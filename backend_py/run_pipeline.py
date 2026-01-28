#!/usr/bin/env python
"""
Convenience script to run the journal index building pipeline.

This is a wrapper that calls the main orchestration script.

Usage:
    python run_pipeline.py
"""

import sys
import subprocess
from pathlib import Path

def main():
    """Run the pipeline orchestrator."""
    pipeline_script = Path(__file__).parent / "pipeline_scripts" / "build_journal_index.py"
    
    if not pipeline_script.exists():
        print(f"[ERROR] Pipeline script not found: {pipeline_script}")
        sys.exit(1)
    
    # Run the pipeline
    result = subprocess.run([sys.executable, str(pipeline_script)], cwd=str(Path(__file__).parent))
    sys.exit(result.returncode)

if __name__ == "__main__":
    main()
