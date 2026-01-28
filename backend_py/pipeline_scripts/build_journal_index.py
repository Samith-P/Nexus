"""
build_journal_index.py

Purpose:
- Single entry point for Phase 2 pipeline
- Orchestrates all journal processing steps in order
- Produces ONE final master file: journals_master.json

Pipeline steps:
1. csv_to_scimago_json.py       → journals_scimago_raw.json
2. clean_scimago.py             → journals_clean.json
3. filter_journals.py           → journals_filtered.json
4. add_domain_text.py           → journals_with_domain_text.json
5. embed_journals.py            → journals_master.json (final output)

Each step reads the previous step's output.
"""

import os
import sys
import json
import subprocess
from pathlib import Path


class PipelineOrchestrator:
    """Orchestrate the journal processing pipeline."""
    
    def __init__(self, base_dir="."):
        """Initialize orchestrator with base directory."""
        self.base_dir = Path(base_dir)
        self.steps = []
        self.success_count = 0
        self.failure_count = 0
        
        # Detect virtual environment Python
        venv_python = self.base_dir / ".venv" / "Scripts" / "python.exe"
        if venv_python.exists():
            self.python_executable = str(venv_python)
            self.log(f"Using virtual environment: {venv_python}")
        else:
            self.python_executable = sys.executable
            self.log(f"Using system Python: {sys.executable}", "WARN")
    
    def log(self, message, level="INFO"):
        """Print formatted log message."""
        icons = {"INFO": "ℹ", "SUCCESS": "✓", "ERROR": "✗", "WARN": "⚠"}
        icon = icons.get(level, "•")
        print(f"{icon} {message}")
    
    def run_step(self, step_num, script_name, input_file, output_file, description):
        """
        Execute a pipeline step.
        
        Args:
            step_num: Step number (for logging)
            script_name: Name of script to run
            input_file: Expected input file (for validation)
            output_file: Expected output file
            description: Human-readable description
        """
        self.log(f"[{step_num}] {description}")
        
        script_path = self.base_dir / "pipeline_scripts" / script_name
        
        # Check if script exists
        if not script_path.exists():
            self.log(f"Script not found: {script_path}", "ERROR")
            self.failure_count += 1
            return False
        
        # Check if input file exists (except for first step)
        if input_file and not (self.base_dir / input_file).exists():
            self.log(f"Input file not found: {input_file}", "ERROR")
            self.failure_count += 1
            return False
        
        # Run script
        try:
            self.log(f"Running {script_name}...")
            
            result = subprocess.run(
                [self.python_executable, str(script_path)],
                cwd=str(self.base_dir),
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout per step
            )
            
            # Print script output
            if result.stdout:
                for line in result.stdout.strip().split("\n"):
                    self.log(f"  {line}")
            
            if result.returncode != 0:
                self.log(f"Script failed with return code {result.returncode}", "ERROR")
                if result.stderr:
                    self.log(f"Error output:\n{result.stderr}", "ERROR")
                self.failure_count += 1
                return False
            
            # Verify output file was created
            output_path = self.base_dir / output_file
            if not output_path.exists():
                self.log(f"Output file not created: {output_file}", "ERROR")
                self.failure_count += 1
                return False
            
            # Get output file size
            file_size = output_path.stat().st_size / (1024 * 1024)  # MB
            self.log(f"✓ Created {output_file} ({file_size:.2f} MB)", "SUCCESS")
            self.success_count += 1
            return True
        
        except subprocess.TimeoutExpired:
            self.log(f"Script timed out: {script_name}", "ERROR")
            self.failure_count += 1
            return False
        except Exception as e:
            self.log(f"Unexpected error running {script_name}: {e}", "ERROR")
            self.failure_count += 1
            return False
    
    def run_pipeline(self):
        """Execute the full pipeline."""
        print("\n" + "="*70)
        print("PHASE 2: JOURNAL INDEX BUILDING PIPELINE")
        print("="*70 + "\n")
        
        # Step 1: CSV to JSON conversion
        if not self.run_step(
            1,
            "csv_to_scimago_json.py",
            None,  # No input file (reads from datasets/)
            "pipeline_cache/journals_scimago_raw.json",
            "Converting Scimago CSV to JSON"
        ):
            self.log("Pipeline aborted at step 1", "ERROR")
            return False
        
        # Step 2: Clean numeric fields and parse SJR
        if not self.run_step(
            2,
            "clean_scimago.py",
            "pipeline_cache/journals_scimago_raw.json",
            "pipeline_cache/journals_clean.json",
            "Cleaning numeric fields and parsing SJR"
        ):
            self.log("Pipeline aborted at step 2", "ERROR")
            return False
        
        # Step 3: Filter journals by quality metrics
        if not self.run_step(
            3,
            "filter_journals.py",
            "pipeline_cache/journals_clean.json",
            "pipeline_cache/journals_filtered.json",
            "Filtering journals by SJR and quartile"
        ):
            self.log("Pipeline aborted at step 3", "ERROR")
            return False
        
        # Step 4: Add domain text
        if not self.run_step(
            4,
            "add_domain_text.py",
            "pipeline_cache/journals_filtered.json",
            "pipeline_cache/journals_with_domain_text.json",
            "Adding natural language domain text"
        ):
            self.log("Pipeline aborted at step 4", "ERROR")
            return False
        
        # Step 5: Generate embeddings and create master file
        if not self.run_step(
            5,
            "embed_journals.py",
            "pipeline_cache/journals_with_domain_text.json",
            "pipeline_cache/journals_embedded.json",
            "Generating sentence embeddings"
        ):
            self.log("Pipeline aborted at step 5", "ERROR")
            return False
        
        return True
    
    def finalize(self):
        """
        Rename final output to journals_master.json.
        
        The master file contains all required fields:
        - title
        - type
        - sjr
        - quartile
        - h_index
        - citations_per_doc_2y
        - domain_text
        - embedding
        """
        self.log("Finalizing master journal index...")
        
        embedded_path = self.base_dir / "pipeline_cache" / "journals_embedded.json"
        master_path = self.base_dir / "journals_master.json"
        
        if not embedded_path.exists():
            self.log("Embedded journals file not found", "ERROR")
            return False
        
        try:
            # Load and validate master file
            with open(embedded_path, "r", encoding="utf-8") as f:
                journals = json.load(f)
            
            # Validate each journal has required fields
            required_fields = [
                "title", "type", "sjr", "quartile", "h_index",
                "citations_per_doc_2y", "domain_text", "embedding"
            ]
            
            invalid_count = 0
            for journal in journals:
                for field in required_fields:
                    if field not in journal:
                        invalid_count += 1
                        self.log(
                            f"Journal '{journal.get('title', 'UNKNOWN')}' missing field: {field}",
                            "WARN"
                        )
                        break
            
            if invalid_count > 0:
                self.log(f"Found {invalid_count} journals with missing fields", "WARN")
            
            # Save as master file
            with open(master_path, "w", encoding="utf-8") as f:
                json.dump(journals, f, indent=2, ensure_ascii=False)
            
            file_size = master_path.stat().st_size / (1024 * 1024)
            self.log(f"Created journals_master.json ({file_size:.2f} MB, {len(journals)} journals)", "SUCCESS")
            
            return True
        
        except json.JSONDecodeError as e:
            self.log(f"Invalid JSON in {embedded_path}: {e}", "ERROR")
            return False
        except Exception as e:
            self.log(f"Error finalizing master file: {e}", "ERROR")
            return False
    
    def print_summary(self):
        """Print summary of pipeline execution."""
        print("\n" + "="*70)
        print("PIPELINE SUMMARY")
        print("="*70)
        self.log(f"Successful steps: {self.success_count}")
        if self.failure_count > 0:
            self.log(f"Failed steps: {self.failure_count}", "WARN")
        
        master_path = self.base_dir / "journals_master.json"
        if master_path.exists():
            with open(master_path, "r") as f:
                journals = json.load(f)
            self.log(f"Master file: journals_master.json ({len(journals)} journals)", "SUCCESS")
        else:
            self.log("Master file not found", "WARN")
        
        print("="*70 + "\n")


def main():
    """Main entry point."""
    # Get base directory (current directory by default)
    base_dir = Path.cwd()
    
    # Create orchestrator
    orchestrator = PipelineOrchestrator(base_dir)
    
    # Run pipeline
    success = orchestrator.run_pipeline()
    
    if success:
        # Finalize master file
        if orchestrator.finalize():
            orchestrator.print_summary()
            print("✓ Pipeline completed successfully!")
            sys.exit(0)
        else:
            orchestrator.print_summary()
            print("✗ Pipeline failed during finalization")
            sys.exit(1)
    else:
        orchestrator.print_summary()
        print("✗ Pipeline failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
