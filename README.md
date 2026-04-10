# Nexus Journal

Nexus Journal is a multi-stage research-paper processing workspace with:

- `backend_py` for PDF parsing, cleaning, section detection, chunking, summarization, insight extraction, and gap detection
- `backend_node` for the Node backend scaffold
- `frontend` for the Vite + React UI

## Local development

Run each part from its own folder:

- `frontend`: install dependencies and run `npm run dev`
- `backend_node`: install dependencies and run the Node entry point or add a script as needed
- `backend_py`: create and activate a Python environment, then run the stage scripts directly

## Notes

- Generated artifacts such as `backend_py/journals_master.json`, caches, virtual environments, and build outputs are ignored by Git.
- Sample PDFs live under `backend_py/data/sample_papers/` for local experimentation.
