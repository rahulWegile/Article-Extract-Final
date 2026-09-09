from pathlib import Path


#
# Project Directories
#

OUTPUT_DIR = Path("output")

DOCUMENTS_DIR = OUTPUT_DIR / "documents"

TEMP_DIR = OUTPUT_DIR / "temp"


UPLOAD_DIR = Path("uploads")

# Ensure runtime directories exist
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
TEMP_DIR.mkdir(parents=True, exist_ok=True)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


#
# API
#

API_PREFIX = ""