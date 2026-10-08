import sys
from pathlib import Path

# Ensure backend root directory is in sys.path for test discovery
BACKEND_DIR = Path(__file__).resolve().parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))
