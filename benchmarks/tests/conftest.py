import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ["CHUNKBENCH_PLUGIN"] = "stubs"
os.environ["PYTHONPATH"] = os.pathsep.join([str(Path(__file__).resolve().parent), os.environ.get("PYTHONPATH", "")])
