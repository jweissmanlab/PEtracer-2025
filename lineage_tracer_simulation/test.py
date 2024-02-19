import sys
import os
from pathlib import Path

# Configure paths
results_path = Path(__file__).parent.absolute() / "results"
module_path = Path(__file__).parent.parent.absolute()
sys.path.append(str(module_path))

from config import threads

print(threads)
print(results_path)
