
import sys

from pathlib import Path


# ------------------------------------------------------------
# ADD PROJECT ROOT TO PYTHON PATH
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ------------------------------------------------------------
# IMPORT EXISTING ML PREDICTION FUNCTION
# ------------------------------------------------------------

from src.predict import predict_customer
