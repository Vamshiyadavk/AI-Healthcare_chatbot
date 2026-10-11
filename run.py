"""
HealthBot Runner
Usage:
    python run.py
    (or: python app.py)
"""

import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from app import app

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    print(f"\n==============================================")
    print(f"  HealthBot AI Symptom Checker")
    print(f"  Running on: http://localhost:{port}")
    print(f"==============================================\n")
    app.run(host="0.0.0.0", port=port, debug=app.config["DEBUG"])
