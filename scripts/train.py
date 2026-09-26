#!/usr/bin/env python3
"""Train the XLoc-CXR model."""
import sys
sys.path.insert(0, "src")

from xloc_cxr.main import main

if __name__ == "__main__":
    sys.argv = ["xloc-cxr", "train"] + sys.argv[1:]
    main()
