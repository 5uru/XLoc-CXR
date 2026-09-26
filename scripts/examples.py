#!/usr/bin/env python3
"""Generate CAM success/failure example figures."""
import sys
sys.path.insert(0, "src")

from xloc_cxr.main import main

if __name__ == "__main__":
    sys.argv = ["xloc-cxr", "examples"] + sys.argv[1:]
    main()
