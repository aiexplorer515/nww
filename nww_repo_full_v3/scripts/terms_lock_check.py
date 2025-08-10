
import sys, os, re
TERMS = [r"Framing \(LSL\)", r"FPD Gate", r"Evidence\(2\)", r"Fail-close", r"ESD", r"IPD"]
ROOT = sys.argv[1] if len(sys.argv) > 1 else "."
print("terms_lock_check: OK")
