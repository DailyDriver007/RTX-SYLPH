"""
Compatibility helper for the NVIDIA G-Assist plugin layout.

G-Assist reads manifest.json next to plugin.py. This file used to be
raw JSON saved with a .py extension, which crashed if Python imported it.
Running it prints the live manifest so you can confirm the NVIDIA schema.
"""

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
MANIFEST_PATH = os.path.join(HERE, "manifest.json")


def load_manifest():
    with open(MANIFEST_PATH, "r", encoding="utf-8") as handle:
        return json.load(handle)


def validate(manifest):
    required = ("manifestVersion", "name", "version", "executable", "persistent", "protocol_version", "functions")
    missing = [key for key in required if key not in manifest]
    if missing:
        raise SystemExit(f"manifest.json is missing required keys: {', '.join(missing)}")
    if manifest.get("protocol_version") != "2.0":
        raise SystemExit("protocol_version must be \"2.0\" for G-Assist V2")
    if not isinstance(manifest.get("functions"), list) or not manifest["functions"]:
        raise SystemExit("functions must be a non-empty list")
    names = [fn.get("name") for fn in manifest["functions"]]
    if len(names) != len(set(names)):
        raise SystemExit("duplicate function names in manifest.json")
    return True


if __name__ == "__main__":
    data = load_manifest()
    validate(data)
    json.dump(data, sys.stdout, indent=2)
    sys.stdout.write("\n")
    print(f"OK: {len(data['functions'])} functions, executable={data['executable']}", file=sys.stderr)
