"""
NVIDIA G-Assist entry point for RTX SYLPH.

G-Assist looks for plugin.py in the plugin folder and runs it with
Protocol V2 (JSON-RPC). This launcher starts the real SYLPH module.

Standalone launches (double-click / py plugin.py on a TTY) wrap the HUD in a
watchdog so a native OpenCV/pygame crash overnight comes back in a few seconds.
G-Assist stdin is not a TTY, so that path is left alone.
"""

import os
import runpy
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
MAIN = os.path.join(HERE, "RTX_SYLPH_G_Assist_V2.py")


def _gassist() -> bool:
    try:
        return sys.stdin is not None and not sys.stdin.isatty()
    except Exception:
        return True


if __name__ == "__main__":
    child = os.environ.get("SYLPH_CHILD") == "1"
    watchdog_off = os.environ.get("SYLPH_WATCHDOG", "1") == "0"
    if child or watchdog_off or _gassist():
        runpy.run_path(MAIN, run_name="__main__")
    else:
        env = os.environ.copy()
        env["SYLPH_CHILD"] = "1"
        while True:
            rc = subprocess.call([sys.executable, os.path.abspath(__file__)], cwd=HERE, env=env)
            if rc == 0:
                break
            print(f"SYLPH exited {rc} — restarting in 3s", flush=True)
            time.sleep(3)
