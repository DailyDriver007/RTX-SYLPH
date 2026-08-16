"""
NVIDIA G-Assist entry point for RTX SYLPH.

G-Assist looks for plugin.py in the plugin folder and runs it with
Protocol V2 (JSON-RPC). This launcher starts the real SYLPH module.
"""

import os
import runpy

HERE = os.path.dirname(os.path.abspath(__file__))
MAIN = os.path.join(HERE, "RTX_SYLPH_G_Assist_V2.py")

if __name__ == "__main__":
    runpy.run_path(MAIN, run_name="__main__")
