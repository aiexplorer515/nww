"""
NWW UI Launcher - Simple launcher for Streamlit UI
"""

import sys
import os

# Add the nwwpkg directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
nwwpkg_dir = os.path.join(current_dir, 'nwwpkg')
sys.path.insert(0, nwwpkg_dir)

# Import and run the UI
if __name__ == "__main__":
    import subprocess
    
    # Launch Streamlit
    ui_path = os.path.join(nwwpkg_dir, 'ui', 'simple_app.py')
    subprocess.run([sys.executable, "-m", "streamlit", "run", ui_path])
