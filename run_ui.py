"""
Direct UI Runner - Run Streamlit UI directly
"""

import subprocess
import sys
import os

def main():
    """Run the Streamlit UI."""
    # Get the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    ui_path = os.path.join(script_dir, 'nwwpkg', 'ui', 'simple_app.py')
    
    print(f"🚀 Launching NWW Dashboard...")
    print(f"📁 UI Path: {ui_path}")
    
    # Run Streamlit
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", ui_path], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Error launching Streamlit: {e}")
        return 1
    except KeyboardInterrupt:
        print("\n👋 Dashboard closed by user")
        return 0
    
    return 0

if __name__ == "__main__":
    sys.exit(main())



