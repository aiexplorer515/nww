#!/usr/bin/env python3
"""
Launcher script for the NWW Total App
"""
import subprocess
import sys
import os

def main():
    """Launch the Streamlit app"""
    try:
        # Change to the project directory
        project_dir = os.path.dirname(os.path.abspath(__file__))
        os.chdir(project_dir)
        
        # Run the Streamlit app
        cmd = [
            sys.executable, "-m", "streamlit", "run", 
            "nwwpkg/ui/app_total.py",
            "--server.port", "8501",
            "--server.headless", "true"
        ]
        
        print("🚀 Starting NWW Total App...")
        print(f"📁 Project directory: {project_dir}")
        print(f"🌐 App will be available at: http://localhost:8501")
        print("Press Ctrl+C to stop the server")
        
        subprocess.run(cmd)
        
    except KeyboardInterrupt:
        print("\n👋 Shutting down NWW Total App...")
    except Exception as e:
        print(f"❌ Error starting app: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()