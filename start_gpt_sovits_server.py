#!/usr/bin/env python3
"""
GPT-SoVITS Server Starter
========================

This script helps start a GPT-SoVITS server for voice synthesis.
For now, it provides instructions to manually start the server.

GPT-SoVITS needs to be installed separately and configured.
"""

import requests
import time
import sys

def check_server_status():
    """Check if GPT-SoVITS server is running"""
    try:
        response = requests.get("http://127.0.0.1:9880", timeout=5)
        return True
    except:
        return False

def main():
    print("="*60)
    print("🎵 GPT-SoVITS Server Setup 🎵")
    print("="*60)
    print()
    
    if check_server_status():
        print("✅ GPT-SoVITS server is already running on port 9880!")
        print("   You can now use Riko's conversation modes.")
        return
    
    print("❌ GPT-SoVITS server is not running.")
    print()
    print("To use Riko's voice synthesis, you need to:")
    print()
    print("1. Install GPT-SoVITS:")
    print("   git clone https://github.com/RVC-Boss/GPT-SoVITS.git")
    print("   cd GPT-SoVITS")
    print("   pip install -r requirements.txt")
    print()
    print("2. Start the GPT-SoVITS server:")
    print("   python api_v2.py")
    print("   (The server should run on http://127.0.0.1:9880)")
    print()
    print("3. Configure your voice model:")
    print("   - Use the web interface to train/load your Riko voice model")
    print("   - Make sure the reference audio path in character_config.yaml is correct")
    print()
    print("4. Once the server is running, you can use the conversation modes!")
    print()
    
    print("Waiting for GPT-SoVITS server to start...")
    print("(Press Ctrl+C to stop checking)")
    
    try:
        while True:
            if check_server_status():
                print("✅ GPT-SoVITS server detected! Ready to go!")
                break
            print("   Still waiting... (make sure to start the server)")
            time.sleep(5)
    except KeyboardInterrupt:
        print("\n⏹️  Stopped waiting for server.")
        print("   Start the GPT-SoVITS server manually and then run conversation_modes.py")

if __name__ == "__main__":
    main()
