#!/usr/bin/env python3
"""
Riko Complete System Startup
============================

This script starts the complete Riko conversational AI system:
- Google Gemini Flash-2.0 integration
- Three conversation modes (Button Voice, Auto Voice Detection, Text)
- GPT-SoVITS voice synthesis integration

Usage: python start_riko.py
"""

import sys
import time
import requests
from pathlib import Path
import subprocess

# Add server path for imports
sys.path.append(str(Path(__file__).parent / "server"))

def check_dependencies():
    """Check if all required dependencies are available"""
    print("🔍 Checking dependencies...")
    
    missing_deps = []
    
    try:
        import google.generativeai
        print("✅ Google Generative AI installed")
    except ImportError:
        missing_deps.append("google-generativeai")
        
    try:
        import sounddevice
        print("✅ Sound device support available")
    except ImportError:
        missing_deps.append("sounddevice")
        
    try:
        import soundfile
        print("✅ Sound file support available")
    except ImportError:
        missing_deps.append("soundfile")
        
    try:
        from faster_whisper import WhisperModel
        print("✅ Faster Whisper available")
    except ImportError:
        missing_deps.append("faster-whisper")
        
    try:
        import tkinter
        print("✅ GUI support (tkinter) available")
    except ImportError:
        missing_deps.append("tkinter")
    
    if missing_deps:
        print(f"❌ Missing dependencies: {', '.join(missing_deps)}")
        print("   Installing missing dependencies...")
        for dep in missing_deps:
            if dep != "tkinter":  # tkinter usually comes with Python
                subprocess.run([sys.executable, "-m", "pip", "install", dep], check=True)
        print("✅ Dependencies installed!")
    
    return len(missing_deps) == 0

def check_configuration():
    """Check if configuration is properly set up"""
    print("\n🔧 Checking configuration...")
    
    config_file = Path("character_config.yaml")
    if not config_file.exists():
        print("❌ character_config.yaml not found!")
        return False
        
    try:
        import yaml
        with open(config_file) as f:
            config = yaml.safe_load(f)
            
        # Check Gemini API key
        gemini_key = config.get('GEMINI_API_KEY', '')
        if not gemini_key or gemini_key == 'YOUR_GEMINI_API_KEY':
            print("❌ Google Gemini API key not configured!")
            return False
        else:
            print("✅ Google Gemini API key configured")
            
        # Check reference audio path
        ref_audio = config.get('sovits_ping_config', {}).get('ref_audio_path', '')
        if ref_audio and Path(ref_audio).exists():
            print("✅ Reference audio file found")
        else:
            print("⚠️  Reference audio file not found (voice synthesis might not work)")
            
    except Exception as e:
        print(f"❌ Error reading configuration: {e}")
        return False
        
    return True

def check_gpt_sovits_server():
    """Check if GPT-SoVITS server is running"""
    print("\n🎵 Checking GPT-SoVITS server...")
    
    try:
        response = requests.get("http://127.0.0.1:9880", timeout=5)
        print("✅ GPT-SoVITS server is running!")
        return True
    except:
        print("⚠️  GPT-SoVITS server is not running")
        print("   Voice synthesis will not work without the server")
        print("   You can still use text mode for testing")
        return False

def test_gemini_integration():
    """Test Google Gemini integration"""
    print("\n🤖 Testing Google Gemini integration...")
    
    try:
        from process.llm_funcs.llm_scr import llm_response
        response = llm_response("Say hi briefly")
        print(f"✅ Gemini test successful! Riko said: '{response[:50]}...'")
        return True
    except Exception as e:
        print(f"❌ Gemini test failed: {e}")
        return False

def start_conversation_modes():
    """Start the conversation modes GUI"""
    print("\n🎌 Starting Riko Conversation Modes...")
    
    try:
        # Import and run the conversation modes
        from conversation_modes import ConversationModes
        app = ConversationModes()
        app.run()
    except Exception as e:
        print(f"❌ Error starting conversation modes: {e}")
        return False
        
    return True

def main():
    """Main startup function"""
    print("="*60)
    print("🎌 Riko - Complete Conversational AI System 🎌")
    print("="*60)
    print()
    
    print("Starting system checks...")
    
    # Check dependencies
    deps_ok = check_dependencies()
    if not deps_ok:
        print("❌ Dependencies check failed!")
        return 1
    
    # Check configuration
    config_ok = check_configuration()
    if not config_ok:
        print("❌ Configuration check failed!")
        print("\nPlease make sure:")
        print("1. character_config.yaml exists")
        print("2. GEMINI_API_KEY is set correctly")
        print("3. Audio paths are configured properly")
        return 1
    
    # Test Gemini integration
    gemini_ok = test_gemini_integration()
    if not gemini_ok:
        print("❌ Google Gemini integration failed!")
        return 1
    
    # Check GPT-SoVITS server (optional)
    sovits_ok = check_gpt_sovits_server()
    
    print("\n" + "="*60)
    print("🎉 System Status:")
    print(f"   ✅ Dependencies: OK")
    print(f"   ✅ Configuration: OK") 
    print(f"   ✅ Google Gemini: OK")
    print(f"   {'✅' if sovits_ok else '⚠️ '} GPT-SoVITS: {'OK' if sovits_ok else 'Not Running'}")
    print("="*60)
    
    if not sovits_ok:
        print("\n📝 Note: Voice synthesis requires GPT-SoVITS server.")
        print("   Text mode will work without it.")
        print("   To enable voice modes, start GPT-SoVITS server separately.")
        
    print("\n🚀 Launching Riko Conversation Modes...")
    print("   Available modes:")
    print("   • Mode 1: Button Voice - Click to record")
    print("   • Mode 2: Auto Voice Detection - Hands-free recording") 
    print("   • Mode 3: Text Mode - Type to chat")
    print()
    
    # Start the conversation interface
    try:
        start_conversation_modes()
    except KeyboardInterrupt:
        print("\n👋 Goodbye senpai! Riko is shutting down...")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
