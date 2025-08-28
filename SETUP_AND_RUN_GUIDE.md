# 🎌 Complete Riko Project Setup & Running Guide

## 📋 Table of Contents
1. [Project Overview](#-project-overview)
2. [Prerequisites](#-prerequisites)
3. [Initial Setup](#-initial-setup)
4. [GPT-SoVITS Server Setup](#-gpt-sovits-server-setup)
5. [Enhanced Chat System Setup](#-enhanced-chat-system-setup)
6. [Running the Complete System](#-running-the-complete-system)
7. [Troubleshooting](#-troubleshooting)
8. [Advanced Usage](#-advanced-usage)

## 🎯 Project Overview

This is your complete Riko AI assistant project featuring:
- **GPT-SoVITS TTS Server** - High-quality voice synthesis
- **Enhanced Chat System** - Voice/text interaction with emotion processing
- **LLM Integration** - Google Gemini for intelligent responses
- **Audio Processing** - Whisper for speech recognition

## 🔧 Prerequisites

### System Requirements:
- **OS**: Windows 10/11
- **Python**: 3.8+ (you have 3.13 ✅)
- **RAM**: 8GB+ recommended
- **Storage**: 5GB+ free space
- **Audio**: Working microphone and speakers

### Required Accounts:
- **Google Gemini API Key** (for LLM responses)
- **Internet Connection** (for API calls)

## 🚀 Initial Setup

### Step 1: Navigate to Project Directory
```powershell
cd "C:\Users\pande\OneDrive\Desktop\Ricko\riko_project"
```

### Step 2: Install Python Dependencies
```powershell
# Core dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install numpy scipy soundfile librosa
pip install transformers accelerate
pip install gradio fastapi uvicorn
pip install openai-whisper faster-whisper
pip install google-generativeai
pip install pydub pygame
pip install colorama keyboard
pip install yaml pathlib uuid
```

### Step 3: Verify Character Configuration
Check if `character_config.yaml` exists in your project root:
```powershell
ls character_config.yaml
```

If missing, create one with your Gemini API key:
```yaml
GEMINI_API_KEY: "your_gemini_api_key_here"
model: "gemini-1.5-flash"
history_file: "server/chat_history.json"
presets:
  default:
    system_prompt: "You are Riko, a helpful but slightly tsundere AI assistant..."
```

## 🎵 GPT-SoVITS Server Setup

### Step 1: Navigate to GPT-SoVITS Directory
```powershell
cd GPT-SoVITS
```

### Step 2: Download Required Models
The system will automatically download models on first run, or manually:
```powershell
python -s GPT_SoVITS/download.py
```

### Step 3: Start the TTS Server
```powershell
# Method 1: Using WebUI (Recommended for beginners)
python webui.py

# Method 2: Direct inference server (Advanced users)
python GPT_SoVITS/inference_webui.py
```

### Step 4: Configure TTS Models
1. Open your browser to `http://localhost:9880` (or shown port)
2. Go to the "1C-推理" (Inference) tab
3. Load your trained models or use pretrained ones
4. Test TTS generation
5. Keep this server running!

## 🤖 Enhanced Chat System Setup

### Step 1: Navigate to Server Directory
```powershell
cd ..\server
# You should now be in: C:\Users\pande\OneDrive\Desktop\Ricko\riko_project\server
```

### Step 2: Verify File Structure
Check that all enhanced chat files exist:
```powershell
ls process/text_processing/
ls process/emotion_system/
ls main_chat.py
ls advanced_chat.py
```

### Step 3: Test Individual Components
```powershell
# Test emotion filtering
python process/text_processing/emotion_filter.py

# Test colored display
python process/text_processing/colored_display.py

# Test emotion handler
python process/emotion_system/emotion_handler.py
```

## 🎮 Running the Complete System

### 🔥 Quick Start (Recommended Path)

#### Terminal 1: Start GPT-SoVITS Server
```powershell
cd "C:\Users\pande\OneDrive\Desktop\Ricko\riko_project\GPT-SoVITS"
python webui.py
```
Wait for "Running on local URL: http://localhost:XXXX" message.

#### Terminal 2: Start Enhanced Chat
```powershell
cd "C:\Users\pande\OneDrive\Desktop\Ricko\riko_project\server"
python main_chat.py
```

### 🌟 Advanced Usage

#### Option A: Advanced 4-Mode Chat System (NEW! 🎉)
```powershell
cd "C:\Users\pande\OneDrive\Desktop\Ricko\riko_project\server"
python advanced_four_mode_chat.py    # GUI version
# OR
python four_mode_chat_cli.py          # Command line version
```
**Features:**
- **Mode 1**: 🔘 Button Voice Mode - Record & train Riko's voice
- **Mode 2**: 🎤 Auto Voice Detection - Automatic speech detection
- **Mode 3**: 💬 Voice-to-Voice - Voice input with voice output  
- **Mode 4**: 📝 Text-to-Text - Pure text interaction
- Voice training capabilities
- Advanced GUI interface (first option)

#### Option B: Three-Feature Chat System
```powershell
cd "C:\Users\pande\OneDrive\Desktop\Ricko\riko_project\server"
python advanced_chat.py
```
**Features:**
- Voice Chat Mode
- Text Chat Mode  
- Mixed Mode
- Interactive menu

#### Option B: Original Simple Chat
```powershell
cd "C:\Users\pande\OneDrive\Desktop\Ricko\riko_project\server"
# If you want the old version without enhancements
# (backup your current main_chat.py first)
```

## 📊 Expected Output

### GPT-SoVITS Server Startup:
```
Defaulting to user installation...
Starting server...
Running on local URL:  http://localhost:9880
```

### Enhanced Chat Startup:
```
============================================================
🎯 Starting Chat with Riko... 🎯
============================================================
[System]: Enhanced Riko Chat with Emotion Processing
[System]: Features: Colored text, emotion filtering, visual feedback
[System]: Initializing Whisper model...
[System]: Ready to chat!
🎙️  Listening... (Press and hold space to talk)
```

### Sample Conversation:
```
🎙️  Listening... (Press and hold space to talk)
[You]: Hello Riko, how are you today?
🤔 Processing...
[Riko]: Oh great, another person wants to chat with me. Fine, I'm doing okay I guess.
✨ [rolls eyes] [sigh]
🙄 😮‍💨
🎵 Generating audio...
🔊 Playing audio...
------------------------------
```

## 🚨 Troubleshooting

### Common Issues & Solutions:

#### 1. **Import Errors**
```
Error: No module named 'colorama'
```
**Solution:**
```powershell
pip install colorama keyboard
```

#### 2. **GPT-SoVITS Server Not Found**
```
Error: Connection refused to localhost:9880
```
**Solutions:**
- Make sure GPT-SoVITS server is running
- Check the port in the server startup message
- Update `sovits_ping.py` with correct port if needed

#### 3. **Whisper Model Download Issues**
```
Error: Failed to download Whisper model
```
**Solutions:**
```powershell
# Try downloading manually
pip install --upgrade faster-whisper
# Or use smaller model
# Change "base.en" to "tiny.en" in main_chat.py
```

#### 4. **Microphone Not Working**
```
Error: No speech detected
```
**Solutions:**
- Check Windows microphone permissions
- Test microphone in other applications
- Verify space bar is held down while speaking

#### 5. **Gemini API Errors**
```
Error: Gemini API key invalid
```
**Solutions:**
- Verify your API key in `character_config.yaml`
- Check Google Cloud Console for API limits
- Ensure billing is enabled for Gemini API

#### 6. **Audio Playback Issues**
```
Error: Failed to play audio
```
**Solutions:**
```powershell
pip install pygame pydub
# Or try alternative audio backends
pip install sounddevice
```

### Debug Mode:
Add debug prints to track issues:
```python
# In main_chat.py, add at the top:
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 🎛️ Advanced Configuration

### Custom Emotion Patterns:
Edit `process/text_processing/emotion_filter.py`:
```python
# Add your custom emotion patterns
self.emotion_patterns.append(r'\*your_custom_emotion\*')
```

### Custom Colors:
Edit `process/text_processing/colored_display.py`:
```python
# Change color schemes
self.riko_color = Fore.LIGHTMAGENTA_EX  # Brighter magenta
self.user_color = Fore.LIGHTCYAN_EX     # Brighter cyan
```

### TTS Server Configuration:
Edit `process/tts_func/sovits_ping.py` if you need to change:
- Server URL/port
- Model parameters
- Audio quality settings

## 📱 Usage Modes

### 1. **Voice Chat Mode** (Default)
- Hold space bar to talk
- Release to process
- Riko responds with voice
- Emotions shown visually

### 2. **Text Chat Mode**
```powershell
python advanced_chat.py
# Choose option 2
```
- Type messages directly
- No voice input needed
- Still gets emotion effects
- Type 'quit' to exit

### 3. **Mixed Mode**
```powershell
python advanced_chat.py  
# Choose option 3
```
- Choose input method each time
- Press V for voice, T for text
- Maximum flexibility

## 🚀 Voice Quality & Speed Optimization

### 🎵 **Automatic Optimization (Recommended)**
Run the optimization scripts to automatically improve voice quality and speed:

```powershell
# Run TTS optimizer (improves quality and speed)
cd "C:\Users\pande\OneDrive\Desktop\Ricko\riko_project\server"
python tts_optimizer.py

# Run GPU accelerator (maximizes performance)
python gpu_accelerator.py
```

### ⚡ **What These Optimizations Do:**
- **🎤 Better Voice Quality**: Enhanced audio processing, noise reduction, 24-bit audio
- **🚀 Faster Generation**: Optimized model settings, GPU acceleration, caching
- **💾 Memory Optimization**: Better memory management, model preloading
- **🔥 GPU Acceleration**: CUDA optimizations, GPU warmup, faster inference
- **📊 Performance Monitoring**: Real-time benchmarking and monitoring tools

### 🎯 Performance Tips

### For Better Performance:
1. **Run optimization scripts first** (see above)
2. **Close unnecessary applications** (frees RAM)
3. **Use smaller Whisper model** if needed:
   ```python
   # In main_chat.py, change:
   WhisperModel("tiny.en")  # Instead of "base.en"
   ```
4. **Restart GPT-SoVITS server** after optimization
5. **Use SSD** for faster model loading

### For Better Audio Quality:
1. **Run tts_optimizer.py first** (automatic audio enhancements)
2. **Use good microphone** in quiet environment
3. **Speak clearly** and not too fast
4. **Keep consistent distance** from microphone
5. **Use headphones** to prevent feedback
6. **Check optimized settings** in character_config.yaml

## 🔄 System Workflow

```
1. User speaks → 2. Whisper STT → 3. Gemini LLM → 
4. Emotion Filter → 5. Display Text → 6. Show Emotions → 
7. GPT-SoVITS TTS → 8. Audio Playback → 9. Cleanup → Repeat
```

## 📞 Quick Commands Reference

### Start Everything:
```powershell
# Terminal 1:
cd "C:\Users\pande\OneDrive\Desktop\Ricko\riko_project\GPT-SoVITS"
python webui.py

# Terminal 2:
cd "C:\Users\pande\OneDrive\Desktop\Ricko\riko_project\server" 
python main_chat.py
```

### Stop Everything:
- Press `Ctrl+C` in both terminals
- Close browser tab with GPT-SoVITS interface

### Reset/Restart:
```powershell
# Clear chat history
rm server/chat_history.json

# Clear audio cache  
rm server/audio/*.wav
```

## 🎉 You're All Set!

Your enhanced Riko chat system is ready to use! 

**Recommended first run:**
1. Start GPT-SoVITS server
2. Wait for "Running on local URL" message
3. Start `main_chat.py` 
4. Try saying "Hello Riko!" 
5. Watch the magic happen! ✨

**Need help?** Check the troubleshooting section or review individual component files for detailed documentation.

**Happy chatting with Riko!** 🎌🤖
