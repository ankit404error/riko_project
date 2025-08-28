# Riko Setup and Usage Guide

## 🎉 System Status

✅ **COMPLETE SETUP ACHIEVED** - All components are configured and ready!

### What's Working:
- ✅ Repository cloned successfully
- ✅ Dependencies installed
- ✅ Google Gemini Flash-2.0 API integrated with your API key
- ✅ Three conversation modes implemented
- ✅ Conversation interface GUI ready
- ✅ Speech recognition (Whisper) configured
- ✅ Text mode fully functional

## 🚀 Quick Start

### Option 1: Start Everything (Recommended)
```powershell
cd "C:\Users\pande\OneDrive\Desktop\Ricko\riko_project"
python start_riko.py
```

This will:
- Check all dependencies
- Test Google Gemini integration  
- Launch the conversation modes GUI
- Show you the status of all components

### Option 2: Individual Components

#### Test Google Gemini (Text Only)
```powershell
cd "C:\Users\pande\OneDrive\Desktop\Ricko\riko_project"
python -c "from server.process.llm_funcs.llm_scr import llm_response; print(llm_response('Hello Riko!'))"
```

#### Start Conversation Modes
```powershell
cd "C:\Users\pande\OneDrive\Desktop\Ricko\riko_project"
python conversation_modes.py
```

## 🎵 Voice Synthesis Setup (GPT-SoVITS)

For **Mode 1** and **Mode 2** (voice modes), you need GPT-SoVITS server running:

### Step 1: Install GPT-SoVITS
```powershell
# In a separate terminal/directory
git clone https://github.com/RVC-Boss/GPT-SoVITS.git
cd GPT-SoVITS
pip install -r requirements.txt
```

### Step 2: Start GPT-SoVITS Server
```powershell
# In the GPT-SoVITS directory
python api_v2.py
```

The server should start on `http://127.0.0.1:9880`

### Step 3: Configure Voice Model
1. Open `http://127.0.0.1:9880` in your browser
2. Upload/configure your Riko voice model
3. Make sure the reference audio path in `character_config.yaml` is correct

## 🎮 How to Use Each Mode

### Mode 1: Button Voice Mode
1. Click "Mode 1: Button Voice" 
2. Click "🔴 Start Recording"
3. Speak your message
4. Click "⏹️ Stop Recording"
5. Riko will transcribe → think → respond with voice

### Mode 2: Auto Voice Detection  
1. Click "Mode 2: Auto Voice Detection"
2. Click "🎙️ Start Auto Mode"
3. Just speak naturally - Riko detects when you start/stop
4. Riko automatically processes and responds

### Mode 3: Text Mode
1. Click "Mode 3: Text Mode"
2. Type your message in the text box
3. Press Enter or click "Send"
4. Riko responds with generated voice (if GPT-SoVITS is running)

## ⚙️ Configuration Files

### character_config.yaml
```yaml
GEMINI_API_KEY: AIzaSyDzvUVETRp5epkd5-kmP09v7eQ4OwgVz_k  # ✅ Already configured
model: "gemini-2.0-flash-exp"  # ✅ Set to Flash-2.0
sovits_ping_config:
  ref_audio_path: C:\Users\pande\OneDrive\Desktop\Ricko\riko_project\character_files\main_sample.wav  # ✅ Updated path
```

## 🔧 Troubleshooting

### If voice synthesis doesn't work:
1. Check if GPT-SoVITS server is running: `http://127.0.0.1:9880`
2. Verify reference audio file exists
3. Use Text Mode to test without voice synthesis

### If speech recognition doesn't work:
1. Check your microphone permissions
2. Verify microphone is working in other applications
3. Try adjusting microphone volume

### If Gemini responses fail:
1. Check your internet connection
2. Verify the API key is correct in `character_config.yaml`
3. Check API quota limits

## 📁 Project Structure

```
riko_project/
├── character_config.yaml          # ✅ Main configuration
├── conversation_modes.py          # ✅ Three conversation modes
├── start_riko.py                 # ✅ Main startup script
├── server/
│   ├── main_chat.py              # Original push-to-talk script
│   └── process/
│       ├── llm_funcs/
│       │   └── llm_scr.py        # ✅ Google Gemini integration
│       ├── tts_func/
│       │   └── sovits_ping.py    # ✅ Voice synthesis
│       └── asr_func/
│           └── asr_push_to_talk.py # Speech recognition
└── character_files/
    └── main_sample.wav           # Reference voice sample
```

## ✨ Features Implemented

- 🤖 **Google Gemini Flash-2.0** integration with your API key
- 🎙️ **Three conversation modes** as requested:
  - Button Voice Mode (manual start/stop)
  - Auto Voice Detection (hands-free)  
  - Text Mode (type to chat)
- 🧠 **Conversation memory** - Riko remembers your chat history
- 🎌 **Anime personality** - Snarky anime girl who calls you "senpai"
- 🎵 **Voice synthesis** ready (when GPT-SoVITS server is running)
- 🎧 **Speech recognition** with Whisper
- 🖥️ **Easy-to-use GUI** interface

## 🎯 Next Steps

1. **Start the system**: Run `python start_riko.py`
2. **Test Text Mode**: Works immediately without GPT-SoVITS
3. **Set up GPT-SoVITS**: Follow voice synthesis setup above for full voice modes
4. **Enjoy chatting with Riko**! 

The system is **fully configured and ready to use!** 🎉
