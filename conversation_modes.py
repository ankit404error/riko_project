#!/usr/bin/env python3
"""
Riko Conversation Modes
======================

This script implements three conversation modes for interacting with Riko:

Mode 1: Button Voice Mode - Press a button to start/stop recording
Mode 2: Auto Voice Detection Mode - Automatically detect when user starts/stops speaking
Mode 3: Text Mode - Type text and get voice responses from Riko

Requirements:
- GPT-SoVITS server running on http://127.0.0.1:9880
- Google Gemini Flash-2.0 API configured
- Audio devices for recording and playback
"""

import os
import sys
import time
import threading
import queue
from pathlib import Path
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import sounddevice as sd
import soundfile as sf
import numpy as np
from faster_whisper import WhisperModel
import uuid
import collections

# Add server directory to path for imports
sys.path.append(str(Path(__file__).parent / "server"))
from process.llm_funcs.llm_scr import llm_response
from process.tts_func.sovits_ping import sovits_gen, play_audio

class ConversationModes:
    def __init__(self):
        self.whisper_model = None
        self.recording = False
        self.auto_mode_active = False
        self.audio_queue = queue.Queue()
        
        # Audio parameters
        self.sample_rate = 44100
        self.chunk_duration = 0.1  # 100ms chunks
        
        # Voice activity detection parameters (simple amplitude-based)
        self.amplitude_threshold = 0.01
        self.silence_threshold = 20  # frames of silence before stopping
        self.voice_threshold = 5    # frames of voice before starting
        
        self.setup_gui()
        self.init_whisper()
        
    def init_whisper(self):
        """Initialize Whisper model for transcription"""
        print("Loading Whisper model...")
        self.whisper_model = WhisperModel("base.en", device="cpu", compute_type="float32")
        print("Whisper model loaded!")
        
    def setup_gui(self):
        """Setup the GUI interface"""
        self.root = tk.Tk()
        self.root.title("Riko Conversation Modes")
        self.root.geometry("800x600")
        
        # Mode selection
        mode_frame = ttk.LabelFrame(self.root, text="Conversation Mode", padding="10")
        mode_frame.pack(fill="x", padx=10, pady=5)
        
        self.mode_var = tk.StringVar(value="button")
        
        ttk.Radiobutton(mode_frame, text="Mode 1: Button Voice", 
                       variable=self.mode_var, value="button",
                       command=self.switch_mode).pack(side="left", padx=10)
        
        ttk.Radiobutton(mode_frame, text="Mode 2: Auto Voice Detection", 
                       variable=self.mode_var, value="auto",
                       command=self.switch_mode).pack(side="left", padx=10)
        
        ttk.Radiobutton(mode_frame, text="Mode 3: Text Mode", 
                       variable=self.mode_var, value="text",
                       command=self.switch_mode).pack(side="left", padx=10)
        
        # Control buttons
        control_frame = ttk.Frame(self.root)
        control_frame.pack(fill="x", padx=10, pady=5)
        
        self.record_button = ttk.Button(control_frame, text="🔴 Start Recording", 
                                       command=self.toggle_recording)
        self.record_button.pack(side="left", padx=5)
        
        self.auto_button = ttk.Button(control_frame, text="🎙️ Start Auto Mode", 
                                     command=self.toggle_auto_mode)
        self.auto_button.pack(side="left", padx=5)
        
        # Text input for Mode 3
        text_frame = ttk.LabelFrame(self.root, text="Text Input (Mode 3)", padding="10")
        text_frame.pack(fill="x", padx=10, pady=5)
        
        self.text_entry = tk.Entry(text_frame, font=("Arial", 12))
        self.text_entry.pack(side="left", fill="x", expand=True, padx=(0, 10))
        self.text_entry.bind("<Return>", self.send_text_message)
        
        ttk.Button(text_frame, text="Send", command=self.send_text_message).pack(side="right")
        
        # Conversation display
        chat_frame = ttk.LabelFrame(self.root, text="Conversation", padding="10")
        chat_frame.pack(fill="both", expand=True, padx=10, pady=5)
        
        self.chat_display = scrolledtext.ScrolledText(chat_frame, height=20, font=("Arial", 10))
        self.chat_display.pack(fill="both", expand=True)
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready - Select a mode to begin")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief="sunken")
        status_bar.pack(fill="x", side="bottom")
        
        self.switch_mode()
        
    def switch_mode(self):
        """Switch between conversation modes"""
        mode = self.mode_var.get()
        
        # Stop any active recordings
        self.recording = False
        self.auto_mode_active = False
        
        if mode == "button":
            self.record_button.config(state="normal")
            self.auto_button.config(state="disabled")
            self.text_entry.config(state="disabled")
            self.status_var.set("Button Voice Mode - Click 'Start Recording' to begin")
            
        elif mode == "auto":
            self.record_button.config(state="disabled")
            self.auto_button.config(state="normal")
            self.text_entry.config(state="disabled")
            self.status_var.set("Auto Voice Detection Mode - Click 'Start Auto Mode' to begin")
            
        elif mode == "text":
            self.record_button.config(state="disabled")
            self.auto_button.config(state="disabled")
            self.text_entry.config(state="normal")
            self.status_var.set("Text Mode - Type your message and press Enter")
            
    def add_to_chat(self, speaker, message, color="black"):
        """Add message to chat display"""
        self.chat_display.config(state="normal")
        self.chat_display.insert("end", f"{speaker}: {message}\n\n")
        self.chat_display.config(state="disabled")
        self.chat_display.see("end")
        
    def toggle_recording(self):
        """Toggle recording for Button Voice Mode"""
        if not self.recording:
            self.start_button_recording()
        else:
            self.stop_recording()
            
    def start_button_recording(self):
        """Start recording for button mode"""
        self.recording = True
        self.record_button.config(text="⏹️ Stop Recording", state="normal")
        self.status_var.set("🔴 Recording... Click 'Stop Recording' when done")
        
        # Start recording in separate thread
        self.record_thread = threading.Thread(target=self.record_audio_button_mode)
        self.record_thread.start()
        
    def record_audio_button_mode(self):
        """Record audio for button mode"""
        try:
            audio_buffer = []
            chunk_size = int(self.sample_rate * self.chunk_duration)
            
            with sd.InputStream(samplerate=self.sample_rate, channels=1, blocksize=chunk_size) as stream:
                while self.recording:
                    audio_chunk, overflowed = stream.read(chunk_size)
                    if overflowed:
                        print("Audio input overflow")
                    audio_buffer.append(audio_chunk)
                    time.sleep(0.01)
                    
            # Combine all audio chunks
            if audio_buffer:
                full_audio = np.concatenate(audio_buffer, axis=0)
                self.process_audio_recording(full_audio)
                
        except Exception as e:
            print(f"Recording error: {e}")
            self.status_var.set("Recording error occurred")
            
    def stop_recording(self):
        """Stop current recording"""
        self.recording = False
        self.record_button.config(text="🔴 Start Recording", state="normal")
        
    def toggle_auto_mode(self):
        """Toggle auto voice detection mode"""
        if not self.auto_mode_active:
            self.start_auto_mode()
        else:
            self.stop_auto_mode()
            
    def start_auto_mode(self):
        """Start auto voice detection mode"""
        self.auto_mode_active = True
        self.auto_button.config(text="⏹️ Stop Auto Mode")
        self.status_var.set("🎙️ Auto Mode Active - Speak naturally, I'll detect when you start/stop")
        
        # Start audio monitoring in separate thread
        self.auto_thread = threading.Thread(target=self.auto_voice_detection)
        self.auto_thread.start()
        
    def stop_auto_mode(self):
        """Stop auto voice detection mode"""
        self.auto_mode_active = False
        self.auto_button.config(text="🎙️ Start Auto Mode")
        self.status_var.set("Auto Voice Detection Mode - Click 'Start Auto Mode' to begin")
        
    def auto_voice_detection(self):
        """Main loop for auto voice detection using simple amplitude detection"""
        try:
            chunk_size = int(self.sample_rate * self.chunk_duration)
            
            with sd.InputStream(samplerate=self.sample_rate, channels=1, blocksize=chunk_size) as stream:
                recording_buffer = []
                voice_frames = 0
                silence_frames = 0
                is_recording = False
                
                while self.auto_mode_active:
                    try:
                        audio_chunk, overflowed = stream.read(chunk_size)
                        if overflowed:
                            print("Audio input overflow")
                            
                        # Simple voice activity detection based on amplitude
                        amplitude = np.max(np.abs(audio_chunk))
                        has_voice = amplitude > self.amplitude_threshold
                        
                        if has_voice:
                            voice_frames += 1
                            silence_frames = 0
                            
                            # Start recording if we detect voice
                            if not is_recording and voice_frames >= self.voice_threshold:
                                is_recording = True
                                recording_buffer = []
                                self.status_var.set("🔴 Voice detected - Recording...")
                                
                        else:
                            silence_frames += 1
                            voice_frames = max(0, voice_frames - 1)
                            
                            # Stop recording if we have enough silence
                            if is_recording and silence_frames >= self.silence_threshold:
                                is_recording = False
                                self.status_var.set("🎯 Processing speech...")
                                
                                # Process the recorded audio
                                if recording_buffer:
                                    full_audio = np.concatenate(recording_buffer, axis=0)
                                    self.process_auto_recording(full_audio)
                                    
                                voice_frames = 0
                                silence_frames = 0
                                
                        # Add to recording buffer if recording
                        if is_recording:
                            recording_buffer.append(audio_chunk)
                            
                    except Exception as e:
                        print(f"Auto detection error: {e}")
                        continue
                        
        except Exception as e:
            print(f"Auto voice detection setup error: {e}")
            self.status_var.set("Error in auto voice detection")
            
    def process_audio_recording(self, audio_data):
        """Process recorded audio and generate response"""
        try:
            # Save audio to temporary file
            audio_file = Path("temp_audio.wav")
            sf.write(audio_file, audio_data, self.sample_rate)
            
            self.status_var.set("🎯 Transcribing speech...")
            
            # Transcribe audio
            segments, _ = self.whisper_model.transcribe(str(audio_file))
            transcription = " ".join([segment.text for segment in segments]).strip()
            
            if transcription:
                self.add_to_chat("You", transcription)
                self.generate_riko_response(transcription)
            else:
                self.status_var.set("No speech detected - try speaking louder")
                
            # Clean up
            audio_file.unlink(missing_ok=True)
            
        except Exception as e:
            print(f"Audio processing error: {e}")
            self.status_var.set("Error processing audio")
            
    def process_auto_recording(self, audio_data):
        """Process auto-detected recording"""
        self.process_audio_recording(audio_data)
        self.status_var.set("🎙️ Auto Mode Active - Speak naturally, I'll detect when you start/stop")
        
    def generate_riko_response(self, user_input):
        """Generate and play Riko's response"""
        try:
            self.status_var.set("🤔 Riko is thinking...")
            
            # Get LLM response
            riko_response = llm_response(user_input)
            self.add_to_chat("Riko", riko_response)
            
            self.status_var.set("🎵 Generating Riko's voice...")
            
            # Generate audio
            uid = uuid.uuid4().hex
            output_path = Path("audio") / f"riko_response_{uid}.wav"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            generated_audio = sovits_gen(riko_response, str(output_path))
            
            if generated_audio:
                self.status_var.set("🔊 Playing Riko's response...")
                play_audio(output_path)
                
                # Clean up
                output_path.unlink(missing_ok=True)
            else:
                self.status_var.set("Error generating voice - check if GPT-SoVITS server is running")
                
        except Exception as e:
            print(f"Response generation error: {e}")
            self.status_var.set("Error generating response")
            
        finally:
            # Reset status based on current mode
            mode = self.mode_var.get()
            if mode == "button":
                self.status_var.set("Button Voice Mode - Click 'Start Recording' to begin")
            elif mode == "auto":
                self.status_var.set("🎙️ Auto Mode Active - Speak naturally, I'll detect when you start/stop")
            elif mode == "text":
                self.status_var.set("Text Mode - Type your message and press Enter")
                
    def send_text_message(self, event=None):
        """Send text message (Mode 3)"""
        user_text = self.text_entry.get().strip()
        if user_text:
            self.text_entry.delete(0, tk.END)
            self.add_to_chat("You", user_text)
            
            # Generate response in separate thread to avoid blocking GUI
            response_thread = threading.Thread(target=self.generate_riko_response, args=(user_text,))
            response_thread.start()
            
    def run(self):
        """Start the GUI application"""
        try:
            self.add_to_chat("System", "Riko conversation modes ready! Select a mode above to begin.")
            self.root.mainloop()
        except KeyboardInterrupt:
            print("\nShutting down...")
        finally:
            self.cleanup()
            
    def cleanup(self):
        """Cleanup resources"""
        self.auto_mode_active = False
        self.recording = False
        
        # Clean up any temporary audio files
        for audio_file in Path(".").glob("temp_*.wav"):
            audio_file.unlink(missing_ok=True)
            
        # Clean up audio directory
        audio_dir = Path("audio")
        if audio_dir.exists():
            for audio_file in audio_dir.glob("*.wav"):
                audio_file.unlink(missing_ok=True)

def main():
    """Main function to run the conversation modes application"""
    print("="*50)
    print("🎌 Riko Conversation Modes 🎌")
    print("="*50)
    print()
    print("Starting application...")
    
    # Check if GPT-SoVITS server is running
    try:
        import requests
        response = requests.get("http://127.0.0.1:9880", timeout=5)
        print("✅ GPT-SoVITS server is running")
    except:
        print("⚠️  Warning: GPT-SoVITS server doesn't seem to be running on port 9880")
        print("   Please start the GPT-SoVITS server first")
        
    app = ConversationModes()
    app.run()

if __name__ == "__main__":
    main()
