#!/usr/bin/env python3
"""
Advanced 4-Mode Chat System for Riko
Modes:
1. Button Voice Mode - Press button to record and train voice
2. Auto Voice Detection - Automatic voice activity detection  
3. Voice-to-Voice Mode - Voice input with voice output
4. Text-to-Text Mode - Pure text interaction
"""

import sys
import os
import time
import threading
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
from pathlib import Path
from typing import Optional, List, Tuple
import uuid
import soundfile as sf
import numpy as np
import queue
import json
from datetime import datetime

# Audio processing imports
import sounddevice as sd
import webrtcvad
from scipy import signal
import librosa

# Add current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import existing modules
from faster_whisper import WhisperModel
from process.llm_funcs.llm_scr import llm_response
from process.tts_func.sovits_ping import sovits_gen, play_audio
from process.text_processing.emotion_filter import EmotionFilter
from process.text_processing.colored_display import ColoredDisplay
from process.emotion_system.emotion_handler import EmotionHandler

class VoiceActivityDetector:
    """Advanced voice activity detection using WebRTC VAD and energy-based detection"""
    
    def __init__(self, sample_rate=16000, frame_duration=30):
        self.sample_rate = sample_rate
        self.frame_duration = frame_duration  # ms
        self.frame_size = int(sample_rate * frame_duration / 1000)
        
        # WebRTC VAD (0=least aggressive, 3=most aggressive)
        self.vad = webrtcvad.Vad(2)  
        
        # Energy-based parameters
        self.energy_threshold = 0.01
        self.silence_duration = 1.0  # seconds of silence to stop
        self.min_speech_duration = 0.5  # minimum speech duration
        
        # State tracking
        self.is_speaking = False
        self.speech_frames = []
        self.silence_frames = 0
        self.speech_start_time = None
        
    def is_speech_frame(self, frame):
        """Determine if a frame contains speech"""
        # Ensure frame is the right size and format
        if len(frame) != self.frame_size:
            return False
            
        # Convert to int16 for WebRTC VAD
        frame_int16 = (frame * 32767).astype(np.int16)
        frame_bytes = frame_int16.tobytes()
        
        try:
            # WebRTC VAD
            vad_result = self.vad.is_speech(frame_bytes, self.sample_rate)
            
            # Energy-based detection
            energy = np.mean(frame ** 2)
            energy_result = energy > self.energy_threshold
            
            # Combine both methods
            return vad_result or energy_result
            
        except Exception as e:
            # Fallback to energy-based only
            energy = np.mean(frame ** 2)
            return energy > self.energy_threshold
    
    def process_frame(self, frame):
        """Process an audio frame and update speech detection state"""
        is_speech = self.is_speech_frame(frame)
        
        if is_speech:
            if not self.is_speaking:
                # Speech started
                self.is_speaking = True
                self.speech_start_time = time.time()
                self.speech_frames = []
                print("🎤 Speech detected - recording started")
            
            self.speech_frames.extend(frame)
            self.silence_frames = 0
            
        else:
            if self.is_speaking:
                self.silence_frames += 1
                silence_duration = self.silence_frames * self.frame_duration / 1000
                
                if silence_duration >= self.silence_duration:
                    # Check if we have enough speech
                    speech_duration = time.time() - self.speech_start_time
                    
                    if speech_duration >= self.min_speech_duration:
                        # Speech ended
                        print("🔇 Speech ended - processing audio")
                        recorded_audio = np.array(self.speech_frames)
                        self.is_speaking = False
                        self.speech_frames = []
                        return recorded_audio
                    else:
                        # Too short, continue listening
                        self.silence_frames = 0
        
        return None

class VoiceTrainer:
    """Voice training system for GPT-SoVITS"""
    
    def __init__(self):
        self.training_samples = []
        self.current_dir = Path(__file__).parent
        self.voice_samples_dir = self.current_dir / "voice_samples"
        self.voice_samples_dir.mkdir(exist_ok=True)
        
    def add_training_sample(self, audio_data, text, sample_rate=16000):
        """Add a voice sample for training"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"voice_sample_{timestamp}.wav"
        filepath = self.voice_samples_dir / filename
        
        # Save the audio sample
        sf.write(filepath, audio_data, sample_rate)
        
        # Add to training database
        sample_data = {
            "filepath": str(filepath),
            "text": text,
            "timestamp": timestamp,
            "duration": len(audio_data) / sample_rate
        }
        
        self.training_samples.append(sample_data)
        self.save_training_database()
        
        print(f"✅ Voice sample added: {filename} ({sample_data['duration']:.2f}s)")
        return filepath
    
    def save_training_database(self):
        """Save training samples database"""
        db_file = self.voice_samples_dir / "training_database.json"
        with open(db_file, 'w') as f:
            json.dump(self.training_samples, f, indent=2)
    
    def load_training_database(self):
        """Load existing training samples"""
        db_file = self.voice_samples_dir / "training_database.json"
        if db_file.exists():
            try:
                with open(db_file, 'r') as f:
                    self.training_samples = json.load(f)
                print(f"📚 Loaded {len(self.training_samples)} training samples")
            except:
                self.training_samples = []
    
    def get_best_reference_sample(self):
        """Get the best reference sample for TTS"""
        if not self.training_samples:
            return None
            
        # For now, return the most recent sample
        # TODO: Implement quality-based selection
        return self.training_samples[-1]["filepath"]

class AdvancedFourModeChat:
    """Advanced 4-Mode Chat System with GUI"""
    
    def __init__(self):
        # Initialize components
        self.emotion_filter = EmotionFilter()
        self.emotion_handler = EmotionHandler()
        self.voice_detector = VoiceActivityDetector()
        self.voice_trainer = VoiceTrainer()
        
        # Load existing training data
        self.voice_trainer.load_training_database()
        
        # Audio settings
        self.sample_rate = 16000
        self.channels = 1
        self.audio_queue = queue.Queue()
        self.is_recording = False
        self.recording_thread = None
        
        # Models
        self.whisper_model = None
        
        # GUI
        self.root = None
        self.current_mode = 1
        
        # Initialize GUI
        self.setup_gui()
        
    def setup_gui(self):
        """Setup the GUI interface"""
        self.root = tk.Tk()
        self.root.title("🎌 Advanced Riko Chat System")
        self.root.geometry("900x700")
        self.root.configure(bg='#2b2b2b')
        
        # Configure style
        style = ttk.Style()
        style.theme_use('clam')
        
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Title
        title_label = tk.Label(main_frame, text="🤖 Advanced Riko Chat System", 
                              font=("Arial", 16, "bold"), bg='#2b2b2b', fg='#ffffff')
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 10))
        
        # Mode selection
        mode_frame = ttk.LabelFrame(main_frame, text="Chat Modes", padding="10")
        mode_frame.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        
        self.mode_var = tk.IntVar(value=1)
        
        modes = [
            (1, "🔘 Button Voice Mode", "Press button to record & train voice"),
            (2, "🎤 Auto Voice Detection", "Automatic voice activity detection"),
            (3, "💬 Voice-to-Voice Mode", "Voice input with voice output"),
            (4, "📝 Text-to-Text Mode", "Pure text interaction")
        ]
        
        for i, (mode_num, title, desc) in enumerate(modes):
            rb = ttk.Radiobutton(mode_frame, text=f"{title}\n{desc}", 
                               variable=self.mode_var, value=mode_num,
                               command=self.on_mode_change)
            rb.grid(row=i//2, column=i%2, sticky=(tk.W), padx=10, pady=5)
        
        # Status display
        self.status_label = tk.Label(main_frame, text="Status: Ready", 
                                   font=("Arial", 10), bg='#2b2b2b', fg='#00ff00')
        self.status_label.grid(row=2, column=0, columnspan=3, pady=(0, 10))
        
        # Chat display
        chat_frame = ttk.LabelFrame(main_frame, text="Chat History", padding="10")
        chat_frame.grid(row=3, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        
        self.chat_display = scrolledtext.ScrolledText(chat_frame, height=20, width=80,
                                                     bg='#1a1a1a', fg='#ffffff', 
                                                     font=("Consolas", 10))
        self.chat_display.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Control panel
        control_frame = ttk.LabelFrame(main_frame, text="Controls", padding="10")
        control_frame.grid(row=4, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Mode 1 & 2 controls (Voice)
        self.record_button = ttk.Button(control_frame, text="🎤 Start Recording", 
                                      command=self.toggle_recording, state="normal")
        self.record_button.grid(row=0, column=0, padx=5)
        
        self.train_button = ttk.Button(control_frame, text="🎓 Train Voice", 
                                     command=self.train_voice_sample, state="disabled")
        self.train_button.grid(row=0, column=1, padx=5)
        
        # Mode 3 & 4 controls (Text)
        self.text_input = tk.Entry(control_frame, width=50, font=("Arial", 10))
        self.text_input.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(10, 0))
        self.text_input.bind('<Return>', self.on_text_submit)
        
        self.send_button = ttk.Button(control_frame, text="📤 Send", 
                                    command=self.on_text_submit, state="disabled")
        self.send_button.grid(row=1, column=2, padx=5, pady=(10, 0))
        
        # Training info
        training_frame = ttk.LabelFrame(main_frame, text="Voice Training Info", padding="10")
        training_frame.grid(row=5, column=0, columnspan=3, sticky=(tk.W, tk.E))
        
        self.training_info = tk.Label(training_frame, 
                                    text=f"Training Samples: {len(self.voice_trainer.training_samples)}",
                                    font=("Arial", 9), bg='#2b2b2b', fg='#ffffff')
        self.training_info.grid(row=0, column=0)
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(3, weight=1)
        chat_frame.columnconfigure(0, weight=1)
        chat_frame.rowconfigure(0, weight=1)
        control_frame.columnconfigure(0, weight=1)
        
        # Initialize mode
        self.on_mode_change()
        
    def add_chat_message(self, sender, message, color="#ffffff"):
        """Add a message to the chat display"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        self.chat_display.configure(state="normal")
        self.chat_display.insert(tk.END, f"[{timestamp}] {sender}: {message}\n")
        
        # Color coding (simplified for tkinter)
        if sender == "You":
            # Could implement colored text here if needed
            pass
        elif sender == "Riko":
            # Could implement colored text here if needed  
            pass
            
        self.chat_display.configure(state="disabled")
        self.chat_display.see(tk.END)
        
    def update_status(self, status, color="#00ff00"):
        """Update status display"""
        self.status_label.configure(text=f"Status: {status}", fg=color)
        self.root.update_idletasks()
        
    def on_mode_change(self):
        """Handle mode change"""
        self.current_mode = self.mode_var.get()
        
        # Reset UI state
        self.stop_recording()
        
        # Configure UI based on mode
        if self.current_mode == 1:  # Button Voice Mode
            self.record_button.configure(state="normal", text="🎤 Start Recording")
            self.train_button.configure(state="disabled")
            self.text_input.configure(state="disabled")
            self.send_button.configure(state="disabled")
            self.update_status("Button Voice Mode - Press record to capture voice for training")
            
        elif self.current_mode == 2:  # Auto Voice Detection
            self.record_button.configure(state="normal", text="🎤 Start Auto Detection")
            self.train_button.configure(state="disabled")
            self.text_input.configure(state="disabled") 
            self.send_button.configure(state="disabled")
            self.update_status("Auto Voice Detection Mode - Automatic speech detection")
            
        elif self.current_mode == 3:  # Voice-to-Voice
            self.record_button.configure(state="normal", text="🎤 Start Voice Chat")
            self.train_button.configure(state="disabled")
            self.text_input.configure(state="disabled")
            self.send_button.configure(state="disabled")
            self.update_status("Voice-to-Voice Mode - Voice input with voice output")
            
        elif self.current_mode == 4:  # Text-to-Text
            self.record_button.configure(state="disabled")
            self.train_button.configure(state="disabled")
            self.text_input.configure(state="normal")
            self.send_button.configure(state="normal")
            self.update_status("Text-to-Text Mode - Pure text interaction")
            
    def initialize_models(self):
        """Initialize AI models"""
        self.update_status("Initializing models...", "#ffff00")
        
        try:
            # Initialize Whisper for speech recognition
            self.whisper_model = WhisperModel("base.en", device="cpu", compute_type="float32")
            self.update_status("Models initialized successfully", "#00ff00")
            return True
        except Exception as e:
            self.update_status(f"Model initialization failed: {e}", "#ff0000")
            return False
    
    def toggle_recording(self):
        """Toggle recording based on current mode"""
        if not self.is_recording:
            self.start_recording()
        else:
            self.stop_recording()
    
    def start_recording(self):
        """Start recording audio"""
        if self.current_mode == 1:
            self.start_button_voice_recording()
        elif self.current_mode == 2:
            self.start_auto_voice_detection()
        elif self.current_mode == 3:
            self.start_voice_to_voice()
    
    def stop_recording(self):
        """Stop recording audio"""
        if self.is_recording:
            self.is_recording = False
            if self.recording_thread:
                self.recording_thread.join(timeout=1.0)
            
            # Update UI
            self.record_button.configure(text=self.get_record_button_text())
            self.update_status("Recording stopped")
    
    def get_record_button_text(self):
        """Get appropriate button text for current mode"""
        if self.current_mode == 1:
            return "🎤 Start Recording"
        elif self.current_mode == 2:
            return "🎤 Start Auto Detection"
        elif self.current_mode == 3:
            return "🎤 Start Voice Chat"
        return "🎤 Record"
    
    def start_button_voice_recording(self):
        """Mode 1: Button Voice Mode with Training"""
        self.is_recording = True
        self.record_button.configure(text="⏹️ Stop Recording")
        self.update_status("Recording for training...", "#ff8800")
        
        # Start recording thread
        self.recording_thread = threading.Thread(target=self.record_for_training)
        self.recording_thread.start()
    
    def record_for_training(self):
        """Record audio for voice training"""
        recorded_audio = []
        
        def audio_callback(indata, frames, time, status):
            if self.is_recording:
                recorded_audio.extend(indata[:, 0])  # Mono channel
        
        try:
            with sd.InputStream(samplerate=self.sample_rate, channels=self.channels, 
                              callback=audio_callback, blocksize=1024):
                while self.is_recording:
                    time.sleep(0.1)
            
            if recorded_audio:
                # Save recorded audio
                audio_array = np.array(recorded_audio)
                self.last_recorded_audio = audio_array
                
                # Enable train button
                self.train_button.configure(state="normal")
                self.update_status(f"Recording complete ({len(audio_array)/self.sample_rate:.2f}s) - Ready to train", "#00ff00")
                
                # Auto-transcribe for training
                self.transcribe_and_prepare_training()
            
        except Exception as e:
            self.update_status(f"Recording error: {e}", "#ff0000")
    
    def transcribe_and_prepare_training(self):
        """Transcribe recorded audio and prepare for training"""
        if not hasattr(self, 'last_recorded_audio'):
            return
            
        try:
            self.update_status("Transcribing audio...", "#ffff00")
            
            # Save temporary audio file for Whisper
            temp_file = Path("temp_recording.wav")
            sf.write(temp_file, self.last_recorded_audio, self.sample_rate)
            
            # Transcribe with Whisper
            if self.whisper_model:
                segments, info = self.whisper_model.transcribe(str(temp_file))
                transcribed_text = " ".join([segment.text for segment in segments])
                
                if transcribed_text.strip():
                    self.last_transcribed_text = transcribed_text.strip()
                    self.add_chat_message("Transcribed", f'"{transcribed_text}"', "#ffff00")
                    self.update_status("Ready to train with transcribed text", "#00ff00")
                else:
                    self.update_status("No speech detected in recording", "#ff8800")
            
            # Clean up
            if temp_file.exists():
                temp_file.unlink()
                
        except Exception as e:
            self.update_status(f"Transcription error: {e}", "#ff0000")
    
    def train_voice_sample(self):
        """Train the voice model with the recorded sample"""
        if not hasattr(self, 'last_recorded_audio') or not hasattr(self, 'last_transcribed_text'):
            messagebox.showerror("Error", "No recorded audio or transcribed text available")
            return
        
        try:
            self.update_status("Adding training sample...", "#ffff00")
            
            # Add to training database
            filepath = self.voice_trainer.add_training_sample(
                self.last_recorded_audio, 
                self.last_transcribed_text,
                self.sample_rate
            )
            
            self.add_chat_message("System", f"Voice sample trained: '{self.last_transcribed_text}'", "#00ff00")
            self.training_info.configure(text=f"Training Samples: {len(self.voice_trainer.training_samples)}")
            self.update_status("Training sample added successfully", "#00ff00")
            
            # Reset state
            self.train_button.configure(state="disabled")
            delattr(self, 'last_recorded_audio')
            delattr(self, 'last_transcribed_text')
            
        except Exception as e:
            self.update_status(f"Training error: {e}", "#ff0000")
    
    def start_auto_voice_detection(self):
        """Mode 2: Auto Voice Detection"""
        self.is_recording = True
        self.record_button.configure(text="⏹️ Stop Detection")
        self.update_status("Auto voice detection active...", "#00ff00")
        
        # Reset voice detector
        self.voice_detector = VoiceActivityDetector()
        
        # Start detection thread
        self.recording_thread = threading.Thread(target=self.auto_voice_detection_loop)
        self.recording_thread.start()
    
    def auto_voice_detection_loop(self):
        """Auto voice detection main loop"""
        def audio_callback(indata, frames, time, status):
            if self.is_recording:
                frame_data = indata[:, 0]  # Mono channel
                self.audio_queue.put(frame_data)
        
        try:
            with sd.InputStream(samplerate=self.sample_rate, channels=self.channels,
                              callback=audio_callback, blocksize=480):  # 30ms frames
                
                while self.is_recording:
                    try:
                        frame = self.audio_queue.get(timeout=0.1)
                        
                        # Process frame for speech detection
                        complete_audio = self.voice_detector.process_frame(frame)
                        
                        if complete_audio is not None:
                            # Speech detected and ended, process it
                            self.process_detected_speech(complete_audio)
                            
                    except queue.Empty:
                        continue
                        
        except Exception as e:
            self.update_status(f"Auto detection error: {e}", "#ff0000")
    
    def process_detected_speech(self, audio_data):
        """Process automatically detected speech"""
        try:
            # Save temporary audio file
            temp_file = Path("temp_auto_speech.wav")
            sf.write(temp_file, audio_data, self.sample_rate)
            
            # Transcribe
            if self.whisper_model:
                segments, info = self.whisper_model.transcribe(str(temp_file))
                transcribed_text = " ".join([segment.text for segment in segments])
                
                if transcribed_text.strip():
                    self.add_chat_message("You", transcribed_text)
                    
                    # Get AI response
                    self.root.after(0, lambda: self.process_ai_response(transcribed_text))
            
            # Clean up
            if temp_file.exists():
                temp_file.unlink()
                
        except Exception as e:
            self.update_status(f"Speech processing error: {e}", "#ff0000")
    
    def start_voice_to_voice(self):
        """Mode 3: Voice-to-Voice Mode"""
        # Similar to auto detection but with voice output
        self.is_recording = True
        self.record_button.configure(text="⏹️ Stop Voice Chat")
        self.update_status("Voice-to-Voice chat active...", "#00ff00")
        
        # Reset voice detector
        self.voice_detector = VoiceActivityDetector()
        
        # Start voice chat thread
        self.recording_thread = threading.Thread(target=self.voice_to_voice_loop)
        self.recording_thread.start()
    
    def voice_to_voice_loop(self):
        """Voice-to-Voice main loop"""
        # Similar implementation to auto_voice_detection_loop
        # but with voice synthesis for responses
        self.auto_voice_detection_loop()  # Reuse the same logic
    
    def on_text_submit(self, event=None):
        """Handle text input submission"""
        text = self.text_input.get().strip()
        if not text:
            return
            
        # Clear input
        self.text_input.delete(0, tk.END)
        
        # Add to chat
        self.add_chat_message("You", text)
        
        # Process with AI
        self.process_ai_response(text)
    
    def process_ai_response(self, user_input):
        """Process user input and generate AI response"""
        try:
            self.update_status("Processing with AI...", "#ffff00")
            
            # Get LLM response
            ai_response = llm_response(user_input)
            
            # Filter emotions
            clean_text, emotions = self.emotion_filter.process_text(ai_response)
            
            # Add to chat
            self.add_chat_message("Riko", clean_text)
            
            # Show emotions
            if emotions:
                emotion_text = " ".join([f"[{e.strip('*')}]" for e in emotions])
                self.add_chat_message("Emotions", emotion_text, "#ffff00")
            
            # Generate voice if not in text-to-text mode
            if self.current_mode != 4 and clean_text.strip():
                self.generate_and_play_voice(clean_text)
            else:
                self.update_status("Response complete", "#00ff00")
                
        except Exception as e:
            self.update_status(f"AI processing error: {e}", "#ff0000")
            self.add_chat_message("System", f"Error: {e}", "#ff0000")
    
    def generate_and_play_voice(self, text):
        """Generate and play voice response"""
        try:
            self.update_status("Generating voice...", "#ffff00")
            
            # Generate unique filename
            uid = uuid.uuid4().hex
            output_file = Path("audio") / f"response_{uid}.wav"
            output_file.parent.mkdir(exist_ok=True)
            
            # Use trained voice reference if available
            reference_sample = self.voice_trainer.get_best_reference_sample()
            
            # Generate TTS
            result = sovits_gen(text, str(output_file))
            
            if result and output_file.exists():
                self.update_status("Playing audio...", "#00ff00")
                
                # Play audio in separate thread to not block UI
                def play_audio_thread():
                    try:
                        play_audio(str(output_file))
                        # Clean up
                        if output_file.exists():
                            output_file.unlink()
                        self.root.after(0, lambda: self.update_status("Response complete", "#00ff00"))
                    except Exception as e:
                        self.root.after(0, lambda: self.update_status(f"Audio playback error: {e}", "#ff0000"))
                
                threading.Thread(target=play_audio_thread, daemon=True).start()
            else:
                self.update_status("Voice generation failed", "#ff0000")
                
        except Exception as e:
            self.update_status(f"Voice generation error: {e}", "#ff0000")
    
    def run(self):
        """Start the application"""
        # Initialize models
        if not self.initialize_models():
            messagebox.showerror("Error", "Failed to initialize models. Some features may not work.")
        
        # Start GUI
        self.root.mainloop()
    
    def __del__(self):
        """Cleanup when application closes"""
        if hasattr(self, 'is_recording'):
            self.is_recording = False

def main():
    """Main entry point"""
    try:
        app = AdvancedFourModeChat()
        app.run()
    except Exception as e:
        print(f"Application error: {e}")
        return 1
    return 0

if __name__ == "__main__":
    exit(main())
