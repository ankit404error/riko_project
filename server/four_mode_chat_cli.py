#!/usr/bin/env python3
"""
Command Line 4-Mode Chat System for Riko
Simple terminal-based interface for the 4 chat modes
"""

import sys
import os
import time
import threading
from pathlib import Path
import uuid
import soundfile as sf
import numpy as np
import sounddevice as sd
from datetime import datetime

# Add current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import existing modules
from faster_whisper import WhisperModel
from process.llm_funcs.llm_scr import llm_response
from process.tts_func.sovits_ping import sovits_gen, play_audio
from process.text_processing.emotion_filter import EmotionFilter
from process.text_processing.colored_display import ColoredDisplay
from process.emotion_system.emotion_handler import EmotionHandler

class SimpleFourModeChat:
    """Simple command line 4-mode chat system"""
    
    def __init__(self):
        # Initialize components
        self.display = ColoredDisplay()
        self.emotion_filter = EmotionFilter()
        self.emotion_handler = EmotionHandler()
        
        # Audio settings
        self.sample_rate = 16000
        self.channels = 1
        self.is_recording = False
        
        # Models
        self.whisper_model = None
        
        # Mode
        self.current_mode = 1
        
    def initialize_models(self):
        """Initialize AI models"""
        self.display.print_system_message("Initializing models...")
        
        try:
            self.whisper_model = WhisperModel("base.en", device="cpu", compute_type="float32")
            self.display.print_system_message("Models initialized successfully!")
            return True
        except Exception as e:
            self.display.print_error(f"Model initialization failed: {e}")
            return False
    
    def show_menu(self):
        """Show the main menu"""
        self.display.print_separator("=", 60)
        print(f"{self.display.system_color}🎌 Advanced 4-Mode Riko Chat System")
        self.display.print_separator("=", 60)
        self.display.print_info("Select Chat Mode:")
        self.display.print_info("1. 🔘 Button Voice Mode - Press ENTER to record & train voice")
        self.display.print_info("2. 🎤 Auto Voice Detection - Automatic voice activity detection")  
        self.display.print_info("3. 💬 Voice-to-Voice Mode - Voice input with voice output")
        self.display.print_info("4. 📝 Text-to-Text Mode - Pure text interaction")
        self.display.print_info("5. 🚪 Exit")
        self.display.print_separator("-", 60)
    
    def get_user_choice(self):
        """Get user's menu choice"""
        try:
            choice = input("Enter your choice (1-5): ").strip()
            return int(choice) if choice.isdigit() else 0
        except:
            return 0
    
    def mode1_button_voice(self):
        """Mode 1: Button Voice Mode with Training"""
        self.display.print_system_message("🔘 Button Voice Mode Active")
        self.display.print_info("Press ENTER to start recording for training, 'q' to quit mode")
        
        while True:
            user_input = input("\n[Press ENTER to record, 'q' to quit]: ").strip().lower()
            
            if user_input == 'q':
                break
            elif user_input == '':
                self.record_and_train_voice()
            else:
                self.display.print_error("Invalid input. Press ENTER to record or 'q' to quit.")
    
    def record_and_train_voice(self):
        """Record voice and add to training"""
        self.display.print_system_message("🎤 Recording for 5 seconds... Speak now!")
        
        try:
            # Record audio
            duration = 5  # seconds
            recorded_audio = sd.rec(int(duration * self.sample_rate), 
                                  samplerate=self.sample_rate, 
                                  channels=self.channels)
            sd.wait()
            
            # Save temporary audio file
            temp_file = Path("temp_training_recording.wav")
            sf.write(temp_file, recorded_audio, self.sample_rate)
            
            self.display.print_system_message("🎯 Transcribing audio...")
            
            # Transcribe with Whisper
            if self.whisper_model:
                segments, info = self.whisper_model.transcribe(str(temp_file))
                transcribed_text = " ".join([segment.text for segment in segments])
                
                if transcribed_text.strip():
                    self.display.print_user_text(f"Transcribed: '{transcribed_text}'")
                    
                    # Save as training sample
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    training_file = Path("voice_samples") / f"training_{timestamp}.wav"
                    training_file.parent.mkdir(exist_ok=True)
                    
                    # Copy the recording to training samples
                    import shutil
                    shutil.copy2(temp_file, training_file)
                    
                    self.display.print_system_message(f"✅ Voice sample saved for training: {training_file.name}")
                    self.display.print_info("This sample can be used to improve voice quality!")
                    
                else:
                    self.display.print_error("No speech detected in recording")
            
            # Clean up
            if temp_file.exists():
                temp_file.unlink()
                
        except Exception as e:
            self.display.print_error(f"Recording error: {e}")
    
    def mode2_auto_voice_detection(self):
        """Mode 2: Auto Voice Detection"""
        self.display.print_system_message("🎤 Auto Voice Detection Mode Active")
        self.display.print_info("Speak naturally - I'll detect when you start and stop talking")
        self.display.print_info("Press Ctrl+C to exit this mode")
        
        try:
            self.is_recording = True
            
            # Simple energy-based voice detection
            def audio_callback(indata, frames, time, status):
                if self.is_recording:
                    # Simple energy threshold detection
                    energy = np.mean(indata ** 2)
                    if energy > 0.01:  # Adjust threshold as needed
                        print("🎤", end="", flush=True)  # Visual indicator
                    
            # Start audio stream
            with sd.InputStream(samplerate=self.sample_rate, 
                              channels=self.channels,
                              callback=audio_callback):
                
                while self.is_recording:
                    # Simulate voice activity detection
                    # In a full implementation, this would use WebRTC VAD
                    time.sleep(0.1)
                    
        except KeyboardInterrupt:
            self.is_recording = False
            self.display.print_system_message("\n🔇 Auto voice detection stopped")
    
    def mode3_voice_to_voice(self):
        """Mode 3: Voice-to-Voice Mode"""
        self.display.print_system_message("💬 Voice-to-Voice Mode Active")
        self.display.print_info("Press ENTER to speak, then I'll respond with voice")
        self.display.print_info("Type 'q' to quit this mode")
        
        while True:
            user_input = input("\n[Press ENTER to speak, 'q' to quit]: ").strip().lower()
            
            if user_input == 'q':
                break
            elif user_input == '':
                self.voice_interaction()
            else:
                self.display.print_error("Invalid input. Press ENTER to speak or 'q' to quit.")
    
    def voice_interaction(self):
        """Handle one voice interaction"""
        try:
            # Record user speech
            self.display.print_system_message("🎤 Listening... (5 seconds)")
            duration = 5
            recorded_audio = sd.rec(int(duration * self.sample_rate), 
                                  samplerate=self.sample_rate, 
                                  channels=self.channels)
            sd.wait()
            
            # Save and transcribe
            temp_file = Path("temp_voice_input.wav")
            sf.write(temp_file, recorded_audio, self.sample_rate)
            
            self.display.print_system_message("🎯 Understanding your speech...")
            
            if self.whisper_model:
                segments, info = self.whisper_model.transcribe(str(temp_file))
                transcribed_text = " ".join([segment.text for segment in segments])
                
                if transcribed_text.strip():
                    # Display what user said
                    self.display.print_user_text(transcribed_text)
                    
                    # Get AI response
                    self.process_and_respond(transcribed_text, with_voice=True)
                else:
                    self.display.print_error("No speech detected. Please try again.")
            
            # Clean up
            if temp_file.exists():
                temp_file.unlink()
                
        except Exception as e:
            self.display.print_error(f"Voice interaction error: {e}")
    
    def mode4_text_to_text(self):
        """Mode 4: Text-to-Text Mode"""
        self.display.print_system_message("📝 Text-to-Text Mode Active")
        self.display.print_info("Type your messages to chat with Riko")
        self.display.print_info("Type 'quit' to exit this mode")
        
        while True:
            user_input = input("\n💬 You: ").strip()
            
            if user_input.lower() in ['quit', 'q', 'exit']:
                break
            elif user_input:
                self.process_and_respond(user_input, with_voice=False)
            else:
                self.display.print_error("Please enter a message or 'quit' to exit.")
    
    def process_and_respond(self, user_input, with_voice=True):
        """Process user input and generate response"""
        try:
            self.display.print_system_message("🤔 Riko is thinking...")
            
            # Get AI response
            ai_response = llm_response(user_input)
            
            # Filter emotions
            clean_text, emotions = self.emotion_filter.process_text(ai_response)
            
            # Display response
            self.display.print_riko_text(clean_text)
            
            # Show emotions
            if emotions:
                self.display.print_emotions(emotions)
                # Trigger emotion effects
                for emotion in emotions:
                    self.emotion_handler.trigger_emotion_effect(emotion)
                    time.sleep(0.3)
            
            # Generate voice if requested
            if with_voice and clean_text.strip():
                self.generate_voice_response(clean_text)
                
        except Exception as e:
            self.display.print_error(f"Response generation error: {e}")
    
    def generate_voice_response(self, text):
        """Generate and play voice response"""
        try:
            self.display.print_system_message("🎵 Generating Riko's voice...")
            
            # Generate unique filename
            uid = uuid.uuid4().hex
            output_file = Path("audio") / f"response_{uid}.wav"
            output_file.parent.mkdir(exist_ok=True)
            
            # Generate TTS
            result = sovits_gen(text, str(output_file))
            
            if result and output_file.exists():
                self.display.print_system_message("🔊 Playing Riko's response...")
                play_audio(str(output_file))
                
                # Clean up
                output_file.unlink()
            else:
                self.display.print_error("Voice generation failed")
                
        except Exception as e:
            self.display.print_error(f"Voice generation error: {e}")
    
    def run(self):
        """Main application loop"""
        # Initialize
        if not self.initialize_models():
            self.display.print_error("Failed to initialize models. Some features may not work.")
            return
        
        # Main loop
        while True:
            try:
                self.show_menu()
                choice = self.get_user_choice()
                
                if choice == 1:
                    self.mode1_button_voice()
                elif choice == 2:
                    self.mode2_auto_voice_detection()
                elif choice == 3:
                    self.mode3_voice_to_voice()
                elif choice == 4:
                    self.mode4_text_to_text()
                elif choice == 5:
                    self.display.print_system_message("Thanks for chatting with Riko! Goodbye! 👋")
                    break
                else:
                    self.display.print_error("Invalid choice. Please try again.")
                    time.sleep(1)
                    
            except KeyboardInterrupt:
                self.display.print_system_message("\n\nExiting... Thanks for chatting with Riko! 👋")
                break
            except Exception as e:
                self.display.print_error(f"Unexpected error: {e}")

def main():
    """Main entry point"""
    try:
        chat_system = SimpleFourModeChat()
        chat_system.run()
        return 0
    except Exception as e:
        print(f"Application error: {e}")
        return 1

if __name__ == "__main__":
    exit(main())
