import sys
import os
import time
import threading
from pathlib import Path
from typing import Optional, List, Tuple
import uuid
import soundfile as sf
import keyboard

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import required modules
from faster_whisper import WhisperModel
from process.asr_func.asr_push_to_talk import record_and_transcribe
from process.llm_funcs.llm_scr import llm_response
from process.tts_func.sovits_ping import sovits_gen, play_audio
from process.text_processing.emotion_filter import EmotionFilter
from process.text_processing.colored_display import ColoredDisplay
from process.emotion_system.emotion_handler import EmotionHandler

class RikoAdvancedChat:
    """
    Advanced Chat Interface with Riko featuring:
    1. Voice Chat - Voice input/output with emotion filtering
    2. Text Chat - Direct text input with colored display
    3. Emotion Visualization - Visual feedback for emotions
    """
    
    def __init__(self):
        self.display = ColoredDisplay()
        self.emotion_filter = EmotionFilter()
        self.emotion_handler = EmotionHandler()
        self.whisper_model = None
        self.chat_mode = "voice"  # voice, text, or mixed
        self.running = True
        
        # Audio settings
        self.audio_dir = Path("audio")
        self.audio_dir.mkdir(parents=True, exist_ok=True)
        
    def initialize_models(self):
        """Initialize the Whisper model and other resources"""
        self.display.print_system_message("Initializing models...")
        try:
            self.whisper_model = WhisperModel("base.en", device="cpu", compute_type="float32")
            self.display.print_system_message("Models initialized successfully!")
            return True
        except Exception as e:
            self.display.print_error(f"Failed to initialize models: {e}")
            return False
            
    def get_wav_duration(self, path: Path) -> float:
        """Get the duration of a WAV file"""
        try:
            with sf.SoundFile(path) as f:
                return len(f) / f.samplerate
        except Exception as e:
            self.display.print_error(f"Error getting audio duration: {e}")
            return 0.0
            
    def cleanup_audio_files(self):
        """Clean up temporary audio files"""
        try:
            for file in self.audio_dir.glob("*.wav"):
                if file.is_file():
                    file.unlink()
        except Exception as e:
            self.display.print_error(f"Error cleaning up audio files: {e}")
            
    def voice_chat_cycle(self) -> bool:
        """Handle one cycle of voice chat"""
        try:
            # Step 1: Record and transcribe user input
            self.display.print_listening()
            conversation_recording = self.audio_dir / "conversation.wav"
            user_spoken_text = record_and_transcribe(self.whisper_model, conversation_recording)
            
            if not user_spoken_text or user_spoken_text.strip() == "":
                self.display.print_error("No speech detected. Try again.")
                return True
                
            self.display.print_user_text(user_spoken_text)
            
            # Step 2: Get LLM response
            self.display.print_processing()
            llm_output = llm_response(user_spoken_text)
            
            # Step 3: Process text for emotions and TTS
            tts_text, emotions = self.emotion_filter.process_text(llm_output)
            
            # Step 4: Display response with colored text
            self.display.print_riko_text(tts_text)
            
            # Step 5: Show emotions
            if emotions:
                self.display.print_emotions(emotions)
                # Trigger emotion effects
                for emotion in emotions:
                    self.emotion_handler.trigger_emotion_effect(emotion)
                    time.sleep(0.5)
                    
            # Step 6: Generate and play audio (only clean text)
            if tts_text.strip():
                self.display.print_generating_audio()
                uid = uuid.uuid4().hex
                filename = f"output_{uid}.wav"
                output_wav_path = self.audio_dir / filename
                
                gen_aud_path = sovits_gen(tts_text, output_wav_path)
                
                if output_wav_path.exists():
                    self.display.print_playing_audio()
                    play_audio(output_wav_path)
                else:
                    self.display.print_error("Failed to generate audio")
                    
            # Step 7: Cleanup
            self.cleanup_audio_files()
            
            return True
            
        except KeyboardInterrupt:
            return False
        except Exception as e:
            self.display.print_error(f"Voice chat error: {e}")
            return True
            
    def text_chat_cycle(self) -> bool:
        """Handle one cycle of text chat"""
        try:
            # Step 1: Get text input from user
            self.display.print_info("Type your message (or 'quit' to exit):")
            user_input = input("> ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                return False
                
            if not user_input:
                return True
                
            self.display.print_user_text(user_input)
            
            # Step 2: Get LLM response
            self.display.print_processing()
            llm_output = llm_response(user_input)
            
            # Step 3: Process text for emotions
            display_text, emotions = self.emotion_filter.process_text(llm_output)
            
            # Step 4: Display response
            self.display.print_riko_text(display_text)
            
            # Step 5: Show emotions
            if emotions:
                self.display.print_emotions(emotions)
                for emotion in emotions:
                    self.emotion_handler.trigger_emotion_effect(emotion)
                    time.sleep(0.3)
                    
            return True
            
        except KeyboardInterrupt:
            return False
        except Exception as e:
            self.display.print_error(f"Text chat error: {e}")
            return True
            
    def mixed_chat_cycle(self) -> bool:
        """Handle mixed mode where user can choose input method"""
        try:
            self.display.print_info("Choose input method:")
            self.display.print_info("1. Voice (press V)")
            self.display.print_info("2. Text (press T)")  
            self.display.print_info("3. Quit (press Q)")
            
            # Wait for key press
            while True:
                if keyboard.is_pressed('v'):
                    return self.voice_chat_cycle()
                elif keyboard.is_pressed('t'):
                    return self.text_chat_cycle()
                elif keyboard.is_pressed('q'):
                    return False
                time.sleep(0.1)
                
        except Exception as e:
            self.display.print_error(f"Mixed chat error: {e}")
            return True
            
    def show_main_menu(self):
        """Show the main menu"""
        self.display.print_separator("=", 70)
        print(f"{ColoredDisplay().system_color}🤖 RIKO ADVANCED CHAT SYSTEM 🤖")
        self.display.print_separator("=", 70)
        self.display.print_info("Select Chat Mode:")
        self.display.print_info("1. Voice Chat - Talk to Riko with voice")
        self.display.print_info("2. Text Chat - Type messages to Riko")
        self.display.print_info("3. Mixed Mode - Choose input method each time")
        self.display.print_info("4. Quit")
        self.display.print_separator("-", 70)
        
    def get_user_choice(self) -> str:
        """Get user's menu choice"""
        try:
            choice = input("Enter your choice (1-4): ").strip()
            return choice
        except KeyboardInterrupt:
            return "4"
            
    def run_chat_mode(self):
        """Run the selected chat mode"""
        if self.chat_mode == "voice":
            self.display.print_conversation_start()
            self.display.print_system_message("Voice Chat Mode - Press space to talk, Ctrl+C to return to menu")
            
            while self.running:
                if not self.voice_chat_cycle():
                    break
                    
        elif self.chat_mode == "text":
            self.display.print_conversation_start()
            self.display.print_system_message("Text Chat Mode - Type messages, 'quit' to return to menu")
            
            while self.running:
                if not self.text_chat_cycle():
                    break
                    
        elif self.chat_mode == "mixed":
            self.display.print_conversation_start() 
            self.display.print_system_message("Mixed Mode - Choose input method for each message")
            
            while self.running:
                if not self.mixed_chat_cycle():
                    break
                    
    def run(self):
        """Main application loop"""
        if not self.initialize_models():
            return
            
        try:
            while True:
                self.show_main_menu()
                choice = self.get_user_choice()
                
                if choice == "1":
                    self.chat_mode = "voice"
                    self.run_chat_mode()
                elif choice == "2":
                    self.chat_mode = "text"
                    self.run_chat_mode()
                elif choice == "3":
                    self.chat_mode = "mixed"
                    self.run_chat_mode()
                elif choice == "4":
                    break
                else:
                    self.display.print_error("Invalid choice. Please try again.")
                    time.sleep(1)
                    
        except KeyboardInterrupt:
            pass
        finally:
            self.display.print_system_message("Thanks for chatting with Riko! Goodbye!")
            self.cleanup_audio_files()


def main():
    """Main entry point"""
    chat_system = RikoAdvancedChat()
    chat_system.run()


if __name__ == "__main__":
    main()
