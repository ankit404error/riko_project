#!/usr/bin/env python3
"""
CLI-based 4-Mode Chat System for Riko
1. Push Talk Mode - Press Enter to start/stop recording
2. Speech Detection Mode - Auto detect speech start/stop
3. Text to Speech Mode - Type text, Riko responds with voice
4. Text to Text Mode - Pure text conversation
"""

import sys
import os
import time
import threading
import keyboard
from pathlib import Path
import soundfile as sf
import numpy as np
import queue
from datetime import datetime
import uuid

# Audio processing
import sounddevice as sd
import webrtcvad

# Add current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import existing modules
try:
    from faster_whisper import WhisperModel
    from process.llm_funcs.llm_scr import llm_response
    from process.tts_func.sovits_ping import sovits_gen, play_audio
    from process.text_processing.emotion_filter import EmotionFilter
except ImportError as e:
    print(f"⚠️  Warning: Could not import module {e}")
    print("Some features may not work properly.")

class VoiceActivityDetector:
    """Simple voice activity detection"""
    
    def __init__(self, sample_rate=16000, frame_duration=30):
        self.sample_rate = sample_rate
        self.frame_duration = frame_duration
        self.frame_size = int(sample_rate * frame_duration / 1000)
        
        try:
            self.vad = webrtcvad.Vad(2)
        except:
            self.vad = None
            print("⚠️  WebRTC VAD not available, using energy detection only")
        
        self.energy_threshold = 0.01
        self.silence_duration = 1.5  # seconds
        self.min_speech_duration = 0.5
        
        self.is_speaking = False
        self.speech_frames = []
        self.silence_frames = 0
        self.speech_start_time = None
        
    def is_speech_frame(self, frame):
        """Check if frame contains speech"""
        if len(frame) != self.frame_size:
            return False
            
        frame = np.clip(frame, -1.0, 1.0)
        energy = np.mean(np.abs(frame))
        
        if self.vad:
            try:
                frame_int16 = (frame * 32767).astype(np.int16)
                frame_bytes = frame_int16.tobytes()
                vad_result = self.vad.is_speech(frame_bytes, self.sample_rate)
                return vad_result or energy > self.energy_threshold
            except:
                pass
        
        return energy > self.energy_threshold
    
    def process_frame(self, frame):
        """Process audio frame and return complete speech if detected"""
        is_speech = self.is_speech_frame(frame)
        
        if is_speech:
            if not self.is_speaking:
                self.is_speaking = True
                self.speech_start_time = time.time()
                self.speech_frames = []
                print("🎤 Speech detected...")
            
            self.speech_frames.extend(frame)
            self.silence_frames = 0
        else:
            if self.is_speaking:
                self.silence_frames += 1
                silence_duration = self.silence_frames * self.frame_duration / 1000
                
                if silence_duration >= self.silence_duration:
                    speech_duration = time.time() - self.speech_start_time
                    
                    if speech_duration >= self.min_speech_duration:
                        print("🔇 Speech ended, processing...")
                        recorded_audio = np.array(self.speech_frames)
                        self.is_speaking = False
                        self.speech_frames = []
                        return recorded_audio
                    else:
                        self.silence_frames = 0
        
        return None

class CLIChatSystem:
    """CLI-based chat system with 4 modes"""
    
    def __init__(self):
        self.sample_rate = 16000
        self.channels = 1
        self.audio_queue = queue.Queue()
        self.is_recording = False
        self.current_mode = 1
        
        # Initialize components
        self.whisper_model = None
        self.emotion_filter = None
        self.voice_detector = VoiceActivityDetector()
        
        # Audio directory
        self.audio_dir = Path("audio")
        self.audio_dir.mkdir(exist_ok=True)
        
        self.init_models()
        
    def init_models(self):
        """Initialize AI models"""
        print("🔄 Initializing models...")
        
        try:
            self.whisper_model = WhisperModel("base.en", device="cpu", compute_type="float32")
            print("✅ Whisper model loaded")
        except Exception as e:
            print(f"❌ Failed to load Whisper: {e}")
        
        try:
            self.emotion_filter = EmotionFilter()
            print("✅ Emotion filter loaded")
        except Exception as e:
            print(f"❌ Failed to load emotion filter: {e}")
        
        print("🚀 Models initialized!")
    
    def clean_transcription(self, text):
        """Clean transcription text"""
        import re
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Remove common artifacts
        artifacts = ["You", "you", "Thank you.", "Thanks for watching!", "Bye", "Hello"]
        words = text.split()
        
        while words and words[-1] in artifacts:
            words.pop()
        
        if len(words) >= 2 and words[-1] == words[-2]:
            words.pop()
            
        cleaned_text = ' '.join(words)
        return cleaned_text if cleaned_text not in artifacts else ""
    
    def transcribe_audio(self, audio_file):
        """Transcribe audio file"""
        if not self.whisper_model:
            return ""
        
        try:
            segments, info = self.whisper_model.transcribe(str(audio_file))
            text = " ".join([segment.text for segment in segments])
            return self.clean_transcription(text)
        except Exception as e:
            print(f"❌ Transcription error: {e}")
            return ""
    
    def get_user_emotion(self, user_input):
        """Analyze user input for emotions and return appropriate emotional context"""
        # Simple emotion detection for user input
        user_emotions = []
        emotion_keywords = {
            "excited": ["wow", "amazing", "awesome", "great", "love", "excited", "fantastic"],
            "happy": ["happy", "glad", "good", "nice", "thank", "thanks", "pleased"],
            "sad": ["sad", "sorry", "upset", "down", "depressed", "disappointed"],
            "angry": ["angry", "mad", "annoyed", "frustrated", "hate", "stupid", "damn"],
            "confused": ["confused", "what", "huh", "don't understand", "unclear"],
            "curious": ["how", "why", "what", "when", "where", "tell me", "explain"]
        }
        
        user_input_lower = user_input.lower()
        for emotion, keywords in emotion_keywords.items():
            if any(keyword in user_input_lower for keyword in keywords):
                user_emotions.append(emotion)
                
        return user_emotions
    
    def display_user_emotions(self, emotions):
        """Display user emotions with visual indicators"""
        if not emotions:
            return
            
        emotion_visuals = {
            "excited": "🤩",
            "happy": "😊", 
            "sad": "😢",
            "angry": "😡",
            "confused": "🤔",
            "curious": "🧐"
        }
        
        visual_emotions = [emotion_visuals.get(emotion, "😐") for emotion in emotions]
        emotion_text = " ".join([f"[{emotion}]{visual}" for emotion, visual in zip(emotions, visual_emotions)])
        print(f"👤 Your emotions: {emotion_text}")
    
    def display_riko_emotions(self, emotions):
        """Display Riko's emotions with visual indicators"""
        if not emotions:
            return
            
        # Emotion mapping for Riko's expressions
        riko_emotion_visuals = {
            "rolls eyes": "🙄",
            "roll eyes": "🙄", 
            "sigh": "😮‍💨",
            "sighs": "😮‍💨",
            "hmph": "😤",
            "angry face": "😠",
            "creates angry face": "😠",
            "pout": "😒",
            "pouts": "😒",
            "smile": "😊",
            "smiles": "😊",
            "laugh": "😂",
            "laughs": "😂",
            "giggle": "😄",
            "giggles": "😄",
            "blush": "😊",
            "blushes": "😊",
            "wink": "😉",
            "winks": "😉",
            "glare": "😡",
            "glares": "😡",
            "shrug": "🤷",
            "shrugs": "🤷",
            "nod": "👍",
            "nods": "👍",
            "crosses arms": "🙅",
            "cross arms": "🙅"
        }
        
        visual_emotions = []
        for emotion in emotions:
            clean_emotion = emotion.strip('*').lower()
            visual = riko_emotion_visuals.get(clean_emotion, "😐")
            visual_emotions.append(f"[{clean_emotion}]{visual}")
            
        emotion_text = " ".join(visual_emotions)
        print(f"🤖 Riko's emotions: {emotion_text}")
    
    def get_ai_response(self, user_input, voice_mode=False):
        """Get AI response with optional short response for voice modes and emotion context"""
        try:
            # Detect user emotions
            user_emotions = self.get_user_emotion(user_input)
            self.display_user_emotions(user_emotions)
            
            # Add emotional context to the prompt
            emotional_context = ""
            if user_emotions:
                emotional_context = f" (User seems {', '.join(user_emotions)}. Respond appropriately with matching emotional tone.)"
            
            # For voice modes, request shorter responses
            if voice_mode:
                modified_input = f"{user_input} (Please respond in 1-2 short sentences only){emotional_context}"
            else:
                modified_input = f"{user_input}{emotional_context}"
                
            response = llm_response(modified_input)
            
            if self.emotion_filter:
                clean_text, emotions = self.emotion_filter.process_text(response)
                return clean_text, emotions
            else:
                return response, []
        except Exception as e:
            print(f"❌ AI response error: {e}")
            return "Sorry, I couldn't process that.", []
    
    def generate_speech(self, text):
        """Generate and play speech"""
        try:
            uid = uuid.uuid4().hex
            output_file = self.audio_dir / f"response_{uid}.wav"
            
            print("🎵 Generating speech...")
            result = sovits_gen(text, str(output_file))
            
            if result and output_file.exists():
                print("🔊 Playing audio...")
                play_audio(str(output_file))
                output_file.unlink()  # Clean up
                print("✅ Audio complete")
            else:
                print("❌ Speech generation failed")
        except Exception as e:
            print(f"❌ Speech error: {e}")
    
    def record_audio(self, duration=None):
        """Record audio"""
        print("🎤 Recording... (Press Enter to stop)" if not duration else f"🎤 Recording for {duration}s...")
        
        recorded_audio = []
        self.is_recording = True
        
        def audio_callback(indata, frames, time, status):
            if self.is_recording:
                recorded_audio.extend(indata[:, 0])
        
        try:
            with sd.InputStream(samplerate=self.sample_rate, channels=self.channels, 
                              callback=audio_callback, blocksize=1024):
                
                if duration:
                    time.sleep(duration)
                    self.is_recording = False
                else:
                    input()  # Wait for Enter key
                    self.is_recording = False
                    
            if recorded_audio:
                audio_array = np.array(recorded_audio)
                temp_file = Path("temp_recording.wav")
                sf.write(temp_file, audio_array, self.sample_rate)
                return temp_file
        except Exception as e:
            print(f"❌ Recording error: {e}")
        
        return None
    
    def auto_speech_detection(self):
        """Auto speech detection mode"""
        print("🎤 Auto speech detection active... (Press Ctrl+C to stop)")
        
        self.voice_detector = VoiceActivityDetector()
        
        def audio_callback(indata, frames, time, status):
            frame_data = indata[:, 0]
            self.audio_queue.put(frame_data)
        
        try:
            with sd.InputStream(samplerate=self.sample_rate, channels=self.channels,
                              callback=audio_callback, blocksize=480):
                
                while True:
                    try:
                        frame = self.audio_queue.get(timeout=0.1)
                        complete_audio = self.voice_detector.process_frame(frame)
                        
                        if complete_audio is not None:
                            # Save and process speech
                            temp_file = Path("temp_auto_speech.wav")
                            sf.write(temp_file, complete_audio, self.sample_rate)
                            
                            text = self.transcribe_audio(temp_file)
                            temp_file.unlink()
                            
                            if text.strip():
                                print(f"👤 You: {text}")
                                response, emotions = self.get_ai_response(text, voice_mode=True)
                                print(f"🤖 Riko: {response}")
                                
                                # Show Riko's emotions
                                if emotions:
                                    self.display_riko_emotions(emotions)
                                
                                if self.current_mode in [2, 3]:  # Modes with speech output
                                    self.generate_speech(response)
                                
                                print("\n" + "="*50 + "\n")
                            
                    except queue.Empty:
                        continue
                    except KeyboardInterrupt:
                        break
                        
        except Exception as e:
            print(f"❌ Auto detection error: {e}")
    
    def mode_1_push_talk(self):
        """Mode 1: Push to Talk"""
        print("\n🎙️  MODE 1: PUSH TO TALK")
        print("Press Enter to start recording, Enter again to stop, then Riko will respond")
        print("Type 'quit' to exit\n")
        
        while True:
            user_input = input("Press Enter to record (or 'quit'): ").strip()
            
            if user_input.lower() == 'quit':
                break
                
            # Record audio
            audio_file = self.record_audio()
            
            if audio_file and audio_file.exists():
                # Transcribe
                text = self.transcribe_audio(audio_file)
                audio_file.unlink()  # Clean up
                
                if text.strip():
                    print(f"👤 You: {text}")
                    
                    # Get AI response (voice mode for speech output)
                    response, emotions = self.get_ai_response(text, voice_mode=True)
                    print(f"🤖 Riko: {response}")
                    
                    # Show Riko's emotions
                    if emotions:
                        self.display_riko_emotions(emotions)
                    
                    # Generate speech
                    self.generate_speech(response)
                    
                    print("\n" + "="*50 + "\n")
                else:
                    print("❌ No speech detected, try again")
            else:
                print("❌ Recording failed")
    
    def mode_2_speech_detection(self):
        """Mode 2: Speech Detection"""
        print("\n🎤 MODE 2: AUTO SPEECH DETECTION")
        print("Speak naturally, system will detect when you start/stop talking")
        print("Press Ctrl+C to exit\n")
        
        self.current_mode = 2
        self.auto_speech_detection()
    
    def mode_3_text_to_speech(self):
        """Mode 3: Text to Speech"""
        print("\n💬 MODE 3: TEXT TO SPEECH")
        print("Type your message, Riko will respond with voice")
        print("Type 'quit' to exit\n")
        
        while True:
            user_input = input("👤 You: ").strip()
            
            if user_input.lower() == 'quit':
                break
            
            if user_input:
                # Get AI response (voice mode for speech output)
                response, emotions = self.get_ai_response(user_input, voice_mode=True)
                print(f"🤖 Riko: {response}")
                
                # Show Riko's emotions
                if emotions:
                    self.display_riko_emotions(emotions)
                
                # Generate speech
                self.generate_speech(response)
                
                print("\n" + "="*30 + "\n")
    
    def mode_4_text_to_text(self):
        """Mode 4: Text to Text"""
        print("\n📝 MODE 4: TEXT TO TEXT")
        print("Pure text conversation")
        print("Type 'quit' to exit\n")
        
        while True:
            user_input = input("👤 You: ").strip()
            
            if user_input.lower() == 'quit':
                break
            
            if user_input:
                # Get AI response
                response, emotions = self.get_ai_response(user_input)
                print(f"🤖 Riko: {response}")
                
                # Show Riko's emotions
                if emotions:
                    self.display_riko_emotions(emotions)
                
                print("\n" + "="*30 + "\n")
    
    def show_menu(self):
        """Show main menu"""
        print("\n" + "="*60)
        print("🎌 RIKO CLI CHAT SYSTEM")
        print("="*60)
        print("1. 🎙️  Push to Talk - Press Enter to record, Riko speaks")
        print("2. 🎤 Speech Detection - Auto detect speech, Riko speaks")  
        print("3. 💬 Text to Speech - Type text, Riko speaks")
        print("4. 📝 Text to Text - Pure text conversation")
        print("5. 🚪 Exit")
        print("="*60)
    
    def run(self):
        """Main application loop"""
        while True:
            self.show_menu()
            
            try:
                choice = input("\nSelect mode (1-5): ").strip()
                
                if choice == '1':
                    self.mode_1_push_talk()
                elif choice == '2':
                    self.mode_2_speech_detection()
                elif choice == '3':
                    self.mode_3_text_to_speech()
                elif choice == '4':
                    self.mode_4_text_to_text()
                elif choice == '5':
                    print("👋 Goodbye!")
                    break
                else:
                    print("❌ Invalid choice, please select 1-5")
                    
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")

def main():
    """Main entry point"""
    try:
        print("🚀 Starting Riko CLI Chat System...")
        chat_system = CLIChatSystem()
        chat_system.run()
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Application error: {e}")
        return 1
    return 0

if __name__ == "__main__":
    exit(main())
