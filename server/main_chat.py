from faster_whisper import WhisperModel
from process.asr_func.asr_push_to_talk import record_and_transcribe
from process.llm_funcs.llm_scr import llm_response
from process.tts_func.sovits_ping import sovits_gen, play_audio
from process.text_processing.emotion_filter import EmotionFilter
from process.text_processing.colored_display import ColoredDisplay
from process.emotion_system.emotion_handler import EmotionHandler
from pathlib import Path
import os
import time
### transcribe audio 
import uuid
import soundfile as sf


def get_wav_duration(path):
    """Get duration of WAV file"""
    try:
        with sf.SoundFile(path) as f:
            return len(f) / f.samplerate
    except Exception as e:
        print(f"Error getting duration: {e}")
        return 0


# Initialize enhanced chat components
display = ColoredDisplay()
emotion_filter = EmotionFilter()
emotion_handler = EmotionHandler()

# Show startup banner
display.print_conversation_start()
display.print_system_message("Enhanced Riko Chat with Emotion Processing")
display.print_system_message("Features: Colored text, emotion filtering, visual feedback")

# Initialize Whisper model
display.print_system_message("Initializing Whisper model...")
whisper_model = WhisperModel("base.en", device="cpu", compute_type="float32")
display.print_system_message("Ready to chat!")

try:
    while True:
        # Step 1: Listen for user input
        display.print_listening()
        conversation_recording = Path("audio") / "conversation.wav"
        conversation_recording.parent.mkdir(parents=True, exist_ok=True)
        
        # Record and transcribe user speech
        user_spoken_text = record_and_transcribe(whisper_model, conversation_recording)
        
        if not user_spoken_text or user_spoken_text.strip() == "":
            display.print_error("No speech detected. Please try again.")
            continue
            
        # Display user input with colored text
        display.print_user_text(user_spoken_text)
        
        # Step 2: Process with LLM
        display.print_processing()
        llm_output = llm_response(user_spoken_text)
        
        # Step 3: Filter emotions from text for TTS
        tts_read_text, detected_emotions = emotion_filter.process_text(llm_output)
        
        # Step 4: Display Riko's response with colors
        display.print_riko_text(tts_read_text)
        
        # Step 5: Show detected emotions with visual effects
        if detected_emotions:
            display.print_emotions(detected_emotions)
            # Trigger emotion effects
            for emotion in detected_emotions:
                emotion_handler.trigger_emotion_effect(emotion)
                time.sleep(0.3)  # Brief pause between emotion effects
        
        # Step 6: Generate audio only from filtered text (no emotions spoken)
        if tts_read_text.strip():
            display.print_generating_audio()
            
            # Generate unique filename
            uid = uuid.uuid4().hex
            filename = f"output_{uid}.wav"
            output_wav_path = Path("audio") / filename
            output_wav_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Generate audio from clean text (emotions filtered out)
            try:
                gen_aud_path = sovits_gen(tts_read_text, output_wav_path)
                
                if output_wav_path.exists():
                    display.print_playing_audio()
                    play_audio(output_wav_path)
                else:
                    display.print_error("Failed to generate audio file")
                    
            except Exception as e:
                display.print_error(f"TTS generation error: {e}")
        
        # Step 7: Cleanup audio files
        try:
            [fp.unlink() for fp in Path("audio").glob("*.wav") if fp.is_file()]
        except Exception as e:
            display.print_error(f"Cleanup error: {e}")
            
        # Add separator for next conversation cycle
        display.print_separator("-", 30)
        
except KeyboardInterrupt:
    display.print_system_message("\nChat session ended by user")
    display.print_system_message("Thanks for chatting with Riko! Goodbye!")
except Exception as e:
    display.print_error(f"Unexpected error: {e}")
    display.print_system_message("Chat session ended due to error")
