import requests
### MUST START SERVERS FIRST USING START ALL SERVER SCRIPT
import time
import soundfile as sf 
import sounddevice as sd
import yaml
import numpy as np
from pathlib import Path
import threading
from queue import Queue
import io
from concurrent.futures import ThreadPoolExecutor

# Load YAML config - find correct path
config_path = None
for potential_path in [
    'character_config.yaml', 
    '../character_config.yaml',
    '../../character_config.yaml', 
    '../../../character_config.yaml',
    '../../../../character_config.yaml'
]:
    try:
        with open(potential_path, 'r') as f:
            char_config = yaml.safe_load(f)
            config_path = potential_path
            break
    except FileNotFoundError:
        continue

if config_path is None:
    raise FileNotFoundError("Could not find character_config.yaml")


# Global settings for optimization
TTS_CACHE = {}  # Simple cache for repeated phrases
AUDIO_BUFFER_SIZE = 4096
THREAD_POOL = ThreadPoolExecutor(max_workers=2)

def enhance_audio(audio_data, samplerate):
    """Apply audio enhancements for better quality"""
    # Normalize audio to prevent clipping
    audio_data = audio_data / np.max(np.abs(audio_data))
    
    # Apply gentle noise gate
    threshold = 0.01
    audio_data = np.where(np.abs(audio_data) < threshold, 0, audio_data)
    
    return audio_data

def play_audio(path):
    """Enhanced audio playback with optimizations"""
    try:
        data, samplerate = sf.read(path)
        
        # Apply audio enhancements
        enhanced_data = enhance_audio(data, samplerate)
        
        # Play with optimized settings (fixed parameter issue)
        sd.play(enhanced_data, samplerate, 
               blocksize=AUDIO_BUFFER_SIZE,
               latency='low')
        sd.wait()
    except Exception as e:
        print(f"Audio playback error: {e}")
        # Fallback to simple playback
        try:
            data, samplerate = sf.read(path)
            sd.play(data, samplerate)
            sd.wait()
        except Exception as e2:
            print(f"Fallback audio playback also failed: {e2}")

def async_play_audio(path):
    """Non-blocking audio playback"""
    return THREAD_POOL.submit(play_audio, path)

def sovits_gen_optimized(in_text, output_wav_pth="output.wav"):
    """Optimized TTS generation with enhanced settings"""
    # Check cache first
    cache_key = hash(in_text)
    if cache_key in TTS_CACHE:
        cached_path = TTS_CACHE[cache_key]
        if Path(cached_path).exists():
            # Copy cached file to output path
            import shutil
            shutil.copy2(cached_path, output_wav_pth)
            return output_wav_pth
    
    url = "http://127.0.0.1:9880/tts"
    
    # Get config with fallbacks
    config = char_config.get('sovits_ping_config', {})
    
    # Enhanced payload with optimization parameters
    payload = {
        "text": in_text,
        "text_lang": config.get('text_lang', 'en'),
        "ref_audio_path": config.get('ref_audio_path', 'riko_voice.wav'),
        "prompt_text": config.get('prompt_text', 'This is a sample voice.'),
        "prompt_lang": config.get('prompt_lang', 'en'),
        # Optimization parameters
        "top_k": config.get('top_k', 15),
        "top_p": config.get('top_p', 1.0),
        "temperature": config.get('temperature', 1.0),
        "speed_factor": config.get('speed_factor', 1.1),
        "batch_size": config.get('batch_size', 1),
        "precision": config.get('precision', 'float16'),
        "stream_chunk_size": config.get('stream_chunk_size', 1024)
    }
    
    try:
        # Use session for connection reuse
        session = requests.Session()
        session.headers.update({
            'Connection': 'keep-alive',
            'Content-Type': 'application/json'
        })
        
        start_time = time.time()
        response = session.post(url, json=payload, timeout=30)
        response.raise_for_status()
        
        generation_time = time.time() - start_time
        print(f"TTS generation took: {generation_time:.2f}s")
        
        # Save the response audio
        with open(output_wav_pth, "wb") as f:
            f.write(response.content)
        
        # Post-process audio for better quality
        try:
            data, samplerate = sf.read(output_wav_pth)
            enhanced_data = enhance_audio(data, samplerate)
            sf.write(output_wav_pth, enhanced_data, samplerate, subtype='PCM_24')
        except Exception as e:
            print(f"Audio post-processing warning: {e}")
        
        # Cache successful generation
        if len(TTS_CACHE) < 50:  # Limit cache size
            TTS_CACHE[cache_key] = output_wav_pth + ".cache"
            import shutil
            shutil.copy2(output_wav_pth, TTS_CACHE[cache_key])
        
        return output_wav_pth
        
    except requests.exceptions.Timeout:
        print("TTS request timed out - server may be overloaded")
        return None
    except requests.exceptions.ConnectionError:
        print("Cannot connect to TTS server - make sure GPT-SoVITS is running")
        return None
    except Exception as e:
        print(f"Error in optimized TTS generation: {e}")
        return None

# Alias for backward compatibility
def sovits_gen(in_text, output_wav_pth="output.wav"):
    """Main TTS function - now uses optimized version"""
    return sovits_gen_optimized(in_text, output_wav_pth)

def test_tts_speed():
    """Test and benchmark TTS generation speed"""
    test_phrases = [
        "Hello senpai, how are you today?",
        "This is a test of the optimized TTS system.",
        "Riko speaking with enhanced voice quality!"
    ]
    
    print("\n🎵 Testing TTS Speed and Quality...")
    total_time = 0
    
    for i, phrase in enumerate(test_phrases):
        print(f"\nTest {i+1}: '{phrase}'")
        start_time = time.time()
        
        result = sovits_gen(phrase, f"test_output_{i}.wav")
        
        if result:
            generation_time = time.time() - start_time
            total_time += generation_time
            print(f"✅ Generated in {generation_time:.2f}s")
            
            # Clean up test files
            try:
                Path(result).unlink()
            except:
                pass
        else:
            print(f"❌ Generation failed")
    
    if total_time > 0:
        avg_time = total_time / len(test_phrases)
        print(f"\n📊 Average generation time: {avg_time:.2f}s")
        print(f"📊 Total test time: {total_time:.2f}s")
    else:
        print("\n❌ All tests failed - check TTS server")



if __name__ == "__main__":

    start_time = time.time()
    output_wav_pth1 = "output.wav"
    path_to_aud = sovits_gen("if you hear this, that means it is set up correctly", output_wav_pth1)
    
    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f"Elapsed time: {elapsed_time:.4f} seconds")
    print(path_to_aud)


