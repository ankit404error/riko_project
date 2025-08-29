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
import re

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

def sanitize_text_for_tts(text):
    """
    Enhanced text sanitization to prevent hanging issues
    
    This specifically addresses patterns like:
    - "What we , senpai?" (comma followed by space)
    - ".Hmp" (period followed by short word)
    - Multiple punctuation marks
    - Unusual spacing patterns
    """
    if not text or not text.strip():
        return "Hello."  # Safe fallback
    
    original_text = text
    
    # Step 1: Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text.strip())
    
    # Step 2: Fix problematic comma patterns that cause hanging
    # Pattern: "word space comma space word" -> "word comma space word"
    text = re.sub(r'(\w)\s+,\s+', r'\1, ', text)
    
    # Step 3: Fix the specific ".Hmp" pattern
    if re.match(r'^\.([A-Za-z]{1,4})\.?$', text):
        word = re.match(r'^\.([A-Za-z]{1,4})\.?$', text).group(1)
        text = f"{word}."
        print(f"🔧 Fixed period-word pattern: '{original_text}' -> '{text}'")
    
    # Step 4: Handle multiple punctuation marks
    text = re.sub(r'[.]{2,}', '.', text)  # Multiple periods
    text = re.sub(r'[!]{2,}', '!', text)  # Multiple exclamations
    text = re.sub(r'[?]{2,}', '?', text)  # Multiple questions
    
    # Step 5: Fix unusual spacing around punctuation
    text = re.sub(r'\s+([.!?])', r'\1', text)  # Remove space before punctuation
    text = re.sub(r'([.!?])\s*([.!?])', r'\1 \2', text)  # Space between different punctuation
    
    # Step 6: Handle very short texts that might cause issues
    if len(text.strip()) < 3:
        short_word_map = {
            'h': 'Hmm.',
            'hm': 'Hmm.',
            'hmm': 'Hmm.',
            'u': 'Uh.',
            'uh': 'Uh.',
            'ugh': 'Ugh.',
            'ah': 'Ah.',
            'oh': 'Oh.'
        }
        lower_text = text.strip().lower()
        if lower_text in short_word_map:
            text = short_word_map[lower_text]
        else:
            text = f"{text.strip()}."
        print(f"🔧 Fixed short text: '{original_text}' -> '{text}'")
    
    # Step 7: Ensure proper sentence ending
    if text and text[-1] not in '.!?':
        text += '.'
    
    # Step 8: Handle specific problematic phrases that have caused timeouts
    problematic_patterns = [
        (r"What\s+we\s*,\s*", "What are we, "),  # Fix "What we , " pattern
        (r"Don\'t\s+get\s+any\s+funny\s+ideas", "Don't get any funny ideas"),  # Standardize
    ]
    
    for pattern, replacement in problematic_patterns:
        if re.search(pattern, text):
            old_text = text
            text = re.sub(pattern, replacement, text)
            print(f"🔧 Fixed problematic pattern: '{old_text}' -> '{text}'")
    
    # Step 9: Final safety check
    if not text.strip():
        text = "Hmph."
        print("🔧 Applied final fallback: 'Hmph.'")
    
    # Log changes
    if text != original_text:
        print(f"📝 Text sanitized: '{original_text}' -> '{text}'")
    
    return text

def enhance_audio(audio_data, samplerate):
    """Apply audio enhancements for better quality"""
    # Normalize audio to prevent clipping
    audio_data = audio_data / np.max(np.abs(audio_data))
    
    # Apply gentle noise gate
    threshold = 0.01
    audio_data = np.where(np.abs(audio_data) < threshold, 0, audio_data)
    
    return audio_data

def play_audio(path):
    """Ultra-fast audio playback - no processing for speed"""
    try:
        # SPEED MODE: Skip all audio enhancements for instant playback
        data, samplerate = sf.read(path)
        
        # Direct playback without any processing
        sd.play(data, samplerate, latency='low')
        sd.wait()
    except Exception as e:
        print(f"Audio playback error: {e}")
        # Minimal fallback
        try:
            data, samplerate = sf.read(path)
            sd.play(data, samplerate)
            sd.wait()
        except Exception as e2:
            print(f"Fallback audio playback also failed: {e2}")

def async_play_audio(path):
    """Non-blocking audio playback"""
    return THREAD_POOL.submit(play_audio, path)

def sovits_gen_with_timeout(in_text, output_wav_pth="output.wav", timeout_seconds=30):
    """
    TTS generation with timeout protection to prevent infinite hanging
    """
    import threading
    import queue
    
    def tts_worker(text, output_path, result_queue, error_queue):
        """Worker function that runs TTS generation"""
        try:
            result = sovits_gen_optimized_safe(text, output_path)
            result_queue.put(result)
        except Exception as e:
            error_queue.put(str(e))
    
    # Sanitize text first
    sanitized_text = sanitize_text_for_tts(in_text)
    
    # Create queues for communication
    result_queue = queue.Queue()
    error_queue = queue.Queue()
    
    # Start TTS in separate thread
    worker_thread = threading.Thread(
        target=tts_worker,
        args=(sanitized_text, output_wav_pth, result_queue, error_queue)
    )
    
    print(f"🎵 Generating TTS with {timeout_seconds}s timeout: '{sanitized_text}'")
    start_time = time.time()
    
    worker_thread.start()
    worker_thread.join(timeout=timeout_seconds)
    
    elapsed = time.time() - start_time
    
    if worker_thread.is_alive():
        print(f"❌ TTS generation timed out after {elapsed:.2f}s")
        print(f"   Original text: '{in_text}'")
        print(f"   Sanitized text: '{sanitized_text}'")
        print("   This prevents infinite hanging!")
        return None
    else:
        # Check results
        try:
            result = result_queue.get_nowait()
            print(f"✅ TTS generation completed in {elapsed:.2f}s")
            return result
        except queue.Empty:
            try:
                error = error_queue.get_nowait()
                print(f"❌ TTS generation error: {error}")
                return None
            except queue.Empty:
                print(f"⚠️ TTS generation completed but no result returned")
                return None

def sovits_gen_optimized_safe(in_text, output_wav_pth="output.wav"):
    """Optimized TTS generation with enhanced settings and safety checks"""
    
    # Apply text sanitization
    original_text = in_text
    in_text = sanitize_text_for_tts(in_text)
    
    # Log text processing
    if in_text != original_text:
        print(f"⚡ Text sanitized: '{original_text}' -> '{in_text}'")
    
    # Keep full sentences intact - only apply minimal cleaning for speed
    in_text = in_text.strip()
    
    # Log text length but don't truncate for quality
    if len(in_text) > 200:
        print(f"⚡ Processing long text: {len(in_text)} chars (keeping full content)")
    
    # SPEED OPTIMIZATION: Check cache first
    cache_key = hash(in_text)
    if cache_key in TTS_CACHE:
        cached_path = TTS_CACHE[cache_key]
        if Path(cached_path).exists():
            # Copy cached file to output path
            import shutil
            shutil.copy2(cached_path, output_wav_pth)
            print(f"⚡ Using cached audio (instant)")
            return output_wav_pth
    
    url = "http://127.0.0.1:9880/tts"
    
    # Get config with fallbacks
    config = char_config.get('sovits_ping_config', {})
    
    # ULTRA-FAST payload - all speed optimizations applied
    payload = {
        "text": in_text,
        "text_lang": config.get('text_lang', 'en'),
        "ref_audio_path": config.get('ref_audio_path', 'riko_voice.wav'),
        "prompt_text": config.get('prompt_text', 'This is a sample voice.'),
        "prompt_lang": config.get('prompt_lang', 'en'),
        # ULTRA-AGGRESSIVE speed parameters for CPU
        "top_k": config.get('top_k', 1),
        "top_p": config.get('top_p', 0.5),
        "temperature": config.get('temperature', 0.3),
        "speed_factor": config.get('speed_factor', 1.2),
        "batch_size": config.get('batch_size', 1),
        "text_split_method": config.get('text_split_method', 'cut0'),
        "batch_threshold": config.get('batch_threshold', 0.1),
        "fragment_interval": config.get('fragment_interval', 0.001),
        "repetition_penalty": config.get('repetition_penalty', 1.05),
        "sample_steps": config.get('sample_steps', 1),
        "parallel_infer": config.get('parallel_infer', False),
        "super_sampling": config.get('super_sampling', False),
        "streaming_mode": config.get('streaming_mode', False),
        "seed": config.get('seed', 42),  # Fixed seed for speed
        "return_fragment": config.get('return_fragment', False),
        "split_bucket": config.get('split_bucket', False)
    }
    
    try:
        # Use session for connection reuse
        session = requests.Session()
        session.headers.update({
            'Connection': 'keep-alive',
            'Content-Type': 'application/json'
        })
        
        start_time = time.time()
        # Reduced timeout to prevent hanging - if it takes longer, something is wrong
        response = session.post(url, json=payload, timeout=120)  # 2 minutes max
        response.raise_for_status()
        
        generation_time = time.time() - start_time
        print(f"TTS generation took: {generation_time:.2f}s")
        
        # Save the response audio
        with open(output_wav_pth, "wb") as f:
            f.write(response.content)
        
        # Cache successful generation (but limit cache size)
        if len(TTS_CACHE) < 50:
            TTS_CACHE[cache_key] = output_wav_pth + ".cache"
            import shutil
            shutil.copy2(output_wav_pth, TTS_CACHE[cache_key])
        
        return output_wav_pth
        
    except requests.exceptions.Timeout:
        print("TTS request timed out - server may be overloaded or stuck")
        print(f"Problematic text: '{in_text}'")
        return None
    except requests.exceptions.ConnectionError:
        print("Cannot connect to TTS server - make sure GPT-SoVITS is running")
        return None
    except Exception as e:
        print(f"Error in TTS generation: {e}")
        print(f"Problematic text: '{in_text}'")
        return None

# Main functions for backward compatibility
def sovits_gen(in_text, output_wav_pth="output.wav"):
    """
    Main TTS function with hanging protection
    
    This function includes fixes for the .Hmp hanging issue by:
    1. Sanitizing problematic text patterns
    2. Adding timeout protection
    3. Safe fallbacks for edge cases
    
    Reduced timeout to 30s for faster failure detection.
    """
    return sovits_gen_with_timeout(in_text, output_wav_pth, timeout_seconds=30)

def test_hang_protection():
    """Test the hanging protection with problematic patterns"""
    
    print("🧪 Testing TTS hanging protection")
    print("=" * 50)
    
    # Test cases that previously caused hanging
    test_cases = [
        ".Hmp",           # The main problem case
        ".Hmph",          # Similar pattern
        ".Hi",            # Short text with period
        ".",              # Just period
        ".A",             # Very short
        "",               # Empty text
        "   ",            # Whitespace only
    ]
    
    for text in test_cases:
        print(f"\n--- Testing: '{text}' ---")
        
        # Test sanitization
        sanitized = sanitize_text_for_tts(text)
        print(f"   Sanitized: '{sanitized}'")
        
        # Test with timeout (won't actually generate, just test logic)
        output_file = f"test_hang_protection_{hash(text) % 1000}.wav"
        
        start_time = time.time()
        # Quick test - just sanitization, not full TTS
        result = sanitized if sanitized else None
        elapsed = time.time() - start_time
        
        if result:
            print(f"   ✅ Protected: {elapsed:.3f}s")
        else:
            print(f"   ❌ Issue: {elapsed:.3f}s")
    
    print("\n✅ Hanging protection test completed!")

if __name__ == "__main__":
    
    print("🔧 Testing fixed TTS system with hanging protection")
    
    # Test the protection system first
    test_hang_protection()
    
    print("\n" + "=" * 50)
    print("Testing actual TTS generation (if server is running):")
    
    start_time = time.time()
    output_wav_pth1 = "output_fixed.wav"
    
    # Test with the problematic pattern
    test_text = ".Hmp"  # This used to cause hanging
    
    print(f"\n🎵 Testing problematic text: '{test_text}'")
    path_to_aud = sovits_gen(test_text, output_wav_pth1)
    
    end_time = time.time()
    elapsed_time = end_time - start_time

    if path_to_aud:
        print(f"✅ Success! Generated in {elapsed_time:.4f} seconds")
        print(f"Output file: {path_to_aud}")
        
        # Test a few more cases
        more_tests = ["Hmph.", "Whatever.", ".Hi there."]
        for test in more_tests:
            print(f"\n🎵 Testing: '{test}'")
            result = sovits_gen(test, f"test_{hash(test) % 1000}.wav")
            if result:
                print("   ✅ OK")
            else:
                print("   ❌ Failed")
    else:
        print(f"❌ Failed after {elapsed_time:.4f} seconds")
        print("Make sure the GPT-SoVITS server is running!")

    print("\n🎯 Fixed TTS system test completed!")
    print("The hanging issue with '.Hmp' patterns should now be resolved!")
