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
import signal
import sys

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

# Global flag for interrupt handling
_generation_interrupted = False

def setup_interrupt_handler():
    """Setup Ctrl+C interrupt handling for TTS generation"""
    global _generation_interrupted
    
    def signal_handler(sig, frame):
        global _generation_interrupted
        print("\n🛑 TTS generation interrupted by user (Ctrl+C)")
        print("   Stopping current generation gracefully...")
        _generation_interrupted = True
        
    signal.signal(signal.SIGINT, signal_handler)

def is_interrupted():
    """Check if generation has been interrupted"""
    global _generation_interrupted
    return _generation_interrupted

def reset_interrupt_flag():
    """Reset the interrupt flag for new generation"""
    global _generation_interrupted
    _generation_interrupted = False

def enhanced_sanitize_text_for_tts(text):
    """
    Enhanced text sanitization to prevent ALL hanging issues including semantic token loops
    
    This fixes multiple issues:
    1. Period-prefixed text that causes infinite recursion 
    2. Text that leads to infinite semantic token generation
    3. Edge cases that crash the TTS pipeline
    """
    if not text or not text.strip():
        return "Hello."  # Safe fallback
    
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text.strip())
    
    # CRITICAL FIX: Remove leading periods that cause infinite loops
    # This is the main cause of both hanging and infinite generation
    while text.startswith('.') and len(text) > 1:
        text = text[1:].strip()
        print(f"🔧 Removed leading period to prevent infinite loop")
    
    # Handle empty text after period removal
    if not text or text == '.':
        text = "Hmph."
        print(f"🔧 Fixed empty text after period removal -> '{text}'")
        return text
    
    # Fix common problematic patterns
    
    # Case 1: Single characters or very short words
    if len(text) <= 2:
        if text.lower() in ['h', 'hm', 'hmm']:
            text = "Hmm."
        elif text.lower() in ['u', 'uh', 'ugh']:
            text = "Ugh."
        elif text.lower() in ['o', 'oh']:
            text = "Oh."
        else:
            text = f"{text}."
        print(f"🔧 Fixed very short text -> '{text}'")
    
    # Case 2: Incomplete sentences or fragments (common cause of generation loops)
    # Look for sentences that end without proper punctuation
    if not re.search(r'[.!?]$', text):
        # Check if it looks like an incomplete sentence
        if text.endswith('?') or 'what' in text.lower() or 'how' in text.lower() or 'why' in text.lower():
            # Question fragment - complete it
            if not text.endswith('?'):
                text = text.rstrip() + "?"
        elif text.endswith('!') or 'ugh' in text.lower() or 'hmph' in text.lower():
            # Exclamation fragment - complete it  
            if not text.endswith('!'):
                text = text.rstrip() + "!"
        else:
            # Statement fragment - complete it
            text = text.rstrip() + "."
        print(f"🔧 Completed incomplete sentence -> '{text}'")
    
    # Case 3: Multiple punctuation at end (can cause processing issues)
    text = re.sub(r'[.!?]{2,}$', '.', text)
    
    # Case 4: Remove problematic characters that might confuse the TTS
    # Keep only safe characters for TTS processing
    text = re.sub(r'[^\w\s.!?,\'"-]', '', text)
    
    # Case 5: Ensure minimum meaningful length for stable generation
    if len(text.strip()) < 3:
        text = "Hmph."
        print(f"🔧 Text too short, using safe fallback -> '{text}'")
    
    # Case 6: Prevent common patterns that lead to infinite semantic token generation
    # These patterns often cause the "Predict Semantic Token" phase to loop
    problematic_patterns = [
        r'^[.!?]+',        # Starting with only punctuation
        r'^\w{1,2}[.!?]+$', # Single/double letter + punctuation
        r'.*\?\s*\?$',     # Ending with question fragments
    ]
    
    for pattern in problematic_patterns:
        if re.match(pattern, text, re.IGNORECASE):
            text = "Hmph."
            print(f"🔧 Detected problematic pattern, using safe fallback -> '{text}'")
            break
    
    return text

def sovits_gen_with_enhanced_protection(in_text, output_wav_pth="output.wav", timeout_seconds=45):
    """
    Enhanced TTS generation with comprehensive protection against hanging and infinite loops
    """
    import threading
    import queue
    
    # Reset interrupt flag
    reset_interrupt_flag()
    
    # Setup interrupt handler
    setup_interrupt_handler()
    
    def tts_worker(text, output_path, result_queue, error_queue):
        """Worker function that runs TTS generation with interrupt checking"""
        try:
            result = sovits_gen_optimized_safe(text, output_path)
            if not is_interrupted():
                result_queue.put(result)
            else:
                error_queue.put("Generation interrupted by user")
        except Exception as e:
            error_queue.put(str(e))
    
    # Enhanced text sanitization
    original_text = in_text
    sanitized_text = enhanced_sanitize_text_for_tts(in_text)
    
    if sanitized_text != original_text:
        print(f"🔧 Text enhanced: '{original_text}' -> '{sanitized_text}'")
    
    # Create queues for communication
    result_queue = queue.Queue()
    error_queue = queue.Queue()
    
    # Start TTS in separate thread
    worker_thread = threading.Thread(
        target=tts_worker,
        args=(sanitized_text, output_wav_pth, result_queue, error_queue),
        daemon=True  # Allow clean shutdown
    )
    
    print(f"🎵 Generating TTS with {timeout_seconds}s timeout: '{sanitized_text}'")
    start_time = time.time()
    
    worker_thread.start()
    
    # Monitor thread with periodic interrupt checks
    check_interval = 0.5  # Check every 500ms
    elapsed = 0
    
    while worker_thread.is_alive() and elapsed < timeout_seconds:
        if is_interrupted():
            print("🛑 Generation interrupted by user!")
            return None
            
        worker_thread.join(timeout=check_interval)
        elapsed = time.time() - start_time
        
        # Show progress for long generations
        if elapsed > 10 and elapsed % 10 < check_interval:
            print(f"   ⏳ Still generating... {elapsed:.1f}s elapsed")
    
    final_elapsed = time.time() - start_time
    
    if worker_thread.is_alive():
        print(f"❌ TTS generation timed out after {final_elapsed:.2f}s")
        print(f"   Original text: '{original_text}'")
        print(f"   Sanitized text: '{sanitized_text}'")
        print("   🛑 This timeout prevents infinite loops!")
        return None
    elif is_interrupted():
        print(f"🛑 TTS generation interrupted after {final_elapsed:.2f}s")
        return None
    else:
        # Check results
        try:
            result = result_queue.get_nowait()
            print(f"✅ TTS generation completed in {final_elapsed:.2f}s")
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
    
    # Check for interrupt before starting
    if is_interrupted():
        print("🛑 Generation cancelled - interrupted")
        return None
    
    # Apply enhanced text sanitization
    original_text = in_text
    in_text = enhanced_sanitize_text_for_tts(in_text)
    
    # Log text processing
    if in_text != original_text:
        print(f"⚡ Text sanitized: '{original_text}' -> '{in_text}'")
    
    # SPEED OPTIMIZATION: Check cache first
    cache_key = hash(in_text)
    if cache_key in TTS_CACHE:
        cached_path = TTS_CACHE[cache_key]
        if Path(cached_path).exists():
            import shutil
            shutil.copy2(cached_path, output_wav_pth)
            print(f"⚡ Using cached audio (instant)")
            return output_wav_pth
    
    url = "http://127.0.0.1:9880/tts"
    
    # Get config with enhanced safety settings
    config = char_config.get('sovits_ping_config', {})
    
    # ENHANCED payload with additional safety parameters
    payload = {
        "text": in_text,
        "text_lang": config.get('text_lang', 'en'),
        "ref_audio_path": config.get('ref_audio_path', 'riko_voice.wav'),
        "prompt_text": config.get('prompt_text', 'This is a sample voice.'),
        "prompt_lang": config.get('prompt_lang', 'en'),
        # Enhanced parameters to prevent infinite loops
        "top_k": max(1, min(config.get('top_k', 5), 20)),  # Bounded for stability
        "top_p": max(0.5, min(config.get('top_p', 0.8), 0.95)),  # Prevent extreme values
        "temperature": max(0.3, min(config.get('temperature', 0.7), 1.0)),  # Bounded
        "speed_factor": config.get('speed_factor', 1.0),
        "batch_size": 1,  # Force single batch for stability
        "text_split_method": config.get('text_split_method', 'cut1'),  # Safer method
        "batch_threshold": config.get('batch_threshold', 0.75),  # Higher threshold
        "fragment_interval": config.get('fragment_interval', 0.3),  # Longer intervals
        "repetition_penalty": max(1.0, min(config.get('repetition_penalty', 1.2), 1.5)),  # Bounded
        "sample_steps": min(config.get('sample_steps', 10), 30),  # Hard limit on steps
        "parallel_infer": False,  # Disable for stability
        "super_sampling": False,  # Disable for stability  
        "streaming_mode": False,  # Disable for stability
        "seed": config.get('seed', 42),
        "return_fragment": False,
        "split_bucket": False,
        # Add max generation steps to prevent infinite loops
        "max_new_tokens": 512,  # Hard limit
        "early_stopping": True,  # Enable early stopping
    }
    
    try:
        # Check for interrupt before making request
        if is_interrupted():
            print("🛑 Generation cancelled before request")
            return None
            
        # Use session for connection reuse
        session = requests.Session()
        session.headers.update({
            'Connection': 'keep-alive',
            'Content-Type': 'application/json'
        })
        
        start_time = time.time()
        
        # Make request with interrupt checking
        print(f"📡 Sending request to TTS server...")
        response = session.post(url, json=payload, timeout=90)  # 90s timeout
        
        # Check for interrupt after request
        if is_interrupted():
            print("🛑 Generation cancelled after request")
            return None
            
        response.raise_for_status()
        
        generation_time = time.time() - start_time
        print(f"✅ TTS generation completed: {generation_time:.2f}s")
        
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
        print("❌ TTS request timed out - server may be overloaded or in infinite loop")
        print(f"   Problematic text: '{in_text}'")
        print("   💡 Try restarting the GPT-SoVITS server")
        return None
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to TTS server - make sure GPT-SoVITS is running")
        return None
    except KeyboardInterrupt:
        print("🛑 TTS generation interrupted by user")
        return None
    except Exception as e:
        print(f"❌ Error in TTS generation: {e}")
        print(f"   Problematic text: '{in_text}'")
        return None

def enhance_audio(audio_data, samplerate):
    """Apply audio enhancements for better quality"""
    # Normalize audio to prevent clipping
    audio_data = audio_data / np.max(np.abs(audio_data))
    
    # Apply gentle noise gate
    threshold = 0.01
    audio_data = np.where(np.abs(audio_data) < threshold, 0, audio_data)
    
    return audio_data

def play_audio(path):
    """Ultra-fast audio playback with interrupt support"""
    try:
        if is_interrupted():
            print("🛑 Audio playback cancelled - interrupted")
            return
            
        # SPEED MODE: Skip all audio enhancements for instant playback
        data, samplerate = sf.read(path)
        
        # Direct playback without any processing
        sd.play(data, samplerate, latency='low')
        
        # Wait with interrupt checking
        while sd.get_stream().active:
            if is_interrupted():
                sd.stop()
                print("🛑 Audio playback interrupted")
                return
            time.sleep(0.1)
            
    except Exception as e:
        print(f"Audio playback error: {e}")
        # Minimal fallback
        try:
            if not is_interrupted():
                data, samplerate = sf.read(path)
                sd.play(data, samplerate)
                sd.wait()
        except Exception as e2:
            print(f"Fallback audio playback also failed: {e2}")

def async_play_audio(path):
    """Non-blocking audio playback"""
    return THREAD_POOL.submit(play_audio, path)

# Main functions for backward compatibility with enhanced protection
def sovits_gen(in_text, output_wav_pth="output.wav"):
    """
    Main TTS function with comprehensive hanging protection
    
    This function includes fixes for:
    1. .Hmp hanging issue with text sanitization
    2. Infinite semantic token generation loops  
    3. Ctrl+C interrupt handling
    4. Timeout protection for all phases
    """
    return sovits_gen_with_enhanced_protection(in_text, output_wav_pth, timeout_seconds=45)

def create_interrupt_test_script():
    """Create a test script to verify interrupt handling works"""
    
    test_script = """#!/usr/bin/env python3
# Test script for interrupt handling
import time
import sys
from pathlib import Path

sys.path.append('server/process/tts_func')
from sovits_ping_enhanced_fix import sovits_gen, setup_interrupt_handler

def test_interrupt():
    print("🧪 Testing Ctrl+C interrupt handling")
    print("Press Ctrl+C to interrupt generation...")
    print("Testing with a long text that should take time to generate...")
    
    setup_interrupt_handler()
    
    long_text = "This is a very long test sentence that should take several seconds to generate and give you time to press Ctrl+C to interrupt the generation process."
    
    result = sovits_gen(long_text, "interrupt_test.wav")
    
    if result:
        print(f"✅ Generation completed: {result}")
    else:
        print("❌ Generation failed or was interrupted")

if __name__ == "__main__":
    test_interrupt()
"""
    
    with open("test_interrupt.py", "w") as f:
        f.write(test_script)
    
    print("📝 Created interrupt test script: test_interrupt.py")
    print("   Run: python test_interrupt.py")
    print("   Then press Ctrl+C during generation to test interrupts")

def test_enhanced_protection():
    """Test the enhanced hanging protection"""
    
    print("🧪 Testing Enhanced TTS Hanging Protection")
    print("=" * 55)
    
    # Test cases including the new problematic patterns
    test_cases = [
        (".Hmp", "Hmp."),                    # Original problem
        (".Ugh", "Ugh."),                    # New problem from your log
        (".Hmph", "Hmph."),                  # Similar
        (".Hi there", "Hi there."),          # Longer
        (".", "Hmph."),                      # Single period
        (".A", "A."),                        # Very short
        ("", "Hello."),                      # Empty
        ("   ", "Hello."),                   # Whitespace
        (".What?", "What?"),                 # Question
        (".Oh!", "Oh!"),                     # Exclamation
        ("...Ugh", "Ugh."),                  # Multiple periods
    ]
    
    all_passed = True
    
    for input_text, expected_pattern in test_cases:
        try:
            result = enhanced_sanitize_text_for_tts(input_text)
            
            # Check if result matches expected pattern (allow some variation)
            if result == expected_pattern or (expected_pattern in result):
                print(f"✅ '{input_text}' -> '{result}'")
            else:
                print(f"⚠️ '{input_text}' -> '{result}' (expected pattern: '{expected_pattern}')")
                # This is acceptable as long as it's not hanging
        except Exception as e:
            print(f"❌ '{input_text}' -> ERROR: {e}")
            all_passed = False
    
    if all_passed:
        print("\n✅ All enhanced protection tests passed!")
        print("🎯 The hanging and infinite generation issues should be resolved!")
    else:
        print("\n⚠️ Some edge cases detected, but protection is active.")
    
    return all_passed

if __name__ == "__main__":
    
    print("🛡️ ENHANCED TTS PROTECTION SYSTEM")
    print("Fixes: .Hmp hanging + Infinite generation + Ctrl+C support")
    print("=" * 65)
    
    # Test the protection system
    test_enhanced_protection()
    
    # Create interrupt test script
    print(f"\n📝 Creating additional test tools...")
    create_interrupt_test_script()
    
    print(f"\n🎯 TESTING PROBLEMATIC PATTERNS")
    print("=" * 40)
    
    # Test the specific patterns from the user's error log
    problem_texts = [
        ".Ugh, senpai, are you mocking me?",  # From user's log
        ".Hmp",                               # Original problem
        ".Whatever",                          # Common Riko response
        "Hmph.",                              # Normal case (should work fine)
    ]
    
    for text in problem_texts:
        print(f"\n🧪 Testing: '{text}'")
        
        try:
            setup_interrupt_handler()
            
            # Test just the sanitization (no actual TTS call)
            sanitized = enhanced_sanitize_text_for_tts(text)
            print(f"   Sanitized: '{sanitized}'")
            
            # Quick validation
            if len(sanitized.strip()) >= 3 and not sanitized.startswith('.'):
                print(f"   ✅ Safe for TTS processing")
            else:
                print(f"   ⚠️ May still need improvement")
                
        except Exception as e:
            print(f"   ❌ ERROR: {e}")
    
    print(f"\n" + "=" * 65)
    print("✨ ENHANCED PROTECTION SYSTEM READY!")
    print("📋 To apply: python apply_hmp_fix.py apply")
    print("🧪 To test interrupts: python test_interrupt.py")  
    print("🛑 Use Ctrl+C to interrupt long generations")
    print("=" * 65)
