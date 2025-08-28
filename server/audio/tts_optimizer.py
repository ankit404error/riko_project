#!/usr/bin/env python3
"""
Advanced TTS Audio Generation Optimizer
=======================================

This module provides comprehensive optimizations for faster and more efficient
text-to-speech generation with Riko, including:

- Intelligent caching system
- Parallel audio processing
- Streaming generation
- Model warming and optimization
- Hardware acceleration
- Audio quality enhancement
"""

import os
import sys
import time
import json
import hashlib
import threading
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, Future
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from collections import OrderedDict
import logging
import queue
import sqlite3
from datetime import datetime, timedelta

import numpy as np
import soundfile as sf
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import yaml

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class CacheEntry:
    """Represents a cached audio entry"""
    text_hash: str
    file_path: str
    timestamp: float
    access_count: int
    file_size: int
    generation_time: float

@dataclass
class AudioGenerationTask:
    """Represents an audio generation task"""
    task_id: str
    text: str
    output_path: str
    priority: int = 1
    callback: Optional[callable] = None
    config_override: Optional[Dict] = None

class TTSAudioCache:
    """
    Intelligent audio caching system with LRU eviction and persistence
    """
    
    def __init__(self, cache_dir: str, max_cache_size_mb: int = 500):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_cache_size = max_cache_size_mb * 1024 * 1024  # Convert to bytes
        
        self.db_path = self.cache_dir / "cache.db"
        self.init_database()
        
        self._cache_lock = threading.Lock()
        self.memory_cache = OrderedDict()  # LRU cache in memory
        
        self.load_cache_index()
        logger.info(f"Audio cache initialized with {len(self.memory_cache)} entries")
    
    def init_database(self):
        """Initialize cache database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS cache_entries (
                    text_hash TEXT PRIMARY KEY,
                    file_path TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    access_count INTEGER DEFAULT 1,
                    file_size INTEGER NOT NULL,
                    generation_time REAL
                )
            """)
            
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON cache_entries(timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_access_count ON cache_entries(access_count)")
            conn.commit()
    
    def _hash_text(self, text: str, config: Dict = None) -> str:
        """Create hash for text and configuration"""
        content = text.lower().strip()
        if config:
            content += json.dumps(config, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()
    
    def get(self, text: str, config: Dict = None) -> Optional[str]:
        """Get cached audio file path if available"""
        text_hash = self._hash_text(text, config)
        
        with self._cache_lock:
            if text_hash in self.memory_cache:
                # Move to end (most recently used)
                entry = self.memory_cache.pop(text_hash)
                self.memory_cache[text_hash] = entry
                
                # Check if file still exists
                if Path(entry.file_path).exists():
                    # Update access count
                    entry.access_count += 1
                    self.update_cache_entry_db(entry)
                    logger.info(f"Cache hit for text hash: {text_hash[:8]}...")
                    return entry.file_path
                else:
                    # File missing, remove from cache
                    del self.memory_cache[text_hash]
                    self.remove_cache_entry_db(text_hash)
        
        return None
    
    def put(self, text: str, file_path: str, generation_time: float, config: Dict = None):
        """Add audio file to cache"""
        if not Path(file_path).exists():
            return
        
        text_hash = self._hash_text(text, config)
        file_size = Path(file_path).stat().st_size
        
        # Create cache file path
        cache_file_path = self.cache_dir / f"{text_hash}.wav"
        
        try:
            # Copy to cache directory
            import shutil
            shutil.copy2(file_path, cache_file_path)
            
            entry = CacheEntry(
                text_hash=text_hash,
                file_path=str(cache_file_path),
                timestamp=time.time(),
                access_count=1,
                file_size=file_size,
                generation_time=generation_time
            )
            
            with self._cache_lock:
                self.memory_cache[text_hash] = entry
                self.store_cache_entry_db(entry)
                
                # Check cache size and evict if necessary
                self._evict_if_needed()
            
            logger.info(f"Cached audio: {text_hash[:8]}... (size: {file_size} bytes)")
            
        except Exception as e:
            logger.error(f"Error caching audio: {e}")
    
    def _evict_if_needed(self):
        """Evict least recently used entries if cache exceeds size limit"""
        current_size = sum(entry.file_size for entry in self.memory_cache.values())
        
        while current_size > self.max_cache_size and self.memory_cache:
            # Remove least recently used (first in OrderedDict)
            text_hash, entry = self.memory_cache.popitem(last=False)
            
            # Remove file
            try:
                Path(entry.file_path).unlink(missing_ok=True)
            except Exception as e:
                logger.error(f"Error removing cached file: {e}")
            
            # Remove from database
            self.remove_cache_entry_db(text_hash)
            
            current_size -= entry.file_size
            logger.info(f"Evicted cache entry: {text_hash[:8]}...")
    
    def store_cache_entry_db(self, entry: CacheEntry):
        """Store cache entry in database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT OR REPLACE INTO cache_entries
                    (text_hash, file_path, timestamp, access_count, file_size, generation_time)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    entry.text_hash,
                    entry.file_path,
                    entry.timestamp,
                    entry.access_count,
                    entry.file_size,
                    entry.generation_time
                ))
                conn.commit()
        except Exception as e:
            logger.error(f"Error storing cache entry in DB: {e}")
    
    def update_cache_entry_db(self, entry: CacheEntry):
        """Update cache entry access count in database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE cache_entries 
                    SET access_count = ?, timestamp = ?
                    WHERE text_hash = ?
                """, (entry.access_count, time.time(), entry.text_hash))
                conn.commit()
        except Exception as e:
            logger.error(f"Error updating cache entry in DB: {e}")
    
    def remove_cache_entry_db(self, text_hash: str):
        """Remove cache entry from database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM cache_entries WHERE text_hash = ?", (text_hash,))
                conn.commit()
        except Exception as e:
            logger.error(f"Error removing cache entry from DB: {e}")
    
    def load_cache_index(self):
        """Load cache index from database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM cache_entries ORDER BY timestamp")
                rows = cursor.fetchall()
                
                for row in rows:
                    text_hash, file_path, timestamp, access_count, file_size, generation_time = row
                    
                    # Check if file still exists
                    if Path(file_path).exists():
                        entry = CacheEntry(
                            text_hash=text_hash,
                            file_path=file_path,
                            timestamp=timestamp,
                            access_count=access_count,
                            file_size=file_size,
                            generation_time=generation_time or 0.0
                        )
                        self.memory_cache[text_hash] = entry
                    else:
                        # Remove stale entry
                        self.remove_cache_entry_db(text_hash)
                        
        except Exception as e:
            logger.error(f"Error loading cache index: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self._cache_lock:
            total_size = sum(entry.file_size for entry in self.memory_cache.values())
            total_entries = len(self.memory_cache)
            
            if total_entries > 0:
                avg_generation_time = sum(entry.generation_time for entry in self.memory_cache.values()) / total_entries
                total_access_count = sum(entry.access_count for entry in self.memory_cache.values())
            else:
                avg_generation_time = 0
                total_access_count = 0
        
        return {
            'total_entries': total_entries,
            'total_size_mb': total_size / (1024 * 1024),
            'max_size_mb': self.max_cache_size / (1024 * 1024),
            'average_generation_time': avg_generation_time,
            'total_access_count': total_access_count,
            'cache_dir': str(self.cache_dir)
        }


class TTSOptimizer:
    """
    Advanced TTS optimization system with caching, streaming, and parallel processing
    """
    
    def __init__(self, cache_dir: str = "audio/cache", config_path: str = None):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Load configuration
        self.config = self.load_config(config_path)
        
        # Initialize components
        self.audio_cache = TTSAudioCache(str(self.cache_dir), self.config.get('max_cache_size_mb', 500))
        
        # Thread pool for parallel processing
        self.thread_pool = ThreadPoolExecutor(
            max_workers=self.config.get('max_parallel_requests', 3),
            thread_name_prefix="TTS-Worker"
        )
        
        # Task queues for different priorities
        self.high_priority_queue = queue.PriorityQueue()
        self.normal_priority_queue = queue.Queue()
        
        # Session for connection pooling
        self.session = requests.Session()
        self.setup_session()
        
        # Warm up TTS server
        if self.config.get('warm_up_on_init', True):
            self.warm_up_server()
        
        # Statistics
        self.stats = {
            'cache_hits': 0,
            'cache_misses': 0,
            'total_generations': 0,
            'total_generation_time': 0.0,
            'average_generation_time': 0.0
        }
        
        logger.info("TTS Optimizer initialized")
    
    def load_config(self, config_path: str = None) -> Dict[str, Any]:
        """Load optimization configuration"""
        default_config = {
            'max_cache_size_mb': 500,
            'max_parallel_requests': 3,
            'connection_timeout': 30,
            'read_timeout': 60,
            'max_retries': 3,
            'retry_backoff_factor': 0.3,
            'warm_up_on_init': True,
            'enable_streaming': True,
            'enable_preprocessing': True,
            'audio_quality_enhancement': True,
            'server_url': 'http://127.0.0.1:9880',
            'optimization_params': {
                'speed_factor': 1.1,
                'temperature': 0.8,
                'top_k': 10,
                'top_p': 0.9,
                'batch_size': 1,
                'streaming_mode': False,
                'parallel_infer': True,
                'precision': 'float16'
            }
        }
        
        if config_path and Path(config_path).exists():
            try:
                with open(config_path, 'r') as f:
                    user_config = yaml.safe_load(f) or {}
                default_config.update(user_config)
            except Exception as e:
                logger.warning(f"Error loading config: {e}, using defaults")
        
        return default_config
    
    def setup_session(self):
        """Setup HTTP session with optimizations"""
        # Retry strategy
        retry_strategy = Retry(
            total=self.config.get('max_retries', 3),
            status_forcelist=[429, 500, 502, 503, 504],
            method_whitelist=["HEAD", "GET", "POST"],
            backoff_factor=self.config.get('retry_backoff_factor', 0.3)
        )
        
        # HTTP adapter with connection pooling
        adapter = HTTPAdapter(
            max_retries=retry_strategy,
            pool_connections=10,
            pool_maxsize=20,
            pool_block=False
        )
        
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        # Set default headers
        self.session.headers.update({
            'Connection': 'keep-alive',
            'Content-Type': 'application/json'
        })
    
    def warm_up_server(self):
        """Warm up the TTS server with a small request"""
        try:
            logger.info("Warming up TTS server...")
            start_time = time.time()
            
            warm_up_text = "Hello"
            result = self._generate_audio_direct(warm_up_text, warm_up=True)
            
            warm_up_time = time.time() - start_time
            
            if result:
                logger.info(f"Server warmed up in {warm_up_time:.2f}s")
                # Clean up warm-up file
                try:
                    Path(result).unlink(missing_ok=True)
                except:
                    pass
            else:
                logger.warning("Server warm-up failed")
                
        except Exception as e:
            logger.error(f"Error during server warm-up: {e}")
    
    def preprocess_text(self, text: str) -> str:
        """Preprocess text for better TTS generation"""
        if not self.config.get('enable_preprocessing', True):
            return text
        
        # Text preprocessing optimizations
        processed_text = text.strip()
        
        # Remove excessive whitespace
        import re
        processed_text = re.sub(r'\s+', ' ', processed_text)
        
        # Handle common abbreviations
        abbreviations = {
            'Dr.': 'Doctor',
            'Mr.': 'Mister',
            'Mrs.': 'Missus',
            'Ms.': 'Miss',
            'Prof.': 'Professor',
            'etc.': 'et cetera',
            'vs.': 'versus'
        }
        
        for abbrev, replacement in abbreviations.items():
            processed_text = processed_text.replace(abbrev, replacement)
        
        # Ensure proper sentence ending
        if processed_text and not processed_text[-1] in '.!?':
            processed_text += '.'
        
        return processed_text
    
    def generate_audio_async(self, text: str, output_path: str, priority: int = 1,
                           config_override: Dict = None) -> Future:
        """Generate audio asynchronously"""
        task_id = hashlib.md5(f"{text}_{time.time()}".encode()).hexdigest()[:8]
        
        task = AudioGenerationTask(
            task_id=task_id,
            text=text,
            output_path=output_path,
            priority=priority,
            config_override=config_override
        )
        
        # Submit to thread pool
        future = self.thread_pool.submit(self._process_generation_task, task)
        return future
    
    def generate_audio_sync(self, text: str, output_path: str, 
                          config_override: Dict = None) -> Optional[str]:
        """Generate audio synchronously (blocking)"""
        return self._process_generation_task(AudioGenerationTask(
            task_id="sync",
            text=text,
            output_path=output_path,
            config_override=config_override
        ))
    
    def _process_generation_task(self, task: AudioGenerationTask) -> Optional[str]:
        """Process a single audio generation task"""
        start_time = time.time()
        
        try:
            # Preprocess text
            processed_text = self.preprocess_text(task.text)
            
            # Check cache first
            config_key = task.config_override or self.config.get('optimization_params', {})
            cached_path = self.audio_cache.get(processed_text, config_key)
            
            if cached_path:
                # Cache hit - copy to output path
                import shutil
                shutil.copy2(cached_path, task.output_path)
                
                self.stats['cache_hits'] += 1
                generation_time = time.time() - start_time
                
                logger.info(f"Task {task.task_id}: Cache hit ({generation_time:.3f}s)")
                return task.output_path
            
            # Cache miss - generate new audio
            self.stats['cache_misses'] += 1
            
            # Generate audio
            result = self._generate_audio_direct(
                processed_text, 
                task.output_path, 
                task.config_override
            )
            
            if result:
                generation_time = time.time() - start_time
                
                # Add to cache
                self.audio_cache.put(processed_text, result, generation_time, config_key)
                
                # Update statistics
                self.stats['total_generations'] += 1
                self.stats['total_generation_time'] += generation_time
                self.stats['average_generation_time'] = (
                    self.stats['total_generation_time'] / self.stats['total_generations']
                )
                
                logger.info(f"Task {task.task_id}: Generated ({generation_time:.3f}s)")
                return result
            else:
                logger.error(f"Task {task.task_id}: Generation failed")
                return None
                
        except Exception as e:
            logger.error(f"Task {task.task_id}: Error - {e}")
            return None
    
    def _generate_audio_direct(self, text: str, output_path: str = None, 
                             config_override: Dict = None, warm_up: bool = False) -> Optional[str]:
        """Generate audio directly via API call"""
        
        if output_path is None:
            output_path = str(self.cache_dir / f"temp_{int(time.time() * 1000)}.wav")
        
        # Merge configuration
        params = self.config.get('optimization_params', {}).copy()
        if config_override:
            params.update(config_override)
        
        # API payload
        payload = {
            "text": text,
            "text_lang": "en",
            "ref_audio_path": "riko_voice.wav",  # This should be configurable
            "prompt_text": "Sample voice",
            "prompt_lang": "en",
            **params
        }
        
        try:
            url = f"{self.config['server_url']}/tts"
            
            response = self.session.post(
                url, 
                json=payload,
                timeout=(
                    self.config.get('connection_timeout', 30),
                    self.config.get('read_timeout', 60)
                )
            )
            
            response.raise_for_status()
            
            # Save audio content
            with open(output_path, 'wb') as f:
                f.write(response.content)
            
            # Enhance audio quality if enabled
            if self.config.get('audio_quality_enhancement', True) and not warm_up:
                self._enhance_audio_quality(output_path)
            
            return output_path
            
        except requests.exceptions.RequestException as e:
            logger.error(f"TTS API request failed: {e}")
            return None
        except Exception as e:
            logger.error(f"Audio generation error: {e}")
            return None
    
    def _enhance_audio_quality(self, audio_path: str):
        """Enhance audio quality post-processing"""
        try:
            # Load audio
            data, samplerate = sf.read(audio_path)
            
            # Normalize audio to prevent clipping
            max_val = np.max(np.abs(data))
            if max_val > 0.95:
                data = data / max_val * 0.95
            
            # Apply gentle noise gate
            threshold = 0.01
            data = np.where(np.abs(data) < threshold, data * 0.1, data)
            
            # Simple high-pass filter to remove low-frequency noise
            if len(data) > 100:
                # Basic high-pass: subtract low-pass filtered signal
                from scipy import ndimage
                low_pass = ndimage.uniform_filter1d(data, size=50, mode='nearest')
                data = data - (low_pass * 0.1)
            
            # Save enhanced audio
            sf.write(audio_path, data, samplerate, subtype='PCM_24')
            
        except Exception as e:
            logger.warning(f"Audio enhancement failed: {e}")
    
    def generate_batch(self, texts: List[str], output_dir: str) -> List[Optional[str]]:
        """Generate multiple audio files in parallel"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        futures = []
        for i, text in enumerate(texts):
            output_path = output_dir / f"audio_{i:03d}.wav"
            future = self.generate_audio_async(text, str(output_path))
            futures.append(future)
        
        # Wait for all to complete
        results = []
        for future in futures:
            try:
                result = future.result(timeout=120)  # 2 minute timeout per audio
                results.append(result)
            except Exception as e:
                logger.error(f"Batch generation error: {e}")
                results.append(None)
        
        return results
    
    def stream_audio_generation(self, text: str, chunk_callback: callable):
        """Stream audio generation in chunks (if supported by server)"""
        # This is a placeholder for streaming implementation
        # The actual implementation would depend on server streaming support
        
        if not self.config.get('enable_streaming', True):
            # Fallback to regular generation
            temp_path = str(self.cache_dir / f"stream_{int(time.time() * 1000)}.wav")
            result = self.generate_audio_sync(text, temp_path)
            if result:
                chunk_callback(result)
            return
        
        logger.info("Streaming audio generation (placeholder implementation)")
        
        # For now, generate normally and call callback
        temp_path = str(self.cache_dir / f"stream_{int(time.time() * 1000)}.wav")
        result = self.generate_audio_sync(text, temp_path)
        if result:
            chunk_callback(result)
    
    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get comprehensive optimization statistics"""
        cache_stats = self.audio_cache.get_stats()
        
        total_requests = self.stats['cache_hits'] + self.stats['cache_misses']
        cache_hit_rate = (self.stats['cache_hits'] / total_requests * 100) if total_requests > 0 else 0
        
        return {
            'cache': cache_stats,
            'performance': {
                'total_requests': total_requests,
                'cache_hits': self.stats['cache_hits'],
                'cache_misses': self.stats['cache_misses'],
                'cache_hit_rate_percent': cache_hit_rate,
                'total_generations': self.stats['total_generations'],
                'total_generation_time': self.stats['total_generation_time'],
                'average_generation_time': self.stats['average_generation_time']
            },
            'configuration': {
                'max_parallel_requests': self.config.get('max_parallel_requests'),
                'server_url': self.config.get('server_url'),
                'optimization_enabled': True,
                'preprocessing_enabled': self.config.get('enable_preprocessing'),
                'enhancement_enabled': self.config.get('audio_quality_enhancement')
            }
        }
    
    def cleanup_old_cache(self, days_old: int = 7):
        """Clean up old cache entries"""
        cutoff_time = time.time() - (days_old * 24 * 60 * 60)
        
        with self.audio_cache._cache_lock:
            to_remove = []
            
            for text_hash, entry in self.audio_cache.memory_cache.items():
                if entry.timestamp < cutoff_time:
                    to_remove.append(text_hash)
            
            for text_hash in to_remove:
                entry = self.audio_cache.memory_cache.pop(text_hash)
                try:
                    Path(entry.file_path).unlink(missing_ok=True)
                except:
                    pass
                
                self.audio_cache.remove_cache_entry_db(text_hash)
        
        logger.info(f"Cleaned up {len(to_remove)} old cache entries")
    
    def shutdown(self):
        """Clean shutdown of optimizer"""
        logger.info("Shutting down TTS Optimizer...")
        
        # Shutdown thread pool
        self.thread_pool.shutdown(wait=True)
        
        # Close session
        self.session.close()
        
        logger.info("TTS Optimizer shutdown complete")


# Singleton instance management
_tts_optimizer_instance = None
_optimizer_lock = threading.Lock()

def get_tts_optimizer(**kwargs) -> TTSOptimizer:
    """Get singleton instance of TTSOptimizer"""
    global _tts_optimizer_instance
    
    with _optimizer_lock:
        if _tts_optimizer_instance is None:
            _tts_optimizer_instance = TTSOptimizer(**kwargs)
        return _tts_optimizer_instance


if __name__ == "__main__":
    # Test the TTS optimizer
    print("Testing TTS Optimizer...")
    
    optimizer = get_tts_optimizer()
    
    # Test texts
    test_texts = [
        "Hello Senpai, how are you today?",
        "This is a test of the optimized TTS system.",
        "Riko speaking with enhanced performance!",
        "The quick brown fox jumps over the lazy dog.",
        "Hello Senpai, how are you today?"  # Duplicate for cache testing
    ]
    
    print("\n1. Testing synchronous generation...")
    for i, text in enumerate(test_texts):
        start_time = time.time()
        output_path = f"test_output_{i}.wav"
        result = optimizer.generate_audio_sync(text, output_path)
        end_time = time.time()
        
        if result:
            print(f"✅ Generated: {text[:30]}... ({end_time - start_time:.2f}s)")
        else:
            print(f"❌ Failed: {text[:30]}...")
    
    print("\n2. Testing batch generation...")
    batch_results = optimizer.generate_batch(test_texts[:3], "batch_output")
    successful_batch = sum(1 for r in batch_results if r is not None)
    print(f"✅ Batch completed: {successful_batch}/{len(test_texts[:3])} successful")
    
    print("\n3. Performance statistics:")
    stats = optimizer.get_optimization_stats()
    print(f"   Cache hit rate: {stats['performance']['cache_hit_rate_percent']:.1f}%")
    print(f"   Average generation time: {stats['performance']['average_generation_time']:.2f}s")
    print(f"   Total cache entries: {stats['cache']['total_entries']}")
    print(f"   Cache size: {stats['cache']['total_size_mb']:.1f} MB")
    
    # Cleanup test files
    import glob
    for test_file in glob.glob("test_output_*.wav") + glob.glob("batch_output/*.wav"):
        try:
            Path(test_file).unlink()
        except:
            pass
    
    try:
        Path("batch_output").rmdir()
    except:
        pass
    
    optimizer.shutdown()
    print("\n✨ TTS Optimizer test completed!")
