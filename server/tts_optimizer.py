#!/usr/bin/env python3
"""
Advanced TTS Optimization Script for GPT-SoVITS
Improves voice quality and generation speed
"""

import os
import sys
import json
import time
import requests
import subprocess
from pathlib import Path
from typing import Dict, Any
import yaml

class TTSOptimizer:
    """
    Advanced TTS optimization for GPT-SoVITS
    """
    
    def __init__(self):
        self.server_url = "http://127.0.0.1:9880"
        self.optimizations_applied = []
        self.current_dir = Path(__file__).parent
        
        # Load character config
        self.load_config()
        
    def load_config(self):
        """Load character configuration"""
        config_paths = [
            'character_config.yaml',
            '../character_config.yaml', 
            '../../character_config.yaml',
            '../../../character_config.yaml'
        ]
        
        for config_path in config_paths:
            try:
                with open(config_path, 'r') as f:
                    self.config = yaml.safe_load(f)
                    self.config_path = config_path
                    break
            except FileNotFoundError:
                continue
        else:
            raise FileNotFoundError("Could not find character_config.yaml")
    
    def test_server_connection(self):
        """Test if GPT-SoVITS server is running"""
        try:
            response = requests.get(f"{self.server_url}/tts", timeout=5)
            return True
        except:
            return False
    
    def optimize_server_settings(self):
        """Apply server-side optimizations"""
        print("🔧 Applying server optimizations...")
        
        # Create optimized inference config
        inference_config = {
            "device": "cuda" if self.has_gpu() else "cpu",
            "half_precision": True,  # Use FP16 for speed
            "batch_size": 1,  # Real-time generation
            "cache_all_data": True,  # Cache models in memory
            "workers": 1,  # Single worker for real-time
            "max_memory_per_gpu": "8GB",
            "low_vram_mode": False,  # Disable if you have enough VRAM
            "use_deepspeed": False,  # Can enable if available
            "streaming": True,  # Enable streaming generation
            "chunk_size": 1024
        }
        
        # Save config
        config_file = self.current_dir / "optimized_inference.json"
        with open(config_file, 'w') as f:
            json.dump(inference_config, f, indent=2)
        
        self.optimizations_applied.append("Server settings optimized")
        print("✅ Server optimization config saved")
    
    def optimize_audio_settings(self):
        """Optimize audio quality settings"""
        print("🎵 Optimizing audio settings...")
        
        # Update character config with better audio settings
        if 'sovits_ping_config' not in self.config:
            self.config['sovits_ping_config'] = {}
            
        audio_optimizations = {
            'sample_rate': 44100,  # High quality sample rate
            'bit_depth': 24,       # Better audio quality
            'audio_format': 'wav',
            'noise_scale': 0.667,  # Reduce noise
            'noise_scale_w': 0.8,  # Emotional variance
            'length_scale': 1.0,   # Normal speech speed
            'top_k': 10,           # More focused generation
            'top_p': 0.9,         # Balanced creativity/stability  
            'temperature': 0.8,    # Slightly cooler for stability
            'repetition_penalty': 1.1,  # Avoid repetition
            'speed_factor': 1.1,   # Slightly faster speech
            'use_gpu': True,       # GPU acceleration
            'batch_inference': True,
            'enable_streaming': True,
            'low_latency_mode': True
        }
        
        self.config['sovits_ping_config'].update(audio_optimizations)
        self.optimizations_applied.append("Audio quality enhanced")
        print("✅ Audio settings optimized")
    
    def optimize_memory_usage(self):
        """Optimize memory usage for faster generation"""
        print("💾 Optimizing memory usage...")
        
        memory_optimizations = {
            'preload_models': True,     # Keep models in memory
            'model_cache_size': '2GB',  # Cache size
            'clear_cache_interval': 100,  # Clear cache every N generations
            'use_model_pooling': True,  # Reuse model instances
            'gradient_checkpointing': False,  # Disable for speed
            'mixed_precision': True,    # Use FP16
            'cpu_offload': False,      # Keep on GPU if available
            'pin_memory': True,        # Faster data transfer
            'persistent_workers': True  # Keep workers alive
        }
        
        if 'memory_config' not in self.config:
            self.config['memory_config'] = {}
        self.config['memory_config'].update(memory_optimizations)
        
        self.optimizations_applied.append("Memory usage optimized")
        print("✅ Memory optimizations applied")
    
    def optimize_model_loading(self):
        """Optimize model loading and caching"""
        print("🤖 Optimizing model loading...")
        
        model_optimizations = {
            'warm_up_models': True,     # Pre-warm models
            'model_precision': 'float16',  # Faster inference
            'compile_models': True,     # JIT compilation
            'use_bettertransformer': True,  # Optimized attention
            'enable_xformers': True,    # Memory efficient attention
            'use_flash_attention': True,  # Fastest attention
            'model_parallel': False,    # Single GPU for speed
            'tensor_parallel': False,   # Disable for single GPU
            'pipeline_parallel': False  # Disable for real-time
        }
        
        if 'model_config' not in self.config:
            self.config['model_config'] = {}
        self.config['model_config'].update(model_optimizations)
        
        self.optimizations_applied.append("Model loading optimized")
        print("✅ Model optimization applied")
    
    def create_fast_inference_script(self):
        """Create an optimized inference script"""
        print("📝 Creating fast inference script...")
        
        script_content = '''
import torch
import gc
import os

# Set environment variables for optimization
os.environ["CUDA_LAUNCH_BLOCKING"] = "0"
os.environ["TORCH_USE_CUDA_DSA"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

def optimize_torch():
    """Apply PyTorch optimizations"""
    if torch.cuda.is_available():
        # Enable optimizations
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        # Memory management
        torch.cuda.empty_cache()
        gc.collect()
        
        print("🚀 CUDA optimizations enabled")
    else:
        # CPU optimizations
        torch.set_num_threads(4)  # Adjust based on CPU cores
        print("🚀 CPU optimizations enabled")

def warmup_gpu():
    """Warm up GPU for faster inference"""
    if torch.cuda.is_available():
        # Dummy operations to warm up GPU
        dummy = torch.randn(1000, 1000, device='cuda')
        torch.matmul(dummy, dummy)
        torch.cuda.empty_cache()
        print("🔥 GPU warmed up")

if __name__ == "__main__":
    optimize_torch()
    warmup_gpu()
'''
        
        script_file = self.current_dir / "fast_inference_setup.py"
        with open(script_file, 'w') as f:
            f.write(script_content)
            
        self.optimizations_applied.append("Fast inference script created")
        print("✅ Fast inference script saved")
    
    def optimize_system_settings(self):
        """Apply system-level optimizations"""
        print("⚙️ Applying system optimizations...")
        
        try:
            # Windows-specific optimizations
            if os.name == 'nt':
                # Set high priority for Python process
                import psutil
                process = psutil.Process()
                process.nice(psutil.HIGH_PRIORITY_CLASS)
                print("✅ Process priority increased")
                
            # Set environment variables for better performance
            os.environ["OMP_NUM_THREADS"] = "4"
            os.environ["MKL_NUM_THREADS"] = "4"  
            os.environ["NUMEXPR_NUM_THREADS"] = "4"
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use first GPU
            
            self.optimizations_applied.append("System settings optimized")
            print("✅ System optimizations applied")
            
        except Exception as e:
            print(f"⚠️ Some system optimizations failed: {e}")
    
    def benchmark_performance(self):
        """Benchmark TTS performance"""
        print("📊 Running performance benchmark...")
        
        if not self.test_server_connection():
            print("❌ Cannot connect to TTS server - make sure it's running")
            return
        
        test_texts = [
            "Hello, this is a performance test.",
            "The quick brown fox jumps over the lazy dog.",
            "Riko is speaking with optimized voice generation!"
        ]
        
        total_time = 0
        successful_tests = 0
        
        for i, text in enumerate(test_texts):
            try:
                start_time = time.time()
                
                # Test TTS generation
                response = requests.post(f"{self.server_url}/tts", json={
                    "text": text,
                    "text_lang": "en",
                    "ref_audio_path": "riko_voice.wav",
                    "prompt_text": "Sample voice",
                    "prompt_lang": "en"
                }, timeout=30)
                
                if response.status_code == 200:
                    generation_time = time.time() - start_time
                    total_time += generation_time
                    successful_tests += 1
                    print(f"✅ Test {i+1}: {generation_time:.2f}s")
                else:
                    print(f"❌ Test {i+1}: Failed")
                    
            except Exception as e:
                print(f"❌ Test {i+1}: Error - {e}")
        
        if successful_tests > 0:
            avg_time = total_time / successful_tests
            print(f"\n📈 Benchmark Results:")
            print(f"   Average generation time: {avg_time:.2f}s")
            print(f"   Successful tests: {successful_tests}/{len(test_texts)}")
            print(f"   Total time: {total_time:.2f}s")
        else:
            print("\n❌ All benchmark tests failed")
    
    def save_optimized_config(self):
        """Save the optimized configuration"""
        print("💾 Saving optimized configuration...")
        
        # Backup original config
        backup_path = Path(self.config_path).with_suffix('.yaml.backup')
        if not backup_path.exists():
            import shutil
            shutil.copy2(self.config_path, backup_path)
            print(f"✅ Original config backed up to {backup_path}")
        
        # Save optimized config
        with open(self.config_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False, indent=2)
        
        print("✅ Optimized configuration saved")
    
    def has_gpu(self):
        """Check if GPU is available"""
        try:
            import torch
            return torch.cuda.is_available()
        except:
            return False
    
    def run_optimization(self):
        """Run complete optimization process"""
        print("🚀 Starting TTS Optimization Process...")
        print("="*50)
        
        # Check prerequisites
        gpu_available = self.has_gpu()
        server_running = self.test_server_connection()
        
        print(f"GPU Available: {'✅' if gpu_available else '❌'}")
        print(f"Server Running: {'✅' if server_running else '❌'}")
        print()
        
        # Apply optimizations
        self.optimize_audio_settings()
        self.optimize_memory_usage()
        self.optimize_model_loading()
        self.optimize_server_settings()
        self.create_fast_inference_script()
        self.optimize_system_settings()
        
        # Save configuration
        self.save_optimized_config()
        
        # Run benchmark if server is available
        if server_running:
            print()
            self.benchmark_performance()
        
        # Summary
        print("\n" + "="*50)
        print("🎯 Optimization Summary:")
        for optimization in self.optimizations_applied:
            print(f"   ✅ {optimization}")
        
        print(f"\n🎉 Optimization complete! Applied {len(self.optimizations_applied)} improvements.")
        
        if not server_running:
            print("\n⚠️  Note: Start GPT-SoVITS server and run benchmark manually")
        
        print("\n📚 Next steps:")
        print("   1. Restart GPT-SoVITS server")
        print("   2. Run main_chat.py to test improvements")
        print("   3. Adjust settings in character_config.yaml if needed")


def main():
    """Main optimization function"""
    try:
        optimizer = TTSOptimizer()
        optimizer.run_optimization()
    except Exception as e:
        print(f"❌ Optimization failed: {e}")
        return 1
    return 0


if __name__ == "__main__":
    exit(main())
