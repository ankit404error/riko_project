#!/usr/bin/env python3
"""
GPU Acceleration Script for GPT-SoVITS
Maximizes GPU utilization for fastest voice generation
"""

import os
import sys
import subprocess
import time
from pathlib import Path

class GPUAccelerator:
    """
    GPU acceleration and optimization for GPT-SoVITS
    """
    
    def __init__(self):
        self.optimizations = []
        
    def check_gpu_availability(self):
        """Check GPU and CUDA availability"""
        print("🔍 Checking GPU configuration...")
        
        try:
            import torch
            gpu_available = torch.cuda.is_available()
            if gpu_available:
                gpu_count = torch.cuda.device_count()
                gpu_name = torch.cuda.get_device_name(0)
                memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                
                print(f"✅ GPU Available: {gpu_name}")
                print(f"✅ GPU Memory: {memory:.1f} GB")
                print(f"✅ CUDA Version: {torch.version.cuda}")
                print(f"✅ GPU Count: {gpu_count}")
                
                return True, gpu_name, memory
            else:
                print("❌ No GPU available - will use CPU optimizations")
                return False, None, 0
                
        except ImportError:
            print("❌ PyTorch not found - installing...")
            self.install_pytorch()
            return False, None, 0
    
    def install_pytorch(self):
        """Install PyTorch with CUDA support"""
        print("📦 Installing PyTorch with CUDA support...")
        
        commands = [
            "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118",
            "pip install xformers",  # Memory efficient attention
            "pip install accelerate",  # Hugging Face acceleration
            "pip install optimum[onnxruntime-gpu]"  # ONNX GPU acceleration
        ]
        
        for cmd in commands:
            try:
                print(f"Running: {cmd}")
                subprocess.run(cmd, shell=True, check=True)
                print("✅ Installation successful")
            except subprocess.CalledProcessError as e:
                print(f"⚠️ Warning: {cmd} failed - {e}")
    
    def optimize_cuda_settings(self):
        """Optimize CUDA settings for maximum performance"""
        print("🚀 Optimizing CUDA settings...")
        
        # CUDA environment variables for maximum performance
        cuda_optimizations = {
            "CUDA_LAUNCH_BLOCKING": "0",  # Async kernel launches
            "TORCH_USE_CUDA_DSA": "1",    # Device-side assertions
            "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:128,caching_allocator:true",
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",  # Deterministic operations
            "CUDA_VISIBLE_DEVICES": "0",  # Use primary GPU
            "NCCL_DEBUG": "WARN",  # Reduce NCCL verbosity
            "OMP_NUM_THREADS": "4",  # OpenMP threads
            "MKL_NUM_THREADS": "4",  # Intel MKL threads
            "NUMEXPR_NUM_THREADS": "4",  # NumExpr threads
            "PYTORCH_NO_CUDA_MEMORY_CACHING": "0",  # Enable memory caching
            "TORCH_CUDNN_V8_API_ENABLED": "1",  # Enable cuDNN v8
            "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE": "1",  # Allow TF32
        }
        
        for key, value in cuda_optimizations.items():
            os.environ[key] = value
            
        self.optimizations.append("CUDA environment optimized")
        print("✅ CUDA settings optimized")
    
    def create_gpu_warmup_script(self):
        """Create GPU warmup script for faster inference"""
        print("🔥 Creating GPU warmup script...")
        
        warmup_script = '''
import torch
import torch.nn.functional as F
import gc
import time

def warmup_gpu():
    """Comprehensive GPU warmup for optimal performance"""
    if not torch.cuda.is_available():
        print("❌ CUDA not available")
        return False
        
    device = torch.device('cuda:0')
    print(f"🔥 Warming up {torch.cuda.get_device_name(0)}")
    
    # Clear cache
    torch.cuda.empty_cache()
    gc.collect()
    
    # Enable optimizations
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    try:
        # Warmup with progressively larger tensors
        sizes = [(100, 100), (500, 500), (1000, 1000), (2000, 2000)]
        
        for size in sizes:
            # Matrix operations
            a = torch.randn(size, device=device, dtype=torch.float16)
            b = torch.randn(size, device=device, dtype=torch.float16)
            c = torch.matmul(a, b)
            
            # Convolution operations (common in neural networks)
            if size[0] >= 64:
                conv_input = torch.randn(1, 64, size[0]//8, size[1]//8, device=device)
                conv = torch.nn.Conv2d(64, 128, 3, padding=1).to(device)
                conv_output = conv(conv_input)
            
            # Attention operations (used in transformers)
            if size[0] >= 512:
                seq_len, d_model = min(size[0], 1024), min(size[1], 768)
                q = torch.randn(1, 8, seq_len, d_model//8, device=device)
                k = torch.randn(1, 8, seq_len, d_model//8, device=device)
                v = torch.randn(1, 8, seq_len, d_model//8, device=device)
                attn_output = F.scaled_dot_product_attention(q, k, v)
            
            torch.cuda.synchronize()
            
        print(f"✅ GPU warmup completed")
        
        # Memory info
        memory_allocated = torch.cuda.memory_allocated(0) / 1024**3
        memory_reserved = torch.cuda.memory_reserved(0) / 1024**3
        print(f"📊 Memory allocated: {memory_allocated:.2f} GB")
        print(f"📊 Memory reserved: {memory_reserved:.2f} GB")
        
        return True
        
    except Exception as e:
        print(f"❌ GPU warmup failed: {e}")
        return False
    finally:
        # Clean up
        torch.cuda.empty_cache()
        gc.collect()

if __name__ == "__main__":
    success = warmup_gpu()
    exit(0 if success else 1)
'''
        
        warmup_file = Path(__file__).parent / "gpu_warmup.py"
        with open(warmup_file, 'w') as f:
            f.write(warmup_script)
            
        self.optimizations.append("GPU warmup script created")
        print("✅ GPU warmup script saved")
        
        return warmup_file
    
    def create_memory_optimizer(self):
        """Create memory optimization utilities"""
        print("💾 Creating memory optimization utilities...")
        
        memory_script = '''
import torch
import gc
import psutil
import time

class MemoryOptimizer:
    """Advanced memory management for GPU inference"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def optimize_memory(self):
        """Apply comprehensive memory optimizations"""
        if torch.cuda.is_available():
            self.optimize_gpu_memory()
        self.optimize_system_memory()
        
    def optimize_gpu_memory(self):
        """GPU memory optimizations"""
        # Clear cache
        torch.cuda.empty_cache()
        
        # Set memory management strategies
        torch.cuda.memory._set_allocator_settings('max_split_size_mb:128')
        
        # Enable memory mapping
        if hasattr(torch.cuda, 'memory_map'):
            torch.cuda.memory_map()
            
        print("✅ GPU memory optimized")
        
    def optimize_system_memory(self):
        """System memory optimizations"""
        # Garbage collection
        gc.collect()
        
        # Set process priority (Windows)
        try:
            import psutil
            process = psutil.Process()
            process.nice(psutil.HIGH_PRIORITY_CLASS)
            print("✅ Process priority increased")
        except:
            print("⚠️ Could not increase process priority")
            
        print("✅ System memory optimized")
        
    def monitor_memory(self, duration=60):
        """Monitor memory usage"""
        print(f"📊 Monitoring memory for {duration} seconds...")
        
        start_time = time.time()
        while time.time() - start_time < duration:
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.memory_allocated(0) / 1024**3
                gpu_reserved = torch.cuda.memory_reserved(0) / 1024**3
                print(f"GPU: {gpu_memory:.2f}GB allocated, {gpu_reserved:.2f}GB reserved")
            
            system_memory = psutil.virtual_memory()
            print(f"RAM: {system_memory.percent}% used ({system_memory.used/1024**3:.2f}GB)")
            
            time.sleep(5)

if __name__ == "__main__":
    optimizer = MemoryOptimizer()
    optimizer.optimize_memory()
'''
        
        memory_file = Path(__file__).parent / "memory_optimizer.py"
        with open(memory_file, 'w') as f:
            f.write(memory_script)
            
        self.optimizations.append("Memory optimizer created")
        print("✅ Memory optimizer saved")
        
    def create_performance_monitor(self):
        """Create performance monitoring script"""
        print("📊 Creating performance monitor...")
        
        monitor_script = '''
import time
import psutil
import torch
import subprocess
from pathlib import Path

class PerformanceMonitor:
    """Real-time performance monitoring for TTS"""
    
    def __init__(self):
        self.gpu_available = torch.cuda.is_available()
        
    def monitor_tts_performance(self, duration=300):  # 5 minutes
        """Monitor TTS performance in real-time"""
        print(f"🔍 Monitoring TTS performance for {duration} seconds...")
        
        start_time = time.time()
        generation_times = []
        
        while time.time() - start_time < duration:
            # System metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            memory_percent = psutil.virtual_memory().percent
            
            metrics = f"CPU: {cpu_percent:.1f}% | RAM: {memory_percent:.1f}%"
            
            if self.gpu_available:
                gpu_memory = torch.cuda.memory_allocated(0) / 1024**3
                gpu_util = self.get_gpu_utilization()
                metrics += f" | GPU: {gpu_util:.1f}% | VRAM: {gpu_memory:.2f}GB"
            
            print(f"📊 {metrics}")
            time.sleep(2)
            
    def get_gpu_utilization(self):
        """Get GPU utilization percentage"""
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'], 
                                 capture_output=True, text=True)
            return float(result.stdout.strip())
        except:
            return 0.0
            
    def benchmark_inference_speed(self):
        """Benchmark inference speed"""
        print("🏁 Running inference speed benchmark...")
        
        # This would integrate with your TTS system
        # For now, just demonstrate the concept
        test_phrases = [
            "Quick test phrase",
            "Medium length test phrase for benchmarking",
            "This is a longer test phrase that will help us measure the performance of the text-to-speech system under various loads"
        ]
        
        total_time = 0
        for i, phrase in enumerate(test_phrases):
            start_time = time.time()
            
            # Simulate TTS generation (replace with actual TTS call)
            time.sleep(len(phrase) * 0.01)  # Simulate processing time
            
            generation_time = time.time() - start_time
            total_time += generation_time
            
            print(f"Test {i+1}: {generation_time:.2f}s ({len(phrase)} chars)")
            
        avg_time = total_time / len(test_phrases)
        print(f"📊 Average generation time: {avg_time:.2f}s")
        
if __name__ == "__main__":
    monitor = PerformanceMonitor()
    monitor.benchmark_inference_speed()
'''
        
        monitor_file = Path(__file__).parent / "performance_monitor.py"
        with open(monitor_file, 'w') as f:
            f.write(monitor_script)
            
        self.optimizations.append("Performance monitor created")
        print("✅ Performance monitor saved")
    
    def run_gpu_acceleration(self):
        """Run complete GPU acceleration setup"""
        print("🚀 Starting GPU Acceleration Setup...")
        print("="*50)
        
        # Check GPU
        gpu_available, gpu_name, gpu_memory = self.check_gpu_availability()
        
        # Apply optimizations
        if gpu_available:
            self.optimize_cuda_settings()
            print(f"🎯 Optimizing for: {gpu_name} ({gpu_memory:.1f}GB)")
        else:
            print("🎯 Applying CPU optimizations...")
            
        # Create utility scripts
        warmup_file = self.create_gpu_warmup_script()
        self.create_memory_optimizer()
        self.create_performance_monitor()
        
        # Run GPU warmup if available
        if gpu_available:
            print("\n🔥 Running GPU warmup...")
            try:
                result = subprocess.run([sys.executable, str(warmup_file)], 
                                      capture_output=True, text=True, timeout=60)
                if result.returncode == 0:
                    print("✅ GPU warmup completed successfully")
                    print(result.stdout)
                else:
                    print("⚠️ GPU warmup had issues:")
                    print(result.stderr)
            except subprocess.TimeoutExpired:
                print("⚠️ GPU warmup timed out")
            except Exception as e:
                print(f"⚠️ GPU warmup failed: {e}")
        
        # Summary
        print("\n" + "="*50)
        print("🎯 GPU Acceleration Summary:")
        for optimization in self.optimizations:
            print(f"   ✅ {optimization}")
        
        print(f"\n🎉 GPU acceleration setup complete!")
        
        # Instructions
        print("\n📚 Next Steps:")
        print("   1. Restart your terminal/command prompt")
        print("   2. Start GPT-SoVITS server")
        print("   3. Run main_chat.py to test performance")
        print("   4. Monitor performance with performance_monitor.py")
        
        if gpu_available:
            print(f"\n🎮 GPU Info:")
            print(f"   Device: {gpu_name}")
            print(f"   Memory: {gpu_memory:.1f}GB")
            print(f"   Optimization: Enabled")
        
        return True

def main():
    """Main function"""
    try:
        accelerator = GPUAccelerator()
        accelerator.run_gpu_acceleration()
        return 0
    except Exception as e:
        print(f"❌ GPU acceleration setup failed: {e}")
        return 1

if __name__ == "__main__":
    exit(main())
