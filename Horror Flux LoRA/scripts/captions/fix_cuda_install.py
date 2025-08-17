#!/usr/bin/env python3
"""
Script to fix PyTorch CUDA installation for CUDA 12.8
Compatible with RTX 4090 and other modern GPUs
"""

import subprocess
import sys
import os
import platform

def run_command(cmd, description):
    print(f"\n🔄 {description}")
    print(f"Running: {cmd}")
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} - Success")
        if result.stdout:
            print(f"Output: {result.stdout.strip()}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} - Failed")
        print(f"Error: {e.stderr}")
        return False

def check_cuda():
    print("🔍 Checking current PyTorch installation...")
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            print(f"cuDNN version: {torch.backends.cudnn.version()}")
            print(f"GPU: {torch.cuda.get_device_name()}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            
            # Test GPU memory allocation
            try:
                test_tensor = torch.randn(1000, 1000, device='cuda', dtype=torch.float16)
                print("✅ GPU memory allocation test passed")
                del test_tensor
                torch.cuda.empty_cache()
                return True, "working"
            except Exception as e:
                print(f"⚠️  GPU memory test failed: {e}")
                return False, "memory_error"
        else:
            print("❌ CUDA not available - PyTorch is CPU-only")
            return False, "no_cuda"
    except ImportError:
        print("❌ PyTorch not installed")
        return False, "not_installed"
    except Exception as e:
        print(f"❌ Error checking CUDA: {e}")
        return False, "error"

def check_system_info():
    print("🖥️  System Information:")
    print(f"OS: {platform.system()} {platform.release()}")
    print(f"Python: {sys.version}")
    
    # Check if CUDA is installed on system
    try:
        result = subprocess.run("nvcc --version", shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"System CUDA: {result.stdout.strip()}")
        else:
            print("System CUDA: Not found")
    except:
        print("System CUDA: Could not check")

def check_pytorch_cuda_compatibility():
    """Check if current PyTorch is compatible with CUDA 12.8"""
    try:
        import torch
        if not torch.cuda.is_available():
            return False, "CUDA not available"
        
        # Check if CUDA version is compatible (12.1 or higher is fine)
        cuda_version = torch.version.cuda
        if cuda_version:
            # Parse CUDA version (e.g., "12.1" -> 12.1)
            try:
                major, minor = map(int, cuda_version.split('.')[:2])
                if major >= 12 and minor >= 1:
                    return True, f"CUDA {cuda_version} is compatible"
                else:
                    return False, f"CUDA {cuda_version} is too old (need 12.1+)"
            except:
                return False, f"Could not parse CUDA version: {cuda_version}"
        else:
            return False, "No CUDA version detected"
    except Exception as e:
        return False, f"Error checking compatibility: {e}"

def install_pytorch_cuda128():
    """Install PyTorch with CUDA 12.8 support"""
    print("\n🔧 Installing PyTorch with CUDA 12.8 support...")
    
    # Check if we actually need to reinstall
    try:
        import torch
        if torch.cuda.is_available():
            cuda_version = torch.version.cuda
            if cuda_version:
                major, minor = map(int, cuda_version.split('.')[:2])
                if major >= 12 and minor >= 1:
                    print(f"✅ Current PyTorch already has CUDA {cuda_version} support")
                    print("   No need to reinstall - your installation is compatible!")
                    return True
    except:
        pass
    
    # Only uninstall if we really need to
    print("⚠️  Current PyTorch installation needs updating...")
    response = input("Do you want to proceed with reinstalling PyTorch? (y/n): ")
    if response.lower() != 'y':
        print("Keeping current installation.")
        return False
    
    # Uninstall current PyTorch
    run_command("pip uninstall torch torchvision torchaudio -y", "Uninstalling current PyTorch")
    
    # Install PyTorch with CUDA 12.8
    success = run_command(
        "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128", 
        "Installing PyTorch with CUDA 12.8"
    )
    
    if not success:
        print("⚠️  Trying alternative installation method...")
        # Alternative: install with specific versions
        run_command(
            "pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu128",
            "Installing specific PyTorch versions with CUDA 12.8"
        )
    
    return success

def install_optional_optimizations():
    """Install optional optimizations (xFormers, triton)"""
    print("\n🚀 Installing optional optimizations...")
    
    # Try to install xFormers
    print("Installing xFormers...")
    xformers_success = run_command(
        "pip install xformers --index-url https://download.pytorch.org/whl/cu128",
        "Installing xFormers with CUDA support"
    )
    
    if not xformers_success:
        print("⚠️  xFormers installation failed. This is optional and training will still work.")
        print("   You can try installing it manually later if needed.")
    
    # Install triton (optional)
    print("Installing triton...")
    run_command("pip install triton", "Installing triton")
    
    return xformers_success

def verify_installation():
    """Verify the installation works correctly"""
    print("\n🔍 Verifying installation...")
    
    try:
        import torch
        print(f"✅ PyTorch {torch.__version__} imported successfully")
        
        if torch.cuda.is_available():
            print(f"✅ CUDA {torch.version.cuda} is available")
            print(f"✅ GPU: {torch.cuda.get_device_name()}")
            
            # Test GPU memory allocation
            try:
                test_tensor = torch.randn(1000, 1000, device='cuda')
                print(f"✅ GPU memory allocation test passed")
                del test_tensor
                torch.cuda.empty_cache()
            except Exception as e:
                print(f"⚠️  GPU memory test failed: {e}")
            
            return True
        else:
            print("❌ CUDA is not available after installation")
            return False
            
    except Exception as e:
        print(f"❌ Verification failed: {e}")
        return False

def main():
    print("🚀 PyTorch CUDA 12.8 Installation Fixer")
    print("=" * 50)
    
    # Check system info
    check_system_info()
    
    # Check current installation
    cuda_working, status = check_cuda()
    
    if cuda_working:
        print("\n✅ CUDA is already working! No need to reinstall.")
        
        # Check if it's compatible with CUDA 12.8
        compatible, reason = check_pytorch_cuda_compatibility()
        if compatible:
            print(f"✅ Current installation is compatible: {reason}")
            print("🎉 Your PyTorch installation is ready for CUDA 12.8!")
            print("\n📊 Current Status:")
            check_cuda()
            return
        else:
            print(f"⚠️  Current installation may have issues: {reason}")
            print("   However, since CUDA is working, you might want to keep it.")
            response = input("Do you want to reinstall anyway? (y/n): ")
            if response.lower() != 'y':
                print("Keeping current installation.")
                print("\n📊 Current Status:")
                check_cuda()
                return
    else:
        print(f"\n❌ CUDA is not working properly: {status}")
        if status == "not_installed":
            print("PyTorch is not installed. Installing now...")
        elif status == "no_cuda":
            print("PyTorch is CPU-only. Installing CUDA version...")
        elif status == "memory_error":
            print("GPU memory test failed. This might be a driver issue.")
            response = input("Do you want to try reinstalling PyTorch? (y/n): ")
            if response.lower() != 'y':
                print("Keeping current installation.")
                return
        else:
            print("Unknown issue. Installing fresh PyTorch...")
    
    # Install PyTorch with CUDA 12.8
    pytorch_success = install_pytorch_cuda128()
    
    if pytorch_success:
        # Install optional optimizations
        install_optional_optimizations()
        
        # Verify installation
        if verify_installation():
            print("\n🎉 Success! PyTorch now has CUDA 12.8 support.")
            print("You can now run your training script.")
            
            # Show final status
            print("\n📊 Final Status:")
            check_cuda()
        else:
            print("\n❌ Installation verification failed.")
            print("Please check the error messages above and try again.")
    else:
        print("\n❌ PyTorch installation failed or was cancelled.")
        print("Please check your internet connection and try again.")

if __name__ == "__main__":
    main() 