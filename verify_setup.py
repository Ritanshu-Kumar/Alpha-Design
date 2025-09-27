def verify_installation():
    required_packages = [
        'torch', 'numpy', 'scipy', 'tqdm', 'matplotlib', 
        'pandas', 'trimesh', 'sklearn', 'psutil', 'yaml',
        'deap', 'sympy', 'plotly', 'seaborn', 'memory_profiler', 'joblib'
    ]
    
    failed_imports = []
    
    for package in required_packages:
        try:
            if package == 'sklearn':
                import sklearn
            elif package == 'yaml':
                import yaml
            elif package == 'memory_profiler':
                import memory_profiler
            else:
                __import__(package)
            print(f"✅ {package}")
        except ImportError as e:
            print(f"❌ {package}: {e}")
            failed_imports.append(package)
    
    try:
        import stl
        print("✅ numpy-stl (imported as 'stl')")
    except ImportError as e:
        print(f"❌ numpy-stl: {e}")
        failed_imports.append('numpy-stl')
    
    if failed_imports:
        print(f"\n⚠️  Failed imports: {failed_imports}")
        print("💡 Try installing individually:")
        for pkg in failed_imports:
            if pkg == 'numpy-stl':
                print(f"   pip install numpy-stl")
            else:
                print(f"   pip install {pkg}")
    else:
        print("\n🎉 All packages successfully installed!")
        
        # Test CUDA if available
        try:
            import torch
            if torch.cuda.is_available():
                print(f"🔥 CUDA available: {torch.cuda.get_device_name(0)}")
                print(f"🔥 CUDA version: {torch.version.cuda}")
            else:
                print("💻 Running on CPU (CUDA not available)")
        except:
            pass
    
    return len(failed_imports) == 0

if __name__ == "__main__":
    verify_installation()