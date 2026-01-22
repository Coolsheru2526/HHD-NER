
import os
import sys
import torch

def check_environment():
    """Check Python environment and dependencies."""
    print("=" * 60)
    print("PRE-FLIGHT CHECK - Hindi Medical NER")
    print("=" * 60)
    
    issues = []
    
    # 1. Check CUDA availability
    print("\n[1/6] Checking CUDA...")
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        print(f"  ✓ CUDA available: {device_name}")
        print(f"  ✓ CUDA version: {torch.version.cuda}")
    else:
        print("  ⚠ WARNING: CUDA not available. Training will be VERY slow on CPU.")
        print("    Consider using A5000 GPU for training.")
    
    # 2. Check data files
    print("\n[2/6] Checking data files...")
    required_files = {
        "data/processed/train_v2.conll": "Main training data",
        "data/raw/Hindi_Health_Data.txt": "Raw dataset",
        "data/raw/Disease_Gazetteer.txt": "Disease gazetteer",
        "data/raw/Symptom_Gazetteer.txt": "Symptom gazetteer",
        "data/raw/Consumable_Gazetteer.txt": "Consumable gazetteer",
        "data/raw/Person_Gazetteer.txt": "Person gazetteer",
    }
    
    for file_path, description in required_files.items():
        if os.path.exists(file_path):
            size_mb = os.path.getsize(file_path) / (1024 * 1024)
            print(f"  ✓ {description}: {size_mb:.2f} MB")
        else:
            print(f"  ✗ MISSING: {file_path}")
            issues.append(f"Missing file: {file_path}")
    
    # 3. Check if splits exist
    print("\n[3/6] Checking train/dev/test splits...")
    split_files = [
        "data/processed/train_split.conll",
        "data/processed/dev.conll",
        "data/processed/test.conll"
    ]
    
    all_splits_exist = all(os.path.exists(f) for f in split_files)
    
    if all_splits_exist:
        print("  ✓ All splits exist")
        for split_file in split_files:
            size_mb = os.path.getsize(split_file) / (1024 * 1024)
            print(f"    - {os.path.basename(split_file)}: {size_mb:.2f} MB")
    else:
        print("  ✗ Splits NOT created yet")
        print("    ACTION REQUIRED: Run 'python src/split_data.py' first!")
        issues.append("Data splits not created - run split_data.py")
    
    # 4. Check disk space
    print("\n[4/6] Checking disk space...")
    try:
        import shutil
        total, used, free = shutil.disk_usage(".")
        free_gb = free / (1024**3)
        
        if free_gb > 5:
            print(f"  ✓ Free disk space: {free_gb:.2f} GB")
        else:
            print(f"  ⚠ WARNING: Low disk space ({free_gb:.2f} GB)")
            print("    Need at least 5GB for model checkpoints")
            issues.append(f"Low disk space: {free_gb:.2f} GB")
    except Exception as e:
        print(f"  ⚠ Could not check disk space: {e}")
    
    # 5. Check dependencies
    print("\n[5/6] Checking Python dependencies...")
    required_modules = {
        "transformers": "Transformers library",
        "torchcrf": "CRF layer",
        "seqeval": "NER evaluation",
        "indicnlp": "Indic NLP library",
        "tqdm": "Progress bars",
        "numpy": "NumPy",
    }
    
    for module, description in required_modules.items():
        try:
            __import__(module)
            print(f"  ✓ {description}")
        except ImportError:
            print(f"  ✗ MISSING: {description} (pip install {module})")
            issues.append(f"Missing dependency: {module}")
    
    # 6. Check output directory
    print("\n[6/6] Checking output directory...")
    if not os.path.exists("weights"):
        os.makedirs("weights")
        print("  ✓ Created 'weights' directory")
    else:
        print("  ✓ 'weights' directory exists")
    
    # Summary
    print("\n" + "=" * 60)
    if not issues:
        print("✓ ALL CHECKS PASSED - Ready to train!")
        print("\nNext steps:")
        if not all_splits_exist:
            print("  1. Run: python src/split_data.py")
            print("  2. Run: python src/train_v2.py")
        else:
            print("  1. Run: python src/train_v2.py")
        print("\nTraining will save:")
        print("  - Model: weights/best_model.pt")
        print("  - Logs: weights/training.log")
        print("  - History: weights/training_history.json")
        return True
    else:
        print(f"✗ FOUND {len(issues)} ISSUE(S) - Fix before training:")
        for i, issue in enumerate(issues, 1):
            print(f"  {i}. {issue}")
        return False
    
    print("=" * 60)


if __name__ == "__main__":
    success = check_environment()
    sys.exit(0 if success else 1)
