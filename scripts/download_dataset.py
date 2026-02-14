import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

def download_kaggle_dataset():
    """Download internship dataset from Kaggle"""
    try:
        import kaggle
    except ImportError:
        print("❌ Kaggle package not installed. Run: pip install kaggle")
        return None
    
    # Create data directories
    Path("data/raw").mkdir(parents=True, exist_ok=True)
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    
    dataset_name = "jayaantanaath/internship-opportunities-in-india-2025"
    output_path = "data/raw"
    
    print(f"📥 Downloading dataset: {dataset_name}")
    print("⚠️  Make sure you have ~/.kaggle/kaggle.json configured")
    
    try:
        kaggle.api.dataset_download_files(dataset_name, path=output_path, unzip=True)
        print(f"✅ Dataset downloaded to {output_path}")
        
        csv_files = list(Path(output_path).glob("*.csv"))
        if csv_files:
            print(f"📄 Found CSV file: {csv_files[0].name}")
            return csv_files[0]
        else:
            print("❌ No CSV file found in downloaded dataset")
            return None
    except Exception as e:
        print(f"❌ Download failed: {e}")
        print("\nSetup instructions:")
        print("1. Go to https://www.kaggle.com/settings")
        print("2. Create API token → download kaggle.json")
        print("3. Move to ~/.kaggle/kaggle.json")
        print("4. Run: chmod 600 ~/.kaggle/kaggle.json (Linux/Mac)")
        return None

if __name__ == "__main__":
    download_kaggle_dataset()
