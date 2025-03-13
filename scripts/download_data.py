import os
from pathlib import Path
from dotenv import load_dotenv
from roboflow import Roboflow

def download_dataset():
    # Load environment variables from roboflow.env
    load_dotenv('roboflow.env')
    
    # Get API key from environment
    api_key = os.getenv('ROBOFLOW_API_KEY')
    if not api_key:
        raise ValueError("ROBOFLOW_API_KEY not found in roboflow.env file")

    # Initialize Roboflow
    rf = Roboflow(api_key=api_key)

    # Project details
    workspace = "kci-technologies"
    project_name = "plant-identification-phrag-cycle-2"
    version_num = 11  # Use specific version as integer
    
    print(f"\nConnecting to workspace '{workspace}' and project '{project_name}' version {version_num}...")
    
    # Get project
    project = rf.workspace(workspace).project(project_name)
    print(f"\nUsing latest version: {version_num}")
    
    # Create data directories if they don't exist
    data_dir = Path(__file__).parent.parent / "data"
    dataset_dir = data_dir / "dataset"
    
    dataset_dir.mkdir(parents=True, exist_ok=True)

    # Download dataset with COCO segmentation format
    print("Starting download...")
    version = project.version(version_num)
    print(f"Version info: {version}")
    print(f"Downloading to: {dataset_dir}")
    print("\nAttempting download with format: coco-segmentation")
    try:
        print("\nChecking available exports...")
        exports = version.exports
        print(f"Available exports: {exports}")
        
        print("\nAttempting download...")
        dataset = version.download(model_format="coco", location=str(dataset_dir))
        print(f"Download result: {dataset}")
        print(f"Dataset downloaded to {dataset_dir}")
        
        # Try to access dataset attributes
        print("\nDataset attributes:")
        print(f"Dataset location: {dataset.location if hasattr(dataset, 'location') else 'N/A'}")
        print(f"Dataset format: {dataset.format if hasattr(dataset, 'format') else 'N/A'}")
        
        # Print all available attributes
        print("\nAll dataset attributes:")
        for attr in dir(dataset):
            if not attr.startswith('_'):  # Skip private attributes
                try:
                    value = getattr(dataset, attr)
                    print(f"{attr}: {value}")
                except Exception as e:
                    print(f"{attr}: Error accessing - {e}")
    except Exception as e:
        print(f"Error during download: {e}")
    
    # List contents
    print("\nContents of download directory:")
    for root, dirs, files in os.walk(dataset_dir):
        print(f"\nDirectory: {root}")
        if dirs:
            print("Subdirectories:", dirs)
        if files:
            print("Files:", files)

if __name__ == "__main__":
    download_dataset()