import os
import shutil
import time
import sys

def safe_move(src, dst):
    """Move file with overwrite safety."""
    if os.path.exists(src):
        if os.path.exists(dst):
            print(f"Overwriting {dst}...")
            os.remove(dst)
        shutil.move(src, dst)
        print(f"Moved {src} -> {dst}")
    else:
        print(f"Source {src} not found, skipping.")

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    legacy_dir = os.path.join(base_dir, "legacy")
    
    print("Starting Project Consolidation...")
    
    # 0. Create legacy directory
    if not os.path.exists(legacy_dir):
        os.makedirs(legacy_dir)
        print(f"Created legacy directory: {legacy_dir}")

    # 1. Archive Old Files
    print("\n[Archiving Legacy Files]")
    files_to_archive = [
        ("train_anomaly_detector.py", "train_anomaly_detector_tf_backup.py"),
        ("Dockerfile.aoi", "Dockerfile.aoi.backup"),
        ("Dockerfile", "Dockerfile.old_backup") # If exists
    ]
    for src, dst_name in files_to_archive:
        src_path = os.path.join(base_dir, src)
        if os.path.exists(src_path):
            shutil.move(src_path, os.path.join(legacy_dir, dst_name))
            print(f"Archived {src} -> legacy/{dst_name}")

    # 2. Delete Unused Test Files
    print("\n[Cleaning Up Test Files]")
    files_to_delete = [
        "test_gpu_pytorch.py",
        "test_gpu_resnet.py",
        "Dockerfile.blackwell_test" # Will be renamed
    ]
    for fname in files_to_delete:
        fpath = os.path.join(base_dir, fname)
        if os.path.exists(fpath):
            # Special case: Dockerfile.blackwell_test is being renamed, not deleted here
            # Wait, logic check: We want to RENAME it to Dockerfile.
            if fname == "Dockerfile.blackwell_test":
                continue
            os.remove(fpath)
            print(f"Deleted {fname}")

    # 3. Promote New Files
    print("\n[Promoting New Files]")
    promotions = [
        ("train_anomaly_detector_pytorch.py", "train_anomaly_detector.py"),
        ("Dockerfile.blackwell_test", "Dockerfile"),
        # We keep model_pytorch.py as is, or rename to model.py?
        # User said "update app.py". Let's verify integration first. 
        # For now, let's keep model_pytorch.py to avoid breaking imports in the newly moved train script without editing content.
        # But for 'train_anomaly_detector.py', we moved the old one, and renamed pytorch one to it.
        # Does the pytorch script import 'model_pytorch'? Yes.
        # So if we rename model_pytorch -> model.py, we must edit train_anomaly_detector.py (the new one).
        # Let's simple keep model_pytorch.py for now, or minimal rename.
    ]
    
    for src, dst in promotions:
        safe_move(os.path.join(base_dir, src), os.path.join(base_dir, dst))

    print("\n[Update Complete]")
    print(f"Legacy files moved to: {legacy_dir}")
    print("Please restart app.py to apply changes.")

if __name__ == "__main__":
    main()
