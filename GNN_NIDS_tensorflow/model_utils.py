import os
import shutil
from datetime import datetime

def backup_model(model_path, backup_dir="model_backups"):
    """
    Create a backup of a model directory
    
    Args:
        model_path: Path to the model directory to backup
        backup_dir: Directory to store backups
    """
    if not os.path.exists(backup_dir):
        os.makedirs(backup_dir)
    
    # Create timestamp for unique backup name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_name = f"model_backup_{timestamp}"
    backup_path = os.path.join(backup_dir, backup_name)
    
    # Create backup
    shutil.copytree(model_path, backup_path)
    print(f"Created backup at: {backup_path}")
    return backup_path

def restore_model(backup_path, restore_path):
    """
    Restore a model from backup
    
    Args:
        backup_path: Path to the backup to restore from
        restore_path: Path where to restore the model
    """
    if os.path.exists(restore_path):
        # Create backup of current state before overwriting
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        current_backup = f"{restore_path}_before_restore_{timestamp}"
        shutil.copytree(restore_path, current_backup)
        print(f"Created backup of current state at: {current_backup}")
        
        # Remove current model
        shutil.rmtree(restore_path)
    
    # Restore from backup
    shutil.copytree(backup_path, restore_path)
    print(f"Restored model from: {backup_path} to: {restore_path}")

def list_backups(backup_dir="model_backups"):
    """List all available model backups"""
    if not os.path.exists(backup_dir):
        print("No backups found")
        return []
    
    backups = [d for d in os.listdir(backup_dir) 
              if os.path.isdir(os.path.join(backup_dir, d))]
    
    for backup in sorted(backups):
        print(f"- {backup}")
    
    return backups
