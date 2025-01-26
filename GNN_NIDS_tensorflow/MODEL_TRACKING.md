# Model Performance Tracking

## Best Model Checkpoint
- **Location**: `logs/ckpt/model_400`
- **Backup Date**: January 4, 2024
- **Backup Location**: `model_backups/best_model_backup_20240104`

### Performance Metrics
- **Categorical Accuracy**: ~0.97
- **Loss**: ~0.068

### Configuration
- Model architecture: GNN-based NIDS
- Training dataset: IDS2017
- Framework: TensorFlow 2.13.0

### Notes
- This is the best performing model as of January 2024
- Shows stable performance on validation set
- Successfully detects network intrusions with high accuracy

## How to Restore
```python
from model_utils import restore_model

# To restore the best model:
restore_model("model_backups/best_model_backup_20240104", "logs/ckpt/model_400")
```

## Future Changes Log
When making changes to the model, please log them here:

| Date | Change Description | Performance Impact | Backup Location |
|------|-------------------|-------------------|-----------------|
| | | | |
