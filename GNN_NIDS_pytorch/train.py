import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
from dataset import IDSDataset
from model import GNN
from config import Config
from tqdm import tqdm
import os
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from torch.cuda.amp import autocast, GradScaler

# Update device configuration
Config.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {Config.DEVICE}")

def train_epoch(model, train_loader, optimizer, criterion, device, scaler, epoch):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    # Adjust learning rate for warmup
    if epoch < Config.WARMUP_EPOCHS:
        warmup_factor = (epoch + 1) / Config.WARMUP_EPOCHS
        for param_group in optimizer.param_groups:
            param_group['lr'] = Config.LEARNING_RATE * warmup_factor
    
    progress_bar = tqdm(train_loader, desc="Training")
    for batch_idx, batch in enumerate(progress_bar):
        # Variables to clean up
        output = None
        loss = None
        pred = None
        loss_value = 0
        
        try:
            # Clear memory before processing each batch
            torch.cuda.empty_cache()
            
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            
            # Forward pass with mixed precision
            with autocast(enabled=Config.USE_AMP):
                output = model(batch)
                loss = criterion(output, batch.y)
            
            # Check for NaNs or infinities in model output
            if torch.isnan(output).any() or torch.isinf(output).any():
                raise ValueError("Model output contains NaN or infinity")
            
            # Backward pass with gradient scaling
            scaler.scale(loss).backward()
            
            # Unscale before gradient clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), Config.GRADIENT_CLIP)
            
            scaler.step(optimizer)
            scaler.update()
            
            # Update metrics
            loss_value = loss.item()
            if not torch.isnan(loss):  # Only update if loss is valid
                total_loss += loss_value
                pred = output.max(1)[1]
                correct += pred.eq(batch.y).sum().item()
                total += batch.y.size(0)
            
            # Memory management
            if Config.EMPTY_CACHE_FREQ > 0 and (batch_idx + 1) % Config.EMPTY_CACHE_FREQ == 0:
                torch.cuda.empty_cache()
            
            # Update progress bar with the saved loss value
            progress_bar.set_postfix({
                'loss': f'{loss_value:.4f}',
                'acc': f'{100. * correct / total:.2f}%' if total > 0 else '0.00%'
            })
            
        except Exception as e:
            print(f"Error in batch {batch_idx}: {str(e)}")
            continue
        
        finally:
            # Clean up variables explicitly
            if output is not None:
                del output
            if loss is not None:
                del loss
            if pred is not None:
                del pred
            torch.cuda.empty_cache()
    
    avg_loss = total_loss / len(train_loader) if total > 0 else float('inf')
    avg_acc = correct / total if total > 0 else 0
    return avg_loss, avg_acc

def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation"):
            batch = batch.to(device)
            with autocast(enabled=Config.USE_AMP):
                output = model(batch)
                loss = criterion(output, batch.y)
            
            total_loss += loss.item()
            pred = output.max(1)[1]
            correct += pred.eq(batch.y).sum().item()
            total += batch.y.size(0)
            
            # Memory management
            torch.cuda.empty_cache()
    
    return total_loss / len(val_loader), correct / total

def main():
    # Clear GPU memory at start
    torch.cuda.empty_cache()
    
    # Create directories for logs and checkpoints
    os.makedirs(Config.LOG_DIR, exist_ok=True)
    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
    
    # Initialize tensorboard writer
    writer = SummaryWriter(Config.LOG_DIR)
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # Load datasets
    print("Loading training dataset...")
    train_dataset = IDSDataset(root=Config.TRAIN_PATH)
    print("Loading validation dataset...")
    val_dataset = IDSDataset(root=Config.VAL_PATH)
    
    # Create data loaders with prefetch factor
    train_loader = DataLoader(
        train_dataset, 
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        prefetch_factor=2
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        num_workers=4,
        pin_memory=True,
        prefetch_factor=2
    )
    
    # Initialize model and move to device
    model = GNN(Config).to(Config.DEVICE)
    if Config.GRADIENT_CHECKPOINTING:
        model.enable_checkpointing()
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Initialize optimizer and loss function
    optimizer = optim.AdamW(
        model.parameters(),
        lr=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY,
        amsgrad=True
    )
    criterion = nn.NLLLoss()  # Use NLL loss since model outputs log_softmax
    scaler = GradScaler(enabled=Config.USE_AMP)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.6,
        patience=5,
        verbose=True,
        min_lr=1e-6
    )
    
    # Early stopping
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(Config.NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{Config.NUM_EPOCHS}")
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, Config.DEVICE, scaler, epoch)
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, Config.DEVICE)
        
        # Log metrics
        writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('Loss/Validation', val_loss, epoch)
        writer.add_scalar('Accuracy/Train', train_acc, epoch)
        writer.add_scalar('Accuracy/Validation', val_acc, epoch)
        writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)
        
        print(f"\nTrain Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save the best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc
            }, os.path.join(Config.CHECKPOINT_DIR, 'best_model.pt'))
        else:
            patience_counter += 1
            if patience_counter >= Config.PATIENCE:
                print("\nEarly stopping triggered!")
                break
    
    print("\nTraining completed!")
    writer.close()

if __name__ == "__main__":
    main()
