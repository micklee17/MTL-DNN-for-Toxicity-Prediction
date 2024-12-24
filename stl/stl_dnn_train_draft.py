import deepchem as dc
import pandas as pd
from hyperopt import hp, fmin, tpe, Trials
import torch
import torch.nn as nn
import torch.optim as optim
import os
from torch.utils.data import DataLoader, TensorDataset
import tempfile
import logging
import json
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define a data loading function
def load_tox21_csv(dataset_file, featurizer='ECFP', split='stratified'):
    tasks = ['NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER', 'NR-ER-LBD', 'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53']
    if featurizer == 'ECFP':
        featurizer = dc.feat.CircularFingerprint()
    elif featurizer == 'MACCS':
        featurizer = dc.feat.MACCSKeysFingerprint()
    
    loader = dc.data.CSVLoader(tasks=tasks, feature_field="smiles", featurizer=featurizer)
    dataset = loader.featurize(dataset_file)
    
    transformers = [dc.trans.BalancingTransformer(dataset=dataset)]
    for transformer in transformers:
        dataset = transformer.transform(dataset)
    
    splitters = {
        'stratified': dc.splits.RandomStratifiedSplitter(),
        'scaffold': dc.splits.ScaffoldSplitter()
    }
    splitter = splitters[split]
    train, test = splitter.train_test_split(dataset, seed=42)  # Added seed parameter for reproducibility
    
    return tasks, (train, test), transformers

# Get combination of layer size
def create_subarrays(arr):
    result = []
    for num in arr:
        for i in range(1, 5):
            subarray = [num] * i
            result.append(subarray)
    return result

# Define a function to create a list of dropout rates
def create_dropout_list(layer_sizes, dropout_rate):
    return [dropout_rate] * len(layer_sizes)

# Define the DNN model for single task learning using PyTorch
class DNNSingleTaskClassifier(nn.Module):
    """Single-task DNN classifier with configurable architecture
    
    Architecture Overview:
    1. Feature Processing Layers (Initial Hidden Layers):
       - Multiple fully connected layers with ReLU activation
       - Process raw molecular fingerprints
       - Gradually reduce dimensionality
       - Learn basic chemical feature representations
       - Dropout for regularization
    
    2. Hidden Processing Layers (Deep Hidden Layers):
       - Additional fully connected layers with ReLU
       - Learn more complex chemical patterns
       - Task-specific feature extraction
       - Progressive dimensionality reduction
       - Deeper layers capture higher-level interactions
       - Dropout prevents overfitting
    
    3. Output Layer:
       - Final binary classification
       - No activation (uses BCEWithLogitsLoss)
    
    Hidden Layer Properties:
    - Feature Layers: Initial feature transformation
    - Processing Layers: Complex pattern recognition
    - All hidden layers use:
      * ReLU activation for non-linearity
      * Dropout for regularization
      * Decreasing sizes for feature abstraction
    
    Parameters:
        n_features (int): Number of input features (2048 for ECFP)
        layer_sizes (list): Sizes of feature processing layers
        dropouts (list): Dropout rates for feature layers
        hidden_layers (list): Sizes of hidden processing layers
        hidden_layer_dropouts (list): Dropout rates for hidden layers
    """
    def __init__(self, n_features, layer_sizes, dropouts, hidden_layers, hidden_layer_dropouts):
        super(DNNSingleTaskClassifier, self).__init__()
        
        # ===== Feature Processing Layers =====
        self.feature_layers = nn.ModuleList()
        prev_size = n_features
        
        # Build initial feature processing layers
        for size, dropout in zip(layer_sizes, dropouts):
            layer_block = nn.Sequential(
                nn.Linear(prev_size, size),
                nn.BatchNorm1d(size),  # Added batch normalization
                nn.ReLU(),
                nn.Dropout(dropout)
            )
            self.feature_layers.append(layer_block)
            prev_size = size
        
        # ===== Hidden Processing Layers =====
        self.hidden_layers = nn.ModuleList()
        
        # Build additional hidden processing layers
        for size, dropout in zip(hidden_layers, hidden_layer_dropouts):
            hidden_block = nn.Sequential(
                nn.Linear(prev_size, size),
                nn.BatchNorm1d(size),  # Added batch normalization
                nn.ReLU(),
                nn.Dropout(dropout)
            )
            self.hidden_layers.append(hidden_block)
            prev_size = size
        
        # ===== Output Layer =====
        self.output_layer = nn.Linear(prev_size, 1)
    
    def forward(self, x):
        # 1. Flatten input if needed
        x = x.view(x.size(0), -1)
        
        # 2. Feature processing
        for feature_layer in self.feature_layers:
            x = feature_layer(x)
        
        # 3. Hidden processing
        for hidden_layer in self.hidden_layers:
            x = hidden_layer(x)
        
        # 4. Final prediction
        return self.output_layer(x)

def count_parameters(model):
    """Count trainable parameters in model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

num_epochs = 30  # Example value, you can change it as needed

# Function to minimize
metric = dc.metrics.Metric(dc.metrics.roc_auc_score, mode='classification')

def evaluate_model(model, data_loader, device):
    model.eval()
    y_preds = []
    y_trues = []
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            y_preds.append(outputs.cpu())
            y_trues.append(targets.cpu())
    y_preds = torch.cat(y_preds, dim=0).numpy()
    y_trues = torch.cat(y_trues, dim=0).numpy()
    return dc.metrics.roc_auc_score(y_trues, y_preds)

def log_model_details(model, layer_sizes, hidden_layers, device, args):
    """Helper function for logging model architecture and training setup"""
    logger.info("\n" + "="*50)
    logger.info("MODEL ARCHITECTURE")
    logger.info("="*50)
    logger.info(f"Input Features: 2048")
    logger.info("\nFeature Processing Layers:")
    for i, size in enumerate(layer_sizes):
        logger.info(f"  Layer {i+1}: {size} neurons")
    
    logger.info("\nHidden Processing Layers:")
    for i, size in enumerate(hidden_layers):
        logger.info(f"  Layer {i+1}: {size} neurons")
    
    logger.info("\nTraining Configuration:")
    logger.info(f"  Device: {device}")
    logger.info(f"  Learning Rate: {args['learning_rate']:.6f}")
    logger.info(f"  Batch Size: {args['batch_size']}")
    logger.info(f"  Dropout Rate: {args['dropout_rate']:.2f}")
    logger.info("="*50 + "\n")

def log_epoch_summary(epoch, num_epochs, epoch_loss, test_score, best_score):
    """Helper function for logging epoch results"""
    logger.info("\n" + "-"*40)
    logger.info(f"Epoch {epoch+1}/{num_epochs} Summary:")
    logger.info(f"  Training Loss: {epoch_loss:.4f}")
    logger.info(f"  Test ROC-AUC: {test_score:.4f}")
    logger.info(f"  Best ROC-AUC: {best_score:.4f}")
    logger.info("-"*40 + "\n")

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')
        self.delta = delta

    def __call__(self, val_loss):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                logger.info(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss)
            self.counter = 0

    def save_checkpoint(self, val_loss):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            logger.info(f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...")
        self.val_loss_min = val_loss

def fm(args, task, train_dataset, test_dataset):
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    layer_sizes = args['layer_sizes']
    dropouts = create_dropout_list(layer_sizes, args['dropout_rate'])
    hidden_layers = args['hidden_layers']
    hidden_layer_dropouts = create_dropout_list(hidden_layers, args['hidden_layer_dropout'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DNNSingleTaskClassifier(n_features=2048, layer_sizes=layer_sizes, dropouts=dropouts, 
                                  hidden_layers=hidden_layers, hidden_layer_dropouts=hidden_layer_dropouts).to(device)
    
    # Enhanced logging of model details
    log_model_details(model, layer_sizes, hidden_layers, device, args)
    num_params = count_parameters(model)
    logger.info(f"Total Trainable Parameters: {num_params:,}\n")
    
    optimizer = optim.Adam(model.parameters(), lr=args['learning_rate'])
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    early_stopping = EarlyStopping(patience=10, verbose=True)
    loss_fn = nn.BCEWithLogitsLoss()
    
    # Prepare DataLoaders with drop_last=True
    train_loader = DataLoader(
        TensorDataset(
            torch.Tensor(train_dataset.X), 
            torch.Tensor(train_dataset.y[:, tasks.index(task)])
        ), 
        batch_size=max(int(args['batch_size']), 2),  # Ensure minimum batch size of 2
        shuffle=True,
        drop_last=True  # Drop last batch if incomplete
    )
    
    test_loader = DataLoader(
        TensorDataset(
            torch.Tensor(test_dataset.X), 
            torch.Tensor(test_dataset.y[:, tasks.index(task)])
        ), 
        batch_size=max(int(args['batch_size']), 2),  # Ensure minimum batch size of 2
        shuffle=False,
        drop_last=False  # Keep all samples for testing
    )
    
    train_losses = []
    test_roc_auc_scores = []
    best_score = float('-inf')
    best_epoch = -1
    
    logger.info(f"\nStarting training for task: {task}")
    logger.info("="*50)
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        batch_count = 0
        
        # Add progress tracking
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs.view(-1), targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            batch_count += 1
            
            # Log progress every 20% of batches
            if (batch_idx + 1) % max(1, len(train_loader)//5) == 0:
                logger.info(f"  Batch {batch_idx+1}/{len(train_loader)} - Loss: {loss.item():.4f}")
        
        epoch_loss /= batch_count
        test_score = evaluate_model(model, test_loader, device)
        test_roc_auc_scores.append(test_score)
        
        # Calculate validation loss for scheduling
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                val_loss += loss_fn(outputs.view(-1), targets).item()
        val_loss /= len(test_loader)

        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Early stopping
        early_stopping(val_loss)
        if early_stopping.early_stop:
            logger.info("Early stopping triggered")
            break

        # Enhanced epoch summary
        log_epoch_summary(epoch, num_epochs, epoch_loss, test_score, best_score)
        
        # Save best model with task-specific name
        if test_score > best_score:
            best_score = test_score
            best_epoch = epoch
            num_params = count_parameters(model)  # Count parameters of best model
            model_save_path = os.path.join(script_dir, f"best_dnn_single_task_model_{task}.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'score': best_score,
                'num_parameters': num_params,  # Save parameter count
                'architecture': {
                    'n_features': 2048,
                    'layer_sizes': layer_sizes,
                    'dropouts': dropouts,
                    'hidden_layers': hidden_layers,
                    'hidden_layer_dropouts': hidden_layer_dropouts
                },
                'hyperparameters': args,
                'task': task  # Save which task this model is for
            }, model_save_path)
            logger.info(f"New best model for {task} saved at epoch {epoch+1}")
            logger.info(f"Score: {best_score:.4f}, Parameters: {num_params:,}")

    # Final summary
    logger.info("\n" + "="*50)
    logger.info("TRAINING COMPLETE")
    logger.info(f"Best ROC-AUC: {best_score:.4f} at epoch {best_epoch+1}")
    logger.info(f"Final Parameters: {num_params:,}")
    logger.info("="*50 + "\n")
    
    # Save task-specific training history
    history = {
        'train_losses': train_losses,
        'test_roc_auc_scores': test_roc_auc_scores,
        'best_epoch': best_epoch,
        'best_score': best_score,
        'hyperparameters': args
    }
    history_path = os.path.join(script_dir, f"training_history_{task}.json")
    with open(history_path, "w") as f:
        json.dump(history, f)
    logger.info(f"Training history saved to {history_path}")

    return -1 * best_score  # Return best score for hyperopt optimization

def save_parameter_counts(tasks, script_dir):
    """Save parameter counts and best architecture details for all tasks to a file"""
    parameter_counts = {}
    total_parameters = 0
    model_details = {}
    
    # Collect parameter counts and model details for each task
    for task in tasks:
        model_path = os.path.join(script_dir, f"best_dnn_single_task_model_{task}.pth")
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path)
            num_params = checkpoint['num_parameters']
            parameter_counts[task] = num_params
            total_parameters += num_params
            
            # Store model architecture details
            model_details[task] = {
                'layer_sizes': checkpoint['architecture']['layer_sizes'],
                'dropout_rate': checkpoint['hyperparameters']['dropout_rate'],
                'hidden_layers': checkpoint['architecture']['hidden_layers'],
                'hidden_layer_dropout': checkpoint['hyperparameters']['hidden_layer_dropout'],
                'batch_size': checkpoint['hyperparameters']['batch_size'],
                'learning_rate': checkpoint['hyperparameters']['learning_rate']
            }
    
    # Write to file with enhanced details
    output_path = os.path.join(script_dir, "model_parameters_summary.txt")
    with open(output_path, "w") as f:
        f.write("=== Model Parameters and Architecture Summary ===\n\n")
        for task in tasks:
            f.write(f"Task: {task}\n")
            f.write("-" * 50 + "\n")
            f.write(f"Total Parameters: {parameter_counts[task]:,}\n")
            f.write(f"Best layer sizes: {model_details[task]['layer_sizes']}\n")
            f.write(f"Best dropout rate: {model_details[task]['dropout_rate']:.4f}\n")
            f.write(f"Best hidden layers: {model_details[task]['hidden_layers']}\n")
            f.write(f"Best hidden layer dropout rate: {model_details[task]['hidden_layer_dropout']:.4f}\n")
            f.write(f"Best batch size: {model_details[task]['batch_size']}\n")
            f.write(f"Best learning rate: {model_details[task]['learning_rate']:.6f}\n\n")
        
        f.write("=" * 50 + "\n")
        f.write(f"Total parameters across all tasks: {total_parameters:,}\n")
        f.write(f"Average parameters per task: {total_parameters/len(tasks):,.2f}\n")
    
    return parameter_counts, total_parameters, model_details

def save_best_hyperparameters(tasks, script_dir):
    """Save best hyperparameters for all tasks to a file"""
    best_hyperparams = {}
    
    # Collect hyperparameters for each task
    for task in tasks:
        model_path = os.path.join(script_dir, f"best_dnn_single_task_model_{task}.pth")
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path)
            hyperparams = checkpoint['hyperparameters']
            best_hyperparams[task] = hyperparams
    
    # Write to file
    output_path = os.path.join(script_dir, "best_hyperparameters_summary.txt")
    with open(output_path, "w") as f:
        f.write("=== Best Hyperparameters Summary ===\n\n")
        for task, params in best_hyperparams.items():
            f.write(f"Task: {task}\n")
            f.write("-" * 50 + "\n")
            f.write(f"Layer Sizes: {params['layer_sizes']}\n")
            f.write(f"Dropout Rate: {params['dropout_rate']:.4f}\n")
            f.write(f"Learning Rate: {params['learning_rate']:.6f}\n")
            f.write(f"Batch Size: {params['batch_size']}\n")
            f.write(f"Hidden Layers: {params['hidden_layers']}\n")
            f.write(f"Hidden Layer Dropout: {params['hidden_layer_dropout']:.4f}\n")
            f.write("\n")
    
    return best_hyperparams

def save_best_scores(tasks, script_dir):
    """Save best ROC-AUC scores for all tasks to a file"""
    best_scores = {}
    
    # Collect scores for each task
    for task in tasks:
        model_path = os.path.join(script_dir, f"best_dnn_single_task_model_{task}.pth")
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path)
            score = checkpoint['score']
            best_scores[task] = score
    
    # Calculate average score
    avg_score = sum(best_scores.values()) / len(best_scores)
    
    # Write to file
    output_path = os.path.join(script_dir, "best_roc_auc_scores.txt")
    with open(output_path, "w") as f:
        f.write("=== Best ROC-AUC Scores ===\n\n")
        for task, score in best_scores.items():
            f.write(f"{task}: {score:.4f}\n")
        f.write(f"\nAverage ROC-AUC Score: {avg_score:.4f}\n")
    
    return best_scores, avg_score

if __name__ == "__main__":
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Load the local csv data file
    data = r'C:\Users\Ford\Desktop\fyp\tox21.csv'
    tasks, datasets, transformers = load_tox21_csv(data)
    train_dataset, test_dataset = datasets

    # Print dataset information
    logger.info("\n=== Dataset Information ===")
    logger.info(f"Number of tasks: {len(tasks)}")
    logger.info(f"Tasks: {tasks}")
    
    logger.info("\n=== Training Dataset ===")
    logger.info(f"Training set shape - X: {train_dataset.X.shape}, y: {train_dataset.y.shape}")
    logger.info("\nFirst 5 samples of training data:")
    for i in range(min(5, len(train_dataset.X))):
        logger.info(f"\nSample {i+1}:")
        logger.info(f"Features (first 10): {train_dataset.X[i][:10]}...")
        logger.info(f"Labels: {dict(zip(tasks, train_dataset.y[i]))}")
    
    logger.info("\nLast 5 samples of training data:")
    for i in range(max(0, len(train_dataset.X)-5), len(train_dataset.X)):
        logger.info(f"\nSample {i+1}:")
        logger.info(f"Features (first 10): {train_dataset.X[i][:10]}...")
        logger.info(f"Labels: {dict(zip(tasks, train_dataset.y[i]))}")

    logger.info("\n=== Test Dataset ===")
    logger.info(f"Test set shape - X: {test_dataset.X.shape}, y: {test_dataset.y.shape}")
    logger.info("\nFirst 5 samples of test data:")
    for i in range(min(5, len(test_dataset.X))):
        logger.info(f"\nSample {i+1}:")
        logger.info(f"Features (first 10): {test_dataset.X[i][:10]}...")
        logger.info(f"Labels: {dict(zip(tasks, test_dataset.y[i]))}")
    
    logger.info("\nLast 5 samples of test data:")
    for i in range(max(0, len(test_dataset.X)-5), len(test_dataset.X)):
        logger.info(f"\nSample {i+1}:")
        logger.info(f"Features (first 10): {test_dataset.X[i][:10]}...")
        logger.info(f"Labels: {dict(zip(tasks, test_dataset.y[i]))}")

    # Print class distribution for each task
    logger.info("\n=== Class Distribution ===")
    for i, task in enumerate(tasks):
        train_pos = (train_dataset.y[:, i] == 1).sum()
        train_neg = (train_dataset.y[:, i] == 0).sum()
        test_pos = (test_dataset.y[:, i] == 1).sum()
        test_neg = (test_dataset.y[:, i] == 0).sum()
        
        logger.info(f"\nTask: {task}")
        logger.info(f"Training - Positive: {train_pos}, Negative: {train_neg}, Ratio: {train_pos/(train_pos+train_neg):.3f}")
        logger.info(f"Test - Positive: {test_pos}, Negative: {test_neg}, Ratio: {test_pos/(test_pos+test_neg):.3f}")

    # Add user prompt
    logger.info("\n=== Verification Complete ===")
    logger.info("Please verify the dataset information above.")
    input("Press Enter to start training...")
    logger.info("\nStarting hyperparameter optimization...")

    # Define the array for layer sizes and get combinations
    arr = [2048, 1024]  # Reduced
    layer_size_comb = create_subarrays(arr)
    logger.info(f"Layer size combinations: {layer_size_comb}")
    
    # Define the hyperparameter search space with balanced architectures
    search_space = {
        'layer_sizes': hp.choice('layer_sizes', layer_size_comb),
        'dropout_rate': hp.uniform('dropout_rate', 0.1, 0.5),
        'learning_rate': hp.loguniform('learning_rate', -4, -2),
        'batch_size': hp.choice('batch_size', [32, 64, 128]),
        'hidden_layers': hp.choice('hidden_layers', [
            [1024, 512, 256],  # Smaller architectures
            [2048, 1024, 512],
            [512, 256, 128],
        ]),
        'hidden_layer_dropout': hp.uniform('hidden_layer_dropout', 0.1, 0.5)
    }
    
    # Update search space to ensure minimum batch size
    search_space = {
        'layer_sizes': hp.choice('layer_sizes', layer_size_comb),
        'dropout_rate': hp.uniform('dropout_rate', 0.1, 0.5),
        'learning_rate': hp.loguniform('learning_rate', -4, -2),
        'batch_size': hp.choice('batch_size', [32, 64, 128]),  # Removed smaller batch sizes
        'hidden_layers': hp.choice('hidden_layers', [
            [1024, 512, 256],
            [2048, 1024, 512],
            [512, 256, 128]
        ]),
        'hidden_layer_dropout': hp.uniform('hidden_layer_dropout', 0.1, 0.5)
    }
    
    # Run hyperparameter optimization for each task
    for task in tasks:
        trials = Trials()
        best = fmin(lambda args: fm(args, task, train_dataset, test_dataset), 
                   space=search_space, algo=tpe.suggest, max_evals=20, trials=trials)
        logger.info(f"Best hyperparameters found for {task}: {best}")
        
        # Load and print details of best model
        model_path = os.path.join(script_dir, f"best_dnn_single_task_model_{task}.pth")
        checkpoint = torch.load(model_path)
        logger.info(f"\nBest model summary for {task}:")
        logger.info(f"Parameters: {checkpoint['num_parameters']:,}")
        logger.info(f"ROC-AUC Score: {checkpoint['score']:.4f}")
        logger.info(f"Architecture: {checkpoint['architecture']}")
    
    # After all tasks are trained, save parameter summary with architecture details
    logger.info("\nGenerating parameter counts and architecture summary...")
    param_counts, total_params, model_details = save_parameter_counts(tasks, script_dir)
    
    # Log the summary
    logger.info("\n=== Model Parameters and Architecture Summary ===")
    for task in tasks:
        logger.info(f"\nTask: {task}")
        logger.info(f"Parameters: {param_counts[task]:,}")
        logger.info(f"Best layer sizes: {model_details[task]['layer_sizes']}")
        logger.info(f"Best dropout rate: {model_details[task]['dropout_rate']:.4f}")
        logger.info(f"Best hidden layers: {model_details[task]['hidden_layers']}")
        logger.info(f"Best hidden layer dropout: {model_details[task]['hidden_layer_dropout']:.4f}")
        logger.info(f"Best batch size: {model_details[task]['batch_size']}")
        logger.info(f"Best learning rate: {model_details[task]['learning_rate']:.6f}")
    
    # After parameter summary, save hyperparameter summary
    logger.info("\nGenerating hyperparameter summary...")
    best_hyperparams = save_best_hyperparameters(tasks, script_dir)
    logger.info(f"Hyperparameter summary saved to: {os.path.join(script_dir, 'best_hyperparameters_summary.txt')}")
    
    # After hyperparameter summary, save ROC-AUC scores
    logger.info("\nGenerating ROC-AUC scores summary...")
    best_scores, avg_score = save_best_scores(tasks, script_dir)
    logger.info("ROC-AUC Scores Summary:")
    for task, score in best_scores.items():
        logger.info(f"{task}: {score:.4f}")
    logger.info(f"Average ROC-AUC Score: {avg_score:.4f}")
    logger.info(f"Scores saved to: {os.path.join(script_dir, 'best_roc_auc_scores.txt')}")

    logger.info(f"Average ROC-AUC Score: {avg_score:.4f}")
    logger.info(f"Scores saved to: {os.path.join(script_dir, 'best_roc_auc_scores.txt')}")
