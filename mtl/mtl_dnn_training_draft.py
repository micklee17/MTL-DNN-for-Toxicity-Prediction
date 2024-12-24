import deepchem as dc
import pandas as pd
from hyperopt import hp, fmin, tpe, Trials
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset
import tempfile
import logging
import json
import os
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

# Simplified model architecture
class DNNMultiTaskClassifier(nn.Module):
    """Multi-task DNN classifier with shared layers and task-specific heads
    
    Architecture Overview:
    1. Shared Feature Processing Layers:
       - Common layers that learn shared representations
       - Multiple FC layers with ReLU activation and dropout
       - Captures common patterns across all tasks
    
    2. Task-Specific Networks:
       - Separate network branches for each task
       - Each branch has its own hidden layers
       - Allows specialization for each task's unique patterns
       - Final output layer for task-specific prediction
    
    Parameters:
        n_features (int): Number of input features (2048 for ECFP)
        layer_sizes (list): Sizes of shared processing layers
        dropouts (list): Dropout rates for shared layers
        hidden_layer_sizes (list): Sizes of task-specific hidden layers
        hidden_layer_dropouts (list): Dropout rates for task-specific layers
        n_tasks (int): Number of prediction tasks
    """
    def __init__(self, n_features, layer_sizes, dropouts, hidden_layer_sizes, hidden_layer_dropouts, n_tasks):
        super(DNNMultiTaskClassifier, self).__init__()
        
        # ===== Shared Feature Processing Layers =====
        self.shared_layers = nn.ModuleList()
        prev_size = n_features
        
        # Build shared layers
        for size, dropout in zip(layer_sizes, dropouts):
            layer_block = nn.Sequential(
                nn.Linear(prev_size, size),
                nn.BatchNorm1d(size),  # Added batch normalization
                nn.ReLU(),
                nn.Dropout(dropout)
            )
            self.shared_layers.append(layer_block)
            prev_size = size
        
        # ===== Task-Specific Networks =====
        self.task_networks = nn.ModuleList()
        final_shared_size = prev_size
        
        # Create separate network for each task
        for _ in range(n_tasks):
            layers = []
            task_size = final_shared_size
            
            # Hidden layers for this task
            for size in hidden_layer_sizes:
                task_block = nn.Sequential(
                    nn.Linear(task_size, size),
                    nn.BatchNorm1d(size),  # Added batch normalization
                    nn.ReLU(),
                    nn.Dropout(hidden_layer_dropouts[0])
                )
                layers.append(task_block)
                task_size = size
            
            # Output layer for this task
            layers.append(nn.Linear(task_size, 1))
            
            # Combine all layers for this task
            self.task_networks.append(nn.Sequential(*layers))

    def forward(self, x):
        # 1. Flatten input if needed
        x = x.view(x.size(0), -1)
        
        # 2. Shared feature processing
        for shared_layer in self.shared_layers:
            x = shared_layer(x)
        
        # 3. Task-specific processing
        task_outputs = []
        for task_net in self.task_networks:
            task_output = task_net(x)
            task_outputs.append(task_output)
        
        return task_outputs

# Add after the DNNMultiTaskClassifier class definition
def count_parameters(model):
    """Count trainable parameters in the model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

num_epochs = 50  # Increase the number of epochs for better training

# Function to minimize
metric = dc.metrics.Metric(dc.metrics.roc_auc_score, mode='classification')

def evaluate_model(model, data_loader, device):
    model.eval()
    y_preds = [[] for _ in range(len(tasks))]
    y_trues = [[] for _ in range(len(tasks))]
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            for i in range(len(tasks)):
                y_preds[i].append(outputs[i].cpu())
                y_trues[i].append(targets[:, i].cpu())
    y_preds = [torch.cat(pred, dim=0).numpy() for pred in y_preds]
    y_trues = [torch.cat(true, dim=0).numpy() for true in y_trues]
    return [dc.metrics.roc_auc_score(y_true, y_pred) for y_true, y_pred in zip(y_trues, y_preds)]

# Simplified training function
def log_model_architecture(model):
    """Log detailed model architecture information"""
    logger.info("\n=== Model Architecture ===")
    logger.info(f"Total layers: {len(list(model.modules()))}")
    logger.info("\nShared Layers:")
    for i, layer in enumerate(model.shared_layers):
        logger.info(f"Layer {i + 1}: {layer}")
    
    logger.info("\nTask-Specific Networks:")
    for i, task_net in enumerate(model.task_networks):
        logger.info(f"\nTask {i + 1} ({tasks[i]}):")
        for j, layer in enumerate(task_net):
            logger.info(f"Layer {j + 1}: {layer}")

class EarlyStopping:
    def __init__(self, patience=7, min_delta=0, verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')

    def __call__(self, val_loss):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.verbose:
                logger.info(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

def fm(args):
    save_dir = os.path.dirname(os.path.abspath(__file__))
    layer_sizes = args['layer_sizes']
    dropouts = create_dropout_list(layer_sizes, args['dropout_rate'])
    hidden_layer_sizes = args['hidden_layers']
    hidden_layer_dropouts = create_dropout_list(hidden_layer_sizes, args['hidden_layer_dropout_rate'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create model
    model = DNNMultiTaskClassifier(n_features=2048, layer_sizes=layer_sizes, dropouts=dropouts, 
                                 hidden_layer_sizes=hidden_layer_sizes, hidden_layer_dropouts=hidden_layer_dropouts, 
                                 n_tasks=len(tasks)).to(device)
    
    # Log model architecture
    log_model_architecture(model)
    total_params = count_parameters(model)
    logger.info(f"\nTotal trainable parameters: {total_params:,}")
    logger.info(f"Device being used: {device}")
    
    optimizer = optim.Adam(model.parameters(), lr=args['learning_rate'])
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    early_stopping = EarlyStopping(patience=10, verbose=True)
    loss_fn = nn.BCEWithLogitsLoss()
    
    # Prepare DataLoader
    train_dataset_targets = [torch.Tensor(train_dataset.y[:, i]).unsqueeze(1) for i in range(len(tasks))]
    train_targets = torch.cat(train_dataset_targets, dim=1)
    test_dataset_targets = [torch.Tensor(test_dataset.y[:, i]).unsqueeze(1) for i in range(len(tasks))]
    test_targets = torch.cat(test_dataset_targets, dim=1)
    
    # Prepare DataLoader with drop_last=True
    train_loader = DataLoader(
        TensorDataset(
            torch.Tensor(train_dataset.X), 
            train_targets
        ), 
        batch_size=max(int(args['batch_size']), 2),  # Ensure minimum batch size of 2
        shuffle=True,
        drop_last=True  # Drop last batch if incomplete
    )
    
    test_loader = DataLoader(
        TensorDataset(
            torch.Tensor(test_dataset.X), 
            test_targets
        ), 
        batch_size=max(int(args['batch_size']), 2),  # Ensure minimum batch size of 2
        shuffle=False,
        drop_last=False  # Keep all samples for testing
    )
    
    train_losses = []
    test_roc_auc_scores = []
    best_score = float('-inf')
    
    logger.info(f"Training with hyperparameters: {args}")
    logger.info(f"Starting training with {num_epochs} epochs")
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        task_losses = [0] * len(tasks)
        batch_count = 0
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            
            # Calculate individual task losses
            losses = []
            for i, (out, target) in enumerate(zip(outputs, targets.t())):
                task_loss = loss_fn(out.view(-1), target)
                task_losses[i] += task_loss.item()
                losses.append(task_loss)
            
            loss = sum(losses) / len(losses)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            batch_count += 1
            
            # Log batch progress every 10 batches
            if (batch_idx + 1) % 10 == 0:
                logger.info(f"Epoch {epoch+1}/{num_epochs} - Batch {batch_idx+1}/{len(train_loader)} - "
                          f"Batch Loss: {loss.item():.4f} - "
                          f"LR: {optimizer.param_groups[0]['lr']:.6f}")

        # Calculate average losses
        epoch_loss /= batch_count
        task_losses = [t_loss / batch_count for t_loss in task_losses]
        
        # Evaluate model
        test_scores = evaluate_model(model, test_loader, device)
        test_roc_auc_scores.append(test_scores)
        
        # Log detailed epoch results
        logger.info(f"\n=== Epoch {epoch+1}/{num_epochs} Summary ===")
        logger.info(f"Average Loss: {epoch_loss:.4f}")
        logger.info("\nPer-Task Losses:")
        for task_idx, task_loss in enumerate(task_losses):
            logger.info(f"{tasks[task_idx]}: {task_loss:.4f}")
        logger.info("\nPer-Task ROC-AUC Scores:")
        for task_idx, score in enumerate(test_scores):
            logger.info(f"{tasks[task_idx]}: {score:.4f}")
        logger.info(f"Average ROC-AUC: {sum(test_scores) / len(test_scores):.4f}")
        
        # Calculate validation loss for scheduling
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                val_losses = []
                for i, (out, target) in enumerate(zip(outputs, targets.t())):
                    val_losses.append(loss_fn(out.view(-1), target))
                val_loss += sum(val_losses) / len(val_losses)
        val_loss /= len(test_loader)

        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Early stopping
        early_stopping(val_loss)
        if early_stopping.early_stop:
            logger.info("Early stopping triggered")
            break

        # Save best model
        avg_test_score = sum(test_scores) / len(test_scores)
        if avg_test_score > best_score:
            best_score = avg_test_score
            torch.save(model.state_dict(), os.path.join(save_dir, "best_dnn_multitask_model.pth"))

    # Save the architecture and hyperparameters
    architecture = {
        'n_features': 2048,
        'layer_sizes': layer_sizes,
        'dropouts': dropouts,
        'hidden_layers': hidden_layer_sizes,
        'hidden_layer_dropouts': hidden_layer_dropouts,
        'batch_size': args['batch_size'],
        'learning_rate': args['learning_rate']
    }
    with open(os.path.join(save_dir, "dnn_multitask_model_architecture.json"), "w") as f:
        json.dump(architecture, f)
    logger.info(f"Model architecture saved to dnn_multitask_model_architecture.json")

    # Save the training losses and test ROC AUC scores
    with open(os.path.join(save_dir, "train_losses.json"), "w") as f:
        json.dump(train_losses, f)
    logger.info(f"Training losses saved to train_losses.json")

    with open(os.path.join(save_dir, "test_roc_auc_scores.json"), "w") as f:
        json.dump(test_roc_auc_scores, f)
    logger.info(f"Test ROC AUC scores saved to test_roc_auc_scores.json")

    # Return the negative mean of the best ROC-AUC score
    return -1 * max([sum(scores) / len(scores) for scores in test_roc_auc_scores])

if __name__ == "__main__":
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

    arr = [1024,2048]
    layer_size_comb = create_subarrays(arr)
    logger.info(f"Layer size combinations: {layer_size_comb}")
    
    # Improved hyperparameter search space
    search_space = {
        'layer_sizes': hp.choice('layer_sizes', layer_size_comb),
        'dropout_rate': hp.uniform('dropout_rate', 0.1, 0.5),  # Adjusted range
        'learning_rate': hp.loguniform('learning_rate', -4, -2),  # Adjusted range
        'batch_size': hp.choice('batch_size', [32, 64, 128]),  # Removed smaller batch sizes
        'hidden_layers': hp.choice('hidden_layers', [
            [1024, 512, 256],
            [2048, 1024, 512],
            [512, 256, 128]
        ]),
        'hidden_layer_dropout_rate': hp.uniform('hidden_layer_dropout_rate', 0.1, 0.5)  # Adjusted range
    }
    
    trials = Trials()
    best = fmin(fm, space=search_space, algo=tpe.suggest, max_evals=20, trials=trials)
    logger.info(f"Best hyperparameters found: {best}")
    
    # Load the best hyperparameters
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "dnn_multitask_model_architecture.json"), "r") as f:
        best_architecture = json.load(f)
    
    best_layer_sizes = best_architecture['layer_sizes']
    best_dropout_rate = best_architecture['dropouts'][0]
    best_hidden_layers = best_architecture['hidden_layers']
    best_hidden_layer_dropout_rate = best_architecture['hidden_layer_dropouts'][0]
    best_batch_size = best_architecture['batch_size']
    best_learning_rate = best_architecture['learning_rate']
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    best_model = DNNMultiTaskClassifier(
        n_features=2048,
        layer_sizes=best_layer_sizes,
        dropouts=create_dropout_list(best_layer_sizes, best_dropout_rate),
        hidden_layer_sizes=best_hidden_layers,
        hidden_layer_dropouts=create_dropout_list(best_hidden_layers, best_hidden_layer_dropout_rate),
        n_tasks=len(tasks)
    ).to(device)
    
    # Add parameter counting here
    total_params = count_parameters(best_model)
    logger.info(f"Best model trainable parameters: {total_params:,}")
    # Add explicit print statement
    print(f"\nTotal trainable parameters in best model: {total_params:,}")
    
    # Save parameters count to file
    params_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_parameters.txt")
    with open(params_file, 'w') as f:
        f.write(f"Total trainable parameters: {total_params:,}\n")
        f.write(f"Best layer sizes: {best_layer_sizes}\n")
        f.write(f"Best dropout rate: {best_dropout_rate:.4f}\n")
        f.write(f"Best hidden layers: {best_hidden_layers}\n")
        f.write(f"Best hidden layer dropout rate: {best_hidden_layer_dropout_rate:.4f}\n")
        f.write(f"Best batch size: {best_batch_size}\n")
        f.write(f"Best learning rate: {best_learning_rate:.6f}\n")
    logger.info(f"Model parameters and hyperparameters saved to {params_file}")
    
    # Ensure the model architecture matches the saved state
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "best_dnn_multitask_model.pth")
    if os.path.exists(model_path):
        best_model.load_state_dict(torch.load(model_path))
        best_model.eval()
    
        test_loader = DataLoader(TensorDataset(torch.Tensor(test_dataset.X), torch.Tensor(test_dataset.y)), batch_size=best_batch_size, shuffle=False)
        test_scores = evaluate_model(best_model, test_loader, device)
        logger.info(f"ROC-AUC score for the best hyperparameters: {test_scores}")
        
        # Save ROC-AUC scores to file
        roc_auc_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "best_model_roc_auc_scores.txt")
        with open(roc_auc_file, 'w') as f:
            f.write("=== Best Model ROC-AUC Scores ===\n\n")
            for task_name, score in zip(tasks, test_scores):
                f.write(f"{task_name}: {score:.4f}\n")
            f.write(f"\nAverage ROC-AUC: {sum(test_scores)/len(test_scores):.4f}")
        logger.info(f"ROC-AUC scores saved to {roc_auc_file}")
    else:
        logger.error(f"Model file {model_path} not found.")
        print(f"Model file {model_path} not found.")
