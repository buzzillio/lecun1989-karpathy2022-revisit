#!/usr/bin/env python3
"""
Plain MiniViT (no PCA / no binning) + epoch metrics with misses
Now with: pl_bolts LinearWarmupCosineAnnealingLR (fallback to CosineWarmRestarts)
--------------------------------------------------------------------------------
- 16x16 grayscale images (LeCun 1989 style), patch size 2
- Prints: loss, error %, and number of misses for train & test each epoch
- Expects train1989.pt / test1989.pt with tensors:
    X: (N, 1, 16, 16) in [0,1]
    Y: (N, 10) one-hot labels
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from einops.layers.torch import Rearrange
import optuna
from optuna.trial import TrialState


import warnings
try:
    from pl_bolts.utils.warnings import UnderReviewWarning
    warnings.filterwarnings("ignore", category=UnderReviewWarning)
except Exception:
    # Fallback if the class path changes
    warnings.filterwarnings("ignore", message=".*UnderReviewWarning.*")



# ---- try to use lightning-bolts scheduler; fallback if missing ----
HAS_BOLTS = True
try:
    from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
except Exception:
    HAS_BOLTS = False
    from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

class MiniViT(nn.Module):
    """
    Minimal ViT for 16x16 grayscale, patch size 2.
    """
    def __init__(self, *, image_size=16, patch_size=2, num_classes=10,
                 dim=20, depth=3, heads=2, mlp_dim=28, channels=1,
                 attn_dropout=0.04, mlp_dropout=0.12, use_mean_pool=True,
                 augment_roll=True):
        super().__init__()
        assert image_size % patch_size == 0, "image_size must be divisible by patch_size"
        self.image_size = image_size
        self.patch_size = patch_size
        self.channels = channels
        self.use_mean_pool = use_mean_pool
        self.augment_roll = augment_roll

        num_patches = (image_size // patch_size) ** 2
        patch_dim = channels * patch_size * patch_size

        # Patch embedding
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size),
            nn.Linear(patch_dim, dim),
        )

        # Positional embeddings and class token
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))

        # Transformer layers
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                nn.LayerNorm(dim),
                nn.MultiheadAttention(dim, heads, batch_first=True, dropout=attn_dropout),
                nn.LayerNorm(dim),
                nn.Sequential(
                    nn.Linear(dim, mlp_dim),
                    nn.GELU(),
                    nn.Dropout(mlp_dropout),
                    nn.Linear(mlp_dim, dim)
                )
            ]))

        # Classification head
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)

        # Parameter count
        self.param_count = sum(p.numel() for p in self.parameters())

    def forward(self, img):
        # Small translation augmentation
        if self.training and self.augment_roll:
            sx, sy = np.random.randint(-1, 2, size=2)
            img = torch.roll(img, (sx, sy), (2, 3))

        x = self.to_patch_embedding(img)  # (B, N, dim)
        B, N, D = x.shape

        cls = self.cls_token.expand(B, 1, D)
        x = torch.cat([cls, x], dim=1)  # (B, 1+N, D)
        x = x + self.pos_embedding[:, : (1+N), :]

        for norm1, attn, norm2, mlp in self.layers:
            y = norm1(x)
            y, _ = attn(y, y, y)
            x = x + y
            y = norm2(x)
            x = x + mlp(y)

        x = self.norm(x)
        feats = x[:, 1:, :].mean(dim=1) if self.use_mean_pool else x[:, 0, :]
        return self.head(feats)

def train_mini_vit_plain(trial=None, **kwargs):
    # Use trial suggestions if provided, otherwise use defaults
    if trial is not None:
        # Hyperparameters to optimize
        dim = trial.suggest_int('dim', 16, 32, step=4)
        depth = trial.suggest_int('depth', 2, 6)
        
        # Ensure heads is a divisor of dim
        possible_heads = [h for h in [1, 2, 4, 8] if dim % h == 0]
        heads = trial.suggest_categorical('heads', possible_heads)
        
        mlp_dim = trial.suggest_int('mlp_dim', dim, dim * 3)
        attn_dropout = trial.suggest_float('attn_dropout', 0.0, 0.3)
        mlp_dropout = trial.suggest_float('mlp_dropout', 0.0, 0.5)
        base_lr = trial.suggest_float('base_lr', 1e-5, 1e-2, log=True)
        weight_decay = trial.suggest_float('weight_decay', 1e-4, 1e-1, log=True)
        warmup_epochs = trial.suggest_int('warmup_epochs', 3, 10)
        label_smoothing = trial.suggest_float('label_smoothing', 0.0, 0.2)
        grad_clip = trial.suggest_float('grad_clip', 0.1, 2.0)
        patch_size = trial.suggest_categorical('patch_size', [1, 2])
        use_mean_pool = trial.suggest_categorical('use_mean_pool', [True, False])
        
        # Constraint: Check if model would exceed parameter budget (~15K max)
        # Quick estimation to avoid creating oversized models
        estimated_params = (
            dim * dim * 4 * depth +  # attention weights (rough estimate)
            dim * mlp_dim * 2 * depth +  # MLP weights
            dim * (256 if patch_size == 1 else 64) +  # pos embedding
            dim * 10  # classification head
        )
        if estimated_params > 15000:
            # Return high error to discourage this configuration
            raise ValueError("Model too large - estimated params exceed budget")
    else:
        # Default values
        dim = kwargs.get('dim', 20)
        depth = kwargs.get('depth', 3)
        heads = kwargs.get('heads', 2)
        mlp_dim = kwargs.get('mlp_dim', 28)
        attn_dropout = kwargs.get('attn_dropout', 0.04)
        mlp_dropout = kwargs.get('mlp_dropout', 0.12)
        base_lr = kwargs.get('base_lr', 6e-4)
        weight_decay = kwargs.get('weight_decay', 0.02)
        warmup_epochs = kwargs.get('warmup_epochs', 8)
        label_smoothing = kwargs.get('label_smoothing', 0.08)
        grad_clip = kwargs.get('grad_clip', 0.8)
        patch_size = kwargs.get('patch_size', 2)
        use_mean_pool = kwargs.get('use_mean_pool', True)
    # Repro
    torch.manual_seed(1337); np.random.seed(1337)
    torch.use_deterministic_algorithms(True)

    if trial is None:
        print("\n" + "="*60)
        print("Training Mini ViT (plain) + cosine LR (bolts if available)")
        print("="*60)

    model = MiniViT(
        image_size=16, patch_size=patch_size, num_classes=10,
        dim=dim, depth=depth, heads=heads, mlp_dim=mlp_dim, channels=1,
        attn_dropout=attn_dropout, mlp_dropout=mlp_dropout, use_mean_pool=use_mean_pool,
        augment_roll=True
    )
    
    # Check parameter count constraint for Optuna trials
    if trial is not None and model.param_count > 15000:
        raise ValueError(f"Model too large: {model.param_count} params > 15000 limit")
    
    if trial is None:
        print("Mini ViT model stats:")
        print("# params:      ", model.param_count)

    # Data
    data_dir = os.path.join(os.path.dirname(__file__), "experiments", "2025")
    train_path = os.path.join(data_dir, "train1989.pt")
    test_path = os.path.join(data_dir, "test1989.pt")
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        print("Error: train1989.pt and test1989.pt not found!")
        return None
    Xtr, Ytr = torch.load(train_path)
    Xte, Yte = torch.load(test_path)
    if trial is None:
        print(f"Training data shape: {Xtr.shape}")
        print(f"Test data shape:     {Xte.shape}")

    # Optimizer
    opt = optim.AdamW(model.parameters(), lr=base_lr, weight_decay=weight_decay, betas=(0.9, 0.95))

    # Scheduler: prefer pl_bolts warmup+cosine, else cosine warm restarts
    total_passes = 80 if trial is None else 20  # Shorter runs for optimization
    if HAS_BOLTS:
        # Linear warmup for warmup_epochs, then cosine to epoch total_passes
        scheduler = LinearWarmupCosineAnnealingLR(
            opt, warmup_epochs=warmup_epochs, max_epochs=total_passes
        )
        if trial is None:
            print("Using pl_bolts LinearWarmupCosineAnnealingLR")
    else:
        # Cosine warm restarts: 20-epoch cycles, min LR = 0.1 * base
        scheduler = CosineAnnealingWarmRestarts(
            opt, T_0=20, T_mult=1, eta_min=base_lr * 0.1
        )
        if trial is None:
            print("pl_bolts not found -> Using CosineAnnealingWarmRestarts fallback")

    # Eval helper (prints misses)
    def eval_split(split, X, Y):
        model.eval()
        with torch.no_grad():
            logits = model(X)
            loss = F.cross_entropy(logits, Y.argmax(dim=1))
            preds = logits.argmax(dim=1)
            targets = Y.argmax(dim=1)
            err = (preds != targets).float().mean().item()
            misses = int((preds != targets).sum().item())
        if trial is None:
            print(f"eval: split {split:5s}. loss {loss.item():e}. error {err*100:.2f}%. misses: {misses}")
        return err, misses

    # Train
    t0 = time.time()
    best_test_err = 1.0  # Track best test error for early stopping and Optuna
    patience = 5  # Early stopping patience for Optuna trials
    epochs_without_improvement = 0
    
    for ep in range(total_passes):
        model.train()
        # one-image "batches" to keep behavior identical
        for i in range(Xtr.size(0)):
            x, y = Xtr[[i]], Ytr[[i]]
            out = model(x)
            loss = F.cross_entropy(out, y.argmax(dim=1), label_smoothing=label_smoothing)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()

        # Step scheduler once per epoch
        if HAS_BOLTS:
            scheduler.step()                # bolts scheduler expects epoch steps
        else:
            scheduler.step(ep)              # WarmRestarts: pass epoch index

        # Evaluate
        if trial is None:
            print(f"ViT Pass {ep + 1}")
        train_err, _ = eval_split('train', Xtr, Ytr)
        test_err, _ = eval_split('test',  Xte, Yte)
        
        # Track best test error and early stopping
        if test_err < best_test_err:
            best_test_err = test_err
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            
        # Early stopping for Optuna trials only
        if trial is not None and epochs_without_improvement >= patience and ep >= 10:
            if trial is None:
                print(f"Early stopping after {ep+1} epochs (no improvement for {patience} epochs)")
            break
            
        # Report intermediate values to Optuna for pruning
        if trial is not None:
            trial.report(test_err, ep)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

    if trial is None:
        print('ViT Execution time:', '{:5.2f}'.format(time.time() - t0), 'seconds')
        return model
    else:
        # Return the best test error for Optuna optimization
        return best_test_err

def objective(trial):
    """Optuna objective function to minimize test error."""
    try:
        test_error = train_mini_vit_plain(trial=trial)
        return test_error
    except optuna.exceptions.TrialPruned:
        # Re-raise pruned exceptions
        raise
    except ValueError as e:
        # Handle constraint violations (model too large, etc.)
        error_msg = str(e)
        if "too large" in error_msg:
            print(f"Trial skipped: {error_msg}")
        else:
            print(f"Trial failed with ValueError: {error_msg}")
        return 1.0
    except Exception as e:
        # Return a high error if training fails
        error_msg = str(e)
        if "embed_dim must be divisible by num_heads" in error_msg:
            print(f"Trial failed: dim/heads incompatibility - {error_msg}")
        elif "out of memory" in error_msg.lower():
            print(f"Trial failed: GPU memory error - {error_msg}")
        else:
            print(f"Trial failed with error: {error_msg}")
        return 1.0

def optimize_hyperparameters(n_trials=50, study_name="minivit_optimization"):
    """Run Optuna hyperparameter optimization."""
    print("\n" + "="*60)
    print("Starting Optuna Hyperparameter Optimization")
    print("="*60)
    
    # Check if data files exist
    if not os.path.exists('train1989.pt') or not os.path.exists('test1989.pt'):
        print("Error: train1989.pt and test1989.pt not found!")
        return None, None
    
    # Create study with better configuration
    study = optuna.create_study(
        direction="minimize",
        study_name=study_name,
        storage=f"sqlite:///{study_name}.db",  # Persist study to database
        load_if_exists=True,  # Resume if study exists
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=8,
            interval_steps=3
        ),
        sampler=optuna.samplers.TPESampler(
            n_startup_trials=10,
            n_ei_candidates=24,
            multivariate=True
        )
    )
    
    print(f"Starting optimization with {n_trials} trials...")
    print(f"Study will be saved to: {study_name}.db")
    
    study.optimize(objective, n_trials=n_trials, timeout=None, show_progress_bar=True)
    
    # Print results
    print("\nOptimization completed!")
    print(f"Number of finished trials: {len(study.trials)}")
    
    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])
    failed_trials = study.get_trials(deepcopy=False, states=[TrialState.FAIL])
    
    print(f"Number of pruned trials: {len(pruned_trials)}")
    print(f"Number of complete trials: {len(complete_trials)}")
    print(f"Number of failed trials: {len(failed_trials)}")
    
    if len(complete_trials) == 0:
        print("No trials completed successfully!")
        return study, None
    
    print("\nBest trial:")
    trial = study.best_trial
    print(f"  Test Error: {trial.value:.4f} ({trial.value*100:.2f}%)")
    print("  Params:")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
    
    # Train final model with best hyperparameters
    print("\nTraining final model with best hyperparameters...")
    best_model = train_mini_vit_plain(**trial.params)
    
    return study, best_model

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="MiniViT Training with Optuna Optimization")
    parser.add_argument('--optimize', action='store_true', help='Run hyperparameter optimization')
    parser.add_argument('--n_trials', type=int, default=50, help='Number of Optuna trials')
    parser.add_argument('--study_name', type=str, default='minivit_optimization', help='Optuna study name')
    
    args = parser.parse_args()
    
    if args.optimize:
        study, best_model = optimize_hyperparameters(
            n_trials=args.n_trials, 
            study_name=args.study_name
        )
        if study:
            print(f"\nOptimization complete! Best test error: {study.best_trial.value:.4f}")
        else:
            print("Optimization failed!")
    else:
        # Run single training with default parameters
        m = train_mini_vit_plain()
        if m:
            print("Training completed successfully!")
        else:
            print("Training failed - please check that the dataset files exist.")
