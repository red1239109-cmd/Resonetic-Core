# ==============================================================================
# File: resonetics_cifar10_rigorous_v1_5.py
# Project: Resonetics Fractal (Scientific Validation)
# Version: 1.5 (Enhanced Science Edition)
# Author: red1239109-cmd
# License: AGPL-3.0
#
# Changes in v1.5:
#   - Added Data Augmentation (Crop, Flip) for generalization
#   - Added CosineAnnealing Learning Rate Scheduler
#   - Fixed Einstein Summation dimension bug
#   - Increased Epochs to 20 for better convergence
# ==============================================================================

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
from scipy import stats
import time
import sys

# Reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TRIALS = 3           # Keeping 3 for speed (Increase to 10 for paper)
EPOCHS = 20          # Increased from 10 to 20
HIDDEN_DIM = 288     # Common multiple of 3, 4, 6, 9
BATCH_SIZE = 128
CANDIDATES = [3, 4, 6, 9]

# ==============================================================================
# 1. Fractal Architecture (Bug Fixed)
# ==============================================================================
class FractalNetCIFAR(nn.Module):
    def __init__(self, base_unit, input_dim=3072, hidden_dim=288, output_dim=10):
        super().__init__()
        self.base_unit = base_unit
        self.num_blocks = hidden_dim // base_unit
        
        # Input Projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Fractal Blocks
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(base_unit, base_unit),
                nn.Tanh(), # Tanh for organic activation
                nn.Linear(base_unit, base_unit)
            )
            for _ in range(self.num_blocks)
        ])
        
        # Resonance Matrix
        self.resonance = nn.Parameter(torch.randn(self.num_blocks, self.num_blocks) * 0.05)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1) 
        x = self.input_proj(x)
        
        chunks = torch.chunk(x, self.num_blocks, dim=1)
        block_outputs = [block(chunk) for block, chunk in zip(self.blocks, chunks)]
        
        stacked = torch.stack(block_outputs, dim=1) # [Batch, N, Base]
        
        # [FIXED] Correct Dimension Matching: n(in_blocks) -> m(out_blocks)
        resonated = torch.einsum('bnf,nm->bmf', stacked, self.resonance)
        
        features = resonated.reshape(x.size(0), -1)
        return self.classifier(features)

# ==============================================================================
# 2. Experiment Engine (Enhanced with Augmentation)
# ==============================================================================
def load_cifar10():
    print("⏳ Loading CIFAR-10 Data with Augmentation...", end="")
    
    # [NEW] Data Augmentation: Crucial for generalization
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    
    # Increase num_workers if you have good CPU
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    print(" Done.")
    return trainloader, testloader

def train_and_evaluate(N, train_loader, test_loader, trial_id):
    model = FractalNetCIFAR(base_unit=N, hidden_dim=HIDDEN_DIM).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # [NEW] Scheduler: Helps convergence
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    for epoch in range(EPOCHS):
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        scheduler.step() # Update LR
            
    # Evaluation
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    accuracy = 100.0 * correct / total
    print(f"   [Trial {trial_id+1}/{TRIALS}] N={N}: Accuracy = {accuracy:.2f}%")
    return accuracy

# ==============================================================================
# 3. Scientific Benchmark
# ==============================================================================
def run_rigorous_science():
    print(f"\n{'='*70}")
    print(f"🔬 RESONETICS RIGOROUS BENCHMARK v1.5 (CIFAR-10)")
    print(f"{'='*70}")
    print(f"   Target: Scientifically validate 'Rule of 3' with Augmentation.")
    print(f"   Conditions: Hidden Dim={HIDDEN_DIM}, Epochs={EPOCHS}, Trials={TRIALS}")
    print(f"   Device: {DEVICE}")
    print(f"{'-'*70}")
    
    try:
        train_loader, test_loader = load_cifar10()
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return

    results = {N: [] for N in CANDIDATES}
    start_time = time.time()
    
    for N in CANDIDATES:
        print(f"\n👉 Testing N={N}")
        for t in range(TRIALS):
            acc = train_and_evaluate(N, train_loader, test_loader, t)
            results[N].append(acc)
        
        mean_acc = np.mean(results[N])
        std_acc = np.std(results[N])
        print(f"   => N={N} Result: {mean_acc:.2f}% ± {std_acc:.2f}")

    total_time = time.time() - start_time
    
    # --- Analysis Phase ---
    print(f"\n{'='*70}")
    print(f"📊 STATISTICAL ANALYSIS REPORT v1.5")
    print(f"{'='*70}")
    print(f"{'N':<5} | {'Mean Acc':<12} | {'Std Dev':<10} | {'Structure'}")
    print(f"{'-'*70}")
    
    sorted_results = sorted(results.items(), key=lambda item: np.mean(item[1]), reverse=True)
    
    for N, accs in sorted_results:
        structure_name = "Triangle" if N==3 else ("Tesla" if N==9 else "Polygon")
        print(f"{N:<5} | {np.mean(accs):<12.2f} | {np.std(accs):<10.2f} | {structure_name}")
        
    print(f"{'='*70}")
    
    # T-test (N=3 vs N=9)
    n3_scores = results[3]
    n9_scores = results[9]
    t_stat, p_value = stats.ttest_ind(n3_scores, n9_scores, equal_var=False)
    
    print(f"🧪 Hypothesis Test: 'Is N=3 better than N=9?'")
    print(f"   p-value: {p_value:.6f}")
    
    if p_value < 0.05 and t_stat > 0:
        print("\n✅ VERDICT: Statistically Significant (p < 0.05)")
    else:
        print("\n⚠️ VERDICT: Not Significant or Inconclusive.")

    print(f"\nTotal Time: {total_time:.1f}s")

if __name__ == "__main__":
    run_rigorous_science()
