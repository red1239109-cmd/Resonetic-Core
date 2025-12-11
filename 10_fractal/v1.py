# ==============================================================================
# File: resonetics_cifar10_rigorous.py
# Project: Resonetics Fractal (Scientific Validation)
# Version: 1.0 (The Verdict)
# Author: red1239109-cmd
# Copyright (c) 2025 red1239109-cmd
#
# License: AGPL-3.0
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License.
#
# Description:
#   A rigorous benchmark on CIFAR-10 to validate the "Rule of Three".
#   It conducts multi-trial training across different base unit sizes (N)
#   and performs a T-test to calculate statistical significance (p-value).
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
TRIALS = 3          # Number of trials per N to ensure statistical validity
EPOCHS = 10         # Epochs per trial (Increase to 50+ for paper publication)
HIDDEN_DIM = 288    # Common multiple of 3, 4, 6, 8, 9, 12 for fair comparison
BATCH_SIZE = 128

# Candidates to test (Focus on N=3 vs N=9)
CANDIDATES = [3, 4, 6, 9] 

# ==============================================================================
# 1. Fractal Architecture (Adapted for CIFAR-10)
# ==============================================================================
class FractalNetCIFAR(nn.Module):
    def __init__(self, base_unit, input_dim=3072, hidden_dim=288, output_dim=10):
        super().__init__()
        self.base_unit = base_unit
        self.num_blocks = hidden_dim // base_unit
        
        # Input Projection (Flattened Image -> Hidden Dim)
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Fractal Blocks (The Core Test Subject)
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(base_unit, base_unit),
                nn.Tanh(), # Tanh is used for organic, bounded activation
                nn.Linear(base_unit, base_unit)
            )
            for _ in range(self.num_blocks)
        ])
        
        # Resonance Matrix (Learnable connections between blocks)
        self.resonance = nn.Parameter(torch.randn(self.num_blocks, self.num_blocks) * 0.05)
        
        # Classifier Head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim)
        )

    def forward(self, x):
        # Flatten: [Batch, 3, 32, 32] -> [Batch, 3072]
        x = x.view(x.size(0), -1) 
        x = self.input_proj(x)
        
        # Split into N-sized chunks
        chunks = torch.chunk(x, self.num_blocks, dim=1)
        block_outputs = [block(chunk) for block, chunk in zip(self.blocks, chunks)]
        
        # Resonance (Cross-Pollination)
        stacked = torch.stack(block_outputs, dim=1)
        resonated = torch.einsum('bnf,ij->bif', stacked, self.resonance)
        
        # Reassemble
        features = resonated.reshape(x.size(0), -1)
        return self.classifier(features)

# ==============================================================================
# 2. Experiment Engine
# ==============================================================================
def load_cifar10():
    print("⏳ Loading CIFAR-10 Data...", end="")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Download and load data
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    print(" Done.")
    return trainloader, testloader

def train_and_evaluate(N, train_loader, test_loader, trial_id):
    """
    Trains a model with base unit N and returns test accuracy.
    """
    model = FractalNetCIFAR(base_unit=N, hidden_dim=HIDDEN_DIM).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # Training Loop
    model.train()
    for epoch in range(EPOCHS):
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
    # Evaluation Loop
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
# 3. Main Scientific Benchmark
# ==============================================================================
def run_rigorous_science():
    print(f"\n{'='*70}")
    print(f"🔬 RESONETICS RIGOROUS BENCHMARK (CIFAR-10)")
    print(f"{'='*70}")
    print(f"   Target: Scientifically validate if N=3 implies better stability.")
    print(f"   Conditions: Hidden Dim={HIDDEN_DIM}, Epochs={EPOCHS}, Trials={TRIALS}")
    print(f"   Candidates: N={CANDIDATES}")
    print(f"{'-'*70}")
    
    try:
        train_loader, test_loader = load_cifar10()
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return

    results = {N: [] for N in CANDIDATES}
    start_time = time.time()
    
    # --- Execution Phase ---
    for N in CANDIDATES:
        structure_name = "Triangle" if N==3 else ("Tesla" if N==9 else "Other")
        print(f"\n👉 Testing N={N} (Structure: {structure_name})")
        
        for t in range(TRIALS):
            acc = train_and_evaluate(N, train_loader, test_loader, t)
            results[N].append(acc)
        
        mean_acc = np.mean(results[N])
        std_acc = np.std(results[N])
        print(f"   => N={N} Result: {mean_acc:.2f}% ± {std_acc:.2f}")

    total_time = time.time() - start_time
    
    # --- Analysis Phase ---
    print(f"\n{'='*70}")
    print(f"📊 STATISTICAL ANALYSIS REPORT")
    print(f"{'='*70}")
    print(f"{'N':<5} | {'Mean Acc':<12} | {'Std Dev':<10} | {'Structure'}")
    print(f"{'-'*70}")
    
    # Sort results by accuracy
    sorted_results = sorted(results.items(), key=lambda item: np.mean(item[1]), reverse=True)
    
    for N, accs in sorted_results:
        structure_name = "Triangle" if N==3 else ("Tesla" if N==9 else "Polygon")
        print(f"{N:<5} | {np.mean(accs):<12.2f} | {np.std(accs):<10.2f} | {structure_name}")
        
    print(f"{'='*70}")
    
    # T-test: The decisive battle (N=3 vs N=9)
    n3_scores = results[3]
    n9_scores = results[9]
    
    t_stat, p_value = stats.ttest_ind(n3_scores, n9_scores, equal_var=False)
    
    print(f"🧪 Hypothesis Test: 'Is N=3 significantly better than N=9?'")
    print(f"   t-statistic: {t_stat:.4f}")
    print(f"   p-value    : {p_value:.6f}")
    
    if p_value < 0.05 and t_stat > 0:
        print("\n✅ VERDICT: Statistically Significant (p < 0.05)")
        print("   Scientific Proof: The 'Rule of Three' outperforms 'Rule of Nine'.")
    elif p_value < 0.05 and t_stat < 0:
        print("\n❌ VERDICT: Statistically Significant - BUT N=9 WON.")
    else:
        print("\n⚠️ VERDICT: Not Statistically Significant. More trials needed.")

    print(f"\nTotal Experiment Time: {total_time:.1f}s")

if __name__ == "__main__":
    run_rigorous_science()
