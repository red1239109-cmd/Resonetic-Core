## License

**Open Source**: AGPL-3.0 License  
→ You can use, modify, and distribute this code for free,  
   but **any derivative work or software that uses Resonetic-Core must also be open-sourced under AGPL-3.0**.

**Commercial / Closed-Source Use**  
→ Want to use Resonetic-Core in a proprietary product, service, or internal tool without disclosing your source code?  
   You need a **Commercial License**.

   Contact: red1239109@gmail.com  
   Price: negotiable (starting from $10,000 USD per year)

> “Free as in freedom for the community.  
>  Paid as in beer for companies.”

Dual-license model (AGPL-3.0 + Commercial)  
© 2025 red1239109-cmd – All rights reserved for commercial use.


# 01_resonant_fluid/resonant_fluid.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

class FluidPhaseEncoder(nn.Module):
    def __init__(self, in_dim=2, hidden_dim=64):
        super().__init__()
        self.embed = nn.Conv2d(in_dim, hidden_dim, kernel_size=1)
        self.phase_transform = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim//2, kernel_size=3, padding=1),
            nn.InstanceNorm2d(hidden_dim//2),
            nn.ReLU(),
            nn.Conv2d(hidden_dim//2, hidden_dim, kernel_size=3, padding=1)
        )
    def forward(self, velocity):
        if velocity.shape[1] != 2:
            velocity = velocity.permute(0, 3, 1, 2)
        encoded = self.embed(velocity)
        return self.phase_transform(encoded)
