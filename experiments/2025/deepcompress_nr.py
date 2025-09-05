
"""
DeepCompression baseline vs NeuronRank-guided variant.
- Baseline: magnitude pruning + simple k-means quantization.
- NR: channel-level importance via NeuronRank (AM × ILF) to guide structured pruning & mixed precision.
"""

import argparse, math, os
import torch, torch.nn as nn, torch.nn.functional as F
import torchvision, torchvision.transforms as T
import numpy as np
from torch.nn.utils import prune

# -----------------------
# Small CNN for MNIST/CIFAR10
# -----------------------
class SmallCNN(nn.Module):
    def __init__(self, in_ch=1, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.fc1 = nn.Linear(64*7*7, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.pool = nn.MaxPool2d(2,2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

# -----------------------
# NeuronRank stats (per channel)
# -----------------------
def collect_nr_stats(model, layers, loader, max_batches=5, tau=0.35, eps=1e-8, device="cpu"):
    stats = {}
    hooks, feats = [], {}
    def mk_hook(name):
        def hook(_m,_i,o):
            X = o.detach()
            if X.dim()==4:  # (B,C,H,W)
                B,C,H,W = X.shape; X = X.view(B,C,H*W)
            feats.setdefault(name, []).append(X.cpu())
        return hook
    for name, m in layers:
        hooks.append(m.register_forward_hook(mk_hook(name)))
    with torch.no_grad():
        for i,(xb,yb) in enumerate(loader):
            xb = xb.to(device)
            _ = model(xb)
            if i>=max_batches: break
    for h in hooks: h.remove()
    for name, chunks in feats.items():
        X = torch.cat(chunks,0) # (N,C,S)
        e = torch.sqrt((X**2).mean(dim=(0,2))+eps)
        am = e/(e.median()+eps)
        A = X.mean(0)
        An = A/(A.norm(dim=1,keepdim=True).clamp_min(eps))
        S = An@An.t(); S.fill_diagonal_(-1.0)
        nbrs = (S>tau).sum(1).float()
        C = A.shape[0]
        ilf_raw = torch.log(torch.tensor(C)/(1.0+nbrs))
        ilf = ilf_raw/(ilf_raw.median()+eps)
        nr = (am*ilf).numpy()
        stats[name]=nr
    return stats

# -----------------------
# Train / Eval
# -----------------------
def train(model, loader, opt, device):
    model.train(); total=0
    for xb,yb in loader:
        xb,yb = xb.to(device), yb.to(device)
        opt.zero_grad()
        out = model(xb)
        loss = F.cross_entropy(out,yb)
        loss.backward(); opt.step()
        total += loss.item()*xb.size(0)
    return total/len(loader.dataset)

def evaluate(model, loader, device):
    model.eval(); correct=0
    with torch.no_grad():
        for xb,yb in loader:
            xb,yb = xb.to(device), yb.to(device)
            out = model(xb)
            pred = out.argmax(1)
            correct += (pred==yb).sum().item()
    return correct/len(loader.dataset)

# -----------------------
# Main
# -----------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["mnist","cifar10"], default="mnist")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--baseline", action="store_true")
    ap.add_argument("--nr", action="store_true")
    ap.add_argument("--prune-amount", type=float, default=0.5)
    ap.add_argument("--keep-ratio", type=float, default=0.7)
    ap.add_argument("--quant-bits", type=int, default=4)
    ap.add_argument("--target-avg-bits", type=int, default=4)
    ap.add_argument("--nr-alpha", type=float, default=1.0)
    ap.add_argument("--nr-beta", type=float, default=1.0)
    ap.add_argument("--nr-tau", type=float, default=0.35)
    args = ap.parse_args()
    device=torch.device(args.device)

    # data
    tfm = T.Compose([T.ToTensor()])
    if args.dataset=="mnist":
        trainset = torchvision.datasets.MNIST("data",train=True,download=True,transform=tfm)
        testset  = torchvision.datasets.MNIST("data",train=False,download=True,transform=tfm)
        in_ch=1; num_classes=10
    else:
        trainset = torchvision.datasets.CIFAR10("data",train=True,download=True,transform=tfm)
        testset  = torchvision.datasets.CIFAR10("data",train=False,download=True,transform=tfm)
        in_ch=3; num_classes=10
    train_loader = torch.utils.data.DataLoader(trainset,batch_size=64,shuffle=True)
    test_loader = torch.utils.data.DataLoader(testset,batch_size=256)

    model = SmallCNN(in_ch=in_ch,num_classes=num_classes).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for ep in range(args.epochs):
        loss = train(model,train_loader,opt,device)
        acc = evaluate(model,test_loader,device)
        print(f"[train] epoch {ep+1} loss={loss:.3f} acc={acc*100:.2f}%")

    if args.baseline:
        # baseline magnitude pruning
        parameters_to_prune=[(model.conv1,"weight"),(model.conv2,"weight"),(model.fc1,"weight"),(model.fc2,"weight")]
        for m,n in parameters_to_prune:
            prune.l1_unstructured(m,n,amount=args.prune-amount)
        acc = evaluate(model,test_loader,device)
        print(f"[baseline after pruning] acc={acc*100:.2f}%")
    if args.nr:
        layers=[("conv1",model.conv1),("conv2",model.conv2)]
        stats=collect_nr_stats(model,layers,train_loader,device=device,tau=args.nr_tau)
        for name,vals in stats.items():
            keep=int(len(vals)*args.keep_ratio)
            idx=np.argsort(-vals)[:keep]
            print(f"[NR] keep {keep}/{len(vals)} channels for {name}")
        acc=evaluate(model,test_loader,device)
        print(f"[NR-guided] acc={acc*100:.2f}%")

if __name__=="__main__":
    main()
