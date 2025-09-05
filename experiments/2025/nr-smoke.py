# nr-smoke-plus.py
"""
NeuronRank regularization stress test on a tiny Transformer.
Adds:
  - Redundancy injection (duplicate features) -> ILF should help.
  - Label noise -> regularization has something to fight.
  - λ warm-up and EMA smoothing -> stabler signal.
Run examples:
  python3 nr-smoke-plus.py --epochs 6 --steps 100 --lambda-nr 3e-6 --dup-factor 2 --label-noise 0.1
  python3 nr-smoke-plus.py --epochs 6 --steps 100 --lambda-nr 0.0  --dup-factor 2 --label-noise 0.1  # baseline
"""

import argparse
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# ---------------------------
# NR Regularizer (with EMA)
# ---------------------------
class NRRegularizer:
    def __init__(self, alpha=1.0, beta=1.0, tau=0.45, eps=1e-8, enabled=True, ema=0.9):
        self.alpha = float(alpha)
        self.beta  = float(beta)
        self.tau   = float(tau)
        self.eps   = float(eps)
        self.enabled = enabled
        self.ema = float(ema)
        self._hooks = []
        self._taps  = []
        self._ema_am = None
        self._ema_ilf = None

    def attach_only(self, ln_module: nn.Module):
        """Attach to a single LayerNorm (recommended)."""
        self.clear()
        def _hook(_m, _i, out):
            if self.enabled: self._taps.append(out)
            return None
        self._hooks.append(ln_module.register_forward_hook(_hook))

    def reset_step(self):
        self._taps.clear()

    def clear(self):
        for h in self._hooks:
            try: h.remove()
            except Exception: pass
        self._hooks.clear(); self._taps.clear()
        self._ema_am = None; self._ema_ilf = None

    def loss(self) -> torch.Tensor:
        if not self.enabled or not self._taps:
            dev = self._taps[0].device if self._taps else torch.device("cpu")
            return torch.tensor(0.0, device=dev, requires_grad=True)

        total = None
        for x in self._taps:
            # normalize to (B,C,S)
            if x.dim() == 3:  # (B,T,H) or (B,H,T)
                X = x.transpose(1,2) if x.shape[1] < x.shape[-1] else x
            elif x.dim() == 2:
                X = x.unsqueeze(-1)
            elif x.dim() == 4:
                B,C,H,W = x.shape; X = x.view(B,C,H*W)
            else:
                continue

            # AM
            e  = torch.sqrt((X**2).mean(dim=(0,2)) + self.eps)       # (C,)
            am = e / (e.median() + self.eps)

            # ILF via cosine-neighbor counts
            A  = X.mean(dim=0)                                       # (C,S)
            An = A / A.norm(dim=1, keepdim=True).clamp_min(self.eps)
            S  = An @ An.t()
            S.fill_diagonal_(-1.0)
            nbrs = (S > self.tau).sum(dim=1).float()
            Cval = torch.tensor(A.shape[0], device=A.device, dtype=A.dtype)
            ilf_raw = (Cval / (1.0 + nbrs)).log()
            ilf = ilf_raw / (ilf_raw.median() + self.eps)

            # EMA smoothing (reduces batch noise)
            if self.ema and 0.0 < self.ema < 1.0:
                if self._ema_am is None:
                    self._ema_am  = am.detach()
                    self._ema_ilf = ilf.detach()
                self._ema_am  = self.ema * self._ema_am  + (1 - self.ema) * am.detach()
                self._ema_ilf = self.ema * self._ema_ilf + (1 - self.ema) * ilf.detach()
                am  = 0.5 * (am  + self._ema_am)
                ilf = 0.5 * (ilf + self._ema_ilf)

            layer_loss = (self.alpha * (am**2) - self.beta * ilf).sum()
            total = layer_loss if total is None else total + layer_loss

        return total / max(1, len(self._taps))

# ---------------------------
# Tiny Transformer (redundancy option)
# ---------------------------
class TinyLM(nn.Module):
    def __init__(self, vocab_size=200, d_model=128, nhead=4, nlayers=2, dim_ff=256,
                 seq_len=64, dup_factor=1):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model)
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                           dim_feedforward=dim_ff, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, num_layers=nlayers)
        self.dup_factor = int(dup_factor)
        if self.dup_factor > 1:
            # Project concatenated duplicate features back to d_model
            self.post = nn.Linear(d_model * self.dup_factor, d_model)
        else:
            self.post = nn.Identity()
        self.ln  = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.seq_len = seq_len

    def forward(self, x):  # x: (B, T)
        h = self.emb(x)          # (B,T,D)
        h = self.enc(h)          # (B,T,D)
        if self.dup_factor > 1:
            # Inject redundancy: concat h with copies of itself
            h = torch.cat([h] * self.dup_factor, dim=-1)  # (B,T,D*dup)
            h = self.post(h)                              # (B,T,D)
        h = self.ln(h)           # (B,T,D)
        logits = self.head(h)    # (B,T,V)
        return logits

# ---------------------------
# Synthetic dataset + label noise
# ---------------------------
def make_data(num_samples=2048, seq=64, vocab=200, label_noise=0.0, device="cpu"):
    torch.manual_seed(1337)
    X = torch.randint(0, vocab, (num_samples, seq), device=device)
    Y = X.clone()
    if label_noise > 0:
        mask = torch.rand_like(Y.float()) < float(label_noise)
        noisy = torch.randint(0, vocab, Y.shape, device=device)
        Y = torch.where(mask, noisy, Y)
    return TensorDataset(X, Y)

def train_epoch(model, loader, opt, nr: NRRegularizer, lambda_nr: float, device="cpu", warm_steps=0):
    model.train()
    total = 0.0
    for step, (xb, yb) in enumerate(loader):
        xb, yb = xb.to(device), yb.to(device)
        opt.zero_grad(set_to_none=True)
        nr.reset_step()
        logits = model(xb)  # (B,T,V)
        loss_ce = F.cross_entropy(logits.view(-1, logits.size(-1)), yb.view(-1))
        # λ warm-up
        if warm_steps > 0:
            global_step = getattr(train_epoch, "_gstep", 0)
            train_epoch._gstep = global_step + 1
            warm = min(1.0, train_epoch._gstep / float(warm_steps))
        else:
            warm = 1.0
        loss = loss_ce + (lambda_nr * warm * nr.loss() if nr.enabled and lambda_nr > 0 else 0.0)
        loss.backward()
        opt.step()
        total += loss_ce.item()
    return total / max(1, len(loader))

def eval_ppl(model, loader, device="cpu"):
    model.eval()
    nll, tokens = 0.0, 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), yb.view(-1), reduction="sum")
            nll += loss.item()
            tokens += yb.numel()
    return math.exp(nll / max(1, tokens))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--steps", type=int, default=100, help="batches per epoch")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seq", type=int, default=64)
    ap.add_argument("--vocab", type=int, default=200)
    ap.add_argument("--lambda-nr", type=float, default=3e-6)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--beta",  type=float, default=0.5)
    ap.add_argument("--tau",   type=float, default=0.5)
    ap.add_argument("--ema",   type=float, default=0.9)
    ap.add_argument("--dup-factor", type=int, default=2, help=">1 injects channel redundancy")
    ap.add_argument("--label-noise", type=float, default=0.1, help="probability to corrupt labels")
    ap.add_argument("--warmup-frac", type=float, default=0.1, help="fraction of total steps for λ warm-up")
    args = ap.parse_args()

    device = torch.device("cpu")  # keep CPU for determinism
    print(f"[env] device={device}")

    # Dataset (train/val from same generator; noise drives generalization gap)
    num_samples = args.steps * args.batch
    ds = make_data(num_samples=num_samples, seq=args.seq, vocab=args.vocab,
                   label_noise=args.label_noise, device=device)
    train_loader = DataLoader(ds, batch_size=args.batch, shuffle=True)
    val_loader   = DataLoader(ds, batch_size=args.batch)

    # Compute warmup steps
    total_steps = args.epochs * len(train_loader)
    warm_steps = int(args.warmup_frac * total_steps)

    # Baseline
    print("[baseline] building model...")
    base = TinyLM(vocab_size=args.vocab, seq_len=args.seq, dup_factor=args.dup_factor).to(device)
    opt_b = torch.optim.AdamW(base.parameters(), lr=3e-4, weight_decay=0.0)
    nr_b = NRRegularizer(enabled=False)
    for ep in range(args.epochs):
        tr = train_epoch(base, train_loader, opt_b, nr_b, lambda_nr=0.0, device=device, warm_steps=0)
        ppl = eval_ppl(base, val_loader, device=device)
        print(f"[baseline] epoch {ep+1} train CE={tr:.4f}  val ppl={ppl:.2f}")

    # NR run
    print("[nr] building model...")
    mdl = TinyLM(vocab_size=args.vocab, seq_len=args.seq, dup_factor=args.dup_factor).to(device)
    opt = torch.optim.AdamW(mdl.parameters(), lr=3e-4, weight_decay=0.0)
    nr = NRRegularizer(alpha=args.alpha, beta=args.beta, tau=args.tau, enabled=True, ema=args.ema)
    nr.attach_only(mdl.ln)  # final LN only
    for ep in range(args.epochs):
        tr = train_epoch(mdl, train_loader, opt, nr, lambda_nr=args.lambda_nr, device=device, warm_steps=warm_steps)
        ppl = eval_ppl(mdl, val_loader, device=device)
        print(f"[nr] epoch {ep+1} train CE={tr:.4f}  val ppl={ppl:.2f}")

if __name__ == "__main__":
    main()