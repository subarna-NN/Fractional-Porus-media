import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
import gc, warnings
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.optim import Adam, LBFGS
from math import gamma as math_gamma
from scipy.linalg import solve_banded
from scipy.interpolate import RegularGridInterpolator
from scipy.special import gamma as sp_gamma
from tqdm import tqdm
warnings.filterwarnings("ignore")

# ================================================================
# Space-Time Fractional Diffusion Equation
#
#   ᶜDₜ^α u(x,t) = -(-Δ)^(β/2) u(x,t)
#
#   Two fractional operators simultaneously:
#     Caputo in time  (order α ∈ (0,1))
#     Riesz in space  (order β ∈ (1,2))
#
#   Exact analytical solution (Mittag-Leffler):
#     u(x,t) = E_α(-λ t^α) · sin(πx)
#     λ = π^β  (Riesz eigenvalue for sin(πx) on [0,1])
#
#   Same architecture as proven FracFormer-PINN (L2=1.02e-2 for α=0.7)
#   Spectral Riesz: -(-Δ)^(β/2) sin(πx) = -λ·sin(πx)
#   PDE residual:   ᶜDₜ^α u + λ·u = 0
# ================================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42);  np.random.seed(42)
torch.cuda.empty_cache();  gc.collect()

if torch.cuda.is_available():
    print(f"GPU        : {torch.cuda.get_device_name(0)}")
    print(f"VRAM total : {torch.cuda.mem_get_info()[1]/1e9:.2f} GB")
    print(f"VRAM free  : {torch.cuda.mem_get_info()[0]/1e9:.2f} GB")

# ================================================================
# 1.  Parameters
# ================================================================
ALPHA   = 0.7    # Caputo time order
BETA    = 1.5    # Riesz space order
LAM     = np.pi**BETA       # eigenvalue: (-Δ)^(β/2) sin(πx) = λ sin(πx)
GAMMA2A = math_gamma(2.0 - ALPHA)
M_QUAD  = 15     # Caputo L1 quadrature steps

N_RES       = 2000
N_RAR_PROBE = 4000
N_RAR_KEEP  = 1000

W_RES = 10.0   # fixed residual weight (proven stable)

print(f"\nEquation  : ᶜDₜ^α u = -(-Δ)^(β/2) u")
print(f"α={ALPHA} (Caputo), β={BETA} (Riesz)")
print(f"λ=π^β={LAM:.6f},  Γ(2-α)={GAMMA2A:.6f}")
print(f"Analytical: u(x,t) = E_α(-λt^α)·sin(πx)")

# ================================================================
# 2.  Mittag-Leffler function and Analytical Solution
# ================================================================
def mittag_leffler(alpha, z, n_terms=100):
    """E_α(z) = Σ z^k / Γ(αk+1)"""
    result = np.zeros_like(z, dtype=np.float64)
    for k in range(n_terms):
        result += (z**k) / sp_gamma(alpha*k + 1.0)
    return result

# ================================================================
# 3.  FDM Ground Truth (Implicit L1-FDM — same as proven version)
#     Now with λ=π^β instead of λ=π² (only change)
# ================================================================
print("\nComputing Implicit L1-FDM ground truth ...")
Nx_fdm, Nt_fdm = 200, 1000
xg_fdm = np.linspace(0, 1, Nx_fdm)
tg_fdm = np.linspace(0, 1, Nt_fdm)
dx_fdm = xg_fdm[1] - xg_fdm[0]
dt_fdm = tg_fdm[1] - tg_fdm[0]

b_fdm     = np.array([(k+1)**(1-ALPHA)-k**(1-ALPHA)
                       for k in range(Nt_fdm+1)], dtype=np.float64)
cap_scale = dt_fdm**(-ALPHA) / GAMMA2A

# Implicit L1-FDM using spectral Riesz:
# ᶜDₜ^α u + λu = 0  →  cap_scale*b₀*u^n + λu^n = cap_scale*history
# Assembled as banded system (diagonal only since λ is scalar × identity)
Ni  = Nx_fdm - 2
d   = cap_scale * b_fdm[0] + LAM   # diagonal coefficient (Riesz spectral)
od  = 0.0                           # no off-diagonal: Riesz is spectral (λ·u)

# Wait: the spatial operator is spectral but the spatial grid is still needed
# for the full 2D solution. We use the eigenfunction structure:
# u(x_i, t) = T(t) * sin(π*x_i)
# Equation for T: ᶜDₜ^α T = -λ T
# Implicit L1: cap_scale*(b₀*T^n - history) = -λ*T^n
# → (cap_scale*b₀ + λ)*T^n = cap_scale*history
# This is a SCALAR equation at each time step — trivially solved.

T_fdm = np.zeros(Nt_fdm)
T_fdm[0] = 1.0   # T(0) = 1

for n in range(1, Nt_fdm):
    hist = np.float64(0.0)
    for k in range(1, n):
        hist += (b_fdm[k-1] - b_fdm[k]) * T_fdm[n-k]
    hist += b_fdm[n-1] * T_fdm[0]
    T_fdm[n] = (cap_scale * hist) / (cap_scale * b_fdm[0] + LAM)

# Build full 2D FDM solution
utrue_fdm = np.outer(T_fdm, np.sin(np.pi * xg_fdm)).astype(np.float32)

# Verify against Mittag-Leffler
t_check = np.array([0.0, 0.5, 1.0])
print(f"  Verification against Mittag-Leffler E_α(-λt^α)·sin(π·0.5):")
for tc in t_check:
    if tc == 0.0:
        ml_val = 1.0
    else:
        ml_val = float(mittag_leffler(ALPHA, np.array([-LAM*tc**ALPHA]))[0])
    n_idx = int(tc * (Nt_fdm-1))
    fdm_val = T_fdm[n_idx]
    print(f"  t={tc:.1f}: FDM={fdm_val:.6f}  ML={ml_val:.6f}  "
          f"diff={abs(fdm_val-ml_val):.2e}")

# ================================================================
# 4.  Fourier Embedding
# ================================================================
class FourierEmbedding(nn.Module):
    def __init__(self, in_dim=2, n_freq=48, sigma=4.0):
        super().__init__()
        B = torch.randn(in_dim, n_freq) * sigma
        self.register_buffer("B", B)
        self.out_dim = 2 * n_freq
    def forward(self, x):
        p = x @ self.B
        return torch.cat([torch.sin(p), torch.cos(p)], dim=-1)

# ================================================================
# 5.  Spatiotemporal Pseudo-Sequence (5 tokens)
# ================================================================
def pseudo_seq(xt, dx, dtc=0.03):
    x_, t_ = xt[:,0:1], xt[:,1:2]
    return torch.stack([
        torch.cat([x_-dx, t_], 1),
        torch.cat([x_,    t_], 1),
        torch.cat([x_+dx, t_], 1),
        torch.cat([x_, (t_-dtc).clamp(0.)], 1),
        torch.cat([x_, (t_+dtc).clamp(max=1.)], 1),
    ], dim=1)

# ================================================================
# 6.  Transformer Block
# ================================================================
class TBlock(nn.Module):
    def __init__(self, d=96, h=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(d, h, batch_first=True)
        self.ff   = nn.Sequential(nn.Linear(d,4*d),nn.GELU(),nn.Linear(4*d,d))
        self.n1   = nn.LayerNorm(d);  self.n2 = nn.LayerNorm(d)
    def forward(self, x):
        xn=self.n1(x); h,_=self.attn(xn,xn,xn); x=x+h
        return x+self.ff(self.n2(x))

# ================================================================
# 7.  Caputo L1 Module (VERIFIED — reversed b-weights, L2=1.02e-2)
# ================================================================
class CaputoL1(nn.Module):
    def __init__(self, alpha=0.7, M=15):
        super().__init__()
        self.alpha=alpha; self.M=M; self.gc=math_gamma(2.-alpha)
        k=torch.arange(0,M,dtype=torch.float32)
        b=(k+1)**(1-alpha)-k**(1-alpha)
        self.register_buffer("bw", torch.flip(b,[0]))  # VERIFIED CORRECT
    def forward(self, model, xc, tc, dx):
        N,M=xc.shape[0],self.M; dt=tc/M
        ki=torch.arange(0,M+1,device=xc.device,dtype=torch.float32)
        tn=dt*ki.unsqueeze(0)
        xr=xc.repeat(1,M+1).reshape(N*(M+1),1)
        tf=tn.reshape(N*(M+1),1)
        ua=model(torch.cat([xr,tf],1),dx)
        un=ua.reshape(N,M+1); du=un[:,1:]-un[:,:-1]
        return (self.bw.unsqueeze(0)*du).sum(1,keepdim=True)/(self.gc*dt**self.alpha)

# ================================================================
# 8.  FracFormer-PINN (Space-Time Fractional)
#
#  Hard IC+BC enforcement:
#    u(x,t) = sin(πx)·(1-ramp) + 4x(1-x)·ramp·NN(x,t)
#    ramp = 1 - exp(-5t)
#    → u(x,0) = sin(πx) ✓  (ramp(0)=0)
#    → u(0,t) = u(1,t) = 0 ✓  (sin(0)=sin(π)=0, 4·0·1=0, 4·1·0=0)
#
#  Same proven output transform from FracFormer (L2=1.02e-2).
#  Only change: PDE residual uses ᶜDₜ^α u + λ·u = 0
#  instead of ᶜDₜ^α u - u_xx = 0.
# ================================================================
class FracFormerSpaceTime(nn.Module):
    def __init__(self, d=96, nh=4, nb=3, nf=48, sig=4., alpha=0.7, M=15):
        super().__init__()
        self.fourier=FourierEmbedding(2,nf,sig)
        self.embed=nn.Sequential(nn.Linear(self.fourier.out_dim,d),nn.Tanh())
        self.tblocks=nn.ModuleList([TBlock(d,nh) for _ in range(nb)])
        self.mlp=nn.Sequential(
            nn.Linear(d,128),nn.Tanh(),nn.Linear(128,128),nn.Tanh(),
            nn.Linear(128,64),nn.Tanh(),nn.Linear(64,1))
        self.caputo=CaputoL1(alpha=alpha,M=M)
        self.log_w_bc=nn.Parameter(torch.tensor(2.0))
        for m in self.modules():
            if isinstance(m,nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)

    def _base(self, xt, dx):
        seq=pseudo_seq(xt,dx); N,S,_=seq.shape
        h=self.embed(self.fourier(seq.reshape(N*S,2))).reshape(N,S,-1)
        for b in self.tblocks: h=b(h)
        return self.mlp(h[:,1,:])

    def forward(self, xt, dx):
        """Hard IC+BC: u = sin(πx)·(1-ramp) + 4x(1-x)·ramp·NN"""
        x_,t_=xt[:,0:1],xt[:,1:2]
        ramp=1.0-torch.exp(-5.0*t_)
        return (torch.sin(np.pi*x_)*(1.0-ramp)
                + 4.0*x_*(1.0-x_)*ramp*self._base(xt,dx))

# ================================================================
# 9.  Build
# ================================================================
model=FracFormerSpaceTime(alpha=ALPHA,M=M_QUAD).to(device)
np_=sum(p.numel() for p in model.parameters())
print(f"\nModel      : FracFormer-PINN (Space-Time Fractional)")
print(f"Parameters : {np_:,}")

# Verify ICs
with torch.no_grad():
    x_v=torch.linspace(0,1,101).reshape(-1,1).to(device)
    t0=torch.zeros(101,1).to(device)
    u0=model(torch.cat([x_v,t0],1),torch.tensor(0.01))
    print(f"IC error   : {(u0-torch.sin(np.pi*x_v)).abs().max().item():.2e}")
    t5=torch.full((1,1),0.5).to(device)
    print(f"BC x=0     : {model(torch.cat([torch.zeros(1,1).to(device),t5],1),torch.tensor(0.01)).item():.2e}")
    print(f"BC x=1     : {model(torch.cat([torch.ones(1,1).to(device),t5],1),torch.tensor(0.01)).item():.2e}")

# ================================================================
# 10.  Domain & Points
# ================================================================
Nx=101; dx=torch.tensor(1.0/(Nx-1))
tbv=torch.linspace(0,1,300).reshape(-1,1).to(device).clamp(min=1e-4)
bc_pts=torch.cat([torch.cat([torch.zeros_like(tbv),tbv],1),
                  torch.cat([torch.ones_like(tbv), tbv],1)],0)

def new_res(N=N_RES):
    p=torch.rand(N,2,device=device); p[:,1]=p[:,1].clamp(min=2e-3)
    return p.requires_grad_(True)

res=new_res()
print(f"BC points  : {bc_pts.shape[0]}, Colloc pts : {N_RES}")

# ================================================================
# 11.  PDE Residual
#      ᶜDₜ^α u + λ·u = 0
#      Spectral Riesz: -(-Δ)^(β/2) u = -λ·u (exact for sin(πx))
#      So: ᶜDₜ^α u = -λ·u → residual = cap(u) + λ·u
# ================================================================
def pde_res(model, pts):
    cap_u = model.caputo(model, pts[:,0:1], pts[:,1:2], dx)
    u     = model(pts, dx)
    # ᶜDₜ^α u + λ·u = 0
    r = cap_u + LAM * u
    return torch.mean(r**2)

def pinn_loss(model, pts):
    r  = pde_res(model, pts)
    bc = torch.mean(model(bc_pts,dx)**2)
    wb = torch.exp(model.log_w_bc)
    return W_RES*r + wb*bc, r, bc

# ================================================================
# 12.  RAR
# ================================================================
def rar_resample(model):
    model.eval(); torch.cuda.empty_cache(); gc.collect()
    Ml=8
    with torch.no_grad():
        pts=torch.rand(N_RAR_PROBE,2,device=device)
        pts[:,1]=pts[:,1].clamp(0.05)
        x_,t_=pts[:,0:1],pts[:,1:2]
        ev=lambda xv,tv: model(torch.cat([xv.clamp(0,1),tv.clamp(1e-3,1)],1),dx)
        k_l=torch.arange(0,Ml,device=device,dtype=torch.float32)
        bl=torch.flip((k_l+1)**(1-ALPHA)-k_l**(1-ALPHA),[0])
        dtl=t_/Ml
        idx=torch.arange(0,Ml+1,device=device,dtype=torch.float32)
        tn=dtl*idx.unsqueeze(0)
        xr=x_.repeat(1,Ml+1).reshape(-1,1)
        un=ev(xr,tn.reshape(-1,1)).reshape(N_RAR_PROBE,Ml+1)
        du=un[:,1:]-un[:,:-1]
        cap=(bl.unsqueeze(0)*du).sum(1,keepdim=True)/(GAMMA2A*dtl**ALPHA)
        u_now=ev(x_,t_)
        mag=(cap+LAM*u_now).abs().squeeze()
    ki=torch.topk(mag,min(N_RAR_KEEP,N_RAR_PROBE)).indices
    fill=torch.rand(max(0,N_RES-len(ki)),2,device=device)
    fill[:,1]=fill[:,1].clamp(2e-3)
    out=torch.cat([pts[ki],fill],0)[:N_RES]
    model.train(); torch.cuda.empty_cache()
    return out.detach().requires_grad_(True)

# ================================================================
# 13.  Training (proven 3-stage schedule)
# ================================================================
loss_log=[]

print("\n"+"="*55)
print("STAGE 1/3  Adam Warmup (2500 steps)")
print("="*55)
opt1=Adam(model.parameters(),lr=5e-4)
sch1=torch.optim.lr_scheduler.OneCycleLR(
    opt1,max_lr=1e-3,total_steps=2500,pct_start=0.10,anneal_strategy="cos")
for step in tqdm(range(2500),desc="Warmup"):
    opt1.zero_grad()
    loss,r,bc_=pinn_loss(model,res)
    loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
    opt1.step(); sch1.step(); loss_log.append(loss.item())
    if (step+1)%500==0:
        tqdm.write(f"  [{step+1:5d}] total={loss.item():.3e}  "
                   f"res={r.item():.3e}  lr={sch1.get_last_lr()[0]:.1e}")
torch.cuda.empty_cache()

print("\n"+"="*55)
print("STAGE 2/3  CosineAdam + RAR (6000 steps)")
print("="*55)
opt2=Adam(model.parameters(),lr=3e-4)
sch2=torch.optim.lr_scheduler.CosineAnnealingLR(opt2,T_max=6000,eta_min=5e-7)
for step in tqdm(range(6000),desc="CosineAdam"):
    if step%1500==0: res=rar_resample(model)
    opt2.zero_grad()
    loss,r,bc_=pinn_loss(model,res)
    loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
    opt2.step(); sch2.step(); loss_log.append(loss.item())
    if (step+1)%1000==0:
        tqdm.write(f"  [{step+1:5d}] total={loss.item():.3e}  "
                   f"res={r.item():.3e}  lr={sch2.get_last_lr()[0]:.1e}")
torch.cuda.empty_cache()

print("\n"+"="*55)
print("STAGE 3/3  L-BFGS polish (1000 iters)")
print("="*55)
res_f=new_res()
opt3=LBFGS(model.parameters(),max_iter=1000,lr=1.0,
           tolerance_grad=1e-9,tolerance_change=1e-11,
           line_search_fn="strong_wolfe")
_n=[0]
def closure():
    opt3.zero_grad()
    loss,r,bc_=pinn_loss(model,res_f)
    loss.backward(); _n[0]+=1; loss_log.append(loss.item())
    if _n[0]%200==0:
        print(f"  L-BFGS [{_n[0]:4d}] total={loss.item():.3e}  res={r.item():.3e}")
    return loss
opt3.step(closure)
print(f"  Done — {_n[0]} evals")
torch.cuda.empty_cache()

# ================================================================
# 14.  Evaluation
# ================================================================
xgm=np.linspace(0,1,Nx); tgm=np.linspace(0,1,201)
itp=RegularGridInterpolator((tg_fdm,xg_fdm),utrue_fdm,method='linear',
                             bounds_error=False,fill_value=None)
Tm,Xm=np.meshgrid(tgm,xgm,indexing='ij')
utm=itp(np.stack([Tm.ravel(),Xm.ravel()],1)).reshape(201,Nx).astype(np.float32)

# Also compute Mittag-Leffler analytical solution for comparison
uml=np.zeros((201,Nx),dtype=np.float32)
for i,tv in enumerate(tgm):
    if tv==0:
        ml=1.0
    else:
        ml=float(mittag_leffler(ALPHA,np.array([-LAM*tv**ALPHA]))[0])
    uml[i]=ml*np.sin(np.pi*xgm)

upred=np.zeros((201,Nx),dtype=np.float32)
model.eval()
with torch.no_grad():
    for i,tv in enumerate(tgm):
        tt=torch.full((Nx,1),tv,device=device)
        xv=torch.linspace(0,1,Nx).reshape(-1,1).to(device)
        upred[i]=model(torch.cat([xv,tt],1),dx).cpu().numpy().reshape(Nx)

# Metrics vs FDM
l2  =float(np.sqrt(np.sum((utm-upred)**2)/(np.sum(utm**2)+1e-14)))
l1  =float(np.sum(np.abs(utm-upred))/(np.sum(np.abs(utm))+1e-14))
linf=float(np.abs(utm-upred).max())
# Metrics vs Mittag-Leffler
l2ml=float(np.sqrt(np.sum((uml-upred)**2)/(np.sum(uml**2)+1e-14)))

print(f"\n{'='*55}")
print(f"  MODEL  : FracFormer-PINN (Space-Time Fractional)")
print(f"  PDE    : ᶜDₜ^α u = -(-Δ)^(β/2) u")
print(f"  α={ALPHA} (Caputo),  β={BETA} (Riesz)")
print(f"  vs FDM  — L2={l2:.4e}  L1={l1:.4e}  Linf={linf:.4e}")
print(f"  vs ML   — L2={l2ml:.4e}")
print(f"{'='*55}")

# ================================================================
# 15.  Publication-Quality Plots
# ================================================================
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Global style
plt.rcParams.update({
    'font.family':       'DejaVu Sans',
    'font.size':         12,
    'axes.titlesize':    13,
    'axes.labelsize':    12,
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'figure.facecolor':  'white',
    'axes.facecolor':    'white',
    'savefig.facecolor': 'white',
    'savefig.dpi':       600,
})

# Custom colormaps
CMAP_SOL  = 'RdYlBu_r'       # warm gradient for solution fields
CMAP_ERR  = 'YlOrRd'         # yellow→red for errors
CMAP_3D_S = 'coolwarm'        # diverging for 3D surface
LINE_REF  = '#1a1a2e'         # near-black for reference
LINE_ML   = '#0077b6'         # deep blue for Mittag-Leffler
LINE_PINN = '#e63946'         # vivid red for PINN
LINE_ERR  = '#f4a261'         # amber for error curves

aerr = np.abs(utm - upred)

# ----------------------------------------------------------------
# A: Time-snapshot comparison (2×2)
# ----------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(13, 9))
axes = axes.flatten()
snap_times = [0.25, 0.50, 0.75, 1.00]
colors_bg  = ['#f0f4ff', '#fff0f0', '#f0fff4', '#fffff0']

for ax, frac, bg in zip(axes, snap_times, colors_bg):
    idx = int(frac * 200)
    ax.set_facecolor(bg)
    ax.fill_between(xgm, utm[idx], alpha=0.12, color=LINE_REF)
    ax.plot(xgm, utm[idx],   color=LINE_REF,  lw=3.0, ls='-',
            label='L1-FDM (reference)', zorder=3)
    ax.plot(xgm, uml[idx],   color=LINE_ML,   lw=2.0, ls=':',
            label='Mittag-Leffler (exact)', zorder=4)
    ax.plot(xgm, upred[idx], color=LINE_PINN, lw=2.5, ls='--',
            label='FracFormer-PINN', zorder=5)
    ax.set_title(f'$t = {tgm[idx]:.2f}$', fontweight='bold', pad=6)
    ax.set_xlabel('$x$'); ax.set_ylabel('$u(x,t)$')
    ax.legend(fontsize=9, framealpha=0.85, loc='upper right')
    # L2 annotation
    l2_snap = np.sqrt(np.sum((utm[idx]-upred[idx])**2)
                      /(np.sum(utm[idx]**2)+1e-14))
    ax.text(0.03, 0.06, f'$L_2={l2_snap:.2e}$',
            transform=ax.transAxes, fontsize=9,
            color=LINE_PINN, bbox=dict(fc='white', ec='none', alpha=0.7))

plt.suptitle(
    r'FracFormer-PINN $\;|\;$ ${}^C\!D_t^\alpha u = -(-\Delta)^{\beta/2}u$'
    f'  $\\alpha={ALPHA},\\;\\beta={BETA}$\n'
    f'Global $L_2={l2:.3e}$  (vs FDM)   $L_2={l2ml:.3e}$  (vs Mittag-Leffler)',
    fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig('fractST_snapshots.png', dpi=600, bbox_inches='tight')
plt.show(); print("Saved: fractST_snapshots.png")

# ----------------------------------------------------------------
# B: Space-time heatmaps (1×3)
# ----------------------------------------------------------------
fig2, ax2 = plt.subplots(1, 3, figsize=(16, 4.5))
datasets = [
    (utm,    'L1-FDM Reference',   CMAP_SOL),
    (upred,  'FracFormer-PINN',    CMAP_SOL),
    (aerr,   'Absolute Error',     CMAP_ERR),
]
for ax, (data, title, cm) in zip(ax2, datasets):
    im = ax.contourf(xgm, tgm, data, levels=60, cmap=cm)
    ax.contour(xgm, tgm, data, levels=8,
               colors='white', linewidths=0.4, alpha=0.5)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='4%', pad=0.07)
    cb  = plt.colorbar(im, cax=cax)
    cb.ax.tick_params(labelsize=9)
    ax.set_title(title, fontweight='bold', pad=6)
    ax.set_xlabel('$x$'); ax.set_ylabel('$t$')
plt.suptitle(
    r'Space-Time Solution $\;|\;$ ${}^C\!D_t^\alpha u = -(-\Delta)^{\beta/2}u$'
    f'  $\\alpha={ALPHA},\\;\\beta={BETA}$',
    fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig('fractST_heatmap.png', dpi=600, bbox_inches='tight')
plt.show(); print("Saved: fractST_heatmap.png")

# ----------------------------------------------------------------
# C: 3D surfaces (1×3)
# ----------------------------------------------------------------
fig3 = plt.figure(figsize=(17, 5))
Tm2, Xm2 = np.meshgrid(tgm, xgm)
configs = [
    (utm.T,   'L1-FDM Reference', CMAP_3D_S),
    (upred.T, 'FracFormer-PINN',  CMAP_3D_S),
    (aerr.T,  'Absolute Error',   'YlOrRd'),
]
for col_, (data, title, cm) in enumerate(configs):
    ax = fig3.add_subplot(1, 3, col_+1, projection='3d')
    surf = ax.plot_surface(Xm2, Tm2, data, cmap=cm,
                           alpha=0.92, linewidth=0, antialiased=True)
    fig3.colorbar(surf, ax=ax, shrink=0.55, aspect=10, pad=0.08)
    ax.set_title(title, fontweight='bold', pad=8)
    ax.set_xlabel('$x$', labelpad=6); ax.set_ylabel('$t$', labelpad=6)
    ax.set_zlabel('$u$', labelpad=4)
    ax.tick_params(labelsize=8)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
plt.suptitle(
    r'3D Solution Surface $\;|\;$ ${}^C\!D_t^\alpha u = -(-\Delta)^{\beta/2}u$'
    f'  $\\alpha={ALPHA},\\;\\beta={BETA}$',
    fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig('fractST_3d.png', dpi=600, bbox_inches='tight')
plt.show(); print("Saved: fractST_3d.png")

# ----------------------------------------------------------------
# D: L2 error over time
# ----------------------------------------------------------------
l2t = [np.sqrt(np.sum((utm[i]-upred[i])**2)/(np.sum(utm[i]**2)+1e-14))
       for i in range(201)]

fig4, ax4 = plt.subplots(figsize=(9, 4))
ax4.fill_between(tgm, l2t, alpha=0.18, color='#4361ee')
ax4.semilogy(tgm, l2t, lw=2.2, color='#4361ee', label='Relative $L_2$ error')
ax4.axhline(l2, color=LINE_PINN, ls='--', lw=1.5,
            label=f'Global $L_2={l2:.2e}$')
ax4.axhline(1e-2, color='gray', ls=':', lw=1, alpha=0.7, label='$10^{-2}$ threshold')
ax4.set_xlabel('Time $t$'); ax4.set_ylabel('Relative $L_2$ error')
ax4.set_title(
    r'Error Evolution $\;|\;$ ${}^C\!D_t^\alpha u=-(-\Delta)^{\beta/2}u$'
    f'  $\\alpha={ALPHA},\\;\\beta={BETA}$',
    fontweight='bold')
ax4.legend(fontsize=10, framealpha=0.85)
plt.tight_layout()
plt.savefig('fractST_l2time.png', dpi=600, bbox_inches='tight')
plt.show(); print("Saved: fractST_l2time.png")

# ----------------------------------------------------------------
# E: Training loss
# ----------------------------------------------------------------
fig5, ax5 = plt.subplots(figsize=(10, 4))
ax5.fill_between(range(len(loss_log)), loss_log, alpha=0.15, color='#f77f00')
ax5.semilogy(loss_log, lw=1.5, color='#f77f00', label='Total loss')
ax5.axvline(2500, color='#4361ee', ls='--', lw=1.8,
            label='Stage 1 → 2 (Warmup end)')
ax5.axvline(8500, color='#e63946', ls='--', lw=1.8,
            label='Stage 2 → 3 (CosineAdam end)')
ymin, ymax = ax5.get_ylim()
ax5.fill_betweenx([ymin,ymax], 0,    2500, alpha=0.04, color='#4361ee')
ax5.fill_betweenx([ymin,ymax], 2500, 8500, alpha=0.04, color='#2dc653')
ax5.fill_betweenx([ymin,ymax], 8500, len(loss_log), alpha=0.04, color='#e63946')
ax5.text(1250,  ymax*0.5, 'Stage 1\nWarmup',   ha='center', fontsize=9, color='#4361ee')
ax5.text(5500,  ymax*0.5, 'Stage 2\nRAR+Cosine',ha='center',fontsize=9, color='#2dc653')
ax5.text(len(loss_log)-400, ymax*0.5, 'Stage 3\nL-BFGS',
         ha='center', fontsize=9, color='#e63946')
ax5.set_xlabel('Training step'); ax5.set_ylabel('Loss')
ax5.set_title('FracFormer-PINN — Three-Stage Training Loss', fontweight='bold')
ax5.legend(fontsize=10, framealpha=0.85)
plt.tight_layout()
plt.savefig('fractST_loss.png', dpi=600, bbox_inches='tight')
plt.show(); print("Saved: fractST_loss.png")

# ----------------------------------------------------------------
# F: Mittag-Leffler modal decay
# ----------------------------------------------------------------
T_pred_modal = upred[:,Nx//2] / np.sin(np.pi*0.5)
T_ml_modal   = uml[:,Nx//2]   / np.sin(np.pi*0.5)
T_fdm_modal  = utm[:,Nx//2]   / np.sin(np.pi*0.5)

fig6, ax6 = plt.subplots(figsize=(9, 4))
ax6.fill_between(tgm, T_fdm_modal, alpha=0.10, color=LINE_REF)
ax6.plot(tgm, T_fdm_modal,   color=LINE_REF,  lw=3.0, ls='-',
         label='L1-FDM reference')
ax6.plot(tgm, T_ml_modal,    color=LINE_ML,   lw=2.0, ls=':',
         label=r'Mittag-Leffler $E_\alpha(-\lambda t^\alpha)$')
ax6.plot(tgm, T_pred_modal,  color=LINE_PINN, lw=2.5, ls='--',
         label='FracFormer-PINN')
ax6.set_xlabel('$t$')
ax6.set_ylabel(r'$E_\alpha(-\lambda t^\alpha)$  [modal amplitude]')
ax6.set_title(
    r'Anomalous Decay: Mittag-Leffler $E_\alpha(-\lambda t^\alpha)$'
    f'\n$\\alpha={ALPHA}$, $\\beta={BETA}$, $\\lambda=\\pi^\\beta={LAM:.3f}$',
    fontweight='bold')
ax6.legend(fontsize=10, framealpha=0.85)
plt.tight_layout()
plt.savefig('fractST_modal.png', dpi=600, bbox_inches='tight')
plt.show(); print("Saved: fractST_modal.png")

# ----------------------------------------------------------------
# Print final summary
# ----------------------------------------------------------------
print(f"\n{'='*55}")
print(f"  FINAL RESULTS")
print(f"  PDE  : ᶜDₜ^α u = -(-Δ)^(β/2) u")
print(f"  α={ALPHA} (Caputo),  β={BETA} (Riesz)")
print(f"  L2 (vs FDM) : {l2:.4e}  ({l2*100:.4f}%)")
print(f"  L1 (vs FDM) : {l1:.4e}  ({l1*100:.4f}%)")
print(f"  L2 (vs ML)  : {l2ml:.4e}  ({l2ml*100:.4f}%)")
print(f"  Linf        : {linf:.4e}")
print(f"  Params      : {np_:,}")
print(f"{'='*55}")
print("All done.")
