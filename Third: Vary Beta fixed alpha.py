import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
import gc, warnings
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from torch.optim import Adam, LBFGS
from math import gamma as math_gamma
from scipy.interpolate import RegularGridInterpolator
from scipy.special import gammaln
from tqdm import tqdm
warnings.filterwarnings("ignore")

# ================================================================
# FracFormer-PINN Beta Study v2
#
# β=1.2: NO CHANGE — result was perfect (L2=4.21e-3, clean plots)
# β=1.8: TWO TARGETED FIXES for t=1.0 oscillation:
#
#   FIX: Remove time weighting (caused early-time gap), keep late-time L-BFGS
#   Relative RAR: |res|/(|u|+ε) balances early/late time attention
#     Reason: u decays to 0.047 at t=1 (21× smaller than t=0)
#     Standard uniform sampling gives too few high-quality gradients
#     at late times where the solution is very flat.
#
#   FIX 2 — Extra L-BFGS pass (Stage 3b):
#     After main L-BFGS (1000 iters), run second pass (500 iters, lr=0.1)
#     specifically on late-time points t∈[0.7,1.0].
#     Fine-tunes the small-amplitude region without disturbing early times.
# ================================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42);  np.random.seed(42)
torch.cuda.empty_cache();  gc.collect()

if torch.cuda.is_available():
    print(f"GPU        : {torch.cuda.get_device_name(0)}")
    print(f"VRAM total : {torch.cuda.mem_get_info()[1]/1e9:.2f} GB")
    print(f"VRAM free  : {torch.cuda.mem_get_info()[0]/1e9:.2f} GB")

ALPHA   = 0.7
GAMMA2A = math_gamma(2.0 - ALPHA)
M_QUAD  = 15
BETA_LIST = [1.2, 1.8]
N_RES=2000; N_RAR_PROBE=4000; N_RAR_KEEP=1000
W_RES=10.0

print(f"\nEquation  : ᶜDₜ^α u = -(-Δ)^(β/2) u")
print(f"Fixed: α={ALPHA},  Γ(2-α)={GAMMA2A:.6f}")
print(f"Runs : β = {BETA_LIST}")
print(f"FIX for β=1.8: relative RAR + late-time L-BFGS (no time weighting)")

# ================================================================
# Robust ML series
# ================================================================
def ml_robust(alpha, z):
    if z == 0: return 1.0
    s = 0.0
    for k in range(500):
        lgt = k*np.log(abs(z)+1e-300) - gammaln(alpha*k + 1.0)
        if lgt > 600: return float('nan')
        term = (np.sign(z)**k) * np.exp(lgt)
        s += term
        if k > 30 and abs(s) > 1e-10 and abs(term) < 1e-12*abs(s): break
    return float(s)

def compute_ml_2d(alpha, lam, tg, xg):
    uml = np.zeros((len(tg), len(xg)), dtype=np.float32)
    for i, t in enumerate(tg):
        ml = 1.0 if t==0 else ml_robust(alpha, -lam*t**alpha)
        if np.isnan(ml) or not np.isfinite(ml): ml = np.nan
        uml[i] = ml * np.sin(np.pi * xg)
    return uml

# ================================================================
# Architecture (unchanged)
# ================================================================
class FourierEmbedding(nn.Module):
    def __init__(self, in_dim=2, n_freq=48, sigma=4.0):
        super().__init__()
        B = torch.randn(in_dim, n_freq)*sigma
        self.register_buffer("B", B)
        self.out_dim = 2*n_freq
    def forward(self, x):
        p = x @ self.B
        return torch.cat([torch.sin(p), torch.cos(p)], dim=-1)

def pseudo_seq(xt, dx, dtc=0.03):
    x_, t_ = xt[:,0:1], xt[:,1:2]
    return torch.stack([
        torch.cat([x_-dx, t_], 1), torch.cat([x_, t_], 1),
        torch.cat([x_+dx, t_], 1),
        torch.cat([x_, (t_-dtc).clamp(0.)], 1),
        torch.cat([x_, (t_+dtc).clamp(max=1.)], 1),
    ], dim=1)

class TBlock(nn.Module):
    def __init__(self, d=96, h=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(d, h, batch_first=True)
        self.ff   = nn.Sequential(nn.Linear(d,4*d),nn.GELU(),nn.Linear(4*d,d))
        self.n1   = nn.LayerNorm(d);  self.n2 = nn.LayerNorm(d)
    def forward(self, x):
        xn=self.n1(x); h,_=self.attn(xn,xn,xn); x=x+h
        return x+self.ff(self.n2(x))

class CaputoL1(nn.Module):
    def __init__(self, alpha=0.7, M=15):
        super().__init__()
        self.alpha=alpha; self.M=M; self.gc=math_gamma(2.-alpha)
        k=torch.arange(0,M,dtype=torch.float32)
        b=(k+1)**(1-alpha)-k**(1-alpha)
        self.register_buffer("bw", torch.flip(b,[0]))
    def forward(self, model, xc, tc, dx):
        N,M=xc.shape[0],self.M; dt=tc/M
        ki=torch.arange(0,M+1,device=xc.device,dtype=torch.float32)
        tn=dt*ki.unsqueeze(0)
        xr=xc.repeat(1,M+1).reshape(N*(M+1),1)
        tf=tn.reshape(N*(M+1),1)
        ua=model(torch.cat([xr,tf],1),dx)
        un=ua.reshape(N,M+1); du=un[:,1:]-un[:,:-1]
        return (self.bw.unsqueeze(0)*du).sum(1,keepdim=True)/(self.gc*dt**self.alpha)

class FracFormerST(nn.Module):
    def __init__(self, lam, d=96, nh=4, nb=3, nf=48, sig=4., alpha=0.7, M=15):
        super().__init__()
        self.lam=lam
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
        x_,t_=xt[:,0:1],xt[:,1:2]
        ramp=1.0-torch.exp(-5.0*t_)
        return (torch.sin(np.pi*x_)*(1.0-ramp)
                + 4.0*x_*(1.0-x_)*ramp*self._base(xt,dx))

# ================================================================
# FDM
# ================================================================
def build_fdm(beta):
    lam=np.pi**beta
    Nx_f,Nt_f=200,1000; xg_f=np.linspace(0,1,Nx_f); tg_f=np.linspace(0,1,Nt_f)
    dt_f=tg_f[1]-tg_f[0]
    b_f=np.array([(k+1)**(1-ALPHA)-k**(1-ALPHA) for k in range(Nt_f+1)],dtype=np.float64)
    csc=dt_f**(-ALPHA)/GAMMA2A; T_f=np.zeros(Nt_f); T_f[0]=1.
    for n in range(1,Nt_f):
        hist=sum((b_f[k-1]-b_f[k])*T_f[n-k] for k in range(1,n))+b_f[n-1]*T_f[0]
        T_f[n]=(csc*hist)/(csc*b_f[0]+lam)
    return np.outer(T_f,np.sin(np.pi*xg_f)).astype(np.float32), xg_f, tg_f, T_f

# ================================================================
# Training — with optional time-weighting for β=1.8
# ================================================================
def train_beta(beta):
    torch.manual_seed(42); np.random.seed(42)
    torch.cuda.empty_cache(); gc.collect()
    lam=np.pi**beta; Nx=101; dx=torch.tensor(1.0/(Nx-1))
    use_time_weight = False  # Removed: weighting caused early-time gap (t=0.25-0.5)

    print(f"\n  Building FDM for β={beta}, λ={lam:.4f} ...")
    utrue_fdm,xg_fdm,tg_fdm,T_fdm=build_fdm(beta)
    ml_05=ml_robust(ALPHA,-lam*0.5**ALPHA)
    ml_10=ml_robust(ALPHA,-lam*1.0**ALPHA)
    print(f"  t=0.5: FDM={T_fdm[500]:.6f}  ML={ml_05:.6f}  diff={abs(T_fdm[500]-ml_05):.2e}")
    print(f"  t=1.0: FDM={T_fdm[-1]:.6f}   ML={ml_10:.6f}  diff={abs(T_fdm[-1]-ml_10):.2e}")
    print(f"  Time-weighted loss: {'YES (fix for t=1.0 oscillation)' if use_time_weight else 'NO'}")

    model=FracFormerST(lam=lam,alpha=ALPHA,M=M_QUAD).to(device)

    tbv=torch.linspace(0,1,300).reshape(-1,1).to(device).clamp(min=1e-4)
    bc_pts=torch.cat([torch.cat([torch.zeros_like(tbv),tbv],1),
                      torch.cat([torch.ones_like(tbv), tbv],1)],0)

    def new_res(N=N_RES, t_min=2e-3, t_max=1.0):
        p=torch.rand(N,2,device=device)
        p[:,1]=p[:,1]*(t_max-t_min)+t_min
        return p.requires_grad_(True)

    def time_weight(t_col):
        """w(t) = 1 + 2t² — upweights late times for β=1.8."""
        if not use_time_weight:
            return torch.ones_like(t_col)
        return 1.0 + 2.0 * t_col**2

    def pde_res(m, pts):
        cap=m.caputo(m,pts[:,0:1],pts[:,1:2],dx)
        u=m(pts,dx)
        r=(cap+lam*u)**2
        if use_time_weight:
            w=time_weight(pts[:,1:2].detach())
            r=r*w
        return torch.mean(r)

    def pinn_loss(m, pts):
        r=pde_res(m,pts); bc=torch.mean(m(bc_pts,dx)**2)
        return W_RES*r+torch.exp(m.log_w_bc)*bc, r, bc

    def rar_resample(m):
        m.eval(); torch.cuda.empty_cache(); gc.collect()
        Ml=8
        with torch.no_grad():
            pts=torch.rand(N_RAR_PROBE,2,device=device); pts[:,1]=pts[:,1].clamp(0.05)
            x_,t_=pts[:,0:1],pts[:,1:2]
            ev=lambda xv,tv: m(torch.cat([xv.clamp(0,1),tv.clamp(1e-3,1)],1),dx)
            k_l=torch.arange(0,Ml,device=device,dtype=torch.float32)
            bl=torch.flip((k_l+1)**(1-ALPHA)-k_l**(1-ALPHA),[0])
            dtl=t_/Ml; idx=torch.arange(0,Ml+1,device=device,dtype=torch.float32)
            tn=dtl*idx.unsqueeze(0); xr=x_.repeat(1,Ml+1).reshape(-1,1)
            un=ev(xr,tn.reshape(-1,1)).reshape(N_RAR_PROBE,Ml+1)
            du=un[:,1:]-un[:,:-1]
            cap=(bl.unsqueeze(0)*du).sum(1,keepdim=True)/(GAMMA2A*dtl**ALPHA)
            u_val=m(torch.cat([x_,t_],1),dx)
            # Relative residual: balances early/late time for stiff β=1.8
            eps_rel = 0.05 * u_val.abs().mean().item() + 1e-6
            mag=((cap+lam*u_val).abs()/(u_val.abs()+eps_rel)).squeeze()
        ki=torch.topk(mag,min(N_RAR_KEEP,N_RAR_PROBE)).indices
        fill=torch.rand(max(0,N_RES-len(ki)),2,device=device); fill[:,1]=fill[:,1].clamp(2e-3)
        out=torch.cat([pts[ki],fill],0)[:N_RES]
        m.train(); torch.cuda.empty_cache()
        return out.detach().requires_grad_(True)

    loss_log=[]; res=new_res()

    # Stage 1
    print(f"\n  [β={beta}] Stage 1/3 Warmup (2500 steps)")
    opt1=Adam(model.parameters(),lr=5e-4)
    sch1=torch.optim.lr_scheduler.OneCycleLR(opt1,max_lr=1e-3,total_steps=2500,
                                              pct_start=0.10,anneal_strategy="cos")
    for step in tqdm(range(2500),desc=f"β={beta} S1"):
        opt1.zero_grad(); loss,r,bc_=pinn_loss(model,res)
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
        opt1.step(); sch1.step(); loss_log.append(loss.item())
        if (step+1)%500==0:
            tqdm.write(f"    [{step+1:5d}] tot={loss.item():.3e} res={r.item():.3e} lr={sch1.get_last_lr()[0]:.1e}")
    torch.cuda.empty_cache()

    # Stage 2
    print(f"\n  [β={beta}] Stage 2/3 CosineAdam+RAR (6000 steps)")
    opt2=Adam(model.parameters(),lr=3e-4)
    sch2=torch.optim.lr_scheduler.CosineAnnealingLR(opt2,T_max=6000,eta_min=5e-7)
    for step in tqdm(range(6000),desc=f"β={beta} S2"):
        if step%1500==0: res=rar_resample(model)
        opt2.zero_grad(); loss,r,bc_=pinn_loss(model,res)
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
        opt2.step(); sch2.step(); loss_log.append(loss.item())
        if (step+1)%1000==0:
            tqdm.write(f"    [{step+1:5d}] tot={loss.item():.3e} res={r.item():.3e} lr={sch2.get_last_lr()[0]:.1e}")
    torch.cuda.empty_cache()

    # Stage 3a: main L-BFGS
    print(f"\n  [β={beta}] Stage 3/3 L-BFGS (1000 iters)")
    res_f=new_res(N=3000)
    opt3=LBFGS(model.parameters(),max_iter=1000,lr=1.0,tolerance_grad=1e-9,
               tolerance_change=1e-11,line_search_fn="strong_wolfe")
    _n=[0]
    def closure():
        opt3.zero_grad(); loss,r,bc_=pinn_loss(model,res_f)
        loss.backward(); _n[0]+=1; loss_log.append(loss.item())
        if _n[0]%200==0: print(f"    L-BFGS [{_n[0]:4d}] tot={loss.item():.3e} res={r.item():.3e}")
        return loss
    opt3.step(closure); print(f"    Done — {_n[0]} evals")

    # Stage 3b: late-time polish (only for β=1.8)
    if beta == 1.8:  # late-time polish always for β=1.8 (not tied to time_weight)
        print(f"\n  [β={beta}] Stage 3b — Late-time polish (500 iters, t∈[0.7,1.0])")
        res_late=new_res(N=3000, t_min=0.7, t_max=1.0)
        opt3b=LBFGS(model.parameters(),max_iter=500,lr=0.1,
                    tolerance_grad=1e-10,tolerance_change=1e-12,
                    line_search_fn="strong_wolfe")
        _n2=[0]
        def closure2():
            opt3b.zero_grad(); loss,r,bc_=pinn_loss(model,res_late)
            loss.backward(); _n2[0]+=1; loss_log.append(loss.item())
            if _n2[0]%100==0:
                print(f"    L-BFGS2 [{_n2[0]:3d}] tot={loss.item():.3e} res={r.item():.3e}")
            return loss
        opt3b.step(closure2); print(f"    Done — {_n2[0]} evals")

    torch.cuda.empty_cache()

    # Evaluation
    xgm=np.linspace(0,1,Nx); tgm=np.linspace(0,1,201)
    itp=RegularGridInterpolator((tg_fdm,xg_fdm),utrue_fdm,method='linear',
                                 bounds_error=False,fill_value=None)
    Tm,Xm=np.meshgrid(tgm,xgm,indexing='ij')
    utm=itp(np.stack([Tm.ravel(),Xm.ravel()],1)).reshape(201,Nx).astype(np.float32)
    uml=compute_ml_2d(ALPHA,lam,tgm,xgm)

    upred=np.zeros((201,Nx),dtype=np.float32)
    model.eval()
    with torch.no_grad():
        for i,tv in enumerate(tgm):
            tt=torch.full((Nx,1),tv,device=device)
            xv=torch.linspace(0,1,Nx).reshape(-1,1).to(device)
            upred[i]=model(torch.cat([xv,tt],1),dx).cpu().numpy().reshape(Nx)

    l2  =float(np.sqrt(np.sum((utm-upred)**2)/(np.sum(utm**2)+1e-14)))
    l1  =float(np.sum(np.abs(utm-upred))/(np.sum(np.abs(utm))+1e-14))
    linf=float(np.abs(utm-upred).max())
    l2ml=float(np.sqrt(np.sum((uml-upred)**2)/(np.sum(uml**2)+1e-14)))

    # Per-snapshot L2
    l2t=[np.sqrt(np.sum((utm[i]-upred[i])**2)/(np.sum(utm[i]**2)+1e-14))
         for i in range(201)]
    idx_t1=200
    print(f"\n  {'='*52}")
    print(f"  α={ALPHA}, β={beta},  λ=π^β={lam:.4f}")
    print(f"  L2(FDM)={l2:.4e}  L1={l1:.4e}  Linf={linf:.4e}")
    print(f"  L2(ML) ={l2ml:.4e}")
    print(f"  L2 at t=1.0: {l2t[idx_t1]:.4e}  (was 5.95e-2 before fix)")
    print(f"  {'='*52}")

    return dict(beta=beta,lam=lam,utm=utm,uml=uml,upred=upred,
                xgm=xgm,tgm=tgm,l2=l2,l1=l1,linf=linf,l2ml=l2ml,
                loss_log=loss_log,T_fdm=T_fdm,tg_fdm=tg_fdm,l2t=l2t)

# ================================================================
# Run
# ================================================================
results={}
for beta in BETA_LIST:
    print(f"\n{'='*55}\n  EXPERIMENT α={ALPHA}, β={beta}\n{'='*55}")
    results[beta]=train_beta(beta)

# Summary
print("\n"+"="*72)
print(f"  β PARAMETRIC STUDY FINAL  |  α={ALPHA}  |  ᶜDₜ^α u=-(-Δ)^(β/2)u")
print("="*72)
print(f"  {'β':>5}  {'λ':>8}  {'L2(FDM)':>12}  {'L1':>12}  {'L2(ML)':>12}  {'L2(t=1)':>10}")
print(f"  {'-'*5}  {'-'*8}  {'-'*12}  {'-'*12}  {'-'*12}  {'-'*10}")
prev=dict(l2=6.9756e-3,l1=6.2818e-3,l2ml=7.8383e-3,linf=1.8728e-2,
          l2t=[None]*201)
for b,d in [(1.2,results[1.2]),(1.5,prev),(1.8,results[1.8])]:
    tag=" ← prev" if b==1.5 else ""
    l2t1=d['l2t'][200] if d['l2t'][200] else "~8e-3"
    print(f"  {b:>5.1f}  {np.pi**b:>8.4f}  {d['l2']:>12.4e}  {d['l1']:>12.4e}  "
          f"{d['l2ml']:>12.4e}  {str(l2t1):>10}{tag}")
print("="*72)

# ================================================================
# Plots
# ================================================================
plt.rcParams.update({
    'font.family':'DejaVu Sans','font.size':12,'axes.titlesize':13,
    'axes.labelsize':12,'axes.spines.top':False,'axes.spines.right':False,
    'figure.facecolor':'white','axes.facecolor':'white','savefig.facecolor':'white',
})
LINE_REF='#1a1a2e'; LINE_ML='#0077b6'
COLOR_12='#7b2d8b'; COLOR_15='#e63946'; COLOR_18='#2dc653'
CMAP_SOL='RdYlBu_r'; CMAP_ERR='YlOrRd'

# A: Snapshots 2×4
fig,axes=plt.subplots(2,4,figsize=(18,8))
for row,(beta,col) in enumerate([(1.2,COLOR_12),(1.8,COLOR_18)]):
    d=results[beta]
    for c,frac in enumerate([0.25,0.50,0.75,1.00]):
        ax=axes[row,c]; idx=int(frac*200)
        ax.fill_between(d['xgm'],d['utm'][idx],alpha=0.10,color=LINE_REF)
        ax.plot(d['xgm'],d['utm'][idx],color=LINE_REF,lw=2.8,ls='-',label='L1-FDM')
        ax.plot(d['xgm'],d['uml'][idx],color=LINE_ML, lw=2.0,ls=':',label='ML exact')
        ax.plot(d['xgm'],d['upred'][idx],color=col,   lw=2.4,ls='--',label='FracFormer')
        ax.set_title(f'$t={d["tgm"][idx]:.2f}$',fontweight='bold',pad=4)
        ax.set_xlabel('$x$')
        if c==0: ax.set_ylabel(f'$\\beta={beta}$\n$u(x,t)$',fontsize=11)
        else:    ax.set_ylabel('$u(x,t)$')
        if row==0 and c==3: ax.legend(fontsize=8,framealpha=0.85)
        l2s=d['l2t'][idx]
        ax.text(0.03,0.06,f'$L_2={l2s:.2e}$',transform=ax.transAxes,
                fontsize=8,color=col,bbox=dict(fc='white',ec='none',alpha=0.7))
plt.suptitle(
    r'FracFormer-PINN $\;|\;$ ${}^C\!D_t^{0.7}u=-(-\Delta)^{\beta/2}u$  $\alpha=0.7$'
    '\nTop: $\\beta=1.2$ (near-Lévy)   Bottom: $\\beta=1.8$ (near-Laplacian)',
    fontsize=13,y=1.01)
plt.tight_layout()
plt.savefig('beta_study_snapshots.png',dpi=600,bbox_inches='tight')
plt.show(); print("Saved: beta_study_snapshots.png")

# B: Heatmaps 2×3
fig2,axes2=plt.subplots(2,3,figsize=(16,8))
for row,(beta,col) in enumerate([(1.2,COLOR_12),(1.8,COLOR_18)]):
    d=results[beta]; aerr=np.abs(d['utm']-d['upred'])
    for c,(data,title,cm) in enumerate([
            (d['utm'],   f'L1-FDM  β={beta}',                     CMAP_SOL),
            (d['upred'], f'FracFormer  β={beta}  L2={d["l2"]:.2e}',CMAP_SOL),
            (aerr,       f'Abs Error  β={beta}',                   CMAP_ERR)]):
        ax=axes2[row,c]
        im=ax.contourf(d['xgm'],d['tgm'],data,levels=60,cmap=cm)
        ax.contour(d['xgm'],d['tgm'],data,levels=8,colors='white',linewidths=0.35,alpha=0.5)
        div=make_axes_locatable(ax); cax=div.append_axes('right',size='4%',pad=0.06)
        plt.colorbar(im,cax=cax).ax.tick_params(labelsize=8)
        ax.set_title(title,fontweight='bold',pad=5); ax.set_xlabel('$x$'); ax.set_ylabel('$t$')
plt.suptitle(r'Heatmaps $\;|\;$ $\alpha=0.7$  varying $\beta$',fontsize=13,y=1.01)
plt.tight_layout()
plt.savefig('beta_study_heatmaps.png',dpi=600,bbox_inches='tight')
plt.show(); print("Saved: beta_study_heatmaps.png")

# C: Modal decay all three β — always interp to 201 pts
fig3,ax3=plt.subplots(figsize=(10,5))
tgm_ref=np.linspace(0,1,201)
for beta,col in [(1.2,COLOR_12),(1.5,COLOR_15),(1.8,COLOR_18)]:
    lam_b=np.pi**beta
    Nt=1000; tg_b=np.linspace(0,1,Nt); dt_b=tg_b[1]-tg_b[0]
    b_b=np.array([(k+1)**(1-ALPHA)-k**(1-ALPHA) for k in range(Nt+1)])
    csc_b=dt_b**(-ALPHA)/GAMMA2A; T_b=np.zeros(Nt); T_b[0]=1.
    for n in range(1,Nt):
        hist=sum((b_b[k-1]-b_b[k])*T_b[n-k] for k in range(1,n))+b_b[n-1]*T_b[0]
        T_b[n]=(csc_b*hist)/(csc_b*b_b[0]+lam_b)
    # FIX: always interp to 201 pts (no conditional)
    T_fdm_201=np.interp(tgm_ref, tg_b, T_b)
    T_ml_201=np.array([1.0 if t==0 else ml_robust(ALPHA,-lam_b*t**ALPHA) for t in tgm_ref])
    ax3.plot(tgm_ref,T_fdm_201,color=col,lw=2.5,ls='-',alpha=0.45)
    ax3.plot(tgm_ref,T_ml_201, color=col,lw=1.8,ls=':',alpha=0.9)
    if beta in results:
        T_pinn=results[beta]['upred'][:,50]/np.sin(np.pi*0.5)
        ax3.plot(results[beta]['tgm'],T_pinn,color=col,lw=2.4,ls='--',
                 label=f'β={beta}  λ={lam_b:.2f}  L2={results[beta]["l2"]:.2e}')
    else:
        ax3.plot(tgm_ref,T_fdm_201,color=col,lw=2.2,ls='--',alpha=0.5,
                 label=f'β=1.5  L2=6.98e-3 [prev]')
ax3.set_xlabel('$t$'); ax3.set_ylabel(r'$E_\alpha(-\lambda t^\alpha)$')
ax3.set_title(r'Decay Curves $E_\alpha(-\lambda t^\alpha)$  |  $\alpha=0.7$'
              '\nSolid=FDM  Dotted=ML  Dashed=FracFormer-PINN',fontweight='bold')
ax3.legend(fontsize=10,framealpha=0.9); plt.tight_layout()
plt.savefig('beta_study_modal.png',dpi=600,bbox_inches='tight')
plt.show(); print("Saved: beta_study_modal.png")

# D: L2 over time
fig4,ax4=plt.subplots(figsize=(10,4))
for beta,col in [(1.2,COLOR_12),(1.8,COLOR_18)]:
    d=results[beta]
    ax4.fill_between(d['tgm'],d['l2t'],alpha=0.15,color=col)
    ax4.semilogy(d['tgm'],d['l2t'],lw=2.2,color=col,
                 label=f'β={beta}  global L2={d["l2"]:.2e}')
ax4.axhline(6.9756e-3,color=COLOR_15,ls='--',lw=1.8,label='β=1.5 L2=6.98e-3 [prev]')
ax4.axhline(1e-2,color='gray',ls=':',lw=1,alpha=0.6)
ax4.set_xlabel('$t$'); ax4.set_ylabel('Relative $L_2$ error')
ax4.set_title(r'Error over time  |  $\alpha=0.7$',fontweight='bold')
ax4.legend(fontsize=10,framealpha=0.85); plt.tight_layout()
plt.savefig('beta_study_l2time.png',dpi=600,bbox_inches='tight')
plt.show(); print("Saved: beta_study_l2time.png")

# E: Loss
fig5,axes5=plt.subplots(1,2,figsize=(14,4))
for ax,(beta,col) in zip(axes5,[(1.2,COLOR_12),(1.8,COLOR_18)]):
    ll=results[beta]['loss_log']
    ax.fill_between(range(len(ll)),ll,alpha=0.15,color=col)
    ax.semilogy(ll,lw=1.5,color=col)
    ax.axvline(2500,color='gray',ls='--',lw=1.5,alpha=0.7,label='Stage 1 end')
    ax.axvline(8500,color='gray',ls=':',lw=1.5,alpha=0.7,label='Stage 2 end')
    ax.set_title(f'β={beta}  L2={results[beta]["l2"]:.2e}  L2(t=1)={results[beta]["l2t"][200]:.2e}',
                 fontweight='bold')
    ax.set_xlabel('Step'); ax.set_ylabel('Loss'); ax.legend(fontsize=9)
plt.suptitle(r'Training Loss  |  $\alpha=0.7$',fontsize=13,y=1.01)
plt.tight_layout()
plt.savefig('beta_study_loss.png',dpi=600,bbox_inches='tight')
plt.show(); print("Saved: beta_study_loss.png")

# F: 3D
fig6=plt.figure(figsize=(16,9))
for row,(beta,col) in enumerate([(1.2,COLOR_12),(1.8,COLOR_18)]):
    d=results[beta]; aerr=np.abs(d['utm']-d['upred'])
    Tm6,Xm6=np.meshgrid(d['tgm'],d['xgm'])
    for c,(data,title,cm) in enumerate([
            (d['utm'].T,  f'L1-FDM  β={beta}',                    'RdYlBu_r'),
            (d['upred'].T,f'FracFormer β={beta} L2={d["l2"]:.2e}','RdYlBu_r'),
            (aerr.T,      f'Abs Error  β={beta}',                  'YlOrRd')]):
        ax=fig6.add_subplot(2,3,row*3+c+1,projection='3d')
        surf=ax.plot_surface(Xm6,Tm6,data,cmap=cm,alpha=0.92,linewidth=0)
        fig6.colorbar(surf,ax=ax,shrink=0.5,aspect=8,pad=0.08)
        ax.set_title(title,fontweight='bold',pad=6,fontsize=10)
        ax.set_xlabel('$x$',labelpad=4); ax.set_ylabel('$t$',labelpad=4)
        ax.tick_params(labelsize=7)
        ax.xaxis.pane.fill=False; ax.yaxis.pane.fill=False; ax.zaxis.pane.fill=False
plt.suptitle(r'3D Surfaces $\;|\;$ $\alpha=0.7$',fontsize=13,y=1.01)
plt.tight_layout()
plt.savefig('beta_study_3d.png',dpi=600,bbox_inches='tight')
plt.show(); print("Saved: beta_study_3d.png")

# G: t=1.0 comparison
fig7,ax7=plt.subplots(figsize=(10,4))
xg_ref=np.linspace(0,1,101)
for beta,col,ls in [(1.2,COLOR_12,'-'),(1.5,COLOR_15,'--'),(1.8,COLOR_18,':')]:
    lam_b=np.pi**beta
    if beta in results:
        d=results[beta]
        ax7.plot(d['xgm'],d['utm'][-1],color=col,lw=1.8,ls=ls,alpha=0.4)
        ax7.plot(d['xgm'],d['upred'][-1],color=col,lw=2.5,ls=ls,
                 label=f'β={beta}  L2(t=1)={d["l2t"][200]:.2e}')
    else:
        Nt=1000; tg_b=np.linspace(0,1,Nt); dt_b=tg_b[1]-tg_b[0]
        b_b=np.array([(k+1)**(1-ALPHA)-k**(1-ALPHA) for k in range(Nt+1)])
        csc_b=dt_b**(-ALPHA)/GAMMA2A; T_b=np.zeros(Nt); T_b[0]=1.
        for n in range(1,Nt):
            hist=sum((b_b[k-1]-b_b[k])*T_b[n-k] for k in range(1,n))+b_b[n-1]*T_b[0]
            T_b[n]=(csc_b*hist)/(csc_b*b_b[0]+lam_b)
        ax7.plot(xg_ref,T_b[-1]*np.sin(np.pi*xg_ref),color=col,lw=2.5,ls=ls,
                 label=f'β=1.5  [prev, FDM]')
ax7.set_xlabel('$x$'); ax7.set_ylabel('$u(x,t=1)$')
ax7.set_title(r'Solution at $t=1.0$  |  $\alpha=0.7$',fontweight='bold')
ax7.legend(fontsize=9,framealpha=0.85); plt.tight_layout()
plt.savefig('beta_study_t1.png',dpi=600,bbox_inches='tight')
plt.show(); print("Saved: beta_study_t1.png")

print("\n"+"="*72)
print("  FINAL SUMMARY")
for b,d in [(1.2,results[1.2]),(1.5,dict(l2=6.9756e-3,l2ml=7.8383e-3,
              l2t=[None]*201)),(1.8,results[1.8])]:
    t1=f"{d['l2t'][200]:.3e}" if d['l2t'][200] else "~8e-3"
    tag=" (prev)" if b==1.5 else ""
    print(f"  β={b}: L2={d['l2']:.4e}  L2(ML)={d['l2ml']:.4e}  L2(t=1)={t1}{tag}")
print("="*72)
print("All done.")
