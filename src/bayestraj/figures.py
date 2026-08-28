"""Regenerate Figures 1--9 of the BayesTraj Aug. 21 submission."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd

from .paper import ADAPTIVE, BUDGETS, CONFIG, DATA, FIXED, OURS, combination_metrics, macro_metrics, read, seed_metrics


DISPLAY = {**CONFIG["paper_names"], "UProp": "UProp", "Degree": "Degree (N=4)"}
METHODS = (FIXED, ADAPTIVE, *CONFIG["baselines"])
COLORS = {
    FIXED: "#004488", ADAPTIVE: "#CC3311", "SNNE": "#56B4E9", "MC-OE": "#332288",
    "BSE-Ciosek-Fixed": "#009E73", "BSE-Ciosek-Adaptive": "#117733", "EigV": "#777777",
    "CoCoA-MaxProb": "#A6761D", "CoCoA-PPL": "#66A61E", "KLE": "#00A087",
    "SAUP": "#00A6D6", "PE": "#006D2C", "SentSAR": "#2B8CBE", "Degree": "#56B4E9",
    "UProp": "#E69F00", "SE": "#807D00", "LS": "#C51B7D", "SD": "#8C510A", "PPL": "#8C2D4A",
}
MARKERS = {method: marker for method, marker in zip(METHODS, "pPh>^>x<>^*^XvPD8vo")}
DATASET_NAMES = {"dbbench": "DBBench", "hotpotqa": "HotpotQA", "webshop": "WebShop", "strategyqa": "StrategyQA"}
BACKBONE_NAMES = {"qwen35": "Qwen-3.5 9B", "gemma3": "Gemma-3 12B", "gptoss20b": "GPT-OSS 20B"}
PP = FuncFormatter(lambda value, _: f"{value:+.2f}%" if value else "0.00%")


def _save(figure: plt.Figure, output: Path, stem: str, *, dpi: int = 300) -> None:
    output.mkdir(parents=True, exist_ok=True)
    figure.savefig(output / f"{stem}.pdf", bbox_inches="tight", pad_inches=.02)
    figure.savefig(output / f"{stem}.png", dpi=dpi, bbox_inches="tight", pad_inches=.02)
    plt.close(figure)


def _with_baseline_b2() -> pd.DataFrame:
    main = seed_metrics()
    b2 = pd.read_csv(DATA / "baseline_b2_seed_metrics.csv")
    return pd.concat([main, b2], ignore_index=True, sort=False)


def _draw_curves(axis: plt.Axes, rows: pd.DataFrame, metric: str, *, compact: bool = False) -> None:
    for method in METHODS:
        block = rows[rows.method.eq(method)].sort_values("budget")
        if block.empty:
            continue
        x = block.mean_trajectories.to_numpy(float)
        y = block[f"{metric}_mean"].to_numpy(float)
        sd = block[f"{metric}_std"].fillna(0).to_numpy(float)
        ours = method in OURS
        axis.plot(
            x, y, marker=MARKERS[method], color=COLORS[method],
            linestyle="-" if ours else ":", linewidth=1.7 if ours else .85,
            markersize=3.6 if ours else 2.4, label=DISPLAY.get(method, method),
            markerfacecolor="white" if method == ADAPTIVE else COLORS[method], zorder=5 if ours else 2,
        )
        axis.fill_between(x, np.clip(y - sd, 0, 1), np.clip(y + sd, 0, 1),
                          color=COLORS[method], alpha=.09 if ours else .045, linewidth=0)
    axis.grid(alpha=.18, linewidth=.45)
    axis.set_xticks((2, *BUDGETS))
    axis.tick_params(labelsize=6.4 if compact else 7)


def figure01(output: Path) -> None:
    navy, text = "#17365D", "#1F2937"
    figure, axis = plt.subplots(figsize=(7.16, 4.05)); axis.set(xlim=(0, 1), ylim=(0, 1)); axis.axis("off")
    def box(x, y, w, h, title, detail, fill, edge):
        axis.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=.006,rounding_size=.018", facecolor=fill, edgecolor=edge, linewidth=1.1))
        axis.text(x+w/2, y+h*.64, title, ha="center", va="center", fontsize=8, fontweight="bold", color=text)
        axis.text(x+w/2, y+h*.29, detail, ha="center", va="center", fontsize=6.3, color=text)
    def arrow(x1, y1, x2, y2):
        axis.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle="-|>",mutation_scale=9,color=navy,linewidth=1.1))
    axis.add_patch(FancyBboxPatch((.012,.69),.976,.285,boxstyle="round,pad=.006",facecolor="#F7FAFC",edgecolor="#AAB7C4"))
    axis.add_patch(FancyBboxPatch((.012,.045),.976,.605,boxstyle="round,pad=.006",facecolor="white",edgecolor="#AAB7C4"))
    axis.text(.027,.944,"A. Cross-fitted learning and calibration",fontsize=9.4,fontweight="bold",color=navy)
    axis.text(.027,.62,"B. Held-out task inference",fontsize=9.4,fontweight="bold",color=navy)
    box(.035,.735,.165,.145,"Training pools","16 trajectories per task","#F2F2F2","#8A8A8A")
    box(.258,.735,.185,.145,"Label-free target",r"$H_i=\mathrm{OE}_{16,i}$","#DDEBF7","#5B9BD5")
    box(.50,.735,.205,.145,"Linear-Gaussian fit",r"$Z_n\mid H\sim\mathcal{N}(a_n+b_nH,\Sigma_n)$","#E4DFEC","#8064A2")
    box(.762,.735,.195,.145,"Adaptive calibration",r"$\sigma_n^2\rightarrow\tau_B$ at $\rho=.80$","#FCE4D6","#ED7D31")
    for a in ((.2,.807,.258,.807),(.443,.807,.5,.807),(.705,.807,.762,.807)): arrow(*a)
    box(.035,.345,.135,.15,"Trajectory prefix",r"$\tau_{i,1:n}$","#F2F2F2","#8A8A8A")
    box(.225,.485,.18,.115,"Outcome buckets",r"counts $c_{i,n}$","#DDEBF7","#5B9BD5")
    box(.225,.28,.18,.115,"Trajectory features","10 surprisal/length summaries","#E4DFEC","#8064A2")
    box(.455,.485,.17,.115,"Count prior",r"Dirichlet $\to p_0(H\mid c)$","#DDEBF7","#5B9BD5")
    box(.455,.28,.17,.115,"Trajectory likelihood",r"$p(Z_n\mid H)$","#E4DFEC","#8064A2")
    box(.675,.365,.145,.155,"Posterior fusion",r"257-point grid; $p(H\mid c,Z)$","#E2F0D9","#70AD47")
    box(.855,.365,.115,.155,"Score + variance",r"$S_n=\mu_n-1.96\sigma_n$","#E2F0D9","#70AD47")
    for a in ((.17,.435,.225,.542),(.17,.405,.225,.337),(.405,.542,.455,.542),(.405,.337,.455,.337),(.625,.542,.675,.472),(.625,.337,.675,.408),(.82,.442,.855,.442)): arrow(*a)
    box(.55,.095,.18,.105,"BayesTraj-Fixed",r"stop at $T_i=B$","#E2F0D9","#70AD47")
    box(.785,.095,.185,.105,"BayesTraj-Adaptive",r"first $\sigma^2\leq\tau_B$; else $B$","#FCE4D6","#ED7D31")
    arrow(.912,.365,.64,.2); arrow(.92,.365,.877,.2)
    axis.text(.035,.075,"Correctness labels are opened only after held-out scores and stopping times are frozen.",fontsize=6.8,color="#7A1F1F",fontweight="bold",bbox=dict(boxstyle="round,pad=.3",facecolor="#F4CCCC",edgecolor="#C0504D"))
    _save(figure, output, "fig01_method_overview")


def figure02(output: Path) -> None:
    macro = macro_metrics(_with_baseline_b2())
    figure, axes = plt.subplots(1, 2, figsize=(7.0, 2.8))
    for axis, metric in zip(axes, ("auroc", "aupr")):
        _draw_curves(axis, macro, metric)
        axis.set_title(metric.upper(), fontweight="bold", fontsize=10)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=5, frameon=False, fontsize=5.6, bbox_to_anchor=(.5,-.02))
    figure.supxlabel("Mean trajectories used", fontsize=8, y=.22)
    figure.subplots_adjust(left=.08,right=.995,top=.92,bottom=.34,wspace=.22)
    _save(figure, output, "fig02_macro_cost_performance")


def figure03(output: Path) -> None:
    rows = read("paired_superiority_summary.csv").set_index(["method","baseline"])
    order=[("BayesTraj-Fixed","CoCoA-PPL",6),("BayesTraj-Fixed","BSE-Fixed",5),("BayesTraj-Fixed","MC-OE",4),("BayesTraj-Adaptive","CoCoA-PPL",2.5),("BayesTraj-Adaptive","BSE-Fixed",1.5),("BayesTraj-Adaptive","MC-OE",.5)]
    figure, axes=plt.subplots(1,2,figsize=(3.55,2.55),sharey=True)
    for axis, metric in zip(axes,("auroc","aupr")):
        axis.axvline(0,color="#333",lw=.8); axis.axvspan(0,8.2,color="#E9F4EC",alpha=.65)
        for method, baseline, y in order:
            row=rows.loc[(method,baseline)]; point=100*row[f"{metric}_delta"]; low=100*row[f"{metric}_ci_low"]; high=100*row[f"{metric}_ci_high"]
            fixed=method.endswith("Fixed")
            color="#174A7E" if fixed else "#D14900"
            axis.errorbar(point,y,xerr=[[point-low],[high-point]],fmt="o" if fixed else "s",mfc=color if fixed else "white",color=color,capsize=2,markersize=4,lw=1.1)
        axis.set_title(f"{metric.upper()} improvement\n(percentage points)",fontsize=7.5,fontweight="bold")
        axis.set_xlim(-.35,8.2); axis.set_xticks([0,2,4,6,8],["0%","+2%","+4%","+6%","+8%"],fontsize=6); axis.grid(axis="x",alpha=.2)
    axes[0].set_yticks([x[2] for x in order],[x[1] for x in order],fontsize=6.5)
    figure.subplots_adjust(left=.24,right=.99,top=.84,bottom=.15,wspace=.14)
    _save(figure, output, "fig03_paired_improvements")


def figure04(output: Path) -> None:
    combination=combination_metrics(_with_baseline_b2())
    figure,axes=plt.subplots(4,3,figsize=(7.16,8.25))
    for axis,(dataset,backbone) in zip(axes.flat,((d,b) for d in CONFIG["datasets"] for b in CONFIG["backbones"])):
        block=combination[(combination.dataset.eq(dataset))&(combination.backbone.eq(backbone))]
        _draw_curves(axis,block,"auroc",compact=True)
        axis.set_title(f"{DATASET_NAMES[dataset]} / {BACKBONE_NAMES[backbone]}",fontsize=7.3,fontweight="bold")
    for row in range(4): axes[row,0].set_ylabel("AUROC",fontsize=7)
    for col in range(3): axes[-1,col].set_xlabel("Mean trajectories used",fontsize=7)
    handles,labels=axes.flat[0].get_legend_handles_labels()
    figure.legend(handles,labels,loc="lower center",ncol=5,frameon=False,fontsize=5.5,bbox_to_anchor=(.5,.005))
    figure.subplots_adjust(left=.075,right=.995,top=.975,bottom=.105,hspace=.30,wspace=.16)
    _save(figure, output, "fig04_dataset_backbone_auroc")


def figure05(output: Path) -> None:
    data=read("efficiency_by_budget.csv")
    fixed=data[data.method.eq("BayesTraj-Fixed")].sort_values("budget")
    adaptive=data[data.method.eq("BayesTraj-Adaptive")].sort_values("budget")
    budgets=adaptive.budget.to_numpy(int)
    figure,axes=plt.subplots(1,2,figsize=(3.55,2.55))
    left,right=axes
    left.axhspan(-1,0,color="#E4F2E7",alpha=.9); left.axhline(0,color="#444",lw=.8); left.axhline(-1,color="#777",ls="--",lw=.7)
    for metric,color,marker in (("auroc","#174A7E","o"),("aupr","#D14900","s")):
        delta=100*(adaptive[metric].to_numpy()-fixed[metric].to_numpy())
        left.plot(budgets,delta,color=color,marker=marker,lw=1.35,ms=3.7,label=f"{metric.upper()} Adaptive − Fixed")
    left.set_title("(a) Ranking retention",fontsize=8,fontweight="bold"); left.set_ylabel("Adaptive − Fixed",fontsize=7)
    left.set_yticks([-1,-.5,0],["−1.0%","−0.5%","0.0%"]); left.set_ylim(-1.4,.15)
    for column,label,color,marker in (("trajectory_saving","Trajectories","#174A7E","o"),("agent_step_saving","Agent steps","#00897B","s"),("output_token_saving","Output tokens","#D14900","^")):
        right.plot(budgets,100*adaptive[column],color=color,marker=marker,lw=1.35,ms=3.7,label=label)
    right.axhline(20,color="#777",ls="--",lw=.7); right.set_ylim(0,22.5); right.set_title("(b) Compute savings",fontsize=8,fontweight="bold"); right.set_ylabel("Saving (%)",fontsize=7)
    for axis in axes:
        axis.set_xticks(budgets); axis.tick_params(labelsize=6); axis.grid(axis="y",alpha=.2); axis.set_xlabel(r"Trajectory budget $B$",fontsize=7)
        axis.legend(frameon=False,fontsize=5.2)
    figure.subplots_adjust(left=.13,right=.99,top=.90,bottom=.22,wspace=.35)
    _save(figure, output, "fig05_adaptive_efficiency")


def figure06(output: Path) -> None:
    data=read("representative_budget_metrics.csv")
    adaptive=data[data.display.eq("BayesTraj-Adaptive")].iloc[0]
    baselines=data[~data.display.str.startswith("BayesTraj")].copy()
    baselines["auroc_gain"]=100*(adaptive.auroc-baselines.auroc)
    baselines["aupr_gain"]=100*(adaptive.aupr-baselines.aupr)
    baselines=baselines.sort_values("auroc_gain")
    y=np.arange(len(baselines)); figure,axes=plt.subplots(1,2,figsize=(3.55,2.25),sharey=True)
    for axis,column,title,color in ((axes[0],"auroc_gain","AUROC gain","#0B559F"),(axes[1],"aupr_gain","AUPR gain","#D94801")):
        values=baselines[column].to_numpy(); axis.barh(y,values,color=color,alpha=.88)
        for position,value in zip(y,values): axis.text(value+.08,position,f"+{value:.1f} pp",va="center",fontsize=5.2)
        axis.set_title(title,fontsize=7.2,fontweight="bold"); axis.set_xlabel("Adaptive − baseline (pp)",fontsize=5.8); axis.tick_params(labelsize=5.5); axis.grid(axis="x",alpha=.2); axis.set_xlim(0,max(values)*1.35)
    axes[0].set_yticks(y,baselines.display,fontsize=5.5)
    figure.text(.5,.02,f"BayesTraj-Adaptive uses {adaptive.trajectories:.2f}/8 trajectories ({100*adaptive.saving:.1f}% saving)",ha="center",fontsize=6.2,fontweight="bold")
    figure.subplots_adjust(left=.25,right=.97,top=.90,bottom=.24,wspace=.20)
    _save(figure, output, "fig06_lower_cost_gain")


def figure07(output: Path) -> None:
    data=read("core_mechanism_tradeoff.csv")
    labels=["Trajectory update","Count-prior fusion","Full covariance","Adaptive stopping","Task-adaptive allocation"]
    positions=np.arange(len(data))[::-1]
    figure,(performance,saving)=plt.subplots(2,1,figsize=(3.5,3.0),sharey=True,gridspec_kw={"height_ratios":(1.45,1),"hspace":.30})
    auroc=100*data.delta_auroc.to_numpy(); aupr=100*data.delta_aupr.to_numpy(); height=.28
    performance.barh(positions+height/2,auroc,height=height,color="#0072B2",label="AUROC")
    performance.barh(positions-height/2,aupr,height=height,color="#D55E00",label="AUPR")
    performance.axvline(0,color="#344054",lw=.65); performance.set_title("(a) Predictive performance",fontsize=7.2,fontweight="bold"); performance.set_xlabel("Full − ablation (percentage points)",fontsize=6)
    performance.legend(loc="lower right",frameon=False,fontsize=5.2,ncol=2)
    for y,a,p in zip(positions,auroc,aupr): performance.text(max(a,.02)+.08,y+height/2,f"{a:+.2f}",va="center",fontsize=4.7); performance.text(max(p,.02)+.08,y-height/2,f"{p:+.2f}",va="center",fontsize=4.7)
    savings=100*data.saving_delta.to_numpy(); saving.barh(positions,savings,height=.48,color=["#087F78" if x>=0 else "#C44E52" for x in savings]); saving.axvline(0,color="#344054",lw=.65)
    saving.set_title("(b) Sampling efficiency",fontsize=7.2,fontweight="bold"); saving.set_xlabel("Δ trajectory saving (percentage points)",fontsize=6)
    for y,value in zip(positions,savings): saving.text(max(value,.02)+.15,y,f"{value:+.2f}",va="center",fontsize=4.8)
    for axis in (performance,saving): axis.set_yticks(positions,labels,fontsize=5.4); axis.tick_params(axis="x",labelsize=5.2); axis.grid(axis="x",alpha=.18); axis.spines[["top","right","left"]].set_visible(False)
    performance.set_xlim(-1,6.3); saving.set_xlim(-1.2,20.8); figure.subplots_adjust(left=.39,right=.985,top=.965,bottom=.12)
    _save(figure, output, "fig07_core_ablation",dpi=400)


def _window_intervals(cell: pd.DataFrame, replicates: int = 5000) -> dict[tuple[str,float],tuple[float,float,float]]:
    rng=np.random.default_rng(20260813); result={}
    for window in ("2","6","all"):
        for rho in sorted(cell.rho.unique()):
            candidate=cell[(cell.window.astype(str)==window)&np.isclose(cell.rho,rho)].set_index("cell")
            reference=cell[(cell.window.astype(str)=="4")&np.isclose(cell.rho,rho)].set_index("cell")
            joined=candidate[["dataset","backbone","seed","delta_auroc"]].join(reference[["delta_auroc"]],rsuffix="_w4")
            joined["effect"]=joined.delta_auroc-joined.delta_auroc_w4
            arrays=np.asarray([joined[(joined.dataset==d)&(joined.backbone==b)].sort_values("seed").effect.to_numpy() for d in CONFIG["datasets"] for b in CONFIG["backbones"]])
            ci=np.empty(replicates)
            for r in range(replicates):
                combos=rng.integers(0,len(arrays),len(arrays)); seeds=rng.integers(0,3,(len(arrays),3)); ci[r]=arrays[combos[:,None],seeds].mean()
            result[(window,float(rho))]=(float(arrays.mean()),float(np.quantile(ci,.025)),float(np.quantile(ci,.975)))
    return result


def figure08(output: Path) -> None:
    summary=read("sensitivity_summary.csv"); cell=read("paired_cell_summary.csv"); colors={"2":"#0072B2","4":"#D55E00","6":"#009E73","all":"#6A3D9A"}
    figure,(front,effect)=plt.subplots(1,2,figsize=(3.5,2.05),gridspec_kw={"width_ratios":(1.08,1)})
    front.axhspan(-1,0,color="#E8F4EC",alpha=.9); front.axhline(0,color="#303642",lw=.8); front.axhline(-1,color="#73808C",ls="--",lw=.7)
    for window in colors:
        block=summary[summary.window.astype(str)==window].sort_values("saving"); x=100*block.saving; y=100*block.delta_auroc
        front.plot(x,y,"o-",color=colors[window],lw=.9,ms=2.4,label=f"w={window}"); front.vlines(x,100*block.delta_auroc_ci_low,100*block.delta_auroc_ci_high,color=colors[window],alpha=.38,lw=.45)
    default=summary[(summary.window.astype(str)=="4")&np.isclose(summary.rho,.8)].iloc[0]; front.scatter(100*default.saving,100*default.delta_auroc,marker="*",s=45,c="#FFD23F",edgecolor="#222",zorder=7)
    front.set(title="(a) Cost–performance",xlabel="Realized saving (%)",ylabel="ΔAUROC vs. fixed (pp)"); front.yaxis.set_major_formatter(PP); front.legend(frameon=False,fontsize=4.4,ncol=2)
    intervals=_window_intervals(cell); effect.axhline(0,color="#303642",lw=.8)
    for window in ("2","6","all"):
        ratios=sorted(cell.rho.unique()); triples=[intervals[(window,float(r))] for r in ratios]; y=100*np.asarray([x[0] for x in triples]); low=100*np.asarray([x[1] for x in triples]); high=100*np.asarray([x[2] for x in triples])
        effect.plot(ratios,y,"o-",color=colors[window],lw=.9,ms=2.4,label=f"w={window}"); effect.fill_between(ratios,low,high,color=colors[window],alpha=.08)
    effect.scatter(.8,0,marker="*",s=45,c="#FFD23F",edgecolor="#222",zorder=7); effect.set(title="(b) Window sensitivity",xlabel=r"Target cost ratio $\rho$",ylabel="ΔAUROC vs. w=4 (pp)"); effect.yaxis.set_major_formatter(PP); effect.legend(frameon=False,fontsize=4.4)
    for axis in (front,effect): axis.title.set_fontsize(6.5); axis.title.set_fontweight("bold"); axis.xaxis.label.set_fontsize(5.8); axis.yaxis.label.set_fontsize(5.8); axis.tick_params(labelsize=5.1); axis.grid(alpha=.16,lw=.35); axis.spines[["top","right"]].set_visible(False)
    figure.text(.5,.055,r"★ default ($\rho=.80,w=4$); bars/ribbons: hierarchical 95% intervals",ha="center",fontsize=5.2,fontweight="bold"); figure.subplots_adjust(left=.12,right=.995,top=.92,bottom=.24,wspace=.42)
    _save(figure, output, "fig08_sensitivity",dpi=400)


def figure09(output: Path) -> None:
    cell=read("posterior_diagnostics_by_cell_budget.csv"); calibration=read("posterior_risk_calibration.csv")
    summary=cell.groupby("budget",as_index=False).agg(count=("count_mse","mean"),count_sd=("count_mse","std"),fused=("fused_mse","mean"),fused_sd=("fused_mse","std"))
    figure,axes=plt.subplots(1,2,figsize=(7,3.15)); left,right=axes; b=summary.budget.to_numpy()
    left.plot(b,summary["count"],"o--",color="#999999",lw=2.2,label="Count only"); left.fill_between(b,summary["count"]-summary.count_sd,summary["count"]+summary.count_sd,color="#999999",alpha=.16)
    left.plot(b,summary.fused,"o-",color="#00796B",lw=2.7,label="Two-view fusion"); left.fill_between(b,summary.fused-summary.fused_sd,summary.fused+summary.fused_sd,color="#00796B",alpha=.16)
    left.set(title="(a) Target estimation",xlabel="Trajectory prefix",ylabel=r"Held-out $H$ MSE"); left.set_xticks(BUDGETS); left.legend(frameon=False)
    x=calibration.predicted.to_numpy(); y=calibration.observed.to_numpy(); low=calibration.observed_ci_low.to_numpy(); high=calibration.observed_ci_high.to_numpy(); upper=max(x.max(),high.max())*1.05
    right.plot([0,upper],[0,upper],"--",color="#777",lw=1.7,label="Ideal"); right.errorbar(x,y,yerr=[y-low,high-y],fmt="o-",color="#CC4C02",lw=2.3,capsize=3,label="BayesTraj")
    right.set(title="(b) Posterior-risk calibration",xlabel="Predicted variance",ylabel="Observed squared error",xlim=(0,None),ylim=(0,None)); right.legend(frameon=False)
    for axis in axes: axis.grid(alpha=.22)
    figure.subplots_adjust(left=.105,right=.985,top=.91,bottom=.20,wspace=.36)
    _save(figure, output, "fig09_posterior_validation")


def generate_all(output: Path) -> list[Path]:
    for function in (figure01,figure02,figure03,figure04,figure05,figure06,figure07,figure08,figure09): function(output)
    return sorted(output.glob("fig*.pdf"))
