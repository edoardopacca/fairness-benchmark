"""Aggregate the completed grid and generate tables/plots; no model selection from test."""
import tempfile
import argparse
import hashlib
import json
import os
from pathlib import Path
os.environ.setdefault("MPLCONFIGDIR",str(Path(tempfile.gettempdir()) / "fairness-benchmark-matplotlib"))
os.environ.setdefault("MPLBACKEND","Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import t

from ..upstream import ROOT

KEYS=["scenario","method","lambda","temperature"]
METHODS=["baseline","gedi","threshold_gedi","mary_chi2"]
LABELS={"baseline":"Baseline","gedi":"Score GeDI","threshold_gedi":"Threshold GeDI","mary_chi2":"Mary chi-square"}
COLORS=dict(zip(METHODS,["#475569","#0d9488","#d97706","#2563eb"]))


def summarize(frame,keys,columns):
    result=frame.groupby(keys,dropna=False)[columns].agg(["mean","std","count"])
    for col in columns:
        n=result[(col,"count")]
        margin=t.ppf(.975,np.maximum(n-1,1))*result[(col,"std")]/np.sqrt(n)
        result[(col,"ci_low")]=result[(col,"mean")]-margin
        result[(col,"ci_high")]=result[(col,"mean")]+margin
    result.columns=["_".join(c) for c in result.columns]
    return result.reset_index()


def save(fig,path):
    fig.savefig(path.with_suffix(".png"),dpi=180,bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"),bbox_inches="tight")
    plt.close(fig)


def holm(pvalues):
    pvalues=np.asarray(pvalues)
    order=np.argsort(pvalues)
    adjusted=np.maximum.accumulate(pvalues[order]*(len(pvalues)-np.arange(len(pvalues))))
    result=np.empty(len(pvalues));result[order]=np.minimum(adjusted,1)
    return result


def contrasts(selected):
    rows=[]
    for (scenario,policy),group in selected.groupby(["scenario","policy"]):
        if policy=="temperature_ablation":continue
        for comparator in ("gedi","mary_chi2"):
            a=group[group.method=="threshold_gedi"].set_index("seed")
            b=group[group.method==comparator].set_index("seed")
            if not len(a) or not len(b):continue
            for metric in ("test_accuracy","test_decision_gap"):
                delta=(a[metric]-b[metric]).dropna().to_numpy()
                n=len(delta);sd=delta.std(ddof=1);mean=delta.mean();margin=t.ppf(.975,n-1)*sd/np.sqrt(n)
                signs=2*((np.arange(2**n)[:,None]>>np.arange(n))&1)-1
                null=(signs*delta).mean(1)
                p=float(np.mean(abs(null)>=abs(mean)-1e-14))
                rows.append({"scenario":scenario,"policy":policy,"contrast":f"threshold_gedi-minus-{comparator}",
                             "metric":metric,"mean_difference":mean,"sd_difference":sd,"ci_low":mean-margin,
                             "ci_high":mean+margin,"n":n,"exact_sign_flip_p":p})
    frame=pd.DataFrame(rows)
    if len(frame):frame["p_holm"]=holm(frame.exact_sign_flip_p)
    return frame


def main():
    parser=argparse.ArgumentParser(__doc__);parser.add_argument("out",type=Path,nargs="?",default=ROOT/"results/threshold_fairness")
    args=parser.parse_args();out=args.out
    config=json.loads((out/"config.json").read_text())
    proof=json.loads((out/"verification.json").read_text());assert proof["all_checks_passed"]
    data=pd.read_csv(out/"per_seed_results.csv")
    decision=pd.read_csv(out/"decision_results.csv")
    decision=decision[decision.split=="test"].copy()
    choices=pd.read_csv(out/"validation_selection.csv")
    selected=data.merge(choices[choices.status=="selected"][KEYS+["policy"]],on=KEYS,validate="many_to_many")
    selected.to_csv(out/"selected_per_seed.csv",index=False)
    columns=[c for c in data if c.startswith("test_") and pd.api.types.is_numeric_dtype(data[c])]
    summary=summarize(data,KEYS,columns);summary.to_csv(out/"aggregated_results.csv",index=False)
    summarize(selected,KEYS+["policy"],columns).to_csv(out/"selected_summary.csv",index=False)
    dcols=["accuracy","prevalence","dp_gap","weighted_variance","score_gap","min_group_n"]
    ds=summarize(decision,KEYS+["threshold","partition"],dcols)
    ds.to_csv(out/"aggregated_decisions.csv",index=False)
    selected_decision=decision.merge(choices[choices.status=="selected"][KEYS+["policy"]],on=KEYS)
    selected_decision.to_csv(out/"selected_decisions_per_seed.csv",index=False)
    sd=summarize(selected_decision,KEYS+["policy","threshold","partition"],dcols)
    sd.to_csv(out/"selected_decision_summary.csv",index=False)
    contrasts(selected).to_csv(out/"paired_contrasts.csv",index=False)
    collapse=data.groupby(["scenario","method"]).agg(runs=("seed","size"),near_constant=("test_near_constant","sum"),
        near_degenerate=("test_near_degenerate","sum"),constant_decision=("test_constant_decision","sum"),
        mean_clip_fraction=("grad_clip_fraction","mean"),maximum_gradient=("grad_norm_max","max"))
    collapse.to_csv(out/"collapse_diagnostics.csv")
    data[data.method=="threshold_gedi"].groupby(["scenario","temperature","lambda"])[
        ["grad_clip_fraction","grad_norm_p95","regularizer_grad_norm_max","soft_saturated_fraction","test_accuracy","test_decision_gap"]].agg(["mean","std"]).to_csv(out/"temperature_diagnostics.csv")
    figures=out/"figures";figures.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size":9,"axes.spines.top":False,"axes.spines.right":False})
    scenarios=config["scenarios"]
    for validation in (False,True):
        fig,axes=plt.subplots(1,3,figsize=(13,3.8))
        source=pd.read_csv(out/"validation_candidates.csv") if validation else summary
        xcol,ycol=("gap","accuracy") if validation else ("test_decision_gap_mean","test_accuracy_mean")
        for ax,scenario in zip(axes,scenarios):
            for method in METHODS:
                sub=source[(source.scenario==scenario)&(source.method==method)].sort_values(xcol)
                ax.scatter(sub[xcol],sub[ycol],s=16,color=COLORS[method],alpha=.55,label=LABELS[method])
                points=list(zip(sub[xcol],sub[ycol]))
                front=sorted((x,y) for x,y in points if not any(x2<=x and y2>=y and (x2<x or y2>y) for x2,y2 in points))
                if front:ax.plot(*zip(*front),color=COLORS[method],linewidth=1)
                if not validation:
                    chosen=selected[(selected.scenario==scenario)&(selected.method==method)&(selected.policy=="performance_matched")]
                    if len(chosen):
                        ax.scatter(chosen.test_decision_gap.mean(),chosen.test_accuracy.mean(),s=65,marker="D",color=COLORS[method],edgecolors="black",linewidths=.8)
            ax.set(title=scenario,xlabel="Max predefined decision gap at t=0.5",ylabel="Accuracy")
        axes[0].legend(fontsize=7)
        fig.suptitle("Validation frontiers" if validation else "Descriptive test frontiers; diamonds: validation performance-matched choices")
        fig.tight_layout();save(fig,figures/("pareto_validation" if validation else "pareto_decisions"))
    fig,axes=plt.subplots(1,3,figsize=(13,3.8))
    for ax,scenario in zip(axes,scenarios):
        for method in METHODS:
            sub=summary[(summary.scenario==scenario)&(summary.method==method)]
            x=np.maximum(sub.test_uniform_score_gap_mean,sub.test_quantile_score_gap_mean)
            ax.scatter(x,sub.test_decision_gap_mean,s=22,color=COLORS[method],alpha=.65,label=LABELS[method])
        ax.set(title=scenario,xlabel="Max predefined mean-score gap",ylabel="Max predefined decision gap")
    axes[0].legend(fontsize=7);fig.tight_layout();save(fig,figures/"score_vs_decision")
    profiles=[]
    for row in selected[selected.policy=="performance_matched"].itertuples():
        profile=pd.read_csv(out/"runs"/row.run_id/"conditional.csv")
        profile=profile[profile.threshold==.5].copy()
        profile["scenario"]=row.scenario;profile["method"]=row.method;profile["seed"]=row.seed
        profiles.append(profile)
    profile=pd.concat(profiles,ignore_index=True)
    profile.to_csv(out/"selected_conditional_profiles.csv",index=False)
    for score in (False,True):
        metric="score_mean" if score else "positive_rate"
        fig,axes=plt.subplots(1,3,figsize=(13,3.8))
        for ax,scenario in zip(axes,scenarios):
            for method in METHODS:
                sub=profile[(profile.scenario==scenario)&(profile.method==method)]
                if not len(sub):continue
                stat=sub.groupby("group")[metric].agg(["mean","std","count"])
                x=np.linspace(-.95,.95,20);margin=t.ppf(.975,stat["count"]-1)*stat["std"]/np.sqrt(stat["count"])
                ax.plot(x,stat["mean"],color=COLORS[method],label=LABELS[method])
                ax.fill_between(x,stat["mean"]-margin,stat["mean"]+margin,color=COLORS[method],alpha=.10)
            if scenario=="controlled":
                x=np.linspace(-1,1,100)
                ax.plot(x,np.full_like(x,.5) if score else .6+.2*x,"k--",linewidth=1,label="Analytic oracle")
            ax.set(title=scenario,xlabel="Sensitive attribute S",ylabel="Mean score" if score else "Positive decision rate",ylim=(0,1))
        axes[0].legend(fontsize=7);fig.suptitle("Performance-matched choices; mean and 95% seed intervals")
        fig.tight_layout();save(fig,figures/("conditional_scores" if score else "conditional_decisions"))
    # Compact two-panel controlled plot for the LaTeX brief, based on learned predictions.
    fig,axes=plt.subplots(1,2,figsize=(7,2.35))
    for ax,metric,label in zip(axes,["score_mean","positive_rate"],["Mean prediction score","Positive decision rate"]):
        for method in METHODS:
            sub=profile[(profile.scenario=="controlled")&(profile.method==method)]
            if len(sub):
                stat=sub.groupby("group")[metric].mean()
                ax.plot(np.linspace(-.95,.95,20),stat,color=COLORS[method],label=LABELS[method],linewidth=1.5)
        x=np.linspace(-1,1,100)
        ax.plot(x,np.full_like(x,.5) if metric=="score_mean" else .6+.2*x,"k--",linewidth=1,label="Oracle")
        ax.set(xlabel="S",ylabel=label,ylim=(.2,.85));ax.grid(alpha=.12)
    axes[1].legend(fontsize=6,loc="lower left",ncol=2)
    fig.tight_layout();save(fig,figures/"controlled_brief")
    fig,axes=plt.subplots(2,3,figsize=(13,7))
    for col,scenario in enumerate(scenarios):
        sub=sd[(sd.scenario==scenario)&(sd.policy=="performance_matched")&(sd.threshold==.5)]
        for j,partition in enumerate(["uniform","quantile","fairgroups"]):
            part=sub[sub.partition==partition].set_index("method")
            for row,metric in enumerate(["dp_gap","weighted_variance"]):
                vals=[part.loc[m,f"{metric}_mean"] if m in part.index else np.nan for m in METHODS]
                err=[part.loc[m,f"{metric}_ci_high"]-part.loc[m,f"{metric}_mean"] if m in part.index else 0 for m in METHODS]
                axes[row,col].bar(np.arange(4)+(j-1)*.25,vals,width=.24,yerr=err,capsize=2,label=partition)
                axes[row,col].set(xticks=np.arange(4),xticklabels=["Base","Score","Thresh","Mary"],ylabel=metric,title=scenario if row==0 else None)
    axes[0,0].legend(fontsize=7);fig.tight_layout();save(fig,figures/"partitions")
    fig,axes=plt.subplots(2,3,figsize=(13,7))
    for col,scenario in enumerate(scenarios):
        sub=selected[(selected.scenario==scenario)&(selected.policy=="temperature_ablation")]
        for row,metric in enumerate(["test_decision_gap","test_accuracy"]):
            stat=sub.groupby("temperature")[metric].agg(["mean","std","count"])
            err=t.ppf(.975,stat["count"]-1)*stat["std"]/np.sqrt(stat["count"])
            axes[row,col].errorbar(stat.index,stat["mean"],yerr=err,marker="o",capsize=3,color=COLORS["threshold_gedi"])
            axes[row,col].set(xscale="log",xlabel="Temperature T",ylabel=metric,title=scenario if row==0 else None)
    fig.suptitle("Temperature ablation: lambda selected separately on validation within accuracy band")
    fig.tight_layout();save(fig,figures/"temperature")
    fig,axes=plt.subplots(2,3,figsize=(13,6.5))
    for col,scenario in enumerate(scenarios):
        sub=data[(data.scenario==scenario)&(data.method=="threshold_gedi")&(data["lambda"]==.2)]
        for row,metric in enumerate(["regularizer_grad_norm_max","soft_saturated_fraction"]):
            stat=sub.groupby("temperature")[metric].agg(["mean","std"])
            axes[row,col].errorbar(stat.index,stat["mean"],yerr=stat["std"],marker="o",capsize=3)
            axes[row,col].set(xscale="log",xlabel="Temperature T",ylabel=metric,title=scenario if row==0 else None)
    fig.suptitle("Optimization at fixed lambda=0.2; mean and seed SD (last-batch gradients, epoch maxima)")
    fig.tight_layout();save(fig,figures/"temperature_optimization")
    fig,axes=plt.subplots(1,3,figsize=(13,3.8))
    for ax,scenario in zip(axes,scenarios):
        for method in METHODS:
            sub=data[(data.scenario==scenario)&(data.method==method)]
            ax.scatter(sub.test_score_std,sub.test_accuracy,s=7,alpha=.35,color=COLORS[method],label=LABELS[method])
        ax.axvline(config["near_constant_std"],color="black",linestyle="--",linewidth=1)
        ax.set(title=scenario,xlabel="Score standard deviation",ylabel="Accuracy")
    axes[0].legend(fontsize=7);fig.tight_layout();save(fig,figures/"collapse")
    fig,axes=plt.subplots(1,3,figsize=(13,3.8))
    for ax,scenario in zip(axes,scenarios):
        for method in METHODS:
            sub=sd[(sd.scenario==scenario)&(sd.method==method)&(sd.policy=="performance_matched")&(sd.partition=="fairgroups")]
            ax.errorbar(sub.threshold,sub.dp_gap_mean,yerr=sub.dp_gap_ci_high-sub.dp_gap_mean,marker="o",capsize=2,color=COLORS[method],label=LABELS[method])
        ax.set(title=scenario,xlabel="Decision threshold (same frozen model)",ylabel="FairGroups maximum gap")
    axes[0].legend(fontsize=7);fig.tight_layout();save(fig,figures/"threshold_sensitivity")
    script_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (out/"analysis_manifest.json").write_text(json.dumps({"script_sha256":script_hash,"verification":proof,
        "intervals":"95% Student-t over independent data/model seeds, conditional on validation selection",
        "paired_inference":"Exact sign-flip, Holm across all available predeclared contrasts"},indent=2)+"\n")
    print(f"Created summaries, paired contrasts and {len(list(figures.glob('*.png')))} figures")


if __name__=="__main__":main()
