#!/usr/bin/env python3
"""Plot every declared fixed-pencil comparison, including regressions."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, NullFormatter

root=Path(__file__).resolve().parent
audit=json.loads((root/"D05/audit.json").read_text())
if audit["issues"]:
    raise SystemExit("Refusing to plot unaudited results")
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"svg.hashsalt":"blaze2d-d05",
                     "axes.spines.top":False,"axes.spines.right":False,"axes.spines.left":False})
names={"hex-gamma-c3":"Γ / C3", "hex-gamma-c6":"Γ / C6", "hex-gamma-c6v":"Γ / C6v",
       "hex-k":"K / little group", "chiral-c4":"chiral C4", "glide-pg":"X / pg", "glide-pgg":"X / pgg"}
labels=[]
for row in audit["comparisons"]:
    if row["study"].endswith("-TM"):
        key=row["study"][:-3]
        if key.startswith("mechanism-"):
            label={"conservative":"adaptive prefixes", "isotypic":"partner reconstruction", "fixed-guards":"total guard budget"}[key.removeprefix("mechanism-")]
            labels.append("Policy: "+label)
        else:
            held=key.startswith("heldout-")
            case=key.split("-",1)[1]
            labels.append(("64/65² · 16 bands: " if held else "32/33² · 8 bands: ")+names[case])
fig,axes=plt.subplots(1,2,figsize=(13,8),sharey=True,gridspec_kw={"wspace":.14})
for ax,pol in zip(axes,["TM","TE"]):
    rows=[r for r in audit["comparisons"] if r["study"].endswith("-"+pol)]
    for i,row in enumerate(rows):
        if not row["valid"]:
            ax.text(.8,len(rows)-i-1,"rejected",va="center",color="#657583")
            continue
        s=row["statistics"];x=s["geometric_speedup"];lo,hi=s["ci95"]
        color={"gain":"#087f75","regression":"#b13d4b"}.get(row["classification"],"#657583")
        ax.errorbar(x,len(rows)-i-1,xerr=[[x-lo],[hi-x]],fmt="o",ms=5,capsize=3,color=color)
        ax.text(1.03,len(rows)-i-1,f"{x:.3f}",transform=ax.get_yaxis_transform(),va="center",fontsize=9,color=color)
    ax.axvline(1,color="#586b7a",lw=1,ls="--")
    for edge in (6.5,2.5):
        ax.axhline(edge,color="#c9d6dd",lw=.8)
    ax.set_xscale("log");ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.set_minor_formatter(NullFormatter())
    intervals=[r["statistics"]["ci95"] for r in rows if r["valid"]]
    ax.set_xticks([.5,.75,1,1.5,2]);ax.set_xlim(min(.45,min(v[0] for v in intervals)*.9),max(2.15,max(v[1] for v in intervals)*1.1))
    ax.set_title(pol,loc="left",weight="bold");ax.set_xlabel("Speedup, baseline time / candidate time")
    ax.grid(axis="x",alpha=.16);ax.tick_params(axis="y",length=0)
axes[0].set_yticks(range(len(labels)),list(reversed(labels)))
fig.suptitle("Qualified Fourier sectors: complete matched solve time",x=.36,y=.99,ha="left",weight="bold")
fig.text(.36,.925,"20 randomized pairs · f64 · same finite pencil · full fields\n95% paired bootstrap intervals; policy controls use their own baselines",ha="left",fontsize=10)
fig.text(.36,.025,"Green: demonstrated gain · Red: regression · Grey: no demonstrated 2% practical change",ha="left",fontsize=9)
fig.subplots_adjust(left=.36,right=.91,top=.86,bottom=.1)
fig.savefig(root/"D05-speedups.svg",metadata={"Date":None})
svg=root/"D05-speedups.svg"
svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines())+"\n")
fig.savefig(root/"D05-speedups.png",dpi=180)
