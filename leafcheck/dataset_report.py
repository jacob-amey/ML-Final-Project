"""Report training support and group counts without opening test predictions."""

import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .common import ROOT, DATA, atomic_json, write_csv
from .data import release


def main():
    rows = list(csv.DictReader((DATA / "split_summary.csv").open()))
    rows.sort(key=lambda row:(-int(row["train"]),row["class_name"]))
    assert len(rows)==177 and sum(int(row["train"]) for row in rows)==59008
    records=[{"rank":i+1,"class_name":row["class_name"],"training_images":int(row["train"]),
              "training_groups":int(row["train_groups"])} for i,row in enumerate(rows)]
    write_csv(ROOT / "docs/training_class_support.csv",records)
    images=np.array([r["training_images"] for r in records]);groups=np.array([r["training_groups"] for r in records])
    summary={"dataset_fingerprint":release()["fingerprint"],"classes":177,"training_images":int(images.sum()),
             "training_groups":int(groups.sum()),"minimum_training_images_per_class":int(images.min()),
             "median_training_images_per_class":float(np.median(images)),"maximum_training_images_per_class":int(images.max()),
             "classes_with_fewer_than_30_training_images":int((images<30).sum()),
             "healthy_labels":sum(r["class_name"].endswith("__healthy") for r in rows),
             "plant_or_crop_prefixes":len({r["class_name"].split("__")[0] for r in rows})}
    atomic_json(ROOT / "docs/dataset_summary.json",summary)
    output=ROOT / "docs/figures";output.mkdir(exist_ok=True)
    fig,axes=plt.subplots(1,2,figsize=(12,4.7),gridspec_kw={"width_ratios":[1.7,1]})
    x=np.arange(1,178)
    axes[0].plot(x,images,label="Training images",color="#32634d",lw=2)
    axes[0].plot(x,groups,label="Known/inferred related-image groups",color="#b38134",lw=1,alpha=.8)
    axes[0].set(yscale="log",xlabel="Class rank by training-image count",ylabel="Count (log scale)",
                title="Training support is highly uneven")
    axes[0].set_xlim(1,177);axes[0].grid(alpha=.2);axes[0].legend(fontsize=8)
    edges=[0,10,30,100,500,np.inf];counts=np.histogram(images,bins=edges)[0]
    axes[1].bar(["<10","10–29","30–99","100–499","500+"],counts,color="#527a67")
    axes[1].set(xlabel="Training images per class",ylabel="Number of classes",title="All 177 labels retained",ylim=(0,max(counts)*1.18))
    axes[1].tick_params(axis="x",rotation=20)
    for i,value in enumerate(counts):axes[1].text(i,value+.6,str(value),ha="center",fontsize=9)
    fig.suptitle("LeafCheck final dataset — training split only")
    fig.text(.5,.005,"Group identity is incomplete; these counts do not establish independent biological specimens.",ha="center",fontsize=8)
    fig.tight_layout(rect=(0,.035,1,1));fig.savefig(output/"class_distribution.png",dpi=180);plt.close(fig)
    print(summary)


if __name__=="__main__":main()
