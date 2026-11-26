import pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.family":"Liberation Serif","font.size":12})
R="docs/runs/riawelc_cpu_20261028/"
h=pd.read_csv(R+"history.csv")
fig,ax=plt.subplots(1,2,figsize=(9,3.6))
ax[0].plot(h.epoch,h["train/loss"],"o-",label="train"); ax[0].plot(h.epoch,h["val/total"],"s-",label="val")
ax[0].set_xlabel("Эпоха"); ax[0].set_ylabel("Потери"); ax[0].legend(); ax[0].grid(alpha=.3)
ax[1].plot(h.epoch,h["val/macro_f1"],"o-",color="tab:green"); ax[1].axvline(5,ls="--",c="gray")
ax[1].annotate("лучшая эпоха (5)",(5,0.8),xytext=(5.3,0.62),fontsize=11)
ax[1].set_xlabel("Эпоха"); ax[1].set_ylabel("macro-F1 (val)"); ax[1].grid(alpha=.3)
plt.tight_layout(); plt.savefig("coursework/fig/train.png",dpi=200)
r=pd.read_csv(R+"referral_test.csv"); print(r.curve.unique(), r["class"].unique())
a=r[r["class"]=="all"]
fig,ax=plt.subplots(figsize=(6.2,4))
for c,l,s in [("model","Модель (по неуверенности)","-"),("random","Случайный отбор","--"),("oracle","Оракул",":")]:
    d=a[a.curve==c]; ax.plot(d.fraction*100,d.recall*100,s,label=l,lw=2)
ax.set_xlabel("Доля снимков, переданных инспектору, %"); ax.set_ylabel("Recall, %")
ax.legend(); ax.grid(alpha=.3); plt.tight_layout(); plt.savefig("coursework/fig/referral.png",dpi=200)
print(a[a.curve=="model"].head(3))
