import json, subprocess, sys, re
heads=json.load(open("headings.json"))
n=int(re.search(r"Pages:\s+(\d+)",subprocess.run(["pdfinfo","WeldSight_kursovaya.pdf"],capture_output=True,text=True).stdout).group(1))
pages={}
txt=[subprocess.run(["pdftotext","-f",str(i),"-l",str(i),"-layout","WeldSight_kursovaya.pdf","-"],capture_output=True,text=True).stdout for i in range(1,n+1)]
def norm(s): return re.sub(r"\s+"," ",s).strip()
for h in heads:
    if h=="СОДЕРЖАНИЕ": continue
    for i in range(2,n):  # пропускаем титул и содержание (стр. 2)
        lines=[norm(l) for l in txt[i].split("\n")]
        key=norm(h)[:40]
        if any(l.startswith(key) for l in lines):
            pages[h]=i+1; break
    else: pages[h]="?"
if len(sys.argv)>1: print("pages total",n); print({k:v for k,v in pages.items() if v=='?'} or "all found")
else: json.dump(pages,open("toc.json","w"),ensure_ascii=False)
