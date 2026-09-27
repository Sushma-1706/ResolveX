#!/usr/bin/env python3
"""Create deterministic development-only data; it never impersonates challenge data."""
from __future__ import annotations
import argparse, random
from pathlib import Path
import pandas as pd
NAMES=["Northstar Trading Corporation","Blue Harbor Cafe","Orchid Technology Private Limited","Maison Verte Boulangerie","Riverstone Medical Clinic","Sunrise Textiles Company","Cedar and Pine Market","Apex Logistics Limited"]
ADDRESSES=["14 Market Street, Springfield 62704","22 Blue Harbor Road, Boston 02110","5 MG Road, Bengaluru 560001","18 Rue de la Paix, Paris 75002","410 River Lane, Austin 78701","9 Textile Avenue, Mumbai 400001","77 Cedar Street, Portland 97205","300 Industrial Boulevard, Lyon 69001"]
COUNTRIES=["US","US","India","France","US","India","US","France"]
def variant(name,address,country,rng):
    n=name.replace("Corporation","Corp.").replace("Company","Co").replace("Private Limited","Pvt Ltd").replace(" and "," & ")
    a=address.replace("Street","St").replace("Road","Rd").replace("Avenue","Ave").replace(","," ")
    if rng.random()<.25: n=" ".join(n.split()[::-1])
    if rng.random()<.18: n=n[:-1]+chr(ord(n[-1])+1) if n[-1].isalpha() else n
    if rng.random()<.18: a="" # missing field
    if rng.random()<.1: country=""
    return n,a,country
def make_split(root,split,count,seed,truth=False):
    rng=random.Random(seed); s1=[]; s2=[]; s3=[]; gt=[]; idx={"S1":1,"S2":1,"S3":1}
    def add(rows,prefix,n,a,c):
        eid=f"{prefix}-{idx[prefix]:05d}";idx[prefix]+=1;rows.append(dict(entity_id=eid,business_name=n,business_address=a,country=c));return eid
    for i in range(count):
        n,a,c=NAMES[i%len(NAMES)],ADDRESSES[i%len(ADDRESSES)],COUNTRIES[i%len(COUNTRIES)]; sid=add(s1,"S1",n,a,c); matches=[]
        if i%5: # intentional singletons
            for rows,prefix in ((s2,"S2"),(s3,"S3")):
                if rng.random()<.72:
                    vn,va,vc=variant(n,a,c,rng);matches.append(add(rows,prefix,vn,va,vc))
            if i%7==0: # one-to-many
                vn,va,vc=variant(n,a,c,rng);matches.append(add(s2,"S2",vn,va,vc))
        if truth: gt.append({"source1_entity_id":sid,"matched_entity_ids":",".join(matches)})
    for rows,prefix in ((s2,"S2"),(s3,"S3")):
        for j in range(max(10,count//3)):
            add(rows,prefix,f"Unrelated {prefix} Business {j}",f"{j+800} Independent Road, Remote {10000+j}",rng.choice(["US","India","France"]))
    root.mkdir(parents=True,exist_ok=True)
    for i,rows in ((1,s1),(2,s2),(3,s3)):pd.DataFrame(rows).to_csv(root/f"{split}_source{i}.tsv",sep="\t",index=False)
    if truth: pd.DataFrame(gt).to_csv(root/f"{split}_ground_truth.tsv",sep="\t",index=False)
def main():
 p=argparse.ArgumentParser();p.add_argument("--data-dir",default="dataset");p.add_argument("--seed",type=int,default=42);p.add_argument("--train-size",type=int,default=80);p.add_argument("--test-size",type=int,default=35);a=p.parse_args();root=Path(a.data_dir);make_split(root/"train","train",a.train_size,a.seed,True);make_split(root/"test","test",a.test_size,a.seed+1);print(f"Synthetic development dataset generated in {root} (seed={a.seed}).")
if __name__=="__main__":main()
