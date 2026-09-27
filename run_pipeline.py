#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, logging, subprocess, sys
from pathlib import Path
import yaml
from src.pipeline.run_pipeline import execute
def main():
 p=argparse.ArgumentParser();p.add_argument("--mode",choices=["synthetic","train","validate","predict","all"],default="all");p.add_argument("--data-dir",default="dataset");p.add_argument("--config",default="configs/config.yaml");a=p.parse_args()
 config=yaml.safe_load(Path(a.config).read_text()); config["data"]={"train_dir":str(Path(a.data_dir)/"train"),"test_dir":str(Path(a.data_dir)/"test")}
 official=Path(config["data"]["train_dir"])/"train_source1.tsv"
 if a.mode=="synthetic" or not official.exists():
  print("[INFO] Real challenge dataset not found.\n[INFO] Using synthetic development dataset."); subprocess.run([sys.executable,"scripts/generate_synthetic_data.py","--data-dir",a.data_dir],check=True); dtype="synthetic"
 else: print("[INFO] Real challenge dataset detected."); dtype="challenge"
 summary=execute(config,dtype);print(json.dumps(summary,indent=2))
if __name__=="__main__":main()
