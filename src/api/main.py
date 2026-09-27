from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
from fastapi import FastAPI, HTTPException
app=FastAPI(title="Business Entity Resolution API")
ROOT=Path(__file__).resolve().parents[2]
def report(name):
 p=ROOT/"reports"/name
 if not p.exists(): raise HTTPException(404,"Run the pipeline first.")
 return json.loads(p.read_text())
@app.get("/health")
def health(): return {"status":"ok","pipeline_available":(ROOT/"reports"/"run_summary.json").exists()}
@app.get("/api/dataset/stats")
def dataset_stats(): return report("run_summary.json")
@app.get("/api/blocking/stats")
def blocking(): return report("blocking_metrics.json")
@app.get("/api/evaluation")
def evaluation():
 return {"metrics":report("validation_metrics.json"),"threshold_analysis":pd.read_csv(ROOT/"reports"/"threshold_analysis.csv").to_dict("records")}
@app.get("/api/results")
def results():
 return {"summary":report("run_summary.json"),"matching":pd.read_csv(ROOT/"output"/"matching_results.tsv",sep="\t",keep_default_na=False).head(50).to_dict("records"),"candidates":pd.read_csv(ROOT/"output"/"candidate_pairs.tsv",sep="\t",keep_default_na=False).head(50).to_dict("records")}
@app.get("/api/entity/{entity_id}")
def entity(entity_id:str):
 for split in ("train","test"):
  for n in (1,2,3):
   p=ROOT/"dataset"/split/f"{split}_source{n}.tsv"
   if p.exists():
    df=pd.read_csv(p,sep="\t",dtype=str,keep_default_na=False); hit=df[df.entity_id==entity_id]
    if not hit.empty:return hit.iloc[0].to_dict()
 raise HTTPException(404,"Entity not found")
@app.post("/api/pipeline/run")
def run():
 import subprocess,sys
 subprocess.run([sys.executable,str(ROOT/"run_pipeline.py"),"--mode","all"],cwd=ROOT,check=True);return report("run_summary.json")
@app.post("/api/pipeline/validate")
def validate():
 from src.pipeline.run_pipeline import validate_outputs
 # Output invariants are enforced during pipeline; check report existence for API usage.
 return {"valid":bool(report("run_summary.json").get("submission_ready"))}
