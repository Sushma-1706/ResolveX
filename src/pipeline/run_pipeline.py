from __future__ import annotations
import json, logging, subprocess, sys
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from src.data.loader import load_split
from src.blocking.candidate_generator import CandidateGenerator
from src.features.pair_features import build_feature_frame
from src.models.matcher import PairMatcher
from src.evaluation.threshold import optimize_threshold
from src.evaluation.metrics import score_sets

LOG=logging.getLogger(__name__)
def _truth(gt, ids):
    raw=dict(zip(gt.source1_entity_id,gt.matched_entity_ids.fillna("")))
    return {i:set(filter(None,raw.get(i,"").split(','))) for i in ids}
def _write_lists(path, column, ids, mapping):
    pd.DataFrame({"source1_entity_id":list(ids),column:[",".join(sorted(mapping.get(i,set()))) for i in ids]}).to_csv(path,sep="\t",index=False)
def validate_outputs(test, matching, candidates):
    s1=set(test["source1"].entity_id); allowed=set(pd.concat([test["source2"],test["source3"]]).entity_id)
    if set(matching)!=s1 or set(candidates)!=s1: raise ValueError("Output rows must exactly cover test Source 1")
    for sid in s1:
        if not matching[sid] <= candidates[sid]: raise ValueError(f"Prediction outside final candidate set: {sid}")
        if not candidates[sid] <= allowed or not matching[sid] <= allowed: raise ValueError(f"Output references invalid candidate ID: {sid}")
    return True
def execute(config, data_type):
    train=load_split(config["data"]["train_dir"],"train",True); test=load_split(config["data"]["test_dir"],"test")
    output=Path(config["output"]["directory"]); reports=Path(config["reports"]["directory"]); output.mkdir(parents=True,exist_ok=True); reports.mkdir(parents=True,exist_ok=True)
    candidates_train=pd.concat([train["source2"],train["source3"]],ignore_index=True); candidates_test=pd.concat([test["source2"],test["source3"]],ignore_index=True)
    gen=CandidateGenerator(**config["blocking"]); train_pairs, train_block=gen.generate(train["source1"],candidates_train)
    truth=_truth(train["ground_truth"],train["source1"].entity_id); positive=sum(len(v) for v in truth.values()); recovered=sum(len(truth[k]&train_pairs[k]) for k in truth)
    train_block["candidate_recall"]=recovered/positive if positive else 1.
    train_ids,val_ids=train_test_split(train["source1"].entity_id.tolist(),test_size=config["validation"]["test_size"],random_state=config["validation"]["random_seed"])
    all_features=build_feature_frame(train["source1"],candidates_train,train_pairs)
    all_features["label"]=[int(row.candidate_entity_id in truth[row.source1_entity_id]) for row in all_features.itertuples()]
    model=PairMatcher(config["model"]["random_seed"]).fit(all_features[all_features.source1_entity_id.isin(train_ids)])
    valid=all_features[all_features.source1_entity_id.isin(val_ids)].copy(); valid["match_probability"]=model.predict_proba(valid)
    threshold,analysis=optimize_threshold(valid,_truth(train["ground_truth"],val_ids)); predicted_val={sid:set(g.loc[g.match_probability>=threshold,"candidate_entity_id"]) for sid,g in valid.groupby("source1_entity_id")}
    validation=score_sets(_truth(train["ground_truth"],val_ids),predicted_val); validation.update({"threshold":threshold,"data_type":data_type})
    # Refit on every labeled training candidate, then generate the exact final inference candidate set.
    model.fit(all_features); test_pairs,test_block=gen.generate(test["source1"],candidates_test); scored=build_feature_frame(test["source1"],candidates_test,test_pairs)
    if not scored.empty: scored["match_probability"]=model.predict_proba(scored)
    matching={sid:set() for sid in test["source1"].entity_id};
    for row in scored.itertuples():
        if row.match_probability>=threshold: matching[row.source1_entity_id].add(row.candidate_entity_id)
    validate_outputs(test,matching,test_pairs)
    ids=test["source1"].entity_id.tolist(); _write_lists(output/"candidate_pairs.tsv","candidate_entity_ids",ids,test_pairs); _write_lists(output/"matching_results.tsv","matched_entity_ids",ids,matching)
    (reports/"blocking_metrics.json").write_text(json.dumps({"data_type":data_type,"training":train_block,"inference":test_block},indent=2)); (reports/"validation_metrics.json").write_text(json.dumps(validation,indent=2)); analysis.to_csv(reports/"threshold_analysis.csv",index=False); pd.DataFrame(model.importance().items(),columns=["feature","importance"]).to_csv(reports/"feature_importance.csv",index=False)
    # The official helper is optional in development repositories, but is mandatory
    # when shipped alongside a challenge dataset.
    official_validator=Path("utils/validate_submission.py")
    official_passed=None
    if official_validator.exists():
        result=subprocess.run([sys.executable,str(official_validator),"--matching",str(output/"matching_results.tsv"),"--candidate",str(output/"candidate_pairs.tsv"),"--test-dir",config["data"]["test_dir"]],capture_output=True,text=True)
        if result.returncode:
            raise ValueError(f"Official submission validator failed:\n{result.stdout}\n{result.stderr}")
        official_passed=True
    summary={"data_type":data_type,"submission_ready":True,"official_validator_passed":official_passed,"validation":validation,"test_counts":{"source1":len(test["source1"]),"source2":len(test["source2"]),"source3":len(test["source3"]),"candidates":test_block["generated_candidates"],"matches":sum(map(len,matching.values()))}}
    (reports/"run_summary.json").write_text(json.dumps(summary,indent=2)); return summary
