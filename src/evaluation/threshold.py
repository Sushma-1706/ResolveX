import numpy as np
import pandas as pd
from .metrics import score_sets
def optimize_threshold(scored, truth):
    rows=[]
    for threshold in np.arange(.10,.951,.05):
        predicted={sid:set(g.loc[g.match_probability>=threshold,"candidate_entity_id"]) for sid,g in scored.groupby("source1_entity_id")}
        m=score_sets(truth,predicted); m["threshold"]=round(float(threshold),2); rows.append(m)
    table=pd.DataFrame(rows); best=table.sort_values(["f0_5","precision"],ascending=False).iloc[0]
    return float(best.threshold),table
