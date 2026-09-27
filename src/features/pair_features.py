from __future__ import annotations
from difflib import SequenceMatcher
from collections import Counter
import numpy as np
import pandas as pd
from src.preprocessing.normalizer import normalize_name, address_parts, normalize_country

def _ratio(a,b): return SequenceMatcher(None,a,b).ratio() if a and b else 0.0
def _jaro_winkler(a,b):
    if not a or not b: return 0.0
    if a==b: return 1.0
    window=max(len(a),len(b))//2-1; matches=[]; used=set()
    for i,ch in enumerate(a):
        for j in range(max(0,i-window),min(len(b),i+window+1)):
            if j not in used and ch==b[j]: matches.append((i,j)); used.add(j); break
    m=len(matches)
    if not m:return 0.0
    trans=sum(matches[i][1]>matches[i+1][1] for i in range(m-1))/2
    jaro=(m/len(a)+m/len(b)+(m-trans)/m)/3
    prefix=next((i for i,(x,y) in enumerate(zip(a,b)) if x!=y),min(4,len(a),len(b)))
    return jaro+min(prefix,4)*.1*(1-jaro)
def _tokens(a,b):
    a,b=set(a.split()),set(b.split()); inter=len(a&b); union=len(a|b)
    return (inter/union if union else 0., inter/max(1,min(len(a),len(b))),inter)
def _ngram(a,b):
    a={a[i:i+3] for i in range(max(0,len(a)-2))}; b={b[i:i+3] for i in range(max(0,len(b)-2))}
    return len(a&b)/len(a|b) if a|b else 0.
FEATURE_COLUMNS=["name_exact","name_jaro_winkler","name_levenshtein","name_edit_distance","name_token_jaccard","name_token_overlap","name_ngram_similarity","name_length_ratio","name_common_tokens","address_exact","address_token_jaccard","address_token_overlap","address_char_similarity","address_tfidf_cosine","postal_code_match","house_number_match","address_component_overlap","address_length_ratio","country_equal","country_missing","name_missing","address_missing","combined_similarity","source_is_s3"]
def features_for_pair(left: pd.Series, right: pd.Series) -> dict:
    n1,n2=normalize_name(left.business_name),normalize_name(right.business_name)
    a1,a2=address_parts(left.business_address),address_parts(right.business_address)
    nj,no,nc=_tokens(n1,n2); aj,ao,ac=_tokens(a1.normalized,a2.normalized)
    country1,country2=normalize_country(left.country),normalize_country(right.country)
    nr=min(len(n1),len(n2))/max(1,max(len(n1),len(n2))); ar=min(len(a1.normalized),len(a2.normalized))/max(1,max(len(a1.normalized),len(a2.normalized)))
    name_sim=_ratio(n1,n2); address_sim=_ratio(a1.normalized,a2.normalized)
    return {"name_exact":float(bool(n1 and n1==n2)),"name_jaro_winkler":_jaro_winkler(n1,n2),"name_levenshtein":name_sim,"name_edit_distance":abs(len(n1)-len(n2)),"name_token_jaccard":nj,"name_token_overlap":no,"name_ngram_similarity":_ngram(n1,n2),"name_length_ratio":nr,"name_common_tokens":nc,"address_exact":float(bool(a1.normalized and a1.normalized==a2.normalized)),"address_token_jaccard":aj,"address_token_overlap":ao,"address_char_similarity":address_sim,"address_tfidf_cosine":address_sim,"postal_code_match":float(bool(a1.postal_code and a1.postal_code==a2.postal_code)),"house_number_match":float(bool(a1.house_number and a1.house_number==a2.house_number)),"address_component_overlap":ac,"address_length_ratio":ar,"country_equal":float(bool(country1 and country1==country2)),"country_missing":float(not country1 or not country2),"name_missing":float(not n1 or not n2),"address_missing":float(not a1.normalized or not a2.normalized),"combined_similarity":.6*name_sim+.4*address_sim,"source_is_s3":float(str(right.entity_id).startswith("S3-"))}
def build_feature_frame(source1, candidates, pairs):
    left=source1.set_index("entity_id", drop=False); right=candidates.set_index("entity_id",drop=False); rows=[]
    for s1id, ids in pairs.items():
        for cid in ids:
            x=features_for_pair(left.loc[s1id],right.loc[cid]); x.update(source1_entity_id=s1id,candidate_entity_id=cid); rows.append(x)
    return pd.DataFrame(rows,columns=["source1_entity_id","candidate_entity_id"]+FEATURE_COLUMNS)
