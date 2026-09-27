"""Inverted-index and sparse retrieval candidate generation (no Cartesian pairs)."""
from __future__ import annotations
from collections import defaultdict
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from src.preprocessing.normalizer import normalize_name, normalize_address, normalize_country

class CandidateGenerator:
    def __init__(self, max_tfidf_candidates=30, max_ngram_candidates=30, max_token_candidates=100):
        self.max_tfidf_candidates, self.max_ngram_candidates, self.max_token_candidates = max_tfidf_candidates, max_ngram_candidates, max_token_candidates

    @staticmethod
    def _index(values):
        idx = defaultdict(set)
        for i, value in enumerate(values):
            if value: idx[value].add(i)
        return idx
    @staticmethod
    def _token_index(values):
        idx = defaultdict(set)
        for i, value in enumerate(values):
            for token in set(value.split()):
                if len(token) >= 2: idx[token].add(i)
        return idx
    @staticmethod
    def _top(matrix, query, limit):
        scores = (query @ matrix.T).toarray().ravel()
        nonzero = np.flatnonzero(scores > 0)
        if len(nonzero) <= limit: return set(nonzero.tolist())
        best = nonzero[np.argpartition(scores[nonzero], -limit)[-limit:]]
        return set(best.tolist())

    def generate(self, source1: pd.DataFrame, candidates: pd.DataFrame) -> tuple[dict[str, set[str]], dict]:
        c = candidates.copy()
        s = source1.copy()
        for frame in (s,c):
            frame["_name"] = frame.business_name.map(normalize_name)
            frame["_addr"] = frame.business_address.map(normalize_address)
            frame["_country"] = frame.country.map(normalize_country)
        name_exact, addr_exact = self._index(c._name), self._index(c._addr)
        name_tokens, addr_tokens = self._token_index(c._name), self._token_index(c._addr)
        country = self._index(c._country)
        combined_c = (c._name + " " + c._addr).tolist()
        combined_s = (s._name + " " + s._addr).tolist()
        word = TfidfVectorizer(analyzer="word", ngram_range=(1,2), min_df=1).fit(combined_c + combined_s)
        char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3,4), min_df=1).fit(combined_c + combined_s)
        wm, cm = normalize(word.transform(combined_c)), normalize(char.transform(combined_c))
        ws, cs = normalize(word.transform(combined_s)), normalize(char.transform(combined_s))
        out, strategies = {}, defaultdict(int)
        for pos, (_, row) in enumerate(s.iterrows()):
            selected = set()
            def add(items, key):
                before=len(selected); selected.update(items); strategies[key] += len(selected)-before
            add(name_exact.get(row._name, set()), "exact_name")
            add(addr_exact.get(row._addr, set()), "exact_address")
            nt = set().union(*(name_tokens.get(t,set()) for t in set(row._name.split()) if len(t)>=2))
            at = set().union(*(addr_tokens.get(t,set()) for t in set(row._addr.split()) if len(t)>=2))
            add(set(list(nt)[:self.max_token_candidates]), "name_token")
            add(set(list(at)[:self.max_token_candidates]), "address_token")
            add(self._top(cm, cs[pos], self.max_ngram_candidates), "character_ngram")
            add(self._top(wm, ws[pos], self.max_tfidf_candidates), "tfidf")
            # Country is a guardrail: only adds records where textual retrieval found none.
            if not selected and row._country: add(set(list(country.get(row._country,set()))[:self.max_token_candidates]), "country_fallback")
            out[row["entity_id"]] = {c.iloc[i]["entity_id"] for i in selected}
        counts=np.array([len(v) for v in out.values()])
        total=len(s)*len(c); generated=int(counts.sum())
        metrics={"theoretical_comparisons":total,"generated_candidates":generated,"candidate_reduction_ratio":1-(generated/total if total else 0),"average_candidates":float(counts.mean()) if len(counts) else 0,"median_candidates":float(np.median(counts)) if len(counts) else 0,"maximum_candidates":int(counts.max()) if len(counts) else 0,"source1_zero_candidates":int((counts==0).sum()),"strategy_new_candidates":dict(strategies)}
        return out, metrics
