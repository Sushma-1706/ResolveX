from __future__ import annotations
from sklearn.ensemble import HistGradientBoostingClassifier
from src.features.pair_features import FEATURE_COLUMNS
class PairMatcher:
    def __init__(self, random_seed=42): self.model=HistGradientBoostingClassifier(max_iter=160, learning_rate=.08, max_leaf_nodes=15, l2_regularization=1., random_state=random_seed)
    def fit(self, pairs): self.model.fit(pairs[FEATURE_COLUMNS],pairs.label); return self
    def predict_proba(self,pairs): return self.model.predict_proba(pairs[FEATURE_COLUMNS])[:,1]
    def importance(self): return {x:0.0 for x in FEATURE_COLUMNS} # HGB has no stable native importances
