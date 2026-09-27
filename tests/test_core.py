import pandas as pd
from src.preprocessing.normalizer import normalize_name, normalize_address, extract_postal
from src.blocking.candidate_generator import CandidateGenerator
from src.evaluation.metrics import score_sets
def test_normalization():
 assert normalize_name("ABC & Company, Inc.")=="abc and"
 assert normalize_address("14 Main St.")=="14 main street"
 assert extract_postal("Paris 75002")=="75002"
def test_blocking_and_metrics():
 s1=pd.DataFrame([{"entity_id":"S1-1","business_name":"Blue Harbor Corp","business_address":"10 Main St","country":"France"}])
 c=pd.DataFrame([{"entity_id":"S2-1","business_name":"Blue Harbor Corporation","business_address":"10 Main Street","country":"France"},{"entity_id":"S3-1","business_name":"Else","business_address":"99 Other Rd","country":"US"}])
 pairs,metrics=CandidateGenerator(3,3,3).generate(s1,c)
 assert "S2-1" in pairs["S1-1"] and metrics["generated_candidates"]<metrics["theoretical_comparisons"]+1
 assert score_sets({"S1-1":set()},{"S1-1":set()})["singleton_accuracy"]==1
