#! /bin/sh

echo "Extractive"
python qasper_evaluator.py --predictions R_extractive_eval.json --gold qasper-test-v0.3.json --text_evidence_only

echo "Abstractive"
python qasper_evaluator.py --predictions R_abstractive_eval.json --gold qasper-test-v0.3.json --text_evidence_only

echo "Boolean"
python qasper_evaluator.py --predictions R_boolean_eval.json --gold qasper-test-v0.3.json --text_evidence_only

echo "All"
python qasper_evaluator.py --predictions R_all_eval.json --gold qasper-test-v0.3.json --text_evidence_only