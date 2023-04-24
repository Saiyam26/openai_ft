import os
import openai
os.environ['OPENAI_API_KEY'] = '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'
openai.api_key = os.getenv("OPENAI_API_KEY")

import urllib3.exceptions
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
from retriever import checker_score

import json
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from time import sleep

with open(f'./qasper-sample.json') as f:
	qasper = json.load(f)
	
with open('ft_retrieved_evidences.json') as f:
	ft_retrieved_evidences = json.load(f)
	
with open('retrieved_evidences.json') as f:
	retrieved_evidences = json.load(f)

with open('op_retrieved_evidences.json') as f:
	op_retrieved_evidences = json.load(f)

# op_retrieved_evidences = {}
p2q = {}
# p2t = {'boolean':[], 'extractive':[], 'abstractive':[], 'all':qasper.keys()}
p2t = {}

def ans_type(answer):
	if len(answer['extractive_spans'])>0:
		return 'extractive'
	elif answer['yes_no'] is not None:
		return 'boolean'
	elif answer['free_form_answer'] != "":
		return 'abstractive'
	else:
		return 'unanswerable'

'''
for i, (paper_id) in enumerate(qasper.keys()):
	print(paper_id)

	col_ev, ftcol_ev, op_ev = [], [], []
	question = qasper[paper_id]['qas'][0]['question']

	for evidence in retrieved_evidences[paper_id]:
		score = checker_score(question, evidence)
		col_ev.append((score, evidence))
	sleep(2)

	for evidence in ft_retrieved_evidences[paper_id]:
		score = checker_score(question, evidence)
		ftcol_ev.append((score, evidence))
	sleep(2)

	for evidence in op_retrieved_evidences[paper_id]:
		score = checker_score(question, evidence)
		op_ev.append((score, evidence))
	sleep(2)

	retrieved_evidences[paper_id] = col_ev
	ft_retrieved_evidences[paper_id] = ftcol_ev
	op_retrieved_evidences[paper_id] = op_ev

	if (i+1)%5==0:
		with open('ft_retrieved_evidences.json', 'w') as f:
			json.dump(ft_retrieved_evidences, f, indent=2)
		with open('retrieved_evidences.json', 'w') as f:
			json.dump(retrieved_evidences, f, indent=2)
		with open('op_retrieved_evidences.json', 'w') as f:
			json.dump(op_retrieved_evidences, f, indent=2)
'''	

for paper_id in qasper.keys():
	# faiss_index = FAISS.load_local(f'./indices/index_{paper_id}', embeddings=OpenAIEmbeddings())
	
	question = qasper[paper_id]['qas'][0]['question']
	question_id = qasper[paper_id]['qas'][0]['question_id']
	
	# docs = faiss_index.similarity_search(question, k=10)
	
	# op_retrieved_evidences[paper_id] = [doc.page_content for doc in docs]
	p2q[paper_id] = question_id
	
	qtype = ans_type(qasper[paper_id]['qas'][0]['answers'][0]['answer'])
	p2t[paper_id]=qtype

# with open('./op_retrieved_evidences', 'w') as f:
# 	json.dump(op_retrieved_evidences, f, indent=2)

emb = {'col':retrieved_evidences, 'ftcol':ft_retrieved_evidences, 'op':op_retrieved_evidences}

# qt = 'extractive'
em = 'ftcol'
# print(p2t)

final_eval = {'extractive':[], 'abstractive':[], 'boolean':[], 'unanswerable':[], 'all':[]}

for paper_id in qasper.keys():
	pred_evidence = sorted([e[0] for e in emb[em][paper_id]], reverse=True)
	if len(pred_evidence)>=5:
		val = pred_evidence[4]
	else:
		val = 0

	final_eval[p2t[paper_id]].append({'question_id': p2q[paper_id],
						'predicted_answer': '',
						'predicted_evidence':[e[1] for e in emb[em][paper_id] if e[0]>=val]})

	final_eval['all'].append({'question_id': p2q[paper_id],
						'predicted_answer': '',
						'predicted_evidence':[e[1] for e in emb[em][paper_id] if e[0]>=val]})					

for qt in ['extractive','abstractive','boolean','all']:
	with open(f'./R_{qt}_eval.json','w') as f:
		for item in final_eval[qt]:
			f.write(json.dumps(item))
			f.write('\n')
