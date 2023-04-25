import json
import os
import openai
import ujson
os.environ['OPENAI_API_KEY'] = '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'
openai.api_key = os.getenv("OPENAI_API_KEY")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'mps' if torch.backend.mps.is_available() else 'cpu'
embed = OpenAIEmbeddings()

with open('qasper-sample.json') as f:
    qasper = json.load(f)

def get_embeddings(text):
    # Returns OpenAI embeddings()
    print('embedding')
    return embed.embed_query(text)
    # return torch.randn(1536)

pid = {}
qid = {}
# embeddings_q, embeddings_p = {}, {}
with open('./query_embeddings.json') as f:
    embeddings_q = json.load(f)

# with open('collection.tsv', encoding='utf8') as f:
#     for line in f:
#         _line = line.split('\t')
#         pid[int(_line[0])] = _line[1]
#         embeddings_p[int(_line[0])] = get_embeddings(_line[1])

# with open('./passage_embeddings.json', 'w') as f:
#     json.dump(embeddings_p, f)

cf = 0
with open('queries.tsv', encoding='utf8') as f:
    for line in f:
        cf+=1
        _line = line.split('\t')
        if _line[0] in embeddings_q.keys():continue
        qid[int(_line[0])] = _line[1]
        embeddings_q[int(_line[0])] = embed.embed_query(_line[1])
        if cf%250==0:
            with open('./query_embeddings.json', 'w') as f:
                json.dump(embeddings_q, f, indent=2)

with open('./query_embeddings.json', 'w') as f:
    json.dump(embeddings_q, f, indent=2)

# print(embeddings_q)
dataset = [] # (qid, pid, label)
val_dataset = []

'''
with open('triples.jsonl') as f:
    for line in f:
        line = ujson.loads(line)
        # print(line)
        dataset.append([embeddings_q[line[0]], embeddings_p[line[1]], 1])
        dataset.append([embeddings_q[line[0]], embeddings_p[line[3]], -1])
        dataset.append([embeddings_q[line[0]], embeddings_p[line[2]], -1])

with open('val_triples.jsonl') as f:
    for line in f:
        line = ujson.loads(line)
        # print(line)
        val_dataset.append([embeddings_q[line[0]], embeddings_p[line[1]], 1])
        val_dataset.append([embeddings_q[line[0]], embeddings_p[line[3]], -1])
        val_dataset.append([embeddings_q[line[0]], embeddings_p[line[2]], -1])

print(len(val_dataset))

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.drop = nn.Dropout(0.4)
        self.lin1 = nn.Linear(1536, 1536)
        self.lin2 = nn.Linear(1536, 1536)

    def forward(self, x):
        out = self.drop(F.relu(self.lin1(x)))
        out = self.lin2(out)
        return out
    
train_dl = DataLoader(dataset, batch_size=32, shuffle=True)
val_dl = DataLoader(val_dataset, batch_size=32, shuffle=True)

model = Model().to(device)
loss_fn = nn.CosineEmbeddingLoss()
opt = torch.optim.Adam(model.parameters(), lr=1e-4)

def validation(model, val_dl):
    model.eval()
    loss_total=0
    with torch.no_grad():
        for i, (x,y,z) in enumerate(train_dl):
            x,y,z = x.to(device), y.to(device), z.to(device)
            qemb = model(x)
            pemb = model(y)
            loss = loss_fn(qemb,pemb,z)

            loss_total = (loss_total*i + loss) / (i+1)
    return loss_total.item()

def train_one_epoch(model, train_dl, val_dl):
    loss_total = 0

    model.train()
    for i, (x,y,z) in enumerate(train_dl):
        x,y,z = x.to(device), y.to(device), z.to(device)
        qemb = model(x)
        pemb = model(y)
        loss = loss_fn(qemb,pemb,z)

        opt.zero_grad()
        loss.backward()
        opt.step()

        loss_total = (loss_total*i + loss) / (i+1)

    val_loss = validation(model, val_dl)
    print('Train Loss: ', loss_total.item())
    print('Validation Loss: ', val_loss)

    return loss_total.item(), val_loss
        
train_loss, val_loss = [], []

for epoch in range(10):
    print('--------------------------------------------------------')
    print(f'Epoch {epoch+1}: \n')
    t_loss, v_loss = train_one_epoch(model, train_dl, val_dl)
    train_loss.append(t_loss)
    val_loss.append(v_loss)

print(train_loss)
print(val_loss)
'''