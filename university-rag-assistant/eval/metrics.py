import re
from collections import Counter
def normalize(text): return re.sub(r"[^a-z0-9]+"," ",text.lower()).strip()
def token_f1(prediction,reference):
    p,r=normalize(prediction).split(),normalize(reference).split()
    if not p or not r: return float(p==r)
    common=Counter(p)&Counter(r); overlap=sum(common.values())
    if not overlap:return 0.0
    precision,recall=overlap/len(p),overlap/len(r)
    return 2*precision*recall/(precision+recall)
def recall_at_k(retrieved_ids,relevant_ids,k):
    relevant=set(relevant_ids)
    return len(set(retrieved_ids[:k])&relevant)/len(relevant) if relevant else 0.0
