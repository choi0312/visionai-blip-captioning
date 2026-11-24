"""Independently recompute corpus BLEU and CIDEr-D arithmetic from saved PTB tokens."""
from pathlib import Path
from collections import Counter,defaultdict
import os,json,math,hashlib
import numpy as np
ROOT=Path(os.environ.get('VISIONAI_ROOT',str(Path(__file__).resolve().parents[1])));R=ROOT/'results'
e=json.loads((R/'tokenized_evidence.json').read_text());m=json.loads((R/'metrics.json').read_text())
refs=e['references'];caps=json.loads((R/'captions.json').read_text());manifest=json.loads((ROOT/'data/manifest.json').read_text())
ids=[str(c['image_id']) for c in caps];assert len(ids)==len(set(ids))==100 and set(ids)==set(refs)
assert all(len(refs[i])==5 for i in ids)
def ngrams(s):
 words=s.split();return len(words),Counter(tuple(words[j:j+n]) for n in range(1,5) for j in range(max(0,len(words)-n+1)))
rgrams={i:[ngrams(r) for r in refs[i]] for i in ids}
df=Counter()
for i in ids:df.update(set(g for _,cs in rgrams[i] for g in cs))
logN=math.log(len(ids))
def vector(cnt):
 v={g:n*(logN-math.log(max(1,df[g]))) for g,n in cnt.items()};norm=[math.sqrt(sum(x*x for g,x in v.items() if len(g)==n)) for n in range(1,5)]
 length=sum(x for g,x in cnt.items() if len(g)==2)
 return v,norm,length
results={};per_expected=json.loads((R/'per_image_metrics.json').read_text())
for name in ['greedy','beam3']:
 total_guess=[0]*4;total_correct=[0]*4;testlen=0;reflen=0;cid=[]
 for k,i in enumerate(ids):
  hyp=e['hypotheses'][name][i][0];L,c=ngrams(hyp);r=rgrams[i]
  testlen+=L;reflen+=min((abs(l-L),l) for l,_ in r)[1]
  maximum={g:max(rc.get(g,0) for _,rc in r) for g in c}
  for n in range(1,5):
   total_guess[n-1]+=max(0,L-n+1);total_correct[n-1]+=sum(min(v,maximum[g]) for g,v in c.items() if len(g)==n)
  v,norm,hl=vector(c);score=0.
  for _,rc in r:
   rv,rnorm,rl=vector(rc);penalty=math.exp(-((hl-rl)**2)/72.)
   for n in range(1,5):
    den=norm[n-1]*rnorm[n-1]
    if den:score+=sum(min(x,rv.get(g,0.))*rv.get(g,0.) for g,x in v.items() if len(g)==n)/den*penalty
  cid.append(10*score/(4*len(r)))
  assert abs(cid[-1]-per_expected[name][k]['CIDEr'])<1e-9
 product=1.
 for a,b in zip(total_correct,total_guess):product*=(a+1e-15)/(b+1e-9)
 ratio=(testlen+1e-15)/(reflen+1e-9);bleu=product**.25*(math.exp(1-1/ratio) if ratio<1 else 1.)
 cider=sum(cid)/len(cid)
 assert abs(bleu-m['conditions'][name]['BLEU_4'])<1e-10
 assert abs(cider-m['conditions'][name]['CIDEr'])<1e-9
 assert all(c[name]['caption'] and c[name]['token_ids'] for c in caps)
 results[name]={'BLEU_4':bleu,'CIDEr':cider,'clipped_ngram_counts':total_correct,'possible_ngram_counts':total_guess,'hypothesis_length':testlen,'closest_reference_length':reflen}
attn=np.load(R/'cross_attention.npz');ar=json.loads((R/'attention_records.json').read_text())
assert len(ar)==4
for a in ar:
 arr=attn[str(a['image_id'])];assert arr.shape==(len(a['sequence_ids'])-1,577)
 assert np.isfinite(arr).all() and (arr>=0).all() and np.allclose(arr.sum(1),1,atol=2e-5)
 assert len(a['predicted_tokens'])==len(arr)
assert all(c['korean']['text'] for c in caps)
assert manifest['known_train_overlap']==manifest['known_val_overlap']==0
report={'status':'passed','n':100,'independent_metrics':results,'checks':['100 distinct selected IDs','five PTB references each','independent corpus BLEU-4 clipping and brevity penalty','independent TF-IDF clipped CIDEr-D arithmetic including per-image scores','cross-attention token-row shape and probability sums','100 translation outputs','declared known train/val split exclusions'],
 'sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [R/'captions.json',R/'cross_attention.npz',R/'tokenized_evidence.json']}}
(R/'verification.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
