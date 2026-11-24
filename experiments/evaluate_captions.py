"""COCO-caption metrics on a fixed 100-image sample, with PTB-tokenized evidence saved."""
from pathlib import Path
import os,json
import numpy as np
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider
ROOT=Path(os.environ.get('VISIONAI_ROOT','/content/visionai_week12'));R=ROOT/'results'
manifest=json.loads((ROOT/'data/manifest.json').read_text());caps=json.loads((R/'captions.json').read_text());ids=[r['image_id'] for r in caps]
assert ids==[r['image_id'] for r in manifest['images']]
refs={r['image_id']:[{'caption':c} for c in r['references']] for r in manifest['images']}
tok=PTBTokenizer();ref_tokens=tok.tokenize(refs)
metrics={};per_image={};evidence={'references':ref_tokens,'hypotheses':{}}
for condition in ['greedy','beam3']:
 hypotheses={r['image_id']:[{'caption':r[condition]['caption']}] for r in caps}
 hyp_tokens=tok.tokenize(hypotheses);evidence['hypotheses'][condition]=hyp_tokens
 bleu,bleu_each=Bleu(4).compute_score(ref_tokens,hyp_tokens,verbose=0)
 cider,cider_each=Cider().compute_score(ref_tokens,hyp_tokens)
 words=[len(hyp_tokens[i][0].split()) for i in ids]
 metrics[condition]={'BLEU_1':float(bleu[0]),'BLEU_4':float(bleu[3]),'CIDEr':float(cider),'mean_ptb_words':float(np.mean(words)),
 'ended_with_eos':sum(bool(r[condition]['ended_with_eos']) for r in caps),'n':len(caps)}
 per_image[condition]=[{'image_id':i,'BLEU_4':float(bleu_each[3][j]),'CIDEr':float(cider_each[j]),'ptb_words':words[j]} for j,i in enumerate(ids)]
change=np.array([r['CIDEr'] for r in per_image['beam3']])-np.array([r['CIDEr'] for r in per_image['greedy']])
result={'conditions':metrics,'paired':{'changed_captions':sum(r['greedy']['caption']!=r['beam3']['caption'] for r in caps),
 'cider_improved':int((change>1e-10).sum()),'cider_regressed':int((change< -1e-10).sum()),'cider_equal':int((np.abs(change)<=1e-10).sum())},
 'evaluation':{'n_images':100,'references_per_image':5,'tokenizer':'Stanford PTBTokenizer (pycocoevalcap 1.2)',
 'BLEU_4':'corpus cumulative 1-4 gram geometric mean, clipped modified precision, closest-reference brevity penalty; raw 0-1 scale',
 'CIDEr':'pycocoevalcap Cider scorer, 1-4 gram TF-IDF, clipping and Gaussian length penalty sigma6, library x10 scale; document frequencies estimated on the same 100 reference groups',
 'limits':'scores not directly comparable with full 5000-image benchmark; no Korean reference or formal human-rating study'}}
for name,obj in [('metrics.json',result),('per_image_metrics.json',per_image),('tokenized_evidence.json',evidence)]:
 (R/name).write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
print('METRICS',json.dumps(result),flush=True);print('EVALUATION_DONE',flush=True)
