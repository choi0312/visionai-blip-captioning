"""Select CC BY 2.0 COCO images from the BLIP author's Karpathy test split."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
import os,json,hashlib,random,urllib.request,re,time
ROOT=Path(os.environ.get('VISIONAI_ROOT',str(Path(__file__).resolve().parents[1])))
D=ROOT/'data';D.mkdir(exist_ok=True);C=ROOT/'source_cache';C.mkdir(exist_ok=True)
SOURCES={'captions':'https://raw.githubusercontent.com/tylin/coco-caption/master/annotations/captions_val2014.json',
 'test':'https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_test.json',
 'train':'https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_train.json',
 'val':'https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_val.json'}
def read(name,url):
 p=C/(name+'.json')
 if not p.exists():p.write_bytes(urllib.request.urlopen(url,timeout=120).read())
 b=p.read_bytes();return json.loads(b),hashlib.sha256(b).hexdigest()
source={};source_sha={}
for name,url in SOURCES.items():source[name],source_sha[name]=read(name,url)
def image_id(p):return int(Path(p).stem.split('_')[-1])
test={image_id(r['image']):r for r in source['test']};train={image_id(r['image']) for r in source['train']};val={image_id(r['image']) for r in source['val']}
metadata={r['id']:r for r in source['captions']['images']}
refs={}
for a in source['captions']['annotations']:refs.setdefault(a['image_id'],[]).append(a)
candidates=sorted(i for i in test if i in metadata and metadata[i]['license']==4 and i not in train and i not in val and len(refs[i])>=5)
selected=sorted(random.Random(42).sample(candidates,100))
records=[]
for i in selected:
 r=metadata[i];references=[a['caption'] for a in sorted(refs[i],key=lambda x:x['id'])[:5]]
 assert all(str(x).strip() in {a['caption'].strip() for a in refs[i]} for x in test[i]['caption'])
 filename=r['file_name'];url='https://s3.amazonaws.com/images.cocodataset.org/val2014/'+filename
 records.append({'image_id':i,'file_name':filename,'url':url,'flickr_url':r['url'],'license':{'id':4,'name':'CC BY 2.0','url':'https://creativecommons.org/licenses/by/2.0/'},'width':r['width'],'height':r['height'],'references':references})
def fetch(r):
 p=D/r['file_name']
 if not p.exists():
  for attempt in range(4):
   try:
    p.write_bytes(urllib.request.urlopen(r['url'],timeout=45).read());break
   except Exception:
    if attempt==3:raise
    time.sleep(attempt+1)
 r['sha256']=hashlib.sha256(p.read_bytes()).hexdigest();r['bytes']=p.stat().st_size
 return r
with ThreadPoolExecutor(max_workers=6) as pool:
 out=[]
 for i,f in enumerate(as_completed([pool.submit(fetch,r) for r in records])):
  out.append(f.result())
  if (i+1)%25==0:print('IMAGES_DOWNLOADED',i+1,flush=True)
manifest={'selection':'100 of CC BY2 images in author Karpathy test, disjoint from author train/val lists; sorted candidate IDs, Python random.Random(42).sample then sort',
 'seed':42,'candidate_count':len(candidates),'n':100,'references_per_image':5,'known_train_overlap':0,'known_val_overlap':0,
 'sources':SOURCES,'source_sha256':source_sha,'annotation_license':'CC BY 4.0 COCO Consortium','images':sorted(out,key=lambda x:x['image_id']),
 'limits':'License-restricted sample; not full Karpathy test. Unknown pretraining/web duplicate overlap cannot be excluded.'}
(D/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
print('DATA_READY',len(candidates),100,sum(r['bytes'] for r in out),flush=True)
