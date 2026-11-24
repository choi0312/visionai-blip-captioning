"""Download only the 100 fixed public images and verify their saved SHA-256 values."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
import os,json,urllib.request,hashlib,time
ROOT=Path(os.environ.get('VISIONAI_ROOT',str(Path(__file__).resolve().parents[1])));D=ROOT/'data';D.mkdir(exist_ok=True)
manifest=json.loads((D/'manifest.json').read_text())
def fetch(r):
 p=D/r['file_name']
 if p.exists() and hashlib.sha256(p.read_bytes()).hexdigest()==r['sha256']:return
 for attempt in range(4):
  try:
   b=urllib.request.urlopen(r['url'],timeout=60).read();assert hashlib.sha256(b).hexdigest()==r['sha256']
   p.write_bytes(b);return
  except Exception:
   if attempt==3:raise
   time.sleep(attempt+1)
with ThreadPoolExecutor(max_workers=6) as pool:
 for i,f in enumerate(as_completed([pool.submit(fetch,r) for r in manifest['images']])):
  f.result()
  if (i+1)%25==0:print('VERIFIED_IMAGES',i+1,flush=True)
