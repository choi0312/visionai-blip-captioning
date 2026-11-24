from pathlib import Path
from zipfile import ZipFile,ZIP_DEFLATED
import os
root=Path(os.environ.get('VISIONAI_ROOT','/content/visionai_week12'))
archive=Path(os.environ.get('VISIONAI_ARCHIVE','/content/week12-results.zip'))
with ZipFile(archive,'w',ZIP_DEFLATED) as z:
 for folder in ['results','figures']:
  for p in sorted((root/folder).rglob('*')):
   if p.is_file() and '__pycache__' not in p.parts:z.write(p,p.relative_to(root))
 z.write(root/'data/manifest.json','data/manifest.json')
print('ARCHIVE_BYTES',archive.stat().st_size,flush=True)
