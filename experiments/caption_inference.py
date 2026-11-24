"""Frozen BLIP caption generation, teacher-forced cross-attention, and EN->KO translation."""
from pathlib import Path
import os,json,time,hashlib,platform,random,importlib.metadata as md
os.environ['HF_HUB_DISABLE_IMPLICIT_TOKEN']='1'
import numpy as np,torch
from PIL import Image
from transformers import BlipProcessor,BlipForConditionalGeneration,M2M100ForConditionalGeneration,M2M100Tokenizer
ROOT=Path(os.environ.get('VISIONAI_ROOT','/content/visionai_week12'));D=ROOT/'data';R=ROOT/'results';R.mkdir(parents=True,exist_ok=True)
MODEL='Salesforce/blip-image-captioning-base';REV='82a37760796d32b1411fe092ab5d4e227313294b'
TRANS='facebook/m2m100_418M';TREV='55c2e61bbf05dfb8d7abccdc3fae6fc8512fd636'
SEED=42;BATCH=8
random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED)
torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
device=torch.device(os.environ.get('VISIONAI_DEVICE','cuda'))
assert (device.type=='cuda' and torch.cuda.is_available()) or (device.type=='mps' and torch.backends.mps.is_available())
def sync():
 if device.type=='cuda':torch.cuda.synchronize()
 else:torch.mps.synchronize()
def reset_peak():
 if device.type=='cuda':torch.cuda.reset_peak_memory_stats()
def cuda_peak():return torch.cuda.max_memory_allocated() if device.type=='cuda' else None
def write(path,obj):path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
manifest=json.loads((D/'manifest.json').read_text());images=manifest['images'];assert len(images)==100
common={'max_new_tokens':30,'min_new_tokens':3,'do_sample':False,'num_return_sequences':1,'length_penalty':1.0,'repetition_penalty':1.0,'no_repeat_ngram_size':0}
conditions={'greedy':{'num_beams':1},'beam3':{'num_beams':3,'early_stopping':True}}
protocol={'model':MODEL,'revision':REV,'seed':SEED,'precision':'float32','batch_size':BATCH,'gradient_updates':0,
 'input':'images only, no text prefix','conditions':{k:{**common,**v} for k,v in conditions.items()},'primary_condition':'beam3',
 'sample':'100 CC BY 2.0 images in Karpathy test, disjoint from author train/val; see data/manifest.json',
 'attention':{'image_ids':[r['image_id'] for r in images[:4]],'rule':'first 4 selected IDs, teacher-force generated beam3 caption, final decoder layer cross-attention mean over all 12 heads',
 'alignment':'row j-1 predicts token sequence[j], with decoder input sequence[:-1]; retain CLS weight separately and renormalize patch weights only for display','causality':'diagnostic association, not causal attribution, segmentation or proof of grounding'},
 'translation':{'model':TRANS,'revision':TREV,'source':'en','target':'ko','input':'beam3 English caption only; translator sees no image','num_beams':4,'max_new_tokens':64,'do_sample':False},
 'evaluation':'5 English references per image; PTB tokenizer; corpus BLEU-4 and CIDEr on these 100 images, IDF on 100 references; no Korean reference scores; no test tuning'}
write(R/'protocol.json',protocol)
for r in images:
 p=D/r['file_name'];assert hashlib.sha256(p.read_bytes()).hexdigest()==r['sha256']
print('DATA_VERIFIED',len(images),flush=True)
processor=BlipProcessor.from_pretrained(MODEL,revision=REV,token=False)
model=BlipForConditionalGeneration.from_pretrained(MODEL,revision=REV,token=False,use_safetensors=False).to(device).eval().requires_grad_(False)
processor.save_pretrained(R/'processor');write(R/'model_config.json',model.config.to_dict())
records=[{'image_id':r['image_id'],'file_name':r['file_name']} for r in images]
# Warm up before measuring the generation loops.
with torch.inference_mode():
 inp=processor(images=Image.open(D/images[0]['file_name']).convert('RGB'),return_tensors='pt').to(device)
 model.generate(**inp,max_new_tokens=3,num_beams=1,do_sample=False)
sync();reset_peak();timings={}
for name,extra in conditions.items():
 sync();start=time.perf_counter()
 with torch.inference_mode():
  for first in range(0,100,BATCH):
   ims=[Image.open(D/r['file_name']).convert('RGB') for r in images[first:first+BATCH]]
   inp=processor(images=ims,return_tensors='pt').to(device)
   out=model.generate(**inp,**common,**extra)
   captions=processor.batch_decode(out,skip_special_tokens=True)
   for k,(seq,caption) in enumerate(zip(out.cpu().tolist(),captions)):
    # Remove only generated trailing pads; retain the BOS and first SEP/EOS for replay.
    eos=model.config.text_config.sep_token_id
    if eos in seq:seq=seq[:seq.index(eos)+1]
    records[first+k][name]={'caption':caption,'token_ids':seq,'new_tokens':len(seq)-1,'ended_with_eos':seq[-1]==eos}
   print('CAPTIONED',name,min(first+BATCH,100),flush=True)
 sync();timings[name+'_loop_s']=time.perf_counter()-start
write(R/'captions.json',records)
print('CAPTIONS_SAVED',flush=True)
attention_meta=[];attention_arrays={};prefix_error=0.
with torch.inference_mode():
 for i in range(4):
  item=images[i];seq=torch.tensor([records[i]['beam3']['token_ids']],device=device)
  inp=processor(images=Image.open(D/item['file_name']).convert('RGB'),return_tensors='pt').to(device)
  vision=model.vision_model(pixel_values=inp['pixel_values'],return_dict=True).last_hidden_state
  out=model.text_decoder(input_ids=seq[:,:-1],attention_mask=torch.ones_like(seq[:,:-1]),encoder_hidden_states=vision,encoder_attention_mask=torch.ones(vision.shape[:2],dtype=torch.long,device=device),output_attentions=True,return_dict=True,use_cache=False)
  a=out.cross_attentions[-1][0].mean(0).float().cpu().numpy()
  assert a.shape[1]==577 and np.allclose(a.sum(-1),1,atol=2e-5)
  attention_arrays[str(item['image_id'])]=a
  attention_meta.append({'image_id':item['image_id'],'sequence_ids':seq[0].tolist(),'predicted_tokens':processor.tokenizer.convert_ids_to_tokens(seq[0,1:].tolist()),'shape':list(a.shape),'layers':len(out.cross_attentions),'heads':out.cross_attentions[-1].shape[1],'image_tokens':vision.shape[1]})
  if i==0:
   j=min(4,seq.shape[1]-2)
   prefix=model.text_decoder(input_ids=seq[:,:j],encoder_hidden_states=vision,return_dict=True,use_cache=False)
   prefix_error=float((prefix.logits[:,-1]-out.logits[:,j-1]).abs().max())
   assert torch.allclose(prefix.logits[:,-1],out.logits[:,j-1],rtol=1e-4,atol=3e-4),prefix_error
np.savez_compressed(R/'cross_attention.npz',**attention_arrays);write(R/'attention_records.json',attention_meta)
print('ATTENTION_SAVED',prefix_error,flush=True)
# Release caption model before loading the translator; all caption/attention outputs are already saved.
blip_parameters=sum(p.numel() for p in model.parameters());blip_peak=cuda_peak()
del model
if device.type=='cuda':torch.cuda.empty_cache()
else:torch.mps.empty_cache()
tokenizer=M2M100Tokenizer.from_pretrained(TRANS,revision=TREV,src_lang='en',token=False)
translator=M2M100ForConditionalGeneration.from_pretrained(TRANS,revision=TREV,token=False,use_safetensors=False).to(device).eval().requires_grad_(False)
translation_parameters=sum(p.numel() for p in translator.parameters())
with torch.inference_mode():
 warm=tokenizer([records[0]['beam3']['caption']],return_tensors='pt').to(device)
 translator.generate(**warm,forced_bos_token_id=tokenizer.get_lang_id('ko'),max_new_tokens=4,num_beams=4,do_sample=False)
sync();reset_peak();start=time.perf_counter()
with torch.inference_mode():
 for first in range(0,100,BATCH):
  texts=[r['beam3']['caption'] for r in records[first:first+BATCH]]
  inp=tokenizer(texts,return_tensors='pt',padding=True,truncation=False).to(device)
  out=translator.generate(**inp,forced_bos_token_id=tokenizer.get_lang_id('ko'),num_beams=4,max_new_tokens=64,do_sample=False,length_penalty=1.0)
  for k,(ko,seq) in enumerate(zip(tokenizer.batch_decode(out,skip_special_tokens=True),out.cpu().tolist())):
   records[first+k]['korean']={'text':ko,'token_ids':seq}
  print('TRANSLATED',min(first+BATCH,100),flush=True)
sync();timings['translation_loop_s']=time.perf_counter()-start
write(R/'captions.json',records)
env={'gpu':torch.cuda.get_device_name(0) if device.type=='cuda' else 'Apple M4 (Metal/MPS)','device':str(device),'python':platform.python_version(),'packages':{k:md.version(k) for k in ['torch','torchvision','transformers','numpy','Pillow','sentencepiece','pycocoevalcap','matplotlib','huggingface_hub']},'seed':SEED,'precision':'float32','batch_size':BATCH,'tf32':False,
 'model_parameters':blip_parameters,'translation_parameters':translation_parameters,'timings':timings,
 'caption_peak_cuda_allocated_bytes':blip_peak,'translation_peak_cuda_allocated_bytes':cuda_peak(),
 'attention_alignment_prefix_logit_max_abs_error':prefix_error,'preprocessing':processor.image_processor.to_dict(),
 'scope':'No additional training. Known Karpathy train/val exclusion; unknown web-pretraining duplication cannot be ruled out.'}
write(R/'environment.json',env);print('INFERENCE_DONE',json.dumps(env),flush=True)
