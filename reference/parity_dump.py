"""Dumps reference outputs for the Kotlin-vs-Python parity harness (faceswap-app/paritytest_aiimage)."""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sd_ref import *
from face_mask import detect_ovals, mask_from_ovals
from canny import canny, gray_u8
ROOT=os.path.abspath(os.path.join(HERE,"..")); OUT=os.path.join(ROOT,"parity"); os.makedirs(OUT,exist_ok=True)
W=os.path.join(ROOT,"work")
def save(a,name): Image.fromarray(a).save(os.path.join(OUT,name))
unit={}
# tokenizer
tok=ClipTokenizer(os.path.join(HERE,"tokenizer","vocab.json"),os.path.join(HERE,"tokenizer","merges.txt"))
texts=["a beach at sunset","Watercolor painting of Two People, café in Paris!!  ","it's 3D-cartoon & neon :) 42 cats", "東京 at night, naïve art", build_prompt("make it a snowy winter night","scene"), full_negative("sketch")]
unit["tokens"]=[[t,tok.encode(t)] for t in texts]
unit["gaussian_1234_3"]=gaussian(1234,3,11).tolist()
unit["gaussian_big"]=[float(x) for x in gaussian(987654321987,0,20000)[-5:]]
unit["timesteps"]={str(s):lcm_timesteps(s,4) for s in [0.3,0.35,0.5,0.55,0.6,0.65,0.8]}
unit["wemb"]=w_embedding(8.0).tolist()[0]
unit["target"]=[[w,h,*target_size(w,h)] for w,h in [(4000,3000),(3000,4000),(1920,1080),(1080,1920),(1000,1000),(4032,3024),(2000,3000),(5000,1000)]]
unit["acp"]=[float(ACP[i]) for i in [0,1,499,999]]
unit["prompts"]=[[u,p,build_prompt(u,p)] for u,p in [("make it a beach at sunset","scene"),("Turn it into a cozy cafe in paris.","watercolor"),("","anime"),("please change me to a knight","oil"),("  two   dogs ","none")]]
unit["negatives"]=[[p,full_negative(p)] for p in PRESETS]+[["watercolor|user",full_negative("watercolor","no hats")],["sketch|empty",full_negative("sketch","")]]
unit["face_strength"]=[[s,d,face_strength_for(s,d)] for s,d in [(0.5,0.25),(0.55,0.3),(0.3,0.25),(0.6,0.3)]]
pipe=Pipeline([os.path.join(ROOT,"models")],threads=8,keep_open=True)
cases=[("portrait","portrait_c2.jpg","watercolor","",True,1234,True,True),
       ("couple","couple_c1.jpg","sketch","",False,42,True,False),
       ("taj","taj.jpg","scene","make it a snowy winter night",True,7,True,True)]
meta=[]
for key,fn,preset,user,best,seed,face,up in cases:
    im=ImageOps.exif_transpose(Image.open(os.path.join(W,fn))).convert("RGB"); a=np.asarray(im)
    save(a,f"{key}_decoded.png")                      # exact decoded pixels (Kotlin tests resize_cover on these)
    Wt,Ht=target_size(a.shape[1],a.shape[0]); rgb=resize_cover(a,Wt,Ht); save(rgb,f"{key}_input.png")
    ov=detect_ovals(rgb) if face else []
    json.dump([[p[:,0].tolist(),p[:,1].tolist()] for p in ov],open(os.path.join(OUT,f"{key}_ovals.json"),"w"))
    m=mask_from_ovals(Ht,Wt,ov) if ov else np.zeros((Ht,Wt),np.float32); save(np.clip(np.floor(m*255+0.5),0,255).astype(np.uint8),f"{key}_mask.png")
    np.save(os.path.join(OUT,f"{key}_mask.npy"),m); m.astype("<f4").tofile(os.path.join(OUT,f"{key}_mask.f32"))
    save((canny(gray_u8(rgb))*255).astype(np.uint8),f"{key}_canny.png")
    kw=preset_kwargs(preset,best)
    out,info=pipe.generate(rgb,build_prompt(user,preset),seed=seed,face_mask=m if ov else None,**kw)
    save(out,f"{key}_base.png")
    kc=keep_colors(rgb[::-1].copy(),rgb,0.5,1.0); save(kc,f"{key}_keepcolors.png")   # standalone keep_colors test (flipped image as 'out')
    save(lanczos2x(rgb),f"{key}_lanczos2x.png")
    if up: save(upscale2x(pipe.m,out),f"{key}_final.png")
    meta.append(dict(key=key,preset=preset,user=user,quality="BEST" if best else "FAST",seed=seed,keepFace=face,upscale=up,
        nsfw=info["nsfw"],timesteps=info["timesteps"],size=info["size"],faces=len(ov),seconds=info["total_s"]))
    print(key,info["size"],"faces",len(ov),"nsfw %.5f"%info["nsfw"],"%.1fs"%info["total_s"],flush=True)
unit["cases"]=meta
json.dump(unit,open(os.path.join(OUT,"reference.json"),"w"),indent=1)
