"""AI Image Create - Python reference pipeline (exactly what the phone will run).
Models: LCM-Dreamshaper-v7 (UNet with T2I-Adapter inputs), CLIP text encoder, SD VAE enc/dec, T2I-Adapter canny, NSFW ViT.
Runtime: ONNX Runtime CPU EP, fp32 compute; weights rebuilt from fp16/int8 blobs into <name>.fp32.bin."""
import numpy as np, onnxruntime as ort, json, os, time, math, argparse, resource
from PIL import Image, ImageOps
from clip_tokenizer import ClipTokenizer
from canny import canny, gray_u8

HERE=os.path.dirname(os.path.abspath(__file__))
DEFAULT_NEG=""  # see presets.json: LCM (w-embedded guidance) runs without CFG; negative prompt only used in "Quality+" mode
# ---------------- sizing / resize (portable) ----------------
def target_size(w, h, long_side=512, mult=64):
    """Pick WxH (multiples of 64, area <= long_side^2, sides 256..1.5*long_side) with the closest aspect ratio
    to the photo (errors < 3 % count as equal, then the largest area). E.g. 3:2 -> 576x384, 4:3 -> 512x384, 16:9 -> 576x320, 1:1 -> 512x512."""
    budget=long_side*long_side; best=None; r=math.log(w/h)
    for W in range(256, int(long_side*1.5)+1, mult):
        for H in range(256, int(long_side*1.5)+1, mult):
            if W*H>budget: continue
            e=abs(math.log(W/H)-r); key=(int(e/0.03), -W*H, e)
            if best is None or key<best[0]: best=(key,W,H)
    return best[1],best[2]
def _weights(n_out, n_in, x0, span):
    """1-D antialiased triangle (bilinear) resampling matrix, source window [x0, x0+span) -> n_out samples."""
    scale=span/n_out; support=max(1.0, scale)
    Wm=np.zeros((n_out,n_in),np.float64)
    for i in range(n_out):
        c=x0+(i+0.5)*scale
        lo=int(math.floor(c-support)); hi=int(math.ceil(c+support))
        tot=0.0; ws=[]
        for k in range(lo,hi+1):
            wv=1.0-abs(k+0.5-c)/support
            if wv>0: ws.append((min(max(k,0),n_in-1),wv)); tot+=wv
        for k,wv in ws: Wm[i,k]+=wv/tot
    return Wm
def resize_cover(rgb_u8, W, H):
    """Scale to cover WxH keeping aspect ratio, centre-crop (no stretching). Returns uint8 HxWx3."""
    h,w,_=rgb_u8.shape; sc=max(W/w,H/h); cw=W/sc; ch=H/sc
    Rx=_weights(W,w,(w-cw)/2.0,cw); Ry=_weights(H,h,(h-ch)/2.0,ch)
    f=rgb_u8.astype(np.float64)
    out=np.einsum("yh,hwc->ywc",Ry,f); out=np.einsum("xw,ywc->yxc",Rx,out)
    return np.clip(np.floor(out+0.5),0,255).astype(np.uint8)
def resize_stretch(rgb_u8, W, H):
    h,w,_=rgb_u8.shape
    Rx=_weights(W,w,0.0,float(w)); Ry=_weights(H,h,0.0,float(h))
    out=np.einsum("yh,hwc->ywc",Ry,rgb_u8.astype(np.float64)); out=np.einsum("xw,ywc->yxc",Rx,out)
    return np.clip(np.floor(out+0.5),0,255).astype(np.uint8)
# ---------------- RNG (portable splitmix64 + Box-Muller) ----------------
M64=(1<<64)-1
def gaussian(seed, stream, n):
    """n standard normals; counter-based so Kotlin can reproduce: key = splitmix64(seed*1000003+stream)."""
    base=np.uint64((seed*1000003+stream)&M64)
    k=np.arange(1,2*((n+1)//2)+1,dtype=np.uint64)
    with np.errstate(over="ignore"):
        z=base+k*np.uint64(0x9E3779B97F4A7C15)
        z=(z^(z>>np.uint64(30)))*np.uint64(0xBF58476D1CE4E5B9)
        z=(z^(z>>np.uint64(27)))*np.uint64(0x94D049BB133111EB)
        z=z^(z>>np.uint64(31))
    u=((z>>np.uint64(11)).astype(np.float64)+1.0)*(1.0/9007199254740992.0)  # (0,1]
    u1=u[0::2]; u2=u[1::2]; r=np.sqrt(-2.0*np.log(u1)); th=2.0*math.pi*u2
    out=np.empty(u1.size*2); out[0::2]=r*np.cos(th); out[1::2]=r*np.sin(th)
    return out[:n].astype(np.float32)
# ---------------- LCM scheduler ----------------
BETAS=(np.linspace(0.00085**0.5,0.012**0.5,1000,dtype=np.float64)**2)
ACP=np.cumprod(1.0-BETAS)
def lcm_timesteps(strength, steps):
    """strength in (0,1]: start noise level t = strength*1000 (LCM 'leading' grid of 50, step 20)."""
    orig=np.arange(1,int(50*strength)+1)*20-1
    idx=np.floor(np.linspace(0,len(orig),num=steps,endpoint=False)).astype(np.int64)
    ts=orig[::-1][idx]
    return [int(t) for t in dict.fromkeys(ts.tolist())]  # unique, keep order
def w_embedding(w, dim=256):
    hd=dim//2; e=np.float32(math.log(10000.0))/np.float32(hd-1)
    f=np.exp(np.arange(hd,dtype=np.float32)*-e); v=np.float32((w-1.0)*1000.0)*f
    return np.concatenate([np.sin(v),np.cos(v)])[None].astype(np.float32)
# ---------------- model store ----------------
def rebuild_fp32(dirp, name):
    """What the phone does once after download: blob (f16 / q8) -> <name>.fp32.bin (16 KiB aligned)."""
    man=json.load(open(os.path.join(dirp,name+".json"))); blob=np.memmap(os.path.join(dirp,name+".wts"),dtype=np.uint8,mode="r")
    out=os.path.join(dirp,name+".fp32.bin")
    with open(out,"wb") as f:
        pos=0
        for t in man["tensors"]:
            f.write(b"\0"*(t["fp32_offset"]-pos)); pos=t["fp32_offset"]; n=t["count"]; o=t["w_offset"]
            if t["kind"]=="f16": a=np.frombuffer(blob[o:o+2*n].tobytes(),np.float16).astype(np.float32)
            else:
                ns=t["nscale"]; sc=np.frombuffer(blob[o:o+4*ns].tobytes(),np.float32); q=np.frombuffer(blob[o+4*ns:o+4*ns+n].tobytes(),np.int8).astype(np.float32)
                shp=t["shape"]
                if t["axis"]==1: a=(q.reshape(shp[0],-1)*sc[None,:]).reshape(-1)
                else: a=(q.reshape(shp[0],-1)*sc[:,None]).reshape(-1)
            f.write(a.astype(np.float32).tobytes()); pos+=4*n
    return out
class Models:
    def __init__(s, dirs, threads=4, keep_open=False):
        s.dirs=dirs; s.threads=threads; s.keep=keep_open; s.sess={}; s.timing={}
    def path(s,name):
        for d in s.dirs:
            p=os.path.join(d,name+".onnx")
            if os.path.exists(p): return p
        raise FileNotFoundError(name)
    def get(s,name):
        if name in s.sess: return s.sess[name]
        so=ort.SessionOptions(); so.intra_op_num_threads=s.threads; so.inter_op_num_threads=1
        so.graph_optimization_level=ort.GraphOptimizationLevel.ORT_ENABLE_ALL; so.enable_cpu_mem_arena=False
        t=time.time(); x=ort.InferenceSession(s.path(name),so,providers=["CPUExecutionProvider"])
        s.timing["load_"+name]=s.timing.get("load_"+name,0)+time.time()-t
        if s.keep: s.sess[name]=x
        return x
    def run(s,name,feed):
        x=s.get(name); t=time.time(); o=x.run(None,feed); s.timing[name]=s.timing.get(name,0)+time.time()-t
        return o
# ---------------- colour keeping (portable post-process) ----------------
def _blur_axis(a, sig, axis):
    rad=int(math.ceil(3*sig)); k=np.exp(-0.5*(np.arange(-rad,rad+1)/sig)**2); k/=k.sum()
    n=a.shape[axis]; idx=np.clip(np.arange(-rad,n+rad),0,n-1)
    p=np.take(a,idx,axis=axis); out=np.zeros_like(a)
    for j,kv in enumerate(k):
        sl=[slice(None)]*a.ndim; sl[axis]=slice(j,j+n); out+=kv*p[tuple(sl)]
    return out
def gblur(a, sig): return _blur_axis(_blur_axis(a,sig,0),sig,1)
def rgb2ycc(x):
    r,g,b=x[...,0],x[...,1],x[...,2]
    return np.stack([0.299*r+0.587*g+0.114*b, -0.168736*r-0.331264*g+0.5*b, 0.5*r-0.418688*g-0.081312*b],-1)
def ycc2rgb(y):
    Y,Cb,Cr=y[...,0],y[...,1],y[...,2]
    return np.stack([Y+1.402*Cr, Y-0.344136*Cb-0.714136*Cr, Y+1.772*Cb],-1)
def keep_colors(out_u8, ref_u8, luma=0.6, chroma=1.0):
    """Move the low-frequency colour/brightness of the result towards the original photo (skin tones, clothes),
    keeping the generated texture/detail. sigma = 2 % of the long side."""
    if luma<=0 and chroma<=0: return out_u8
    sig=0.02*max(out_u8.shape[:2])
    o=rgb2ycc(out_u8.astype(np.float64)); r=rgb2ycc(ref_u8.astype(np.float64))
    d=gblur(r,sig)-gblur(o,sig)
    o[...,0]+=luma*d[...,0]; o[...,1:]+=chroma*d[...,1:]
    return np.clip(np.floor(ycc2rgb(o)+0.5),0,255).astype(np.uint8)
def init_transform(rgb_u8, kind, amount=0.55):
    """Optional per-preset tweak of the image the diffusion starts from (edges/colour-keeping still use the original).
    'sketch': greyscale, lifted towards white paper -> the model can reach a pencil-on-paper look at moderate strength."""
    if not kind: return rgb_u8
    if kind=="sketch":
        g=gray_u8(rgb_u8).astype(np.float64)
        g=255.0-(255.0-g)*amount
        return np.repeat(np.clip(np.floor(g+0.5),0,255).astype(np.uint8)[...,None],3,axis=2)
    raise ValueError(kind)
# ---------------- pipeline ----------------
class Pipeline:
    def __init__(s, model_dirs, threads=4, keep_open=False):
        s.m=Models(model_dirs,threads,keep_open)
        T=os.path.join(HERE,"tokenizer"); s.tok=ClipTokenizer(os.path.join(T,"vocab.json"),os.path.join(T,"merges.txt"))
    def prepare(s, path, long_side=512):
        im=ImageOps.exif_transpose(Image.open(path)).convert("RGB"); a=np.asarray(im)
        W,H=target_size(a.shape[1],a.shape[0],long_side)
        return resize_cover(a,W,H)
    def generate(s, rgb, prompt, strength=0.5, steps=4, w=8.0, adapter_scale=0.6, seed=1234, decoder="vae_decoder", encoder="vae_encoder",
                 negative=None, cfg=1.0, safety=True, cb=None, keep_luma=0.0, keep_chroma=0.0, cfg_steps=99, init=None, init_amount=0.55, face_mask=None, face_strength=None, face_delta=None):
        t0=time.time(); H,W,_=rgb.shape; s.m.timing={}
        ids=np.array([s.tok.encode(prompt)],np.int64)
        cond=s.m.run("text_encoder",{"input_ids":ids})[0]
        unc=None
        if cfg>1.0: unc=s.m.run("text_encoder",{"input_ids":np.array([s.tok.encode(negative or "")],np.int64)})[0]
        x=(init_transform(rgb,init,init_amount).astype(np.float32)/127.5-1.0).transpose(2,0,1)[None]
        lat=s.m.run(encoder,{"image":x})[0]
        h,wl=lat.shape[2],lat.shape[3]
        if adapter_scale>0:
            ed=canny(gray_u8(rgb)).astype(np.float32)[None,None]
            res=[r*np.float32(adapter_scale) for r in s.m.run("adapter_canny",{"edges":ed})]
        else:
            res=[np.zeros((1,c,h//f,wl//f),np.float32) for c,f in [(320,1),(640,2),(1280,4),(1280,8)]]
        ts=lcm_timesteps(strength,steps); wemb=w_embedding(w)
        n=lat.size; a0=ACP[ts[0]]; eps0=gaussian(seed,0,n).reshape(lat.shape)
        z=(np.float32(math.sqrt(a0))*lat+np.float32(math.sqrt(1-a0))*eps0).astype(np.float32)
        if face_strength is None and face_delta is not None and face_mask is not None: face_strength=face_strength_for(strength,face_delta)
        smap=None
        if face_mask is not None and face_strength is not None and face_strength<strength:
            # per-latent-pixel strength map (differential diffusion): faces change less -> likeness is kept
            mk=face_mask.reshape(h,8,wl,8).mean(axis=(1,3)).astype(np.float32)
            smap=(np.float32(strength)*(1-mk)+np.float32(face_strength)*mk)[None,None]
        for i,t in enumerate(ts):
            if smap is not None:
                keep=(smap*1000.0<t)  # these pixels are not allowed to change yet: reset to the noised original
                if keep.any():
                    at=ACP[t]; orig=(np.float32(math.sqrt(at))*lat+np.float32(math.sqrt(1-at))*eps0).astype(np.float32)
                    z=np.where(keep,orig,z).astype(np.float32)
            feed={"sample":z,"timestep":np.array([t],np.int64),"encoder_hidden_states":cond,"timestep_cond":wemb}
            for k in range(4): feed[f"adapter_res{k}"]=res[k]
            eps=s.m.run("unet",feed)[0]
            if unc is not None and i<cfg_steps:
                feed["encoder_hidden_states"]=unc; eu=s.m.run("unet",feed)[0]; eps=eu+np.float32(cfg)*(eps-eu)
            a=ACP[t]; pt=ts[i+1] if i+1<len(ts) else -1; ap=ACP[pt] if pt>=0 else 1.0
            st=t*10.0; cskip=0.25/(st*st+0.25); cout=st/math.sqrt(st*st+0.25)
            x0=(z-np.float32(math.sqrt(1-a))*eps)/np.float32(math.sqrt(a))
            den=(np.float32(cout)*x0+np.float32(cskip)*z).astype(np.float32)
            if pt>=0: z=(np.float32(math.sqrt(ap))*den+np.float32(math.sqrt(1-ap))*gaussian(seed,i+1,n).reshape(z.shape)).astype(np.float32)
            else: z=den
            if cb: cb(i,len(ts))
        if decoder=="vae_decoder": img=s.m.run("vae_decoder",{"latent":z})[0][0]
        else: img=s.m.run("taesd_decoder",{"latent":z})[0][0]  # TAESD output is already [-1,1]
        out=np.clip(np.floor((np.clip(img,-1,1).transpose(1,2,0)+1.0)*127.5+0.5),0,255).astype(np.uint8)
        out=keep_colors(out,rgb,keep_luma,keep_chroma)
        nsfw=None
        if safety:
            sm=resize_stretch(out,224,224).astype(np.float32)/255.0
            px=((sm-0.5)/0.5).transpose(2,0,1)[None].astype(np.float32)
            nsfw=float(s.m.run("safety",{"pixel_values":px})[0][0,1])
        info={"timesteps":ts,"size":[W,H],"total_s":time.time()-t0,"nsfw":nsfw,"timing":dict(s.m.timing)}
        return out,info

# ---------------- 2x upscaler (portable parts; mirrored in the app's SdPipeline.kt / Upscale) ----------------
def _lanczos_taps(n_in):
    out=[]
    for i in range(n_in*2):
        c=(i+0.5)/2.0-0.5; f0=math.floor(c); idx=[]; ws=[]
        for j in range(6):
            k=f0-2+j; d=c-k
            wv=(1.0 if d==0 else math.sin(math.pi*d)/(math.pi*d))*(1.0 if d==0 else math.sin(math.pi*(d/3.0))/(math.pi*(d/3.0))) if abs(d)<3.0 else 0.0
            idx.append(min(max(k,0),n_in-1)); ws.append(wv)
        t=sum(ws); out.append((idx,[w/t for w in ws]))
    return out
def lanczos2x(rgb_u8):
    H,W,_=rgb_u8.shape; f=rgb_u8.astype(np.float64)
    ry=_lanczos_taps(H); rx=_lanczos_taps(W)
    tmp=np.zeros((2*H,W,3))
    for j in range(6):
        idx=np.array([r[0][j] for r in ry]); w=np.array([r[1][j] for r in ry])
        tmp+=w[:,None,None]*f[idx]
    out=np.zeros((2*H,2*W,3))
    for j in range(6):
        idx=np.array([r[0][j] for r in rx]); w=np.array([r[1][j] for r in rx])
        out+=w[None,:,None]*tmp[:,idx]
    return np.clip(np.floor(out+0.5),0,255).astype(np.uint8)
UPSCALE_BLEND=0.7
def upscale2x(models, rgb_u8):
    """Real-ESRGAN general x4v3 at denoise strength 0.5 -> antialiased /2 -> 70/30 blend with Lanczos-3 2x."""
    x=(rgb_u8.astype(np.float32)/255.0).transpose(2,0,1)[None]
    y=models.run("upscaler",{"image":x})[0][0]
    x4=np.clip(np.floor(y.transpose(1,2,0)*255.0+0.5),0,255).astype(np.uint8)
    H,W,_=rgb_u8.shape
    esr=resize_stretch(x4,W*2,H*2).astype(np.float64); lz=lanczos2x(rgb_u8).astype(np.float64)
    return np.clip(np.floor(UPSCALE_BLEND*esr+(1-UPSCALE_BLEND)*lz+0.5),0,255).astype(np.uint8)
PRESETS=json.load(open(os.path.join(HERE,"presets.json")))
DEFAULTS=PRESETS.pop("_defaults")
def preset_kwargs(preset, quality=True):
    """Full generate() settings for a preset (what the app uses when the user doesn't touch the sliders)."""
    p=PRESETS[preset]
    kw=dict(strength=p["strength"],steps=DEFAULTS["steps"],w=p["w"],adapter_scale=p["adapter"],keep_luma=p["keep_luma"],
            keep_chroma=p["keep_chroma"],init=p.get("init"),init_amount=p.get("init_amount",0.55),encoder=DEFAULTS["encoder"],decoder=DEFAULTS["decoder"])
    # Best: CFG with the negative prompt on every step (8 UNet evals). Fast: CFG only on the first step (5 evals) so the
    # (always-on) nudity-avoidance negative terms still steer the composition.
    kw.update(cfg=p.get("cfg",DEFAULTS["cfg"]),negative=full_negative(preset),cfg_steps=99 if quality else 1)
    kw["face_delta"]=p.get("face_delta",0.25)
    return kw
NSFW_NEGATIVE="nude, naked, nsfw, nipples, genitals, sexual"  # always appended, not user-removable
NSFW_THRESHOLDS={"standard":0.5,"relaxed":0.85}  # block if P(nsfw) > threshold; app default = relaxed
def full_negative(preset, user_negative=None):
    """user_negative=None -> app default quality terms. Preset extras + NSFW terms are always appended."""
    p=PRESETS.get(preset,{}) if preset else {}
    parts=[DEFAULTS["negative"] if user_negative is None else user_negative.strip(), p.get("negative_extra",""), NSFW_NEGATIVE]
    return ", ".join(x for x in parts if x)
def face_strength_for(strength, face_delta):
    """'Keep face likeness': faces get a lower strength than the rest of the picture."""
    return max(DEFAULTS["face_min"], min(strength, strength-face_delta))
import re
_INSTR=re.compile(r"^(please\s+)?(make|turn|change|convert|transform|put)\s+(it|this|the photo|the picture|the image|me|him|her|them|us)\s*(into|to|look like|like|in|at|on)?\s*", re.I)
def clean_user_prompt(t):
    """SD is not instruction-tuned: 'make it a beach at sunset' -> 'a beach at sunset'."""
    t=re.sub(r"\s+"," ",(t or "")).strip().rstrip(".!")
    return _INSTR.sub("",t).strip()
def build_prompt(user, preset):
    p=PRESETS[preset] if preset else None
    user=clean_user_prompt(user)
    if p is None: return user
    return p["prompt"].replace("{}",user) if user else p["prompt_empty"]
if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("image"); ap.add_argument("out"); ap.add_argument("--prompt",default=""); ap.add_argument("--preset",default=None)
    ap.add_argument("--strength",type=float); ap.add_argument("--steps",type=int,default=4); ap.add_argument("--w",type=float)
    ap.add_argument("--adapter",type=float); ap.add_argument("--seed",type=int,default=1234); ap.add_argument("--long",type=int,default=512)
    ap.add_argument("--threads",type=int,default=4); ap.add_argument("--fast",action="store_true",help="Fast: CFG on the first step only")
    ap.add_argument("--no-face",action="store_true",help="disable 'keep face likeness'")
    ap.add_argument("--models",default=os.path.join(HERE,"..","models"))
    a=ap.parse_args()
    kw=preset_kwargs(a.preset or "none", not a.fast)
    if a.strength is not None: kw["strength"]=a.strength
    if a.adapter is not None: kw["adapter_scale"]=a.adapter
    if a.w is not None: kw["w"]=a.w
    kw["steps"]=a.steps
    pipe=Pipeline(a.models.split(","),a.threads,keep_open=True)
    rgb=pipe.prepare(a.image,a.long)
    fm=None
    if not a.no_face:
        from face_mask import face_mask
        m,nf=face_mask(rgb); fm=m if nf else None
    out,info=pipe.generate(rgb,build_prompt(a.prompt,a.preset or "none"),seed=a.seed,face_mask=fm,**kw)
    Image.fromarray(out).save(a.out); info["peak_rss_mb"]=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
    print(json.dumps(info))
