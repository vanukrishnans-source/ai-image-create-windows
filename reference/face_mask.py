"""Face 'likeness' mask: MediaPipe FaceLandmarker (same model file the face-swap core ships/uses)
-> face-oval polygon (grown 12 %, forehead extended), filled + Gaussian-feathered.
The polygon fill + blur are portable (no OpenCV): the phone (FaceMask.kt) gets the same landmarks from core/FaceDetector
and runs exactly this math."""
import numpy as np, math, os
MODEL="/workspace/tools/models/face_landmarker.task"
OVAL=[10,338,297,332,284,251,389,356,454,323,361,288,397,365,379,378,400,377,152,148,176,149,150,136,172,58,132,93,234,127,162,21,54,103,67,109]
_lm=None
def detect_ovals(rgb_u8):
    """List of 36x2 oval point arrays (pixels) for every face MediaPipe finds."""
    global _lm
    import mediapipe as mp
    from mediapipe.tasks.python import vision, BaseOptions
    if _lm is None:
        _lm=vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(base_options=BaseOptions(model_asset_path=MODEL),running_mode=vision.RunningMode.IMAGE,num_faces=6,min_face_detection_confidence=0.4,min_face_presence_confidence=0.4))
    H,W,_=rgb_u8.shape
    res=_lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB,data=np.ascontiguousarray(rgb_u8)))
    return [np.array([[f[i].x*W,f[i].y*H] for i in OVAL],np.float64) for f in res.face_landmarks]
def fill_polygon(H, W, pts):
    """Even-odd scanline fill; pixel (x,y) is inside if its centre (x+.5,y+.5) is."""
    m=np.zeros((H,W),np.float64); n=len(pts)
    for y in range(H):
        yc=y+0.5; xs=[]
        for i in range(n):
            x0,y0=pts[i]; x1,y1=pts[(i+1)%n]
            if (y0<=yc<y1) or (y1<=yc<y0): xs.append(x0+(yc-y0)*(x1-x0)/(y1-y0))
        xs.sort()
        for k in range(0,len(xs)-1,2):
            a=max(0,math.ceil(xs[k]-0.5)); b=min(W,math.ceil(xs[k+1]-0.5))
            if b>a: m[y,a:b]=1.0
    return m
def blur_axis(a, sig, axis):
    from sd_ref import _blur_axis
    return _blur_axis(a, sig, axis)
def mask_from_ovals(H, W, ovals, grow=0.12, feather=0.08):
    m=np.zeros((H,W),np.float64)
    for pts in ovals:
        pts=np.asarray(pts,np.float64); c=pts.mean(0); size=float(np.max(pts.max(0)-pts.min(0)))
        p=c+(pts-c)*(1+grow)
        top=p[:,1]<c[1]; p[top,1]-=0.10*size  # extend forehead (hairline)
        mm=fill_polygon(H,W,p)
        k=max(3,int(feather*size)|1); sig=k/2.0
        mm=blur_axis(blur_axis(mm,sig,0),sig,1)
        m=np.maximum(m,mm)
    return np.clip(m,0,1).astype(np.float32)
def face_mask(rgb_u8, grow=0.12, feather=0.08):
    ov=detect_ovals(rgb_u8); H,W,_=rgb_u8.shape
    return mask_from_ovals(H,W,ov,grow,feather), len(ov)
