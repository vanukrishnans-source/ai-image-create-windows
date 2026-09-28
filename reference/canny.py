"""Portable Canny (same semantics as OpenCV cv2.Canny(img, low, high), aperture 3, L1 gradient).
Written with plain loops-friendly logic so the Kotlin port is line-for-line."""
import numpy as np
def gray_u8(rgb_u8):
    # ITU-R 601 luma with OpenCV's fixed-point rounding (R*4899 + G*9617 + B*1868 + 8192) >> 14
    r=rgb_u8[...,0].astype(np.int32); g=rgb_u8[...,1].astype(np.int32); b=rgb_u8[...,2].astype(np.int32)
    return ((r*4899+g*9617+b*1868+8192)>>14).astype(np.uint8)
def canny(gray, low=100, high=200):
    H,W=gray.shape
    p=np.pad(gray.astype(np.int32),1,mode="edge")
    dx=(p[:-2,2:]+2*p[1:-1,2:]+p[2:,2:])-(p[:-2,:-2]+2*p[1:-1,:-2]+p[2:,:-2])
    dy=(p[2:,:-2]+2*p[2:,1:-1]+p[2:,2:])-(p[:-2,:-2]+2*p[:-2,1:-1]+p[:-2,2:])
    mag=np.abs(dx)+np.abs(dy)
    M=np.zeros((H+2,W+2),np.int64); M[1:-1,1:-1]=mag
    ax=np.abs(dx).astype(np.int64); ay=np.abs(dy).astype(np.int64)<<15
    tg22x=ax*13573; tg67x=tg22x+(ax<<16)
    c=M[1:-1,1:-1]; L=M[1:-1,:-2]; R=M[1:-1,2:]; U=M[:-2,1:-1]; D=M[2:,1:-1]
    s=np.where((dx^dy)<0,-1,1)
    # diagonal neighbours: prev row col-s, next row col+s
    UL=M[:-2,:-2]; UR=M[:-2,2:]; DL=M[2:,:-2]; DR=M[2:,2:]
    Up_s=np.where(s<0,UR,UL); Dn_s=np.where(s<0,DL,DR)
    horiz=ay<tg22x; vert=(~horiz)&(ay>tg67x); diag=(~horiz)&(~vert)
    keep=np.zeros((H,W),bool)
    keep|=horiz&(c>L)&(c>=R); keep|=vert&(c>U)&(c>=D); keep|=diag&(c>Up_s)&(c>Dn_s)
    keep&=c>low
    strong=keep&(c>high); weak=keep&~strong
    out=np.zeros((H,W),np.uint8); out[strong]=1
    stack=list(zip(*np.nonzero(strong)))
    while stack:
        y,x=stack.pop()
        for yy in (y-1,y,y+1):
            if yy<0 or yy>=H: continue
            for xx in (x-1,x,x+1):
                if 0<=xx<W and weak[yy,xx] and not out[yy,xx]:
                    out[yy,xx]=1; stack.append((yy,xx))
    return out
