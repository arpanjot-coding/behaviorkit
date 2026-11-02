"""Generate a rights-free plumbing fixture, not a vision accuracy dataset."""
import argparse
from pathlib import Path
import av
import numpy as np

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--output',type=Path,default=Path('examples/synthetic-scene.mp4'))
a=p.parse_args();a.output.parent.mkdir(parents=True,exist_ok=True)
with av.open(str(a.output),'w') as container:
    stream=container.add_stream('mpeg4',rate=10)
    stream.width=640;stream.height=360;stream.pix_fmt='yuv420p'
    for i in range(61):
        image=np.full((360,640,3),45,dtype=np.uint8)
        image[50:320,65:575]=[57,66,75]
        x=80+i*4
        image[160:260,x:x+45]=[100,130,160]
        frame=av.VideoFrame.from_ndarray(image,format='rgb24')
        for packet in stream.encode(frame):container.mux(packet)
    for packet in stream.encode():container.mux(packet)
print(a.output)
