from decord import VideoReader
from decord import cpu, gpu

vr = VideoReader('abseiling_k400.mp4', ctx=cpu(0))

print('video frames:', len(vr))
