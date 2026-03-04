import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
os.environ['MAX_PIXELS'] = '1003520'
os.environ['VIDEO_MAX_PIXELS'] = '50176'
os.environ['FPS_MAX_FRAMES'] = '12'
os.environ['MODELSCOPE_CACHE'] = '/data/ZS/model'
from swift.infer_engine import TransformersEngine, RequestConfig, InferRequest
model = 'Qwen/Qwen3-VL-8B-Instruct'

# 加载推理引擎
engine = TransformersEngine(model, max_batch_size=2)
request_config = RequestConfig(max_tokens=512, temperature=0)

# 这里使用了3个infer_request来展示batch推理
infer_requests = [
    InferRequest(messages=[{'role': 'user', 'content': 'who are you?'}]),
    InferRequest(messages=[{'role': 'user', 'content': '<image><image>两张图的区别是什么？'}],
                 images=['http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/cat.png',
                         'http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/animal.png']),
    InferRequest(messages=[{'role': 'user', 'content': '<video>describe the video'}],
                 videos=['https://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/baby.mp4']),
]
resp_list = engine.infer(infer_requests, request_config)
query0 = infer_requests[0].messages[0]['content']
print(f'response0: {resp_list[0].choices[0].message.content}')
print(f'response1: {resp_list[1].choices[0].message.content}')
print(f'response2: {resp_list[2].choices[0].message.content}')