# RWKV-Batch-infer
- 支持pytorch，cuda，fla 3种batch推理
os.environ['RWKV_JIT_ON'] = '1'
os.environ["RWKV_CUDA_ON"] = '1' # '1' to compile CUDA kernel (10x faster), requires c++ compiler & cuda libraries
os.environ["RWKV_fla_ON"] = '0' # use fla
!!!cuda不可和fla同时开启
# Usage
- 下面是快速使用，utils里新增了encode_bsz、decode_bsz、gen_bsz、sample_bsz函数以便快速使用，但未优化如果你要接入自己的后端还需自行优化，可以直接使用model.forward即可
```
import os, sys, torch, time
import numpy as np
np.set_printoptions(precision=4, suppress=True, linewidth=200)
current_path = os.path.dirname(os.path.abspath(__file__))
print(current_path)

# set these before import RWKV
os.environ['RWKV_JIT_ON'] = '1'
os.environ["RWKV_CUDA_ON"] = '1' # '1' to compile CUDA kernel (10x faster), requires c++ compiler & cuda libraries
os.environ["RWKV_fla_ON"] = '0' # use fla
from rwkv.model import RWKV # pip install rwkv
from rwkv.utils import PIPELINE, PIPELINE_ARGS

model = RWKV(model='/home/rwkv/JL/model/RWKV-x060-World-7B-v2.1-20240507-ctx4096.pth', strategy='cuda bf16')
pipeline = PIPELINE(model, "rwkv_vocab_v20230424")


args = PIPELINE_ARGS(temperature = 1.0, top_p = 0.0, top_k=0, # top_k = 0 then ignore
                     alpha_frequency = 0.0,
                     alpha_presence = 0.0,
                     token_ban = [0], # ban the generation of some tokens
                     token_stop = [], # stop generation whenever you see any token here
                     chunk_len = 256) # split input into chunks to save VRAM (shorter -> slower)

########################################################################################################

msg2 = ["Q: 你能做什么？\n\nA:","Q: 你是谁？\n\nA:","Q: 介绍一下openai这个公司，以及他的产品\n\nA:"]
msg3 = ["Q: 介绍一下openai这个公司，以及他的产品\n\nA:"]

token, mask = pipeline.encode_bsz(msg2)
print('tokens:',token)
print('mask:', mask)

prefill,states = model.forward(token, state=None, mask=mask)

answer, state = pipeline.gen_bsz(msg2, token_count=500, args=args)

print(answer, len(state))
```
