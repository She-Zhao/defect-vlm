"""
读取ms swift格式的jsonl文件，推理并保存结果（包含Logprobs）。适用于SFT之后的模型
输入：原始模型权重+适配器权重输入+待推理数据
输出：包含推理结果的数据（包含Logprobs）
"""
import os
import math
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2'
os.environ['MAX_PIXELS'] = '1003520'
import json
from tqdm import tqdm
from swift.infer_engine import TransformersEngine, RequestConfig, InferRequest
from swift import get_model_processor, get_template
from swift.utils import safe_snapshot_download
from peft import PeftModel

def get_val(obj, key, default=None):
    """万能安全提取器：兼容 Object 属性和 Dict 键值两种访问模式"""
    if hasattr(obj, key):
        return getattr(obj, key)
    elif isinstance(obj, dict):
        return obj.get(key, default)
    return default

def init_engine(model_path: str, adapter_path: str=None, max_batch_size: int=2):
    """
    初始化模型引擎和配置参数
    建议：配置 max_batch_size (比如4或8) 和 temperature=0
    """
    template_type = None                # None: 使用对应模型默认的template_type
    default_system = None               # None: 使用对应模型默认的default_system
    
    model, tokenizer = get_model_processor(model_path, device_map='auto')
    if adapter_path is not None:
        model = PeftModel.from_pretrained(model, adapter_path)
    
        # 👇 加入这行神仙代码：将 LoRA 永久熔合进底座物理内存中！
        print("⚡ 正在执行权重合并 (Merge) 以恢复满血推理速度...")
        model = model.merge_and_unload()

    template_type = template_type or model.model_meta.template
    template = get_template(tokenizer, template_type=template_type, default_system=default_system)
    
    # 加载推理引擎
    print(f"正在加载底座模型: {model_path} ...")
    print(f"正在挂载 LoRA 权重: {adapter_path} ...")
    engine = TransformersEngine(model, template=template, max_batch_size=max_batch_size)
    request_config = RequestConfig(
        max_tokens=4096,
        temperature=0,
        logprobs=True,       # 开启对数概率输出
        top_logprobs=5       # 保存排名前 5 的 Token 和概率
    )
    
    return engine, request_config

def build_requests(data_chunk: list) -> list[InferRequest]:
    """
    【你的 load_dataset 的进阶版】
    将从 jsonl 读出的原始 dict 列表，转化为 engine 认识的 InferRequest 列表
    """
    infer_requests = []
    for item in data_chunk:
        # 确保喂给模型的 messages 里只有 user 的提问，不能有 assistant 的历史答案
        prompt_messages = [msg for msg in item['messages'] if msg['role'] != 'assistant']
        
        infer_requests.append(InferRequest(
            messages = prompt_messages,
            images = item['images']
        ))

    return infer_requests


def infer_and_save_chunk(engine, request_config, data_chunk: list, output_path: str):
    """
    【你的 infer_batch 的进阶版】
    1. 调用 build_requests 组装请求
    2. engine.infer() 批量推理
    3. 将预测结果塞回 data_chunk 的每一项中 (比如新增一个 "prediction" 字段)
    4. 以追加模式 ('a') 将这批数据写入 output_path
    """
    # 组装请求
    infer_requests = build_requests(data_chunk)
    
    # 推理
    resp_list = engine.infer(infer_requests, request_config)
    
    for i, resp in enumerate(resp_list):
        # 1. message 是对象，继续用点号访问
        data_chunk[i]['pred'] = resp.choices[0].message.content
    
        # 2. 提取概率数据
        token_probs_list = []
        logprobs_data = resp.choices[0].logprobs
        
        # 确保模型返回了 logprobs
        if logprobs_data is not None:
            # 【关键修改】：logprobs_data 是字典，必须用 ['content']
            for token_info in logprobs_data['content']:
                
                # token_info 也是字典，必须用 ['token'] 和 ['logprob']
                t_logprob = token_info['logprob']
                clean_token_info = {
                    "token": token_info['token'],
                    "logprob": t_logprob,
                    "probability": round(math.exp(t_logprob), 6),
                    "top_candidates": []
                }
                
                # 遍历它的 top 候选词
                if 'top_logprobs' in token_info:
                    for candidate in token_info['top_logprobs']:
                        c_logprob = candidate['logprob']
                        clean_token_info["top_candidates"].append({
                            "token": candidate['token'],
                            "probability": round(math.exp(c_logprob), 6)
                        })
                
                token_probs_list.append(clean_token_info)
                
        # 截取最后20个Token的概率流
        data_chunk[i]['pred_token_probs'] = token_probs_list[-20:]

    # 保存处理结果
    with open(output_path, 'a', encoding='utf-8') as f:
        for item in data_chunk:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

def main(input_path, output_path, model_path, adapter_path, chunk_size=2, max_batch_size=2):
    """
    主控流：
    1. 定义 input_path, output_path, model_path
    2. 全量读取 input_path 的 jsonl 数据到列表 all_data
    3. 检查断点续跑记录，过滤出待处理数据
    4. 如果有待处理数据，加载模型 engine
    5. 用 for 循环和 tqdm 分块遍历 rest_data，调用 infer_and_save_chunk
    """
    # 1. 加载所有数据
    with open(input_path, 'r', encoding='utf-8') as f:
        all_data = [json.loads(line) for line in f]
    
    # 2. 获取已经处理的item的id
    processed_id = set()
    if os.path.exists(output_path): # 修复: 加上了 s
        with open(output_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    processed_id.add(item['id'])
                    
        print(f"🔄 检测到历史进度，已完成 {len(processed_id)} 条数据。") # 修复: 统一了变量名
    
    # 3. 过滤掉已经处理的数据
    rest_data = [item for item in all_data if item['id'] not in processed_id]
    total_items = len(all_data)
    rest_items = len(rest_data)
    
    print(f"总计有 {total_items} 条数据，剩余 {rest_items} 条数据待处理，chunk_size大小为: {chunk_size}")
    
    # 4. 如果所有数据都处理完了，直接退出，不加载模型
    if rest_items == 0:
        print("🎉 所有数据已推理完毕，无需加载模型！")
        return
        
    # 5. 只有在需要推理时，才消耗时间加载模型 (优化点)
    engine, request_config = init_engine(model_path, adapter_path, max_batch_size)
    
    # 6. 开始分块推理
    for i in tqdm(range(0, rest_items, chunk_size), desc='Processing'):
        try:
            chunk = rest_data[i: i+chunk_size]
            infer_and_save_chunk(engine, request_config, chunk, output_path)
        
        except Exception as e:
            print(f"❌ 处理第 {i} 个chunk时发生错误: {e}")
            

if __name__ == "__main__":
    # input_path = '/data/ZS/defect_dataset/7_swift_dataset/test/012_pos400_neg300_rect300.jsonl'
    input_path = '/data/ZS/defect_dataset/12_vlm_message/stripe_phase012/val_0p1.jsonl'
    model_path = 'Qwen/Qwen3-VL-4B-Instruct'
    output_path = '/data/ZS/defect_dataset/13_vlm_response/stripe_phase012/v2_qwen3_4b_LM.jsonl' # 修改
    adapter_path = '/data/ZS/defect-vlm/output/weights/v1-20260308-204436_qwen3_4b_LM/checkpoint-4800_best'     # 修改
    chunk_size = 16
    max_batch_size = 16
    main(
        input_path = input_path,
        output_path = output_path,
        model_path = model_path,
        adapter_path = adapter_path,
        chunk_size = chunk_size,            # 一次加载多少条数据，和max_batch_size保持一致即可（或者是整数倍）
        max_batch_size = max_batch_size     # 一次并行推理多少条数据
    )

