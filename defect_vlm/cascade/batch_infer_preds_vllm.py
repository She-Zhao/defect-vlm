"""
读取ms swift格式的jsonl文件，使用vllm推理并保存结果。适用于SFT之后的模型
输入：离线合并后的完整模型权重 + 待推理数据
输出：包含推理结果的数据（无 logprobs）
"""
import os

# ==================== 环境配置 ====================
os.environ['CUDA_VISIBLE_DEVICES'] = '1,2,3'  # 保持你原来的 3 张卡
os.environ['MAX_PIXELS'] = '1003520'
os.environ['TORCH_COMPILE_DISABLE'] = '1'     # 全局强制禁用 Torch Compile 以防 vllm 报错
TENSOR_PARALLEL_SIZE = 2                      # 对应你上面配置的 3 张显卡
# ==================================================

import json
from tqdm import tqdm
from typing import List

from swift import InferRequest, RequestConfig
from swift.infer_engine import VllmEngine

def get_val(obj, key, default=None):
    """万能安全提取器：兼容 Object 属性和 Dict 键值两种访问模式"""
    if hasattr(obj, key):
        return getattr(obj, key)
    elif isinstance(obj, dict):
        return obj.get(key, default)
    return default

def init_engine(model_path: str, adapter_path: str=None):
    """
    初始化 vLLM 模型引擎和配置参数
    """
    print(f"🚀 正在加载 vLLM 底座模型: {model_path} ...")
    if adapter_path:
        print(f"⚠️ 警告: 检测到 adapter_path ({adapter_path})。vLLM 不支持动态融合，请确保传入的是离线 Merge 后的全量模型路径。")

    # 加载推理引擎，tensor_parallel_size 必须与物理显卡数量一致
    engine = VllmEngine(model_path, max_model_len=8192, enforce_eager=True, tensor_parallel_size=TENSOR_PARALLEL_SIZE)
    
    # 移除了 logprobs 相关配置，保持基本的生成参数
    request_config = RequestConfig(max_tokens=4096, temperature=0)
    
    return engine, request_config

def build_requests(data_chunk: list) -> List[InferRequest]:
    """
    将从 jsonl 读出的原始 dict 列表，转化为 engine 认识的 InferRequest 列表
    """
    infer_requests = []
    for item in data_chunk:
        # 确保喂给模型的 messages 里只有 user 的提问，不能有 assistant 的历史答案
        prompt_messages = [msg for msg in item['messages'] if msg['role'] != 'assistant']
        
        infer_requests.append(InferRequest(
            messages = prompt_messages,
            images = item.get('images', []) # 兼容可能没有图像的情况
        ))

    return infer_requests

def infer_and_save_chunk(engine: VllmEngine, request_config: RequestConfig, data_chunk: list, output_path: str):
    """
    分块推理并保存（已剥离 logprobs 逻辑）
    """
    # 组装请求
    infer_requests = build_requests(data_chunk)
    
    # vLLM 批量推理
    resp_list = engine.infer(infer_requests, request_config)
    
    # 将预测结果塞回 data_chunk
    for i, resp in enumerate(resp_list):
        if resp and resp.choices and resp.choices[0].message.content:
            data_chunk[i]['pred'] = resp.choices[0].message.content
        else:
            data_chunk[i]['pred'] = ""
            print(f"⚠️ 第 {i} 条数据推理返回异常，已置为空字符串。")
    
    # 追加写入文件
    with open(output_path, 'a', encoding='utf-8') as f:
        for item in data_chunk:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

def main(input_path, output_path, model_path, adapter_path=None, chunk_size=32):
    """
    主控流：vLLM 的并发能力极强，推荐较大的 chunk_size
    """
    # 1. 加载所有数据
    with open(input_path, 'r', encoding='utf-8') as f:
        all_data = [json.loads(line) for line in f if line.strip()]
    
    # 2. 获取已经处理的item的id，实现断点续跑
    processed_id = set()
    if os.path.exists(output_path):
        with open(output_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    if 'id' in item:
                        processed_id.add(item['id'])
                    
        print(f"🔄 检测到历史进度，已完成 {len(processed_id)} 条数据。")
    
    # 3. 过滤掉已经处理的数据
    rest_data = [item for item in all_data if item.get('id') not in processed_id]
    total_items = len(all_data)
    rest_items = len(rest_data)
    
    print(f"📊 总计: {total_items} | 剩余待处理: {rest_items} | 每次保存步长(chunk_size): {chunk_size}")
    
    # 4. 如果所有数据都处理完了，直接退出
    if rest_items == 0:
        print("🎉 所有数据已推理完毕，无需加载模型！")
        return
        
    # 5. 加载 vLLM 引擎
    engine, request_config = init_engine(model_path, adapter_path)
    
    # 6. 开始分块推理
    for i in tqdm(range(0, rest_items, chunk_size), desc='VLLM Inferencing'):
        try:
            chunk = rest_data[i: i+chunk_size]
            infer_and_save_chunk(engine, request_config, chunk, output_path)
        
        except Exception as e:
            print(f"❌ 处理索引 [{i}:{i+chunk_size}] 时发生错误: {e}")

if __name__ == "__main__":
    # ==================== 路径配置 ====================
    input_path = '/data/ZS/defect_dataset/12_vlm_message/stripe_phase012/val_0p01_crop0.jsonl'
    output_path = '/data/ZS/defect_dataset/13_vlm_response/stripe_phase012/v11_val_0p01_crop0.jsonl'
    
    # 【核心注意】：这里必须填入你通过 swift export 命令合并后的模型文件夹路径！
    # 如果填原版 'Qwen/Qwen3-VL-4B-Instruct' 它是没有你的微调知识的。
    model_path = '/data/ZS/defect-vlm/output/merged_model/v11-stage3-LM-PRO-VIT-26k' 
    
    chunk_size = 64  # vLLM 推荐使用较大的 batch size 喂饱显存
    # ==================================================
    
    main(
        input_path = input_path,
        output_path = output_path,
        model_path = model_path,
        adapter_path = None,  # vLLM 下设为 None，强制使用 merge 后的模型
        chunk_size = chunk_size
    )