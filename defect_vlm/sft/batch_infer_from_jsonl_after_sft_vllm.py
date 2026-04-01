"""
读取ms swift格式的jsonl文件，使用vllm推理并保存结果。适用于SFT之后的模型
输入：原始模型权重+适配器权重输入+待推理数据
输出：包含推理结果的数据
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
os.environ['MAX_PIXELS'] = '1003520'
TENSOR_PARALLEL_SIZE = 2
import json
from tqdm import tqdm
from typing import List

# 导入必要的 Swift 组件
from swift import InferRequest, RequestConfig
from swift.infer_engine import VllmEngine # 改用 VllmEngine

def init_engine(model_path: str, adapter_path: str=None):
    """
    初始化 vLLM 模型引擎和配置参数
    
    ⚠️ 重要提示：
    vLLM 不支持在内存中动态融合 PyTorch 模型。如果使用了 LoRA，
    建议先通过终端命令将模型离线合并，例如：
    swift export --model_type qwen2-vl-7b-instruct --adapters /path/to/lora --merge_lora true --dest_dir /path/to/merged_model
    然后将合并后的 `/path/to/merged_model` 直接作为这里的 `model_path`。
    """
    print(f"🚀 正在加载 vLLM 底座模型: {model_path} ...")
    if adapter_path:
        print(f"⚠️ 警告: 检测到 adapter_path ({adapter_path})。vLLM 官方建议直接加载离线合并后的全量模型以获得最佳性能。")

    # 加载推理引擎，max_model_len 可根据你的显存和上下文长度需求调整
    engine = VllmEngine(model_path, max_model_len=8192, enforce_eager=True, tensor_parallel_size=2)
    
    # max_tokens 设为你需要生成的最大长度，temperature=0 保证输出稳定性
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
    1. 调用 build_requests 组装请求
    2. engine.infer() 批量推理
    3. 将预测结果塞回 data_chunk
    4. 写入文件
    """
    # 组装请求
    infer_requests = build_requests(data_chunk)
    
    # vLLM 推理 (这里内部会自动处理并发和 PagedAttention)
    resp_list = engine.infer(infer_requests, request_config)
    
    # 将处理结果追加到原始数据中
    for i, resp in enumerate(resp_list):
        # 防止因被屏蔽等意外情况导致没生成内容
        if resp and resp.choices and resp.choices[0].message.content:
            data_chunk[i]['pred'] = resp.choices[0].message.content
        else:
            data_chunk[i]['pred'] = ""
            print(f"⚠️ 第 {i} 条数据推理返回异常，已置为空字符串。")
    
    # 保存处理结果
    with open(output_path, 'a', encoding='utf-8') as f:
        for item in data_chunk:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

def main(input_path, output_path, model_path, adapter_path=None, chunk_size=32):
    """
    主控流：vLLM 的吞吐量极大，可以将 chunk_size 设置得大一点（比如 32 或 64），
    既能充分利用 vLLM 的并发批处理优势，又能兼顾断点保存的安全性。
    """
    # 1. 加载所有数据
    with open(input_path, 'r', encoding='utf-8') as f:
        all_data = [json.loads(line) for line in f if line.strip()]
    
    # 2. 获取已经处理的item的id
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
        
    # 5. 加载引擎
    engine, request_config = init_engine(model_path, adapter_path)
    
    # 6. 开始分块推理
    for i in tqdm(range(0, rest_items, chunk_size), desc='VLLM Inferencing'):
        try:
            chunk = rest_data[i: i+chunk_size]
            infer_and_save_chunk(engine, request_config, chunk, output_path)
        
        except Exception as e:
            print(f"❌ 处理索引 [{i}:{i+chunk_size}] 时发生错误: {e}")

if __name__ == "__main__":
    # 使用示例
    # 注意：为了让 vLLM 跑出满血速度，如果用了 LoRA，请将 model_path 替换为 merge 后的文件夹路径
    main(
        input_path = '/data/ZS/defect_dataset/7_swift_dataset/test/val_merged.jsonl',
        output_path = '/data/ZS/defect_dataset/8_model_reponse/val_merged/v1_qwen3_4b_LM.jsonl', # 修改
        model_path = "/data/ZS/defect-vlm/output/merged_model/v1_qwen3_4b_LM", # vllm 加载的最佳姿势
        chunk_size = 128                           # vLLM 并发极强，这个值建议设大（32~128）以减少 IO 频次
    )
