"""
读取 ms swift 格式的 jsonl 文件，使用 vLLM 推理并保存结果。
特性：取消概率输出，温度固定为0，最大化并发吞吐量。
输入：离线合并后的全量模型权重 + 待推理数据
输出：包含推理结果的数据 (pred字段)
"""
import os
import json
from tqdm import tqdm
from typing import List

# 显卡配置
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
os.environ['MAX_PIXELS'] = '1003520'
TENSOR_PARALLEL_SIZE = 2  # 对应 4 张卡，开启张量并行

# 导入必要的 Swift 和 vLLM 组件
from swift import InferRequest, RequestConfig
from swift.infer_engine import VllmEngine


def init_engine(model_path: str):
    """
    初始化 vLLM 模型引擎和配置参数
    """
    print(f"🚀 正在加载 vLLM 底座模型: {model_path} ...")
    print(f"⚙️ 当前张量并行度 (Tensor Parallel Size): {TENSOR_PARALLEL_SIZE}")

    # 加载推理引擎，max_model_len 可根据你的显存和上下文长度需求调整
    engine = VllmEngine(
        model_path, 
        max_model_len=8192, 
        enforce_eager=True, 
        tensor_parallel_size=TENSOR_PARALLEL_SIZE
    )
    
    # max_tokens 设为你需要生成的最大长度，temperature=0 保证输出稳定性，不需要 logprobs
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
    3. 将预测结果塞回 data_chunk 的 'pred' 字段中
    4. 写入文件
    """
    # 组装请求
    infer_requests = build_requests(data_chunk)
    
    # vLLM 推理 (内部自动处理并发和 PagedAttention)
    resp_list = engine.infer(infer_requests, request_config)
    
    # 将处理结果追加到原始数据中
    for i, resp in enumerate(resp_list):
        # 确保模型成功返回了结果
        if resp and resp.choices and resp.choices[0].message.content:
            data_chunk[i]['pred'] = resp.choices[0].message.content
        else:
            data_chunk[i]['pred'] = ""
            print(f"⚠️ 警告: 第 {i} 条数据推理返回异常，已置为空字符串。")
    
    # 保存处理结果
    with open(output_path, 'a', encoding='utf-8') as f:
        for item in data_chunk:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def main(input_path, output_path, model_path, chunk_size=64):
    """
    主控流：
    vLLM 的吞吐量极大，可以将 chunk_size 设置得大一点（比如 64~128），
    既能充分利用 vLLM 的并发批处理优势，又能兼顾断点保存的安全性。
    """
    # 1. 加载所有数据
    with open(input_path, 'r', encoding='utf-8') as f:
        all_data = [json.loads(line) for line in f if line.strip()]
    
    # 2. 获取已经处理的item的id (断点续传)
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
    engine, request_config = init_engine(model_path)
    
    # 6. 开始分块推理
    for i in tqdm(range(0, rest_items, chunk_size), desc='VLLM Inferencing'):
        try:
            chunk = rest_data[i: i+chunk_size]
            infer_and_save_chunk(engine, request_config, chunk, output_path)
        
        except Exception as e:
            print(f"❌ 处理索引 [{i}:{i+chunk_size}] 时发生错误: {e}")


if __name__ == "__main__":
    # 1. 输入数据路径
    INPUT_PATH = '/data/ZS/flywheel_dataset/5_vlm_message/iter2/0p1_chunk12.jsonl'
    
    # 2. 输出结果路径
    OUTPUT_PATH = '/data/ZS/flywheel_dataset/6_vlm_response/iter2/0p1_chunk12_v1_LM.jsonl'
    
    # 3. 模型路径
    # ⚠️ 请确保这里是已经将 Qwen3-VL-4B-Instruct 与 checkpoint-4800_best 合并后的完整权重文件夹
    MERGED_MODEL_PATH = '/data/ZS/defect-vlm/output/merged_model/v1_qwen3_4b_LM' 
    
    # 4. 设置分块大小
    CHUNK_SIZE = 64  # 根据你的 4x4090 算力，这里开到 64 甚至 128 速度会起飞

    main(
        input_path = INPUT_PATH,
        output_path = OUTPUT_PATH,
        model_path = MERGED_MODEL_PATH,
        chunk_size = CHUNK_SIZE
    )
