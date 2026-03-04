"""
读取ms swift格式的jsonl文件，推理并保存结果
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
os.environ['MAX_PIXELS'] = '1003520'
import json
from tqdm import tqdm
from swift.infer_engine import TransformersEngine, RequestConfig, InferRequest


def init_engine(model_path: str):
    """
    初始化模型引擎和配置参数
    建议：配置 max_batch_size (比如4或8) 和 temperature=0
    """
    # 加载推理引擎
    print(f"正在加载模型: {model_path} ...")
    engine = TransformersEngine(model_path, max_batch_size=24, model_kwargs={'device_map': 'auto'})
    request_config = RequestConfig(max_tokens=4096, temperature=0)
    
    return engine, request_config

def build_requests(data_chunk: list) -> list[InferRequest]:
    """
    【你的 load_dataset 的进阶版】
    将从 jsonl 读出的原始 dict 列表，转化为 engine 认识的 InferRequest 列表
    """
    infer_requests = []
    for item in data_chunk:
        infer_requests.append(InferRequest(
            messages = item['messages'],
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
    
    # 将处理结果追加到原始数据中
    for i, resp in enumerate(resp_list):
        data_chunk[i]['pred'] = resp.choices[0].message.content
    
    # 保存处理结果
    with open(output_path, 'a', encoding='utf-8') as f:
        for item in data_chunk:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

def main(input_path, output_path, model_path, chunk_size=2):
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
    engine, request_config = init_engine(model_path)
    
    # 6. 开始分块推理
    for i in tqdm(range(0, rest_items, chunk_size), desc='Processing'):
        try:
            chunk = rest_data[i: i+chunk_size]
            infer_and_save_chunk(engine, request_config, chunk, output_path)
        
        except Exception as e:
            print(f"❌ 处理第 {i} 个chunk时发生错误: {e}")
            

if __name__ == "__main__":
    input_path = '/data/ZS/defect_dataset/7_swift_dataset/student/test/012_pos400_neg300_rect300.jsonl'
    output_path = '/data/ZS/defect_dataset/8_model_reponse/student/test/qwen2.5-vl-7b-instrusct.jsonl'
    model_path = 'Qwen/Qwen2.5-VL-7B-Instruct'
    chunk_size = 24
    main(input_path, output_path, model_path, chunk_size)
