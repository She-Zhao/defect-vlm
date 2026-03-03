"""
调用在线LLM API处理多模态数据

输入构造好的用于调用api的jsonl数据，调用api并解析返回的结果，保存到新的jsonl文件中

author:zhaoshe
"""
import os
import json
import base64
import asyncio
from tqdm import tqdm
import argparse
from typing import List, Dict, Any
from openai import AsyncOpenAI, APIError
from defect_vlm.utils import APIConfigManager  

# 如果采用openai等外网官方网站作为base_url，需要设置下代理的映射端口
# os.environ['HTTP_PROXY'] = 'http://127.0.0.1:xxxx'
# os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:xxxx'
# os.environ['ALL_PROXY'] = 'http://127.0.0.1:xxxx'

# 设置在通过该网址访问时不使用任何代理，否则在开启vpn通过该网站调用api会出错
os.environ['NO_PROXY'] = 'api.agicto.cn'

def initialize_client(api_key: str, base_url: str) -> AsyncOpenAI:
    if not api_key:
        raise ValueError("API KEY为空!")
    
    return AsyncOpenAI(
        api_key = api_key,
        base_url = base_url,
    )

def encode_image_to_base64(image_path: str) -> str:
    """读取图像文件并编码为base64格式

    Args:
        image_path (str): 图像文件的路径

    Returns:
        str: "data:jpeg;base64,{base64_string}格式的字符串
    """
    with open(image_path, 'rb') as image_file:
        binary_data = image_file.read()
        base64_string = base64.b64encode(binary_data).decode('utf-8')
        
        mime_type = 'image/jpeg'
        if image_path.lower().endswith('.png'):
            mime_type = 'image/png'
        elif image_path.lower().endswith('.bmp'):
            mime_type = 'image/bmp'
        elif image_path.lower().endswith('.webp'):
            mime_type = 'image/webp'            
            
    return f'data:{mime_type};base64,{base64_string}'

def build_send_message(sample: Dict[str, Any]) -> List[Dict[str, Any]]:
    """根据sft.jsonl中的每条json数据，构造输入给api的数据

    Args:
        sample (Dict[str, Any]): 单个json数据

    Returns:
        List[Dict[str, Any]]: 输入给模型的数据
    """
    image_paths = sample['image']
    human_prompt_raw = sample['conversation'][0]['value']
    human_prompt_list = human_prompt_raw.split('<image>')   # 调用api的时候，不需要包含<image>
    
    content = []
    for idx, human_prompt in enumerate(human_prompt_list):
        if human_prompt.strip():        # human_prompt.strip()是非空字符串时，才为True
            content.append({
                'type': 'text',
                'text': human_prompt
            })
        
        if idx < len(image_paths):
            try:
                image_path = image_paths[idx]
                base64_image = encode_image_to_base64(image_path)
                content.append({
                    'type': 'image_url',
                    'image_url': {'url': base64_image}
                })
                
            except Exception as e:
                raise ValueError(f"图像编码失败: {image_path}, 错误: {e}")
    
    message = [{
        'role': 'user',
        'content': content
    }]
    
    return message


async def process_single_task(
    client: AsyncOpenAI,
    sample: Dict[str, Any],
    model: str,
    semaphore: asyncio.Semaphore
) -> Dict[str, Any]:
    """提交单个API调用的协程函数, 当调用失败的时候会自动保存'ERROR',
    后续处理的时候可以通过判断sample['conversation'][1]['value'] == 'ERROR'来剔除调用失败的数据

    Args:
        client (AsyncOpenAI): OpenAI客户端（支持异步）
        sample (Dict[str, Any]): 待处理的单个样本（任务）
        model (str): 调用API的名称（调用模型的名称）
        semaphore(asyncio.Semaphore): 接收信号量

    Returns:
        Dict[str, Any]:包含模型回复的数据
    """
    async with semaphore:       # 确保在任何时候，最多都只有semaphore个任务同时执行with内的代码
        try:
            messages = build_send_message(sample)
            response = await client.chat.completions.create(
                model = model,
                messages = messages,
                temperature=0.0,
                max_tokens=8192
            )
            
            # 检查 response 和 choices 是否有效
            if response and response.choices and len(response.choices) > 0:
                # 检查 message 和 content 是否有效
                if response.choices[0].message and response.choices[0].message.content:
                    ai_response = response.choices[0].message.content
                    sample['conversation'][1]['value'] = ai_response
                else:
                    # API 成功了，但 message.content 为空
                    print(f"❌ API 警告 (ID: {sample['id']}): 响应中缺少 message.content。")
                    sample['conversation'][1]['value'] = 'ERROR: Empty message content'
            else:
                # API 成功了，但返回了空的 'choices' 列表或 None
                print(f"❌ API 警告 (ID: {sample['id']}): 响应中缺少 'choices'。")
                # sample['conversation'][1]['value'] = 'ERROR: Empty choices list'
                
                # 【新增这行】把服务器到底回了什么原封不动地打印出来
                try:
                    print(f"🔍 原始响应内容: {response.model_dump_json(indent=2)}")
                except:
                    print(f"🔍 原始响应内容 (Raw): {response}")
                    
                sample['conversation'][1]['value'] = 'ERROR: Empty choices list'
        
        except APIError as e: # 更具体地捕获 API 错误
            print(f"❌ API 错误 (ID: {sample['id']}): {e} ")
            sample['conversation'][1]['value'] = f'ERROR: APIError {e}'
        except Exception as e:
            print(f"❌ 未知错误 (ID: {sample['id']}): {e} ")
            sample['conversation'][1]['value'] = f'ERROR: Exception {e}'
        
    return sample
    
    
async def process_batch_task(args):
    """
    异步批处理的主协调函数。

    该函数负责：
    1. 初始化客户端。
    2. 读取已完成的任务ID（断点续传）。
    3. 读取并过滤待处理的任务。
    4. 创建信号量以控制并发。
    5. 使用 `asyncio.as_completed` 并发执行所有任务。
    6. 在任务完成时立即将其结果追加写入输出文件。

    Args:
        args (argparse.Namespace): 
            从命令行解析的参数, 必须包含:
            - provider (str): API 提供商
            - model (str): 模型名称
            - input_file (str): 输入的 .jsonl 任务文件
            - output_file (str): 输出的 .jsonl 结果文件
            - concurrency (int): 最大并发数
    """
    print(f"🚀 开始调用API（异步）...")
    print(f"    并发数量: {args.concurrency}")
    
    # 初始化模型
    config_manager = APIConfigManager()
    model_config = config_manager.get_model_config(args.provider, args.model)
    client = initialize_client(api_key=model_config['api_key'], base_url=model_config['base_url'])
    print(f"    模型: {args.provider} - {args.model}")

    # 读取已完成的任务
    completed_ids = set()
    
    if os.path.exists(args.output_file):
        with open(args.output_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    line_data = json.loads(line)
                    completed_ids.add(line_data['id'])
                except Exception as e:
                    pass
    print(f"已加载 {len(completed_ids)} 个已完成的任务")
    
    # 读取所有待调用api的数据
    task_to_process = []
    with open(args.input_file, 'r', encoding = 'utf-8') as f_in:
        for line in f_in:
            task = json.loads(line)
            if task['id'] not in completed_ids:
                task_to_process.append(task)

    total_tasks = len(task_to_process)
    if total_tasks == 0:
        print(f"✔️所有任务均已完成，无需处理!")
    else:
        print(f"⚡共找到 {total_tasks} 个待处理任务")
    
    # 创建信号量
    semaphore = asyncio.Semaphore(args.concurrency)
    
    # 创建所有任务的协程，存储在list中
    coroutines = [process_single_task(client, sample, model_config['model'], semaphore) for sample in task_to_process]
    
    # 并发执行所有任务并保存结果
    print(f"✨✨开始并发执行 {len(coroutines)} 个任务，最大并发数: {args.concurrency}")
    results_count = 0
    try:
        with open(args.output_file, 'a', encoding='utf-8') as f_out:
            for future in tqdm(asyncio.as_completed(coroutines), total=total_tasks, desc="Processing tasks"):
                result = await future           # await获取事件循环中的一个处理结果
            
                if result:
                    f_out.write(json.dumps(result, ensure_ascii=False) + '\n')
                    f_out.flush()           # 立刻将文件写入
                    results_count += 1
    except Exception as e:
        print(f"❌  循环处理过程中遇到错误: {e}")
        print(f"    已处理  {results_count} / {total_tasks} 个任务")
        return
    
    print(f"\n✅ 任务处理完成，{results_count} 个新结果已追加至 {args.output_file}")
    await client.close()

def main():
    """
    主入口函数：解析命令行参数。
    """
    parser = argparse.ArgumentParser(description="批量调用LLM API, 异步控制, 支持断点续跑")
    parser.add_argument('--provider', type=str, required=True, help='API提供商')
    parser.add_argument('--model', type=str, required=True, help='模型名称')
    parser.add_argument('--input_file', type=str, required=True, help='输入的 .jsonl 待处理文件')
    parser.add_argument('--output_file', type=str, required=True, help='输出的处理结果文件')
    parser.add_argument('--concurrency', type=int, default=2, help='并发调用数量, 默认为10')

    args = parser.parse_args()
    asyncio.run(process_batch_task(args))

if __name__ == "__main__":
    # 通过命令行运行
    main()
    
    # 通过代码运行
    # test_args = argparse.Namespace()

    # # 修改下面这部分参数即可
    # test_args.provider = 'qwen'                                                       # 提供商
    # # test_args.model = 'qwen3-vl-235b-a22b-instruct'                                   # 模型名称
    # # test_args.input_file = '/data/ZS/defect_dataset/4_api_request/retry/qwen3-vl-235b-a22b-instruct.jsonl'            # 输入文件
    # # test_args.output_file = '/data/ZS/defect_dataset/5_api_response/retry/qwen3-vl-235b-a22b-instruct.jsonl'    # 调用api后得到的输出文件
    
    # test_args.model = 'qwen3-vl-235b-a22b-instruct'                                   # 模型名称
    # test_args.input_file = '/data/ZS/defect_dataset/4_api_request/teacher/retry/qwen3-vl-235b-a22b-instruct.jsonl'            # 输入文件
    # test_args.output_file = '/data/ZS/defect_dataset/5_api_response/teacher/retry/qwen3-vl-235b-a22b-instruct.jsonl'

    # test_args.concurrency = 1                                                         # 并发数
    
    # print(f"--- 正在从脚本中启动 call_llm_api_robust (调试模式) ---")
    # print(f"   Provider: {test_args.provider}")
    # print(f"   Model: {test_args.model}")
    # print(f"   Input: {test_args.input_file}")
    # print(f"   Output: {test_args.output_file}")
    # print(f"   Concurrency: {test_args.concurrency}")
    
    # try:
    #     asyncio.run(process_batch_task(test_args))
    #     print(f"--- 脚本调用执行完毕 ---")
    # except Exception as e:
    #     print(f"--- 脚本调用时发生错误: {e} ---")
