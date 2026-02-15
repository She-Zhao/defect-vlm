"""
将多个文件的打标结果进行合并重新排列id，构成完整的训练集和测试集
"""
import os
import json
from pathlib import Path
from tqdm import tqdm

