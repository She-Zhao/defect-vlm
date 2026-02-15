from setuptools import setup, find_packages

setup(
    name="defect_vlm",        # 包名，随便起
    version="0.1",
    packages=find_packages(), # 自动查找根目录下的所有包（包含 __init__.py 的文件夹）
)