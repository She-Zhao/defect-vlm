import os
import yaml
from dotenv import load_dotenv

load_dotenv()

class APIConfigManager:
    """
    模型API配置管理器
    """
    def __init__(self, config_path='config/api_config.yaml'):
        self.config_path = config_path
        self.config = self._load_config()
        
    def _load_config(self):
        """加载yaml配置文件"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            raise FileNotFoundError(f"配置文件 {self.config_path} 未找到！")
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"配置文件 {self.config_path} 格式错误: {e}")
    
    def get_model_config(self, provider, model):
        """获取指定模型配置"""
        if provider not in self.config['providers']:
            raise ValueError(f"提供商 {provider} 尚不支持。可用的提供商: {self.list_providers()}")
        
        provider_config = self.config['providers'][provider]
        if model not in provider_config['models']:
            raise ValueError(f"{provider} 尚不支持 {model}。可用的模型：{self.list_models(provider)}")
        
        model_config = {
            "provider": provider,
            "api_key": os.environ.get(provider_config['api_key_env']),
            "base_url": provider_config["base_url"],
            "model": model,
            "description": provider_config['models'][model]
        }
        
        return model_config

    def list_providers(self):
        """列出所有的提供商"""
        return list(self.config['providers'].keys())
   
    def list_models(self, provider):
        """列出指定提供商的模型"""
        if provider not in self.config['providers']:
            raise ValueError(f"提供商 '{provider}' 不存在")
        return self.config['providers'][provider]['models']


if __name__ == "__main__":
    apiconfigmanager = APIConfigManager()
    all_providers = apiconfigmanager.list_providers()
    print(f"所有的供应商: {all_providers}")
    
    provider = "qwen"
    print(apiconfigmanager.list_models(provider))
    
    model = "qwen3-vl-plus"
    print(apiconfigmanager.get_model_config(provider, model))
