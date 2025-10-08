import yaml
import os


class ConfigLoader:
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.config = None

    def load_config(self):
        """加载并解析配置文件"""
        try:
            with open(self.config_path, 'r', encoding="utf-8") as file:
                self.config = yaml.safe_load(file)
        except FileNotFoundError:
            raise FileNotFoundError(f"配置文件 {self.config_path} 不存在")
        except Exception as e:
            raise Exception(f"加载配置文件时出错: {str(e)}")

        # 解析配置
        return self.config


if __name__ == "__main__":
    # 加载配置
    config_loader = ConfigLoader("config.yaml")
    config = config_loader.load_config()
    print(config)

