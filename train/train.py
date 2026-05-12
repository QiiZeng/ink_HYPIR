# 导入命令行参数解析模块
from argparse import ArgumentParser
# 常用于深度学习配置文件读取
from omegaconf import OmegaConf

# 从HYPIR训练器模块中导入SD2模型对应的训练器类
from HYPIR.trainer.sd2 import SD2Trainer

# 1. 初始化命令行参数解析器
parser = ArgumentParser()
# 2. 添加必选的配置文件参数：--config
parser.add_argument("--config", type=str, required=True)
# 3. 解析用户传入的命令行参数（例如：python xxx.py --config config.yaml）
args = parser.parse_args()

# 4. 加载配置文件：通过OmegaConf读取指定路径的配置文件（支持yaml/yml/json等格式）
config = OmegaConf.load(args.config)

# 5. 判断配置中指定的基础模型类型是否为"sd2"
if config.base_model_type == "sd2":
    # 6. 初始化SD2模型的训练器，传入配置参数
    trainer = SD2Trainer(config)
    # 7. 启动训练器的主运行流程（包含训练、验证、保存等核心逻辑）
    trainer.run()
else:
    # 8. 若模型类型不支持，抛出值错误异常并提示不支持的模型类型
    raise ValueError(f"Unsupported model type: {config.base_model_type}")