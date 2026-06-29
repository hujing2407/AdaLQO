import os
import logging
import platform
from pathlib import Path


class Config:
    # 1. 路径锚点：无论从哪运行，都以 config.py 所在位置为准
    ROOT = Path(__file__).resolve().parent

    # 2. 子目录定义
    DATASET_DIR = ROOT / "dataset"
    LOG_DIR = ROOT / "logs"
    RES_DIR = ROOT / "results"

    # 默认数据库配置（作为 Fallback）
    DEFAULT_DB_PARAMS = {
        "dbname": "tpcds",
        "user": "postgres",
        "password": "123",  # 建议实际生产环境使用环境变量
        "host": "192.168.0.225" if platform.system() == "Darwin" else "localhost",
        "port": "5432",
        "options": "-c statement_timeout=60000"
    }

    HINTS_OFF = [
        '',
        'set enable_nestloop = off;',
        'set enable_nestloop = off; set enable_indexscan = off;',
        'set enable_hashjoin = off;',
        'set enable_hashjoin = off; set enable_indexscan = off;',
        'set enable_mergejoin = off;',
        'set enable_mergejoin = off; set enable_indexscan = off;',
        'set enable_nestloop = off; set enable_mergejoin = off;',
        'set enable_nestloop = off; set enable_mergejoin = off; set enable_indexscan = off;',
        'set enable_nestloop = off; set enable_hashjoin = off;',
        'set enable_nestloop = off; set enable_hashjoin = off; set enable_indexscan = off;',
        'set enable_mergejoin = off; set enable_hashjoin = off;',
        'set enable_mergejoin = off; set enable_hashjoin = off; set enable_indexscan = off;'
    ]

    QUERY_DIR = "dataset/queries/tpc-h_sf1_10"  # SQL文件路径
    EXPLAIN_OR_NOT = False  # 是否开启解释器
    SAVE_MODEL = True  # 是否保存模型

    @staticmethod
    def get_db_params(dbname=None, user=None, password=None, host=None, port=None):
        """
        根据传入参数构建 db_params 字典。
        如果外部没有传入（None），则使用默认配置。
        """
        return {
            "dbname": dbname or Config.DEFAULT_DB_PARAMS["dbname"],
            "user": user or Config.DEFAULT_DB_PARAMS["user"],
            "password": password or Config.DEFAULT_DB_PARAMS["password"],
            "host": host or Config.DEFAULT_DB_PARAMS["host"],
            "port": port or Config.DEFAULT_DB_PARAMS["port"],
            "options": Config.DEFAULT_DB_PARAMS["options"]
        }

    @staticmethod
    def setup_logging():
        """全局日志配置：同时输出到控制台和文件"""
        Config.LOG_DIR.mkdir(parents=True, exist_ok=True)

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
            handlers=[
                logging.FileHandler(Config.LOG_DIR / "app.log"),
                logging.StreamHandler()  # 控制台输出
            ]
        )
        return logging.getLogger("DB")

    @staticmethod
    def get_data_path(db_name):
        """生成并自动创建统计数据存放路径"""
        path = Config.DATASET_DIR / db_name
        path.mkdir(parents=True, exist_ok=True)
        return path

    @staticmethod
    def get_result_path(db_name):
        """生成并自动创建统计数据存放路径"""
        path = Config.RES_DIR / db_name
        path.mkdir(parents=True, exist_ok=True)
        return path


# 预创建基础文件夹
Config.DATASET_DIR.mkdir(exist_ok=True)
