"""
Alicia Inference API - 用户级 API

提供统一的推理接口，开发者可以直接调用方法进行模型加载和推理。
"""
import logging
import time
from typing import TYPE_CHECKING, Dict, List, Optional, Union

import numpy as np

from .core.config import DeviceConfig, PolicyLoadConfig, RuntimeConfig
from .core.types import Observation, PolicyMetadata, PolicyStatus

if TYPE_CHECKING:
    from .session import InferenceSession

logger = logging.getLogger(__name__)


class InferenceAPI:
    """推理 API - 提供统一的用户接口"""

    def __init__(self, auto_start_async: bool = False):
        """初始化推理 API

        :param auto_start_async: 加载模型后自动启动异步推理
        """
        self._session: Optional["InferenceSession"] = None
        self._auto_start_async = auto_start_async
        self._model_loaded = False

    # ==================== 连接管理 ====================

    def load_model(
        self,
        model_type: str,
        checkpoint_dir: str,
        device: str = "cuda:0",
        tokenizer_path: Optional[str] = None,
        instruction: Optional[str] = None,
        control_fps: float = 20.0,
        enable_async: bool = True,
        temporal_ensemble_coeff: Optional[float] = None,
        robot_type: Optional[str] = None,
        policy_robot_type: Optional[str] = None,
    ) -> Dict:
        """加载推理模型

        :param model_type: 模型类型 (act, pi0, smolvla)
        :param checkpoint_dir: 检查点目录路径
        :param device: 设备 (cuda:0, cpu 等)
        :param tokenizer_path: tokenizer 路径 (PI0/SmolVLA)
        :param instruction: 语言指令 (PI0/SmolVLA)
        :param control_fps: 控制频率
        :param enable_async: 是否启用异步推理
        :param temporal_ensemble_coeff: 通用时间集成系数。设置后每一步都会重新推理并对重叠动作做指数加权融合
        :param robot_type: 运行时机械臂类型 (ACT 可选, 如 Alicia-D / Alicia-M)
        :param policy_robot_type: 模型动作空间对应的机械臂类型 (ACT 可选)
        :return: 包含模型元数据的字典
        """
        try:
            # 卸载已有模型
            if self._session is not None:
                self.unload_model()

            # 创建配置
            config = PolicyLoadConfig(
                checkpoint_dir=checkpoint_dir,
                model_type=model_type,
                device=DeviceConfig(device=device),
                runtime=RuntimeConfig(
                    control_fps=control_fps,
                    enable_async_inference=enable_async,
                    temporal_ensemble_coeff=temporal_ensemble_coeff,
                ),
                tokenizer_path=tokenizer_path,
                instruction=instruction,
                robot_type=robot_type,
                policy_robot_type=policy_robot_type,
            )

            # 加载模型
            from .session import InferenceSession

            self._session = InferenceSession()
            self._session.load(config=config)
            self._model_loaded = True

            # 自动启动异步推理
            if self._auto_start_async and enable_async:
                self._session.start_async_inference()

            # 获取元数据
            metadata = self._session.get_metadata()

            logger.info(f"模型加载成功: {model_type}")
            return {
                "success": True,
                "model_type": metadata.model_type,
                "required_cameras": metadata.required_cameras,
                "state_dim": metadata.state_dim,
                "action_dim": metadata.action_dim,
                "chunk_size": metadata.chunk_size,
            }

        except Exception as e:
            logger.error(f"加载模型时发生未知错误: {str(e)}")
            raise

    def unload_model(self):
        """卸载当前模型"""
        if self._session is not None:
            try:
                self._session.close()
                logger.info("模型已卸载")
            except Exception as e:
                logger.error(f"卸载模型失败: {str(e)}")
            finally:
                self._session = None
                self._model_loaded = False

    def is_loaded(self) -> bool:
        """检查模型是否已加载"""
        return self._model_loaded and self._session is not None

    # ==================== 推理接口 ====================

    def infer(
        self,
        images: Dict[str, np.ndarray],
        state: Union[List[float], np.ndarray],
        instruction: Optional[str] = None,
    ) -> np.ndarray:
        """执行推理，返回动作

        :param images: 图像字典 {camera_name: np.ndarray (BGR)}
        :param state: 机器人状态向量
        :param instruction: 可选的语言指令
        :return: 动作向量 (numpy array)
        """
        if not self.is_loaded():
            raise RuntimeError("模型未加载，请先调用 load_model()")

        try:
            # 确保 state 是 numpy 数组
            if isinstance(state, list):
                state = np.array(state, dtype=np.float32)

            # 构造观测
            observation = Observation(
                images=images,
                state=state,
                instruction=instruction,
            )

            # 执行推理
            action = self._session.infer(observation)
            return action

        except Exception as e:
            logger.error(f"推理时发生未知错误: {str(e)}")
            raise

    def step(
        self,
        images: Dict[str, np.ndarray],
        state: Union[List[float], np.ndarray],
        instruction: Optional[str] = None,
    ) -> np.ndarray:
        """执行单步推理（用于控制环）

        :param images: 图像字典 {camera_name: np.ndarray (BGR)}
        :param state: 机器人状态向量
        :param instruction: 可选的语言指令
        :return: 单步动作向量
        """
        if not self.is_loaded():
            raise RuntimeError("模型未加载，请先调用 load_model()")

        try:
            if isinstance(state, list):
                state = np.array(state, dtype=np.float32)

            observation = Observation(
                images=images,
                state=state,
                instruction=instruction,
            )

            action = self._session.step(observation)
            return action

        except Exception as e:
            logger.error(f"单步推理时发生未知错误: {str(e)}")
            raise

    # ==================== 状态查询 ====================

    def get_metadata(self) -> PolicyMetadata:
        """获取模型元数据"""
        if not self.is_loaded():
            raise RuntimeError("模型未加载")
        return self._session.get_metadata()

    def get_status(self) -> PolicyStatus:
        """获取模型运行状态"""
        if not self.is_loaded():
            raise RuntimeError("模型未加载")
        return self._session.get_status()

    def get_model_info(self) -> Dict:
        """获取模型信息（简化版）"""
        if not self.is_loaded():
            return {
                "is_loaded": False,
                "model_type": None,
            }

        try:
            metadata = self.get_metadata()
            return {
                "is_loaded": True,
                "model_type": metadata.model_type,
                "required_cameras": metadata.required_cameras,
                "state_dim": metadata.state_dim,
                "action_dim": metadata.action_dim,
                "chunk_size": metadata.chunk_size,
            }
        except Exception as e:
            logger.error(f"获取模型信息失败: {str(e)}")
            return {"is_loaded": False, "error": str(e)}

    # ==================== 异步推理控制 ====================

    def start_async_inference(self):
        """启动异步推理"""
        if not self.is_loaded():
            raise RuntimeError("模型未加载")
        self._session.start_async_inference()
        logger.info("异步推理已启动")

    def stop_async_inference(self):
        """停止异步推理"""
        if not self.is_loaded():
            raise RuntimeError("模型未加载")
        self._session.stop_async_inference()
        logger.info("异步推理已停止")

    # ==================== 工具方法 ====================

    def print_info(self):
        """打印当前模型信息"""
        info = self.get_model_info()
        if not info["is_loaded"]:
            print("模型未加载")
            return

        print("=" * 50)
        print(f"模型类型: {info['model_type']}")
        print(f"需要相机: {info['required_cameras']}")
        print(f"状态维度: {info['state_dim']}")
        print(f"动作维度: {info['action_dim']}")
        print(f"动作块大小: {info['chunk_size']}")
        print("=" * 50)

    def __enter__(self):
        """上下文管理器入口"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器退出"""
        self.unload_model()
        return False

    def __del__(self):
        """析构函数"""
        try:
            self.unload_model()
        except Exception:
            pass
