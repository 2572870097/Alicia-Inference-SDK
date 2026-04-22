"""
InferenceAPI 使用示例
演示如何使用封装好的 API 类进行推理
"""
from pathlib import Path

import numpy as np
import cv2
from inference_sdk import InferenceAPI


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ACT_CHECKPOINT_DIR = PROJECT_ROOT / "models" / "ACT_pick_and_place_v2"
PI0_CHECKPOINT_DIR = Path("/path/to/pi0/checkpoint")
PI0_TOKENIZER_DIR = Path("/path/to/pi0/tokenizer")
SMOLVLA_CHECKPOINT_DIR = Path("/path/to/smolvla/checkpoint")
SMOLVLA_VLM_DIR = Path("/path/to/HuggingFaceTB/SmolVLM2-500M-Video-Instruct")
SPARKMIND_CHECKOUT = PROJECT_ROOT.parent / "SparkMind"


def print_runtime_prerequisites():
    """打印运行示例前的依赖和资产要求。"""
    print("运行前提:")
    print(f"1. 在当前虚拟环境安装 SDK extras 和 SparkMind: pip install -e .[all] && pip install -e {SPARKMIND_CHECKOUT}")
    print(f"2. ACT 示例默认使用本地 checkpoint: {ACT_CHECKPOINT_DIR}")
    print(f"3. PI0 需要 checkpoint + tokenizer 资产: {PI0_CHECKPOINT_DIR} / {PI0_TOKENIZER_DIR}")
    print("   也可以通过环境变量 PI0_TOKENIZER_PATH 指向 tokenizer 目录")
    print(f"4. SmolVLA 需要 checkpoint + 基础 VLM 资产: {SMOLVLA_CHECKPOINT_DIR} / {SMOLVLA_VLM_DIR}")
    print("   离线环境建议设置环境变量 SMOLVLA_VLM_MODEL_PATH")
    print()


def print_model_prerequisites(model_type: str):
    """针对具体模型打印运行提示。"""
    normalized = model_type.lower()
    print(f"{normalized} 运行提示:")
    print(f"- 当前虚拟环境里需要可导入的 SparkMind 依赖；推荐执行 `pip install -e {SPARKMIND_CHECKOUT}`")

    if normalized == "act":
        print(f"- ACT 需要合法 checkpoint，例如: {ACT_CHECKPOINT_DIR}")
        return

    if normalized == "pi0":
        print(f"- PI0 需要 checkpoint，例如: {PI0_CHECKPOINT_DIR}")
        print(f"- PI0 还需要 tokenizer 资产，例如: {PI0_TOKENIZER_DIR}")
        print("- 可通过 load_model(..., tokenizer_path=...) 或环境变量 PI0_TOKENIZER_PATH 指向本地 tokenizer")
        return

    if normalized == "smolvla":
        print(f"- SmolVLA 需要 checkpoint，例如: {SMOLVLA_CHECKPOINT_DIR}")
        print(f"- SmolVLA 还需要基础 VLM 资产，例如: {SMOLVLA_VLM_DIR}")
        print("- 离线环境请设置环境变量 SMOLVLA_VLM_MODEL_PATH 指向本地基础模型目录")
        return

    print("- 请确认 checkpoint 路径、模型依赖和相关资产已经准备完成")


def load_model_with_help(api: InferenceAPI, **kwargs):
    """加载模型并在失败时给出可执行的排障提示。"""
    model_type = kwargs.get("model_type", "unknown")
    try:
        return api.load_model(**kwargs)
    except Exception as exc:
        print(f"加载模型失败: {exc}")
        print_model_prerequisites(model_type)
        return None


def example_basic_usage():
    """基础使用示例"""
    print("=" * 50)
    print("示例 1: 基础使用")
    print("=" * 50)

    # 创建 API 实例
    api = InferenceAPI()

    # 加载模型
    result = load_model_with_help(
        api,
        model_type="act",
        checkpoint_dir=str(ACT_CHECKPOINT_DIR),
        device="cpu"
    )
    if result is None:
        return
    print(f"模型加载结果: {result}")

    # 打印模型信息
    api.print_info()

    # 准备数据
    head_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    wrist_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    robot_state = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]

    # 执行推理
    action = api.infer(
        images={"head": head_img, "wrist": wrist_img},
        state=robot_state
    )
    print(f"推理结果: {action[:5]}... (共 {len(action)} 维)")

    # 卸载模型
    api.unload_model()
    print("模型已卸载\n")


def example_with_context_manager():
    """使用上下文管理器"""
    print("=" * 50)
    print("示例 2: 使用上下文管理器")
    print("=" * 50)

    with InferenceAPI() as api:
        # 加载模型
        result = load_model_with_help(
            api,
            model_type="pi0",
            checkpoint_dir=str(PI0_CHECKPOINT_DIR),
            tokenizer_path=str(PI0_TOKENIZER_DIR),
            instruction="pick up the apple"
        )
        if result is None:
            return

        # 执行推理
        images = {
            "head": np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),
            "wrist": np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        }
        state = np.random.randn(7)

        action = api.infer(images=images, state=state)
        print(f"动作: {action[:5]}...")

    # 自动卸载
    print("上下文退出，模型自动卸载\n")


def example_control_loop():
    """控制环示例"""
    print("=" * 50)
    print("示例 3: 控制环")
    print("=" * 50)

    api = InferenceAPI()

    # 加载模型
    result = load_model_with_help(
        api,
        model_type="act",
        checkpoint_dir=str(ACT_CHECKPOINT_DIR),
        device="cpu",
        enable_async=False,
    )
    if result is None:
        return

    # 模拟控制环
    for i in range(10):
        # 获取观测
        images = {
            "head": np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),
            "wrist": np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        }
        state = np.random.randn(7)

        # 执行单步推理
        action = api.step(images=images, state=state)
        print(f"步骤 {i+1}: 动作 = {action[:3]}...")

    api.unload_model()
    print("控制环结束\n")


def example_error_handling():
    """错误处理示例"""
    print("=" * 50)
    print("示例 4: 错误处理")
    print("=" * 50)

    api = InferenceAPI()

    try:
        # 尝试在未加载模型时推理
        api.infer(
            images={"head": np.zeros((480, 640, 3), dtype=np.uint8)},
            state=[0.0] * 7
        )
    except Exception as e:
        print(f"预期的错误: {e}")

    try:
        # 加载模型
        api.load_model(
            model_type="act",
            checkpoint_dir="/path/to/checkpoint"
        )

        # 正常推理
        action = api.infer(
            images={
                "head": np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),
                "wrist": np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            },
            state=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
        )
        print(f"推理成功: {action[:3]}...")

    except Exception as e:
        print(f"错误: {e}")
    finally:
        api.unload_model()

    print()


def example_real_camera():
    """真实相机示例"""
    print("=" * 50)
    print("示例 5: 使用真实相机")
    print("=" * 50)

    api = InferenceAPI()

    # 加载模型
    result = load_model_with_help(
        api,
        model_type="act",
        checkpoint_dir=str(ACT_CHECKPOINT_DIR)
    )
    if result is None:
        return

    # 打开相机
    cap = cv2.VideoCapture(0)

    try:
        for i in range(5):
            # 读取图像
            ret, frame = cap.read()
            if not ret:
                print("无法读取相机")
                break

            # 执行推理
            action = api.infer(
                images={"head": frame, "wrist": frame},  # 示例：使用同一相机
                state=np.random.randn(7)
            )
            print(f"帧 {i+1}: 动作 = {action[:3]}...")

    finally:
        cap.release()
        api.unload_model()

    print()


if __name__ == "__main__":
    # 运行示例（需要根据实际情况修改路径）
    print("\n注意: 这些示例除了 checkpoint 之外，还可能需要 SparkMind / tokenizer / 基础模型资产。\n")
    print_runtime_prerequisites()

    # example_basic_usage()
    # example_with_context_manager()
    example_control_loop()
    # example_error_handling()
    # example_real_camera()

    print("请取消注释相应的示例函数来运行")
