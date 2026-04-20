"""
InferenceAPI 使用示例
演示如何使用封装好的 API 类进行推理
"""
import numpy as np
import cv2
from inference_sdk import InferenceAPI


def example_basic_usage():
    """基础使用示例"""
    print("=" * 50)
    print("示例 1: 基础使用")
    print("=" * 50)

    # 创建 API 实例
    api = InferenceAPI()

    # 加载模型
    result = api.load_model(
        model_type="act",
        checkpoint_dir="/path/to/checkpoint",
        device="cuda:0"
    )
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
        api.load_model(
            model_type="pi0",
            checkpoint_dir="/path/to/checkpoint",
            tokenizer_path="/path/to/tokenizer",
            instruction="pick up the apple"
        )

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

    api = InferenceAPI(auto_start_async=True)

    # 加载模型
    api.load_model(
        model_type="act",
        checkpoint_dir="/path/to/checkpoint",
        enable_async=True
    )

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
    api.load_model(
        model_type="act",
        checkpoint_dir="/path/to/checkpoint"
    )

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
    print("\n注意: 这些示例需要有效的模型检查点路径才能运行\n")

    # example_basic_usage()
    # example_with_context_manager()
    # example_control_loop()
    # example_error_handling()
    # example_real_camera()

    print("请取消注释相应的示例函数来运行")
