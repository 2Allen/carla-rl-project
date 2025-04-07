"""自動駕駛模型測試腳本。

此模組用於評估在CARLA環境中訓練的強化學習模型的效能。它提供以下功能：
- 載入已訓練的PPO模型
- 在CARLA模擬環境中執行測試回合
- 收集並顯示效能指標（成功率、平均獎勵等）
- 支援視覺化和錄影功能

典型用法：
    python test.py --model-path ./models/ppo_model.zip --episodes 5 --render
"""

import argparse
import os
import sys
import time

import numpy as np
from stable_baselines3 import PPO

from src.environment import CarlaEnv

def parse_args():
    """解析命令行參數"""
    parser = argparse.ArgumentParser(description='測試訓練好的 CARLA 自動駕駛模型')
    parser.add_argument(
        '--model-path',
        type=str,
        required=True,
        help='訓練好的模型路徑'
    )
    parser.add_argument(
        '--episodes',
        type=int,
        default=5,
        help='測試回合數'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='隨機種子'
    )
    parser.add_argument(
        '--render',
        action='store_true',
        help='啟用視覺化界面（啟用時不會錄影，未啟用時自動錄影到 ./videos 目錄）'
    )
    return parser.parse_args()

def evaluate_model(model, num_episodes=5, render_mode=False, seed=None):
    """在CARLA環境中評估強化學習模型的效能。
       注意：此函數現在會在每個回合內部創建和銷毀環境。

    Args:
        model: 已訓練的PPO模型實例。
        num_episodes: 要執行的測試回合數量，預設為5。
        render_mode: 是否啟用視覺化。
        seed: 隨機種子。

    Returns:
        dict: 包含評估結果的字典，包括：
            - mean_reward: 平均回合獎勵
            - mean_steps: 平均回合步數
            - success_rate: 成功到達目標的百分比
            - episode_rewards: 每個回合的獎勵列表
            - episode_lengths: 每個回合的步數列表
    """
    episode_rewards = []
    episode_lengths = []
    success_count = 0
    
    for episode in range(num_episodes):
        print(f"\n{'='*20} 回合 {episode + 1}/{num_episodes} {'='*20}")
        env = None # 確保 env 在 try 之前被定義
        try:
            # --- 在每個回合開始時初始化環境 ---
            print("重新初始化 CARLA 環境...")
            env = CarlaEnv(render_mode=render_mode)
            # 注意：如果需要在環境中使用種子，需要在這裡傳遞
            # obs, info = env.reset(seed=seed) # 如果 CarlaEnv 支持 reset 時設置種子
            obs, info = env.reset() # 假設 reset 會處理內部種子或不需要每次設置
            # ------------------------------------

            terminated = truncated = False
            total_reward = 0
            steps = 0

            while not (terminated or truncated):
                action, _ = model.predict(obs, deterministic=True)
                # 確保 env 存在才執行 step
                if env is None:
                    print("\n錯誤：環境未成功初始化，跳過此回合。")
                    break
                obs, reward, terminated, truncated, info = env.step(action)

                total_reward += reward
                steps += 1

                # 獲取並顯示狀態資訊
                distance = info.get('distance_to_target', float('inf'))
                collision = info.get('collision_detected', False)
                status_msg = (
                    f"\r進度: 步數={steps:4d} | "
                    f"距離={distance:6.2f}m | "
                    f"獎勵={reward:6.2f} | "
                    f"總獎勵={total_reward:8.2f}"
                )
                print(status_msg, end='', flush=True)

                # 檢查特殊事件
                if collision:
                    print("\n[警告] 發生碰撞！")

                if distance < 5.0 and not collision:
                    success_count += 1
                    print("\n[成功] 已到達目標！")

                time.sleep(0.05)

            episode_rewards.append(total_reward)
            episode_lengths.append(steps)

            # 回合結束統計
            print(f"\n{'-'*10} 回合統計 {'-'*10}")
            print(f"總步數: {steps:4d}")
            print(f"總獎勵: {total_reward:8.2f}")

        finally:
            # --- 確保每個回合結束時關閉環境 ---
            if env is not None:
                print("\n清理當前回合環境資源...")
                env.close()
            # ------------------------------------
    # 計算整體統計結果
    mean_reward = np.mean(episode_rewards)
    mean_steps = np.mean(episode_lengths)
    success_rate = (success_count / num_episodes) * 100
    
    # 輸出最終結果
    print(f"\n{'='*20} 測試結果 {'='*20}")
    print(f"測試回合數: {num_episodes}")
    print(f"平均獎勵  : {mean_reward:8.2f}")
    print(f"平均步數  : {mean_steps:8.2f}")
    print(f"成功率    : {success_rate:8.2f}%")
    print('='*50)
    
    return {
        'mean_reward': mean_reward,
        'mean_steps': mean_steps,
        'success_rate': success_rate,
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths
    }

def main():
    """主程式進入點。

    處理命令行參數解析、環境初始化、模型載入和評估，以及錯誤處理。
    支援以下功能：
    - 設置隨機種子以確保可重現性
    - 使用指定的訓練模型
    - 配置是否啟用視覺化界面
    - 完整的錯誤處理和資源清理
    """
    args = parse_args()

    try:
        # 設置隨機種子（如果有指定）
        # 注意：種子現在主要影響模型加載和可能的環境內部隨機性（如果支持）
        if args.seed is not None:
            print(f"設置隨機種子: {args.seed}")
            np.random.seed(args.seed)
            # 如果需要，也可以考慮設置 torch/tensorflow 的種子

        # 驗證模型文件是否存在
        if not os.path.exists(args.model_path):
            raise FileNotFoundError(f"找不到模型文件: {args.model_path}")

        # 載入預訓練模型
        print(f"載入模型: {args.model_path}")
        # 可以在這裡傳遞 device='cpu' 如果不想用 GPU
        model = PPO.load(args.model_path)

        # 執行評估並獲取結果
        # 將環境相關參數傳遞給 evaluate_model
        print("\n開始模型評估...")
        results = evaluate_model(
            model=model,
            num_episodes=args.episodes,
            render_mode=args.render,
            seed=args.seed # 傳遞種子以供潛在的環境內部使用
        )
        
        # 可以在這裡添加結果的進一步處理，例如：
        # - 保存到文件
        # - 生成詳細報告
        # - 繪製圖表等
        
    except FileNotFoundError as e:
        print(f"\n錯誤: {e}")
        return 1
    except KeyboardInterrupt:
        print("\n測試被使用者中斷")
        return 130  # 標準的 SIGINT 返回碼
    except Exception as e:
        print(f"\n發生未預期的錯誤: {str(e)}")
        return 1
    finally:
        # 環境清理邏輯已移至 evaluate_model 內部
        print("\n測試執行完畢。")
    
    return 0  # 成功完成

if __name__ == "__main__":
    sys.exit(main())  # 使用返回碼作為程式退出狀態