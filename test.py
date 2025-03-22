"""
CARLA 自動駕駛測試腳本。

用於載入訓練好的模型並在環境中進行測試，展示訓練成果。
可以通過設置 RENDER_MODE 來選擇是否要顯示 CARLA 的視覺化界面。
"""

import os
import time
import argparse
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
        help='啟用視覺化界面'
    )
    return parser.parse_args()

def evaluate_model(model, env, num_episodes=5):
    """
    評估模型效果。
    
    參數:
        model: 訓練好的PPO模型
        env: CARLA環境實例
        num_episodes: 測試回合數
    """
    episode_rewards = []
    episode_lengths = []
    success_count = 0
    
    for episode in range(num_episodes):
        print(f"\n開始測試回合 {episode + 1}/{num_episodes}")
        obs, info = env.reset()
        terminated = truncated = False
        total_reward = 0
        steps = 0
        
        while not (terminated or truncated):
            # 使用模型預測動作
            action, _ = model.predict(obs, deterministic=True)
            
            # 執行動作
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            
            # 更新狀態
            done = terminated or truncated
            
            # 顯示即時資訊
            distance = info.get('distance_to_target', float('inf'))
            collision = info.get('collision_detected', False)
            print(f"\r步數: {steps}, 距離目標: {distance:.2f}m, "
                  f"獎勵: {reward:.2f}, 總獎勵: {total_reward:.2f}",
                  end='', flush=True)
            
            if collision:
                print("\n發生碰撞！")
            
            # 如果到達目標
            if distance < 5.0 and not collision:
                success_count += 1
                print("\n成功到達目標！")
            
            time.sleep(0.05)  # 放慢顯示速度，便於觀察
            
        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        
        print(f"\n回合 {episode + 1} 結束")
        print(f"總步數: {steps}")
        print(f"總獎勵: {total_reward:.2f}")
    
    # 顯示總結果
    print("\n===== 測試結果 =====")
    print(f"測試回合數: {num_episodes}")
    print(f"平均回合獎勵: {np.mean(episode_rewards):.2f}")
    print(f"平均回合步數: {np.mean(episode_lengths):.2f}")
    print(f"成功率: {success_count/num_episodes*100:.2f}%")
    print("==================")

def main():
    """主函數"""
    args = parse_args()
    
    if args.seed is not None:
        np.random.seed(args.seed)
    # 創建環境
    env = CarlaEnv(render_mode=args.render)
    
    
    try:
        # 載入模型
        print(f"載入模型: {args.model_path}")
        model = PPO.load(args.model_path)
        
        # 運行測試
        evaluate_model(model, env, args.episodes)
        
    except KeyboardInterrupt:
        print("\n測試被中斷")
    
    finally:
        # 清理環境
        env.close()

if __name__ == "__main__":
    main()