"""
CARLA 自動駕駛強化學習訓練腳本。

使用 Stable-Baselines3 的 PPO 算法進行訓練，並整合 Tensorboard 用於觀察訓練過程。
"""

import os
import time
from datetime import datetime
import numpy as np
import carla
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from src.environment import CarlaEnv

def make_env():
    """創建並配置 CARLA 環境"""
    env = CarlaEnv()
    # 使用 Monitor 包裝器來記錄訓練數據
    log_dir = os.path.join("logs", "monitor")
    os.makedirs(log_dir, exist_ok=True)
    env = Monitor(env, log_dir)
    return env

def main():
    """主訓練函數"""
    # 創建日誌目錄
    current_time = datetime.now().strftime('%Y%m%d-%H%M%S')
    log_dir = os.path.join("logs", f"PPO_{current_time}")
    os.makedirs(log_dir, exist_ok=True)
    
    # 創建環境
    env = DummyVecEnv([make_env])
    
    # 配置 PPO 模型
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,          # 學習率
        n_steps=2048,               # 每次更新收集的步數
        batch_size=64,              # 每次優化的批次大小
        n_epochs=10,                # 每次更新的訓練回合
        gamma=0.99,                 # 折扣因子
        gae_lambda=0.95,            # GAE lambda 參數
        clip_range=0.2,             # PPO clip 參數
        ent_coef=0.01,              # 熵係數，用於鼓勵探索
        verbose=1,
        tensorboard_log=log_dir
    )
    
    # 設置回調函數
    # 每 10000 步保存一次檢查點
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=os.path.join(log_dir, "checkpoints"),
        name_prefix="carla_model"
    )
    
    # 創建評估環境
    eval_env = DummyVecEnv([make_env])
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(log_dir, "best_model"),
        log_path=os.path.join(log_dir, "eval_results"),
        eval_freq=10000,             # 每 10000 步評估一次
        n_eval_episodes=5,           # 每次評估 5 個回合
        deterministic=True
    )
    
    # 訓練模型
    total_timesteps = 1_000_000     # 總訓練步數
    try:
        print(f"開始訓練，總步數: {total_timesteps}")
        print(f"可以使用以下指令查看 Tensorboard:")
        print(f"tensorboard --logdir={log_dir}")
        
        model.learn(
            total_timesteps=total_timesteps,
            callback=[checkpoint_callback, eval_callback],
            progress_bar=True
        )
        
        # 保存最終模型
        final_model_path = os.path.join(log_dir, "final_model")
        model.save(final_model_path)
        print(f"訓練完成！模型已保存至: {final_model_path}")
        
    except KeyboardInterrupt:
        print("\n訓練被中斷")
        # 保存中斷時的模型
        interrupted_model_path = os.path.join(log_dir, "interrupted_model")
        model.save(interrupted_model_path)
        print(f"已保存中斷時的模型至: {interrupted_model_path}")
    
    finally:
        # 清理環境
        env.close()
        eval_env.close()

if __name__ == "__main__":
    main()