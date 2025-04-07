# CARLA 強化學習專案

這個專案實作了一個基於強化學習的自動駕駛系統，使用CARLA模擬器環境訓練智能代理進行自動駕駛。本專案採用PPO (Proximal Policy Optimization) 算法，實現了完整的環境感知、決策和控制功能。

## 環境需求

- Python 3.8+
- CARLA 0.9.13+
- CUDA 11.0+ (用於GPU訓練，建議)
- 主要依賴套件：
  * stable-baselines3
  * gymnasium
  * numpy
  * pygame
  * tensorboard

## 專案架構

```
carla-rl-project/
├── src/
│   ├── environment.py  # CARLA環境封裝
│   ├── agent.py       # 強化學習代理實作
│   └── utils.py       # 工具函數
├── train.py           # 訓練腳本
├── test.py           # 測試腳本
└── requirements.txt   # 專案依賴
```

### 核心組件說明

#### 環境設定 (environment.py)
- **場景配置**：
  * 使用CARLA Town04地圖
  * 使用賓士轎車模型
  * 支援即時渲染和視頻錄製

- **觀察空間**：
  * 車輛位置 (x, y)
  * 航向角
  * 當前速度
  * 到目標距離
  * 與目標方向夾角

- **動作空間**：
  * 轉向控制 (-1.0 到 1.0)
  * 油門/剎車控制 (-1.0 到 1.0)

- **獎勵機制設計**：
  * 車道保持獎勵 (-2.0 到 2.0)：根據與車道中心的距離計算
  * 速度控制獎勵 (0.0 到 1.0)：鼓勵維持5-15 m/s的適當速度
  * 方向獎勵 (0.0 到 1.0)：基於與目標方向的夾角
  * 碰撞懲罰 (-50.0)：發生碰撞時給予
  * 到達目標獎勵 (100.0)：成功到達目標位置

#### 訓練設定 (train.py)
- **算法配置**：
  * 使用Stable-Baselines3的PPO實現
  * MLP策略網絡
  * 支援Tensorboard監控

- **訓練參數**：
  * 總訓練步數：1,000,000步
  * 批次大小：64
  * 學習率：3e-4
  * GAE lambda：0.95
  * Clip range：0.2
  * 熵係數：0.01

- **檢查點與評估**：
  * 每50,000步保存一次檢查點
  * 每10,000步進行一次評估
  * 自動保存最佳模型
  * 支援訓練中斷恢復

#### 測試功能 (test.py)
- **命令行參數**：
  * `--model-path`：指定要測試的模型路徑（必需）
  * `--episodes`：測試回合數（預設：5）
  * `--seed`：隨機種子（可選）
  * `--render`：是否啟用視覺化顯示

- **評估指標**：
  * 即時顯示：當前步數、距離、獎勵等
  * 回合統計：總步數、總獎勵
  * 整體統計：平均獎勵、平均步數、成功率

- **視頻記錄**：
  * 非渲染模式下自動錄製視頻
  * 保存至`./videos`目錄
  * 使用時間戳命名，便於管理

## 安裝步驟

1. 確保已安裝CARLA模擬器：
```bash
# 下載並解壓CARLA
wget https://carla-releases.s3.eu-west-3.amazonaws.com/Linux/CARLA_0.9.13.tar.gz
tar xf CARLA_0.9.13.tar.gz
```

2. 克隆此專案：
```bash
git clone https://github.com/2Allen/carla-rl-project.git
cd carla-rl-project
```

3. 安裝依賴套件：
```bash
pip install -r requirements.txt
```

## 使用說明

### 訓練模型
1. 啟動CARLA伺服器：
```bash
cd /path/to/carla
./CarlaUE4.sh
```

2. 執行訓練腳本：
```bash
python train.py
```

3. 監控訓練過程：
```bash
tensorboard --logdir=logs
```

### 測試模型
```bash
# 使用視覺化界面測試
python test.py --model-path logs/PPO_<timestamp>/best_model/best_model.zip --render

# 測試特定回合數
python test.py --model-path logs/PPO_<timestamp>/best_model/best_model.zip --episodes 10

# 使用固定隨機種子測試
python test.py --model-path logs/PPO_<timestamp>/best_model/best_model.zip --seed 42
```

## 訓練建議

- **硬體配置建議**：
  * CPU：8核心以上
  * 記憶體：16GB以上
  * GPU：8GB顯存以上（NVIDIA）

- **訓練策略**：
  * 建議進行長時間訓練（至少24小時）
  * 理想訓練步數：500萬到1000萬步
  * 通過Tensorboard監控以下指標：
    - 平均回合獎勵
    - 成功率
    - 策略損失
    - 值函數損失

- **PPO算法優勢**：
  * 訓練穩定性高
  * 超參數不敏感
  * 適合長時間訓練
  * 具有良好的探索-利用平衡

- **常見問題處理**：
  * 如果訓練不穩定，嘗試調整學習率
  * 如果探索不足，可以增加熵係數
  * 如果收斂過慢，可以調整batch_size和n_steps
