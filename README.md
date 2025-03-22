# CARLA 強化學習專案

這個專案實作了一個基於強化學習的自動駕駛模型，目標是在 CARLA 模擬器中訓練一個能夠有效進行車道保持的智能代理。本專案使用 PPO (Proximal Policy Optimization) 算法來訓練模型，並整合了完整的獎勵機制。

## 專案架構

```
carla-rl-project/
├── src/
│   ├── environment.py  # CARLA 環境封裝
│   ├── agent.py       # 強化學習代理實作
│   └── utils.py       # 工具函數
├── train.py           # 訓練腳本
├── test.py           # 測試腳本
└── requirements.txt   # 專案依賴
```

### 核心組件說明

#### 環境設定 (environment.py)
- 使用 CARLA Town04 地圖和賓士車輛模型
- 獎勵函數設計：
  * 距離獎勵：鼓勵車輛接近目標
  * 方向獎勵：鼓勵車輛朝向目標
  * 速度獎勵：維持適當速度
  * 碰撞懲罰：避免碰撞
  * 完成獎勵：成功到達目標

#### 訓練設定 (train.py)
- 使用 Stable-Baselines3 的 PPO 算法
- 整合 Tensorboard 用於訓練監控
- 訓練參數配置：
  * 總訓練步數：1,000,000 步
  * 批次大小：64
  * 學習率：3e-4
  * 每 10000 步儲存檢查點
  * 每 10000 步進行評估

#### 測試功能 (test.py)
- 支援載入訓練模型進行測試
- 即時顯示：距離、獎勵等資訊
- 提供成功率和平均表現統計

## 安裝步驟

1. 確保已安裝 CARLA 模擬器
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
1. 啟動 CARLA 伺服器
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
python test.py --model-path logs/PPO_<timestamp>/best_model/best_model.zip
```

## 訓練建議

- 建議進行長時間訓練（如整晚）以獲得更好的效果
- 訓練量建議：
  * 理想訓練步數：500萬到1000萬步
  * 可透過 Tensorboard 觀察訓練曲線判斷收斂情況
- PPO 算法特性：
  * 訓練穩定性高
  * 適合長時間訓練
  * 有良好的探索-利用平衡
