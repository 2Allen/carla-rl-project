"""CARLA強化學習環境。

此模組實現了一個基於CARLA的強化學習環境，使用Town04地圖和賓士車輛。
環境設計符合OpenAI Gym介面標準，便於與Stable-Baselines3整合。
"""

import math
import random
import numpy as np
import pygame
import gymnasium as gym
from gymnasium import spaces
import carla
import queue
import weakref

class CarlaCameraCallback:
    """相機感應器回調處理類"""
    def __init__(self):
        self._queue = queue.Queue()
        
    def __call__(self, image):
        """將相機圖像數據加入佇列"""
        # 獲取原始圖像數據並轉換成RGBA格式
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]  # 只保留RGB通道
        self._queue.put(array)
        
    def get_image(self):
        """從佇列中獲取最新的圖像"""
        try:
            return self._queue.get(timeout=2.0)
        except queue.Empty:
            return None

class CarlaEnv(gym.Env):
    """
    CARLA強化學習環境類。
    
    屬性:
        observation_space: 狀態空間，包含車輛位置、速度、方向等資訊
        action_space: 動作空間，包含轉向、油門、剎車
    """
    
    def __init__(self, host='localhost', port=2000, timeout=10.0, render_mode=False):
        """
        初始化CARLA環境。

        參數:
            host: CARLA伺服器主機名
            port: CARLA伺服器端口
            timeout: 連接超時時間（秒）
            render_mode: 是否啟用視覺化顯示
        """
        super().__init__()

        # 連接CARLA伺服器
        self.client = carla.Client(host, port)
        self.client.set_timeout(timeout)
        
        # 設定Town04地圖和基本配置
        self.world = self.client.load_world('Town04')
        self.map = self.world.get_map()
        
        # 設定渲染模式
        if render_mode:
            # 設置同步模式
            settings = self.world.get_settings()
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = 0.05
            self.world.apply_settings(settings)
            
            # 初始化pygame
            pygame.init()
            self.display = pygame.display.set_mode((800, 600))
            pygame.display.set_caption("CARLA 自動駕駛")
        
        # 設定天氣
        weather = carla.WeatherParameters(
            cloudiness=30.0,
            precipitation=0.0,
            sun_altitude_angle=70.0
        )
        self.world.set_weather(weather)
        
        # 車輛和感應器
        self.vehicle = None
        self.collision_sensor = None
        self.rgb_camera = None
        self.camera_callback = None
        self._collision_detected = False
        
        # 定義觀察空間 (車輛狀態)
        # [x座標, y座標, 航向角, 速度, 距離目標點, 與目標方向夾角]
        self.observation_space = spaces.Box(
            low=np.array([-1000, -1000, -math.pi, 0, 0, -math.pi]),
            high=np.array([1000, 1000, math.pi, 30, 1000, math.pi]),
            dtype=np.float32
        )
        
        # 定義動作空間 [轉向(-1~1), 油門/剎車(-1~1)]
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0]),
            high=np.array([1.0, 1.0]),
            dtype=np.float32
        )
        
        # 設定目標點
        self.target_location = None
        self.max_episode_steps = 1000
        self.current_step = 0
        
        # 記錄上一步的資訊用於計算reward
        self.prev_distance = None
        self.render_mode = render_mode
        
    def _setup_camera(self):
        """設置RGB攝影機感應器"""
        if not self.render_mode:
            return
            
        # 創建相機藍圖
        blueprint_library = self.world.get_blueprint_library()
        camera_bp = blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', '800')
        camera_bp.set_attribute('image_size_y', '600')
        camera_bp.set_attribute('fov', '90')
        
        # 設置相機位置（車輛後上方）
        camera_transform = carla.Transform(
            carla.Location(x=-5.5, z=2.8),
            carla.Rotation(pitch=-15)
        )
        
        # 創建相機並附加到車輛上
        self.camera_callback = CarlaCameraCallback()
        self.rgb_camera = self.world.spawn_actor(
            camera_bp,
            camera_transform,
            attach_to=self.vehicle
        )
        self.rgb_camera.listen(self.camera_callback)
        
    def reset(self, *, seed=None, options=None):
        """
        重置環境到初始狀態。
        
        參數:
            seed: 隨機數種子
            options: 額外的配置選項
            
        返回:
            tuple: (observation, info) 初始狀態和額外信息
        """
        super().reset(seed=seed)
        # 清理現有的車輛和感應器
        self._cleanup_actors()
        
        # 在Town04生成賓士車輛
        blueprint_library = self.world.get_blueprint_library()
        vehicle_bp = blueprint_library.filter('vehicle.mercedes.coupe')[0]
        
        # 隨機選擇生成點並嘗試生成車輛
        spawn_points = self.map.get_spawn_points()
        random.shuffle(spawn_points)  # 隨機打亂生成點順序
        
        # 嘗試在不同的生成點生成車輛
        for spawn_point in spawn_points:
            try:
                self.vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)
                break
            except RuntimeError:
                continue
                
        if self.vehicle is None:
            raise RuntimeError("無法找到可用的車輛生成點")
        
        # 設置碰撞感應器
        collision_bp = blueprint_library.find('sensor.other.collision')
        self.collision_sensor = self.world.spawn_actor(
            collision_bp,
            carla.Transform(),
            attach_to=self.vehicle
        )
        self.collision_sensor.listen(lambda event: self._on_collision(event))
        self._collision_detected = False
        
        # 設置RGB相機
        if self.render_mode:
            self._setup_camera()
        
        # 隨機選擇目標點
        self.target_location = random.choice(spawn_points).location
        self.current_step = 0
        self.prev_distance = self._get_distance_to_target()
        
        initial_obs = self._get_observation()
        info = {
            'distance_to_target': initial_obs[4],
            'collision_detected': False,
            'steps': 0
        }
        return initial_obs, info
    
    def render(self):
        """更新pygame視窗顯示"""
        if not self.render_mode:
            return
            
        # 獲取最新的相機圖像
        image = self.camera_callback.get_image()
        if image is not None:
            # 將numpy陣列轉換為pygame表面
            surface = pygame.surfarray.make_surface(image.swapaxes(0, 1))
            self.display.blit(surface, (0, 0))
            pygame.display.flip()
            
        # 處理pygame事件
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                
    def _on_collision(self, event):
        """碰撞事件處理器"""
        self._collision_detected = True
    
    def _get_distance_to_target(self):
        """計算車輛到目標點的距離"""
        return self.vehicle.get_location().distance(self.target_location)
    
    def _get_observation(self):
        """
        獲取當前環境狀態。
        
        返回:
            observation: 包含位置、方向、速度等資訊的陣列
        """
        location = self.vehicle.get_location()
        velocity = self.vehicle.get_velocity()
        transform = self.vehicle.get_transform()
        
        # 計算車輛到目標的方向角
        target_vector = self.target_location - location
        target_angle = math.atan2(target_vector.y, target_vector.x)
        heading_angle = math.radians(transform.rotation.yaw)
        angle_diff = target_angle - heading_angle
        
        # 標準化角度到[-pi, pi]
        angle_diff = (angle_diff + math.pi) % (2 * math.pi) - math.pi
        
        speed = math.sqrt(velocity.x**2 + velocity.y**2)
        distance = self._get_distance_to_target()
        
        return np.array([
            location.x,
            location.y,
            heading_angle,
            speed,
            distance,
            angle_diff
        ], dtype=np.float32)
    
    def step(self, action):
        """
        執行一步動作。
        
        參數:
            action: [轉向, 油門/剎車] 的numpy陣列
            
        返回:
            (observation, reward, terminated, truncated, info): 執行動作後的結果
        """
        self.current_step += 1
        
        # 控制車輛
        steer = float(action[0])
        throttle = float(max(0, action[1]))
        brake = float(max(0, -action[1]))
        
        self.vehicle.apply_control(carla.VehicleControl(
            throttle=throttle,
            steer=steer,
            brake=brake,
            hand_brake=False,
            reverse=False,
            manual_gear_shift=False
        ))
        
        # 等待一個時間步並更新顯示
        self.world.tick()
        if self.render_mode:
            self.render()
        
        # 獲取新的觀察和獎勵
        observation = self._get_observation()
        reward = self._compute_reward()
        
        # 判斷是否結束
        terminated = self._is_done()
        
        info = {
            'distance_to_target': observation[4],
            'collision_detected': self._collision_detected,
            'steps': self.current_step
        }
        
        return observation, reward, terminated, False, info
    
    def _compute_reward(self):
        """
        計算獎勵。
        
        獎勵設計說明：
        1. 距離獎勵：根據車輛是否在接近目標
        2. 角度獎勵：鼓勵車輛朝向目標
        3. 速度獎勵：鼓勵開始移動和保持適當速度
        4. 碰撞懲罰：發生碰撞時給予巨大懲罰
        5. 完成獎勵：到達目標時給予額外獎勵
        6. 探索獎勵：鼓勵車輛探索環境
        """
        observation = self._get_observation()
        current_distance = observation[4]
        speed = observation[3]
        angle_diff = observation[5]
        
        # 距離變化獎勵 (-2 到 2)，增加獎勵幅度
        distance_diff = self.prev_distance - current_distance
        distance_reward = np.clip(distance_diff, -2, 2)
        self.prev_distance = current_distance
        
        # 方向獎勵 (最大1.5，增加權重)
        angle_reward = 1.5 * (math.pi - abs(angle_diff)) / math.pi
        
        # 速度獎勵 (鼓勵開始移動和保持適當速度)
        min_speed = 30.0  # 最小期望速度
        optimal_speed = 90.0  # 最佳速度
        
        # 如果速度小於最小期望速度，給予負獎勵
        if speed < min_speed:
            speed_reward = -1.0 + (speed / min_speed)  # 線性增加到0
        else:
            # 速度在理想範圍內時給予正獎勵
            speed_reward = 1.0 * (1 - min(abs(speed - optimal_speed) / optimal_speed, 1))
        
        # 探索獎勵：只要有移動就給予小獎勵
        exploration_reward = 0.1 if speed > 0.5 else -0.1
        
        # 碰撞懲罰（保持不變）
        collision_penalty = -50.0 if self._collision_detected else 0.0
        
        # 到達目標獎勵（增加獎勵）
        goal_reward = 100.0 if current_distance < 5.0 else 0.0
        
        # 總獎勵（調整權重）
        reward = (
            distance_reward * 3.0 +     # 增加距離獎勵權重
            angle_reward * 2.0 +        # 增加方向獎勵權重
            speed_reward * 2.0 +        # 增加速度獎勵權重
            exploration_reward +        # 新增探索獎勵
            collision_penalty +         # 碰撞懲罰
            goal_reward                # 完成獎勵
        )
        
        return reward
    
    def _is_done(self):
        """
        判斷回合是否結束。
        
        結束條件：
        1. 到達目標
        2. 發生碰撞
        3. 超過最大步數
        4. 卡在原地不動
        """
        observation = self._get_observation()
        distance = observation[4]
        speed = observation[3]
        
        # 檢查是否到達目標
        if distance < 5.0:
            return True
            
        # 檢查是否發生碰撞
        if self._collision_detected:
            return True
            
        # 檢查是否超過最大步數
        if self.current_step >= self.max_episode_steps:
            return True
            
        # 檢查是否卡住（連續過長時間速度很低）
        if self.current_step > 200 and speed < 0.5:  # 增加時間閾值和速度閾值
            return True
            
        return False
        
    def _cleanup_actors(self):
        """安全地清理所有actors"""
        try:
            if self.rgb_camera is not None:
                self.rgb_camera.destroy()
                self.rgb_camera = None
        except RuntimeError:
            pass
            
        try:
            if self.collision_sensor is not None:
                self.collision_sensor.destroy()
                self.collision_sensor = None
        except RuntimeError:
            pass
            
        try:
            if self.vehicle is not None:
                self.vehicle.destroy()
                self.vehicle = None
        except RuntimeError:
            pass
            
    def close(self):
        """清理環境資源"""
        if self.render_mode:
            pygame.quit()
        self._cleanup_actors()