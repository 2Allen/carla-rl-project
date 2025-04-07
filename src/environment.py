"""CARLA強化學習環境。

此模組實現了一個基於CARLA的強化學習環境，使用Town04地圖和賓士車輛。
環境設計符合OpenAI Gym介面標準，便於與Stable-Baselines3整合。
"""

import datetime
import math
import os
import queue
import random
import sys
import weakref
import time

import cv2
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame

import carla

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
            render_mode: 是否啟用視覺化顯示（True時顯示界面不錄影，False時錄影不顯示）
        """
        super().__init__()
        
        # 檢查是否為測試模式（透過查看調用堆疊）
        try:
            caller_filename = sys._getframe().f_back.f_back.f_code.co_filename
            is_testing = 'test.py' in os.path.basename(caller_filename)
        except (AttributeError, ValueError):
            is_testing = False  # 如果無法確定，預設為非測試模式
        
        # 只在測試模式且未啟用渲染時才錄影
        self.record_path = './videos' if (is_testing and not render_mode) else None
        self.video_writer = None
        self.current_episode = 0
        self.camera_setup_done = False  # 追踪相機是否已設置
        print(f"初始化環境 - render_mode: {render_mode}, record_path: {self.record_path}, is_testing: {is_testing}")

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
        
        # 定義觀察空間
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
        # 檢查是否需要相機（渲染模式或測試模式的錄影）
        need_camera = self.render_mode or self.record_path
        if not need_camera:
            return
            
        # 避免重複設置
        if self.camera_setup_done and self.rgb_camera is not None:
            return
            
        try:
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
            self.camera_setup_done = True
            print(f"相機設置完成 - render_mode: {self.render_mode}")
        except Exception as e:
            print(f"相機設置失敗: {str(e)}")
            self.camera_setup_done = False
            
    def _setup_video_writer(self):
        """設置視訊寫入器"""
        if not self.record_path:
            return
            
        if not os.path.exists(self.record_path):
            os.makedirs(self.record_path)
            
        # 生成影片檔案名稱（使用時間戳）
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        video_filename = os.path.join(self.record_path, f'episode_{self.current_episode}_{timestamp}.mp4')
        
        # 創建視訊寫入器
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv2.VideoWriter(video_filename, fourcc, 20.0, (800, 600))
        
    def _close_video_writer(self):
        """關閉視訊寫入器"""
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None
            
    def reset(self, *, seed=None, options=None):
        """
        重置環境到初始狀態。

        參數:
            seed: 隨機數種子
            options: 額外的配置選項

        返回:
            tuple: (observation, info) 初始狀態和額外信息
        """
        # 暫存相機設置狀態
        was_camera_setup = self.camera_setup_done

        # 關閉上一個回合的視訊寫入器
        self._close_video_writer()
        self.current_episode += 1

        # 在同步模式下，確保上一步驟完成
        if self.render_mode:
            self.world.tick()

        super().reset(seed=seed)
        # 清理現有的車輛和感應器
        self._cleanup_actors()
        # 在同步模式下，等待清理完成
        if self.render_mode:
            self.world.tick()
        # 清理完成後立即重置碰撞標誌
        self._collision_detected = False

        # 在Town04生成賓士車輛
        blueprint_library = self.world.get_blueprint_library()
        vehicle_bp = blueprint_library.filter('vehicle.mercedes.coupe')[0]

        # 隨機選擇生成點並嘗試生成車輛
        spawn_points = self.map.get_spawn_points()
        random.shuffle(spawn_points)  # 隨機打亂生成點順序

        # 嘗試在不同的生成點生成車輛
        self.vehicle = None # 確保先設為 None
        for spawn_point in spawn_points:
            try:
                self.vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)
                # 在同步模式下，等待生成完成
                if self.render_mode:
                    self.world.tick()
                break
            except RuntimeError:
                continue # 確保繼續嘗試下一個點

        if self.vehicle is None:
            raise RuntimeError("無法找到可用的車輛生成點")

        # 設置碰撞感應器
        collision_bp = blueprint_library.find('sensor.other.collision')
        self.collision_sensor = self.world.spawn_actor(
            collision_bp,
            carla.Transform(),
            attach_to=self.vehicle
        )
        # 使用 weakref 避免循環引用
        self.collision_sensor.listen(lambda event: self._on_collision(weakref.ref(self), event))
        # 在同步模式下，等待感應器設置完成
        if self.render_mode:
            self.world.tick()

        # 只在需要時重新設置相機（避免重複打印日誌）
        # 恢復相機設置狀態，避免重複創建
        self.camera_setup_done = was_camera_setup
        if (self.render_mode or self.record_path) and not self.camera_setup_done:
            self._setup_camera()
            # 在同步模式下，等待相機設置完成
            if self.render_mode and self.rgb_camera is not None:
                 self.world.tick()

        # 隨機選擇目標點
        self.target_location = random.choice(spawn_points).location
        self.current_step = 0
        self.prev_distance = self._get_distance_to_target()

        initial_obs = self._get_observation()
        info = {
            'distance_to_target': initial_obs[4],
            'collision_detected': self._collision_detected, # 使用已在清理後重置的值
            'steps': 0
        }
        # 在同步模式下，確保所有設置完成後再返回
        if self.render_mode:
            self.world.tick()
        return initial_obs, info

    def render(self):
        """更新pygame視窗顯示和錄影"""
        # 獲取最新的相機圖像
        if self.camera_callback is None:
            print("警告：相機回調未初始化")
            return
            
        image = self.camera_callback.get_image()
        if image is None:
            print("警告：未獲取到相機圖像")
            return
            
        # 非渲染模式時處理錄影
        if self.record_path and not self.render_mode:
            try:
                # 初始化視訊寫入器（如果需要）
                if self.video_writer is None:
                    self._setup_video_writer()
                    
                # 寫入影片幀
                if self.video_writer is not None:
                    frame = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                    self.video_writer.write(frame)
            except Exception as e:
                print(f"錄影出錯：{str(e)}")
                
        # 如果是渲染模式，更新顯示和處理事件
        if self.render_mode:
            try:
                # 更新顯示
                surface = pygame.surfarray.make_surface(image.swapaxes(0, 1))
                self.display.blit(surface, (0, 0))
                pygame.display.flip()
                
                # 處理pygame事件
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self.close()
            except pygame.error as e:
                print(f"Pygame 錯誤：{str(e)}")
            except Exception as e:
                print(f"渲染出錯：{str(e)}")
                
    def _on_collision(self, weak_self, event):
        """碰撞事件處理器"""
        self = weak_self()
        if not self:
            return
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
        
        # 等待一個時間步並更新
        self.world.tick()
        
        # 只在需要時渲染或錄影
        if self.render_mode or self.record_path:
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
        """計算當前狀態下的獎勵值"""
        observation = self._get_observation()
        current_distance = observation[4]
        speed = observation[3]
        angle_diff = observation[5]
        
        # 獲取當前車道信息
        vehicle_location = self.vehicle.get_location()
        waypoint = self.map.get_waypoint(vehicle_location)
        
        # 車道中心距離獎勵 (-2 到 2)
        if waypoint is not None:
            lane_distance = abs(waypoint.transform.location.distance(vehicle_location))
            lane_width = waypoint.lane_width
            # print(f"車道中心距離: {lane_distance}, 車道寬度: {lane_width}")
            normalized_lane_distance = lane_distance / (lane_width / 2)  # 標準化到 [0, 2]範圍
            lane_reward = 2.0 * (1 - min(normalized_lane_distance, 1))  # 距離越近獎勵越高
        else:
            lane_reward = -2.0  # 完全偏離道路給予負獎勵
        
        # 速度獎勵：鼓勵保持穩定前進速度
        target_speed = 8.0  # 降低目標速度為 8 m/s（約 29 km/h）
        min_speed = 3.0    # 最低期望速度
        
        if speed < min_speed:
            # 當速度太低時給予更大的負獎勵
            speed_reward = -2.0 + (speed / min_speed)
        else:
            # 使用高斯函數計算速度獎勵，使獎勵曲線更平滑
            speed_diff = abs(speed - target_speed)
            speed_reward = math.exp(-(speed_diff * speed_diff) / 8.0)
        
        # 方向獎勵：優化轉向行為
        normalized_angle = abs(angle_diff) / math.pi  # 標準化到 [0,1] 範圍
        # 使用餘弦函數使獎勵在角度接近目標時更敏感
        angle_reward = math.cos(normalized_angle * math.pi/2)
        
        # 當速度過低時降低角度獎勵，避免原地轉圈
        speed_factor = min(1.0, speed / 5.0)  # 速度達到5m/s才有完整轉向獎勵
        angle_reward *= speed_factor
        
        # 計算進展獎勵
        progress_reward = 0.0
        if self.prev_distance is not None:
            progress = self.prev_distance - current_distance
            progress_reward = progress * 2.0  # 根據距離目標的進展給予獎勵
        self.prev_distance = current_distance

        # 碰撞懲罰
        collision_penalty = -50.0 if self._collision_detected else 0.0

        # 到達目標獎勵
        goal_reward = 100.0 if current_distance < 5.0 else 0.0

        # 計算最終獎勵（調整權重）
        reward = (
            lane_reward * 1.5 +       # 適度降低車道中心獎勵
            speed_reward * 3.0 +      # 增加速度獎勵權重
            angle_reward * 2.0 +      # 增加方向獎勵權重
            progress_reward * 2.0 +    # 降低前進獎勵權重
            collision_penalty +       # 碰撞懲罰
            goal_reward              # 目標獎勵
        )

        return float(reward)

    def _is_done(self):
        """判斷回合是否結束。"""
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
            
        # 移除卡在原地不動的終止條件
        # if speed < 0.1 and self.current_step > 100:
        #     return True
            
        # 檢查是否偏離車道超過一半
        vehicle_location = self.vehicle.get_location()
        waypoint = self.map.get_waypoint(vehicle_location)
        if waypoint is not None:
            lane_distance = waypoint.transform.location.distance(vehicle_location)
            if lane_distance > waypoint.lane_width:
                return True
                
        return False

    def _cleanup_actors(self):
        """清理所有創建的演員（車輛和感應器），使用批量操作提高效率和可靠性"""
        actors_to_destroy = []
        if self.collision_sensor is not None:
            if self.collision_sensor.is_listening:
                self.collision_sensor.stop() # 停止監聽
            actors_to_destroy.append(self.collision_sensor)
            self.collision_sensor = None # 清除引用
        if self.rgb_camera is not None:
            if self.rgb_camera.is_listening:
                self.rgb_camera.stop() # 停止監聽
            actors_to_destroy.append(self.rgb_camera)
            self.rgb_camera = None # 清除引用
        if self.vehicle is not None:
            actors_to_destroy.append(self.vehicle)
            self.vehicle = None # 清除引用

        # 批量銷毀
        if actors_to_destroy:
            # print(f"正在銷毀 {len(actors_to_destroy)} 個 actor...") # 避免過多打印
            # 過濾掉可能已經被銷毀的 actor
            valid_actors = [actor for actor in actors_to_destroy if actor is not None and actor.is_alive]
            if valid_actors:
                self.client.apply_batch_sync([carla.command.DestroyActor(actor) for actor in valid_actors])
                # print(f"成功銷毀 {len(valid_actors)} 個 actor。")
            # else:
                # print("沒有有效的 actor 需要銷毀。")

        # 重置相機設置狀態，以便下次 reset 時可以重新設置
        self.camera_setup_done = False

    def close(self):
        """關閉環境，清理資源"""
        self._cleanup_actors()
        self._close_video_writer()  # 確保關閉視訊寫入器
        if self.render_mode:
            pygame.quit()