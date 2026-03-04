import numpy as np
import heapq
import matplotlib.pyplot as plt
import os

class AStarPlanner:
    def __init__(self, resolution=0.05, safe_margin=1.88, debug=True):
        """
        resolution: 栅格分辨率 (米/格)
        safe_margin: 障碍物膨胀半径 (米)
        """
        self.res = resolution
        self.safe_margin = safe_margin
        self.debug = debug
        self.print_count = 0 
        
    def plan(self, start_pos, target_pos, obstacles):
        forced_pt = np.array([1.7, 5.2])
        
        # 1. 自适应确定地图边界
        all_x =[start_pos[0], target_pos[0], forced_pt[0]] + [obs['pos'][0] for obs in obstacles]
        all_y = [start_pos[1], target_pos[1], forced_pt[1]] + [obs['pos'][1] for obs in obstacles] 
        
        min_x, max_x = min(all_x) - 3.0, max(all_x) + 3.0
        
        center_y = (start_pos[1] + target_pos[1]) / 2.0
        min_y = min(min(all_y) - 3.0, center_y - 8.0) 
        max_y = max(max(all_y) + 3.0, center_y + 8.0)
        
        width = int((max_x - min_x) / self.res)
        height = int((max_y - min_y) / self.res)
        
        # 2. 构建栅格地图
        grid = np.zeros((width, height), dtype=np.int8)
        
        for obs in obstacles:
            ox, oy = obs['pos']
            r = obs['radius'] + self.safe_margin 
            
            min_ix = max(0, int((ox - r - min_x) / self.res))
            max_ix = min(width, int((ox + r - min_x) / self.res) + 1)
            min_iy = max(0, int((oy - r - min_y) / self.res))
            max_iy = min(height, int((oy + r - min_y) / self.res) + 1)
            
            for ix in range(min_ix, max_ix):
                for iy in range(min_iy, max_iy):
                    gx = min_x + ix * self.res
                    gy = min_y + iy * self.res
                    if np.linalg.norm([gx - ox, gy - oy]) <= r:
                        grid[ix, iy] = 1
                        
        # 3. 运行 2D A* 搜索
        def get_grid_idx(pos_x, pos_y):
            ix = int((pos_x - min_x) / self.res)
            iy = int((pos_y - min_y) / self.res)
            return (np.clip(ix, 0, width-1), np.clip(iy, 0, height-1))
            
        start_idx = get_grid_idx(start_pos[0], start_pos[1])
        mid_idx   = get_grid_idx(forced_pt[0], forced_pt[1])
        goal_idx  = get_grid_idx(target_pos[0], target_pos[1])

        grid[mid_idx[0], mid_idx[1]] = 0 
        
        path1_idx = self._astar_search(grid, start_idx, mid_idx, min_y)
        path2_idx = self._astar_search(grid, mid_idx, goal_idx, min_y)
        
        if not path1_idx or not path2_idx:
            if self.print_count < 1:
                print("🚨[A* Planner] 警告: 未找到路径！返回直线！")
            return [start_pos, target_pos]
            
        path_idx = path1_idx[:-1] + path2_idx
            
        # 4. 获取 2D 原始路径点 (这是密集的栅格点，台阶状)
        raw_waypoints_2d =[]
        for (ix, iy) in path_idx:
            wx = min_x + ix * self.res
            wy = min_y + iy * self.res
            raw_waypoints_2d.append(np.array([wx, wy]))
            
        # =======================================================
        # 5. [核心优化] 执行极度丝滑的曲线平滑处理
        # =======================================================
        smoothed_2d = self._smooth_path(raw_waypoints_2d)
        smoothed_2d[-1] = target_pos[:2].copy() 

        # 终端可视化 Debug (放在平滑之后，你可以欣赏完美的曲线)
        if self.debug and self.print_count < 1:
            self._print_debug_map(grid, path_idx, smoothed_2d, min_x, min_y, mid_idx=mid_idx)
            self.print_count += 1 
        
        # 6. Z轴基于累计距离的线性插值
        start_z = start_pos[2]
        target_z = target_pos[2]
        
        cumulative_dists = [0.0]
        for i in range(1, len(smoothed_2d)):
            dist = np.linalg.norm(smoothed_2d[i] - smoothed_2d[i-1])
            cumulative_dists.append(cumulative_dists[-1] + dist)
            
        total_dist = cumulative_dists[-1]
        
        final_3d_waypoints =[]
        for i, pt_2d in enumerate(smoothed_2d):
            if total_dist == 0:
                z = target_z
            else:
                progress_ratio = cumulative_dists[i] / total_dist
                z = start_z + progress_ratio * (target_z - start_z)
            final_3d_waypoints.append(np.array([pt_2d[0], pt_2d[1], z]))
            
        final_3d_waypoints[-1][2] = target_z
        return final_3d_waypoints

    def _astar_search(self, grid, start, goal, min_y):
        motions =[(-1,0), (1,0), (0,-1), (0,1), (-1,-1), (-1,1), (1,-1), (1,1)]
        open_set =[]
        heapq.heappush(open_set, (0, start))
        came_from = {}
        g_score = {start: 0}
        
        while open_set:
            _, current = heapq.heappop(open_set)
            if current == goal:
                path =[]
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                return path[::-1]
                
            for dx, dy in motions:
                neighbor = (current[0] + dx, current[1] + dy)
                if 0 <= neighbor[0] < grid.shape[0] and 0 <= neighbor[1] < grid.shape[1]:
                    if grid[neighbor[0], neighbor[1]] == 1:
                        continue 
                    
                    cost = 1.414 if dx != 0 and dy != 0 else 1.0
                    
                    real_y = min_y + neighbor[1] * self.res
                    centerline_penalty = 0.05 * abs(real_y) 
                    
                    tentative_g = g_score[current] + cost + centerline_penalty
                    
                    if neighbor not in g_score or tentative_g < g_score[neighbor]:
                        came_from[neighbor] = current
                        g_score[neighbor] = tentative_g
                        h = np.linalg.norm([neighbor[0]-goal[0], neighbor[1]-goal[1]])
                        heapq.heappush(open_set, (tentative_g + h, neighbor))
        return None

    def _smooth_path(self, path):
        """
        [全新重写] 双重核滑动平均曲线平滑 (Dual Moving Average Smoothing)
        能将尖锐的直角和台阶状栅格点，变成极其顺滑的近似 Bezier 弧线！
        """
        if len(path) < 10: 
            return [p.copy() for p in path]
            
        path_arr = np.array(path)
        x = path_arr[:, 0]
        y = path_arr[:, 1]
        
        # 窗口大小决定了转弯半径，30个点(1.5米)能切出非常完美的弯道弧度
        window = min(30, len(path_arr) // 3)
        if window < 3:
            return[p.copy() for p in path]
            
        # 1. Padding 延长：在首尾复制端点，防止平滑后路径两端往回缩短
        x_pad = np.pad(x, (window, window), mode='edge')
        y_pad = np.pad(y, (window, window), mode='edge')
        
        kernel = np.ones(window) / window
        
        # 2. 第一重平滑：滤除栅格带来的锯齿台阶效应
        x_smooth = np.convolve(x_pad, kernel, mode='same')
        y_smooth = np.convolve(y_pad, kernel, mode='same')
        
        # 3. 第二重平滑：把直角切成高阶连续的优雅平滑弧线 (类 B-Spline)
        x_smooth = np.convolve(x_smooth, kernel, mode='same')
        y_smooth = np.convolve(y_smooth, kernel, mode='same')
        
        # 4. 掐头去尾，剥离刚刚 padding 加上的辅助点
        x_final = x_smooth[window:-window]
        y_final = y_smooth[window:-window]
        
        smoothed_points = np.vstack((x_final, y_final)).T
        
        # 5. 均匀重采样：每隔约 0.2 米提取一个点，降低数据量并保持间距均匀
        final_path = [path_arr[0]] 
        for pt in smoothed_points:
            if np.linalg.norm(pt - final_path[-1]) > 0.2:
                final_path.append(pt)
                
        # 确保强力接驳终点
        if np.linalg.norm(final_path[-1] - path_arr[-1]) > 0.05:
            final_path.append(path_arr[-1])
            
        return final_path

    def _print_debug_map(self, grid, raw_path_idx, smoothed_2d, min_x, min_y, mid_idx=None):
        print("\n" + "="*50)
        print("📸 正在生成 A* 规划高清调试图 (已开启极度平滑曲线)...")
        
        plt.figure(figsize=(10, 10))
        plt.imshow(grid.T, cmap='Blues', origin='lower', alpha=0.6)
        
        # 绘制原始栅格粗糙路径 (浅色细线，用于对比)
        if raw_path_idx:
            raw_x = [p[0] for p in raw_path_idx]
            raw_y = [p[1] for p in raw_path_idx]
            plt.plot(raw_x, raw_y, color='pink', linewidth=1.5, linestyle='--', label='Raw Grid Path')
            
            # 绘制极度平滑后的弧线！需要将坐标转换回栅格索引系用于显示
            smooth_ix = [(p[0] - min_x) / self.res for p in smoothed_2d]
            smooth_iy = [(p[1] - min_y) / self.res for p in smoothed_2d]
            plt.plot(smooth_ix, smooth_iy, color='red', linewidth=3.5, label='Smoothed Curved Path')
            
            plt.scatter(smooth_ix[0], smooth_iy[0], color='green', s=150, zorder=5, label='Start')
            plt.scatter(smooth_ix[-1], smooth_iy[-1], color='purple', s=200, marker='*', zorder=5, label='Target')
            
            if mid_idx is not None:
                plt.scatter(mid_idx[0], mid_idx[1], color='orange', s=150, marker='D', zorder=6, label='Forced Waypoint')
            
        plt.title(f"A* Planner Smooth Trajectory\n(Res: {self.res}m, Safe Margin: {self.safe_margin}m)")
        plt.xlabel("X (Grid Index)")
        plt.ylabel("Y (Grid Index)")
        plt.legend(loc='upper right')
        plt.grid(True, linestyle=':', alpha=0.5)
        
        save_path = "astar_debug_map.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close() 
        
        print(f"✅ 平滑轨迹图已保存至: {os.path.abspath(save_path)}")
        print("="*50 + "\n")