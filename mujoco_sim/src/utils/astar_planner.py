import numpy as np
import heapq
import matplotlib.pyplot as plt
import os
import math  # [新增] 引入底层数学库，取代缓慢的 numpy 小数组运算

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
        
        # =======================================================
        # [核心优化 1] Numpy 广播机制取代纯 Python 双层循环
        # =======================================================
        for obs in obstacles:
            ox, oy = obs['pos']
            r = obs['radius'] + self.safe_margin 
            r_sq = r ** 2  # 比较平方根以避免开方运算
            
            min_ix = max(0, int((ox - r - min_x) / self.res))
            max_ix = min(width, int((ox + r - min_x) / self.res) + 1)
            min_iy = max(0, int((oy - r - min_y) / self.res))
            max_iy = min(height, int((oy + r - min_y) / self.res) + 1)
            
            if min_ix >= max_ix or min_iy >= max_iy:
                continue
                
            # 生成局部包围盒坐标轴，利用 Broadcasting 一次性算出所有点到圆心的平方距离
            ix_coords = np.arange(min_ix, max_ix)
            iy_coords = np.arange(min_iy, max_iy)
            gx = min_x + ix_coords * self.res
            gy = min_y + iy_coords * self.res
            
            # dist_sq 形状为 (max_ix - min_ix, max_iy - min_iy)
            dist_sq = (gx[:, None] - ox)**2 + (gy[None, :] - oy)**2
            
            # 使用布尔索引，一步完成局部栅格赋值
            grid[min_ix:max_ix, min_iy:max_iy][dist_sq <= r_sq] = 1
                        
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
            
        # 4. 获取 2D 原始路径点
        raw_waypoints_2d =[]
        for (ix, iy) in path_idx:
            wx = min_x + ix * self.res
            wy = min_y + iy * self.res
            raw_waypoints_2d.append(np.array([wx, wy]))
            
        # 5. 执行极度丝滑的曲线平滑处理
        smoothed_2d = self._smooth_path(raw_waypoints_2d)
        smoothed_2d[-1] = target_pos[:2].copy() 

        if self.debug and self.print_count < 1:
            self._print_debug_map(grid, path_idx, smoothed_2d, min_x, min_y, mid_idx=mid_idx)
            self.print_count += 1 
        
        # =======================================================
        # [核心优化 2] Z轴计算全盘 Numpy 向量化，消灭累加的 For 循环
        # =======================================================
        start_z = start_pos[2]
        target_z = target_pos[2]
        
        smoothed_arr = np.array(smoothed_2d)
        
        # np.diff 计算相邻两点向量，np.linalg.norm(..., axis=1) 计算所有段长度
        diffs = np.diff(smoothed_arr, axis=0)
        dists = np.linalg.norm(diffs, axis=1)
        
        # np.cumsum 直接算出每个点当前的累计里程
        cumulative_dists = np.zeros(len(smoothed_arr))
        cumulative_dists[1:] = np.cumsum(dists)
        total_dist = cumulative_dists[-1]
        
        if total_dist == 0:
            zs = np.full(len(smoothed_arr), target_z)
        else:
            progress_ratio = cumulative_dists / total_dist
            zs = start_z + progress_ratio * (target_z - start_z)
            
        # 组装回原本 list of numpy array 的数据格式
        final_3d_waypoints = [np.array([pt[0], pt[1], z]) for pt, z in zip(smoothed_arr, zs)]
        final_3d_waypoints[-1][2] = target_z
        
        return final_3d_waypoints

    def _astar_search(self, grid, start, goal, min_y):
        width, height = grid.shape
        goal_x, goal_y = goal
        
        # 提前把代价值封存进元组，免得循环里做 if-else 判定
        motions = [(-1,0, 1.0), (1,0, 1.0), (0,-1, 1.0), (0,1, 1.0), 
                   (-1,-1, 1.414), (-1,1, 1.414), (1,-1, 1.414), (1,1, 1.414)]
        
        open_set =[]
        heapq.heappush(open_set, (0, start))
        came_from = {}
        g_score = {start: 0}
        
        # =======================================================
        # [核心优化 3] 预计算中心惩罚表，并在内部使用 math.hypot 加速
        # =======================================================
        # 避免在几万次寻路计算中反复执行：0.05 * abs(min_y + y * res)
        y_penalties = 0.05 * np.abs(min_y + np.arange(height) * self.res)
        
        while open_set:
            _, current = heapq.heappop(open_set)
            if current == goal:
                path =[]
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                return path[::-1]
                
            cx, cy = current
            for dx, dy, cost in motions:
                nx, ny = cx + dx, cy + dy
                
                if 0 <= nx < width and 0 <= ny < height:
                    if grid[nx, ny] == 1:
                        continue 
                    
                    # O(1) 直接查表获取预处理过的惩罚值
                    tentative_g = g_score[current] + cost + y_penalties[ny]
                    
                    neighbor = (nx, ny)
                    if neighbor not in g_score or tentative_g < g_score[neighbor]:
                        came_from[neighbor] = current
                        g_score[neighbor] = tentative_g
                        
                        # 放弃慢吞吞的 np.linalg.norm 构建微型数组，直接调用底层 C 的 hypot
                        h = math.hypot(nx - goal_x, ny - goal_y)
                        heapq.heappush(open_set, (tentative_g + h, neighbor))
        return None

    def _smooth_path(self, path):
        if len(path) < 10: 
            return [p.copy() for p in path]
            
        path_arr = np.array(path)
        x = path_arr[:, 0]
        y = path_arr[:, 1]
        
        window = min(30, len(path_arr) // 3)
        if window < 3:
            return[p.copy() for p in path]
            
        x_pad = np.pad(x, (window, window), mode='edge')
        y_pad = np.pad(y, (window, window), mode='edge')
        
        kernel = np.ones(window) / window
        
        x_smooth = np.convolve(x_pad, kernel, mode='same')
        y_smooth = np.convolve(y_pad, kernel, mode='same')
        
        x_smooth = np.convolve(x_smooth, kernel, mode='same')
        y_smooth = np.convolve(y_smooth, kernel, mode='same')
        
        x_final = x_smooth[window:-window]
        y_final = y_smooth[window:-window]
        
        smoothed_points = np.vstack((x_final, y_final)).T
        
        final_path = [path_arr[0]] 
        for pt in smoothed_points:
            if np.linalg.norm(pt - final_path[-1]) > 0.2:
                final_path.append(pt)
                
        if np.linalg.norm(final_path[-1] - path_arr[-1]) > 0.05:
            final_path.append(path_arr[-1])
            
        return final_path

    def _print_debug_map(self, grid, raw_path_idx, smoothed_2d, min_x, min_y, mid_idx=None):
        print("\n" + "="*50)
        print("📸 正在生成 A* 规划高清调试图 (已开启极度平滑曲线)...")
        
        plt.figure(figsize=(10, 10))
        plt.imshow(grid.T, cmap='Blues', origin='lower', alpha=0.6)
        
        if raw_path_idx:
            raw_x = [p[0] for p in raw_path_idx]
            raw_y = [p[1] for p in raw_path_idx]
            plt.plot(raw_x, raw_y, color='pink', linewidth=1.5, linestyle='--', label='Raw Grid Path')
            
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