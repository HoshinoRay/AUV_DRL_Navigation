import numpy as np
import heapq

class AStarPlanner:
    def __init__(self, resolution=0.05, safe_margin=1.88, debug=True):
        """
        resolution: 栅格分辨率 (米/格)
        safe_margin: 障碍物膨胀半径 (米)
        """
        self.res = resolution
        self.safe_margin = safe_margin
        self.debug = debug
        self.print_count = 0  # [核心新增] 控制打印频率，防止刷屏
        
    def plan(self, start_pos, target_pos, obstacles):
        # 1. 自适应确定地图边界 
        all_x =[start_pos[0], target_pos[0]] + [obs['pos'][0] for obs in obstacles]
        all_y = [start_pos[1], target_pos[1]] + [obs['pos'][1] for obs in obstacles] # [修复] 提取所有Y坐标
        
        min_x, max_x = min(all_x) - 3.0, max(all_x) + 3.0
        
        # [修复] 动态推导Y轴，但维持至少宽度为 16 的走廊
        center_y = (start_pos[1] + target_pos[1]) / 2.0
        min_y = min(min(all_y) - 3.0, center_y - 8.0) 
        max_y = max(max(all_y) + 3.0, center_y + 8.0)
        
        width = int((max_x - min_x) / self.res)
        height = int((max_y - min_y) / self.res)
        
        # 2. 构建栅格地图 (0: 空闲, 1: 障碍物)
        grid = np.zeros((width, height), dtype=np.int8)
        
        for obs in obstacles:
            ox, oy = obs['pos']
            r = obs['radius'] + self.safe_margin # 包含安全膨胀半径
            
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
        start_idx = (int((start_pos[0] - min_x)/self.res), int((start_pos[1] - min_y)/self.res))
        goal_idx = (int((target_pos[0] - min_x)/self.res), int((target_pos[1] - min_y)/self.res))
        
        start_idx = (np.clip(start_idx[0], 0, width-1), np.clip(start_idx[1], 0, height-1))
        goal_idx = (np.clip(goal_idx[0], 0, width-1), np.clip(goal_idx[1], 0, height-1))

        path_idx = self._astar_search(grid, start_idx, goal_idx, min_y)
        
        if not path_idx:
            if self.print_count < 1:
                print("🚨[A* Planner] 警告: 未找到路径！可能 safe_margin 过大。返回直线！")
            return[start_pos, target_pos]
            
        # 4. 终端可视化 Debug (只打印1次)
        if self.debug and self.print_count < 1:
            self._print_debug_map(grid, path_idx)
            self.print_count += 1 
            
        # 5. 获取 2D 原始路径点
        raw_waypoints_2d =[]
        for (ix, iy) in path_idx:
            wx = min_x + ix * self.res
            wy = min_y + iy * self.res
            raw_waypoints_2d.append(np.array([wx, wy]))
            
        # 6. 路径抽稀/平滑
        smoothed_2d = self._smooth_path(raw_waypoints_2d)
        smoothed_2d[-1] = target_pos[:2].copy() 
        
        # 7. Z轴基于累计距离的线性插值 (极度丝滑的俯仰角)
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
                    
                    # 中心轴引力：轻微惩罚偏离 Y=0 轴的行为，鼓励在中间钻缝
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
        if len(path) < 3: return path
        smoothed = [path[0]]
        for i in range(1, len(path)-1):
            v1 = path[i] - smoothed[-1]
            v2 = path[i+1] - path[i]
            if np.linalg.norm(v1) > 0 and np.linalg.norm(v2) > 0:
                cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                if cos_theta < 0.95: 
                    smoothed.append(path[i])
        smoothed.append(path[-1])
        return smoothed

    def _print_debug_map(self, grid, path_idx):
        print("\n" + "="*50)
        print("🗺️  A* 路径规划地图预览 (仅打印1次)")
        print("🟩=安全  🟥=障碍物(含膨胀)  🟦=规划路线")
        print("="*50)
        path_set = set(path_idx)
        for y in range(grid.shape[1]-1, -1, -1):
            row_str = ""
            for x in range(grid.shape[0]):
                if (x, y) in path_set: row_str += "🟦"
                elif grid[x, y] == 1: row_str += "🟥"
                else: row_str += "🟩"
            print(row_str)
        print("="*50 + "\n")