import numpy as np
import heapq
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from collections import deque

class MazeSolver:
    def __init__(self, maze):
        self.maze = maze
        self.fig = plt.figure(figsize=(len(maze[0]), len(maze)))
        self.ax = self.fig.add_subplot(111)
        self.ax.set_xticks(range(len(maze[0])))
        self.ax.set_yticks(range(len(maze)))
        self.ax.set_xticks([x - 0.5 for x in range(1, len(maze[0]))], minor=True)
        self.ax.set_yticks([y - 0.5 for y in range(1, len(maze))], minor=True)
        self.ax.grid(which="minor", color="black", linestyle='-', linewidth=2)
        self.ax.axis('on')
        
        self.directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        self.n = len(maze)
        self.m = len(maze[0])
        
        self.queue = []
        self.visited = set()
        
        self.states = [] 
        self.state_index = -1
        
        self.btn_next = Button(plt.axes([0.7, 0.025, 0.1, 0.04]), '->')
        self.btn_next.on_clicked(self.next_state)
        self.btn_next.set_active(False)
        
        self.btn_prev = Button(plt.axes([0.59, 0.025, 0.1, 0.04]), '<-')
        self.btn_prev.on_clicked(self.prev_state)
        self.btn_prev.set_active(False)
        
    def next_state(self, event=None):
        self.state_index = min(self.state_index + 1, len(self.states) - 1)
        self.update_state()
        
    def prev_state(self, event=None):
        if self.state_index == -1:
            self.state_index = len(self.states) - 1
        self.state_index = max(self.state_index - 1, 0)
        self.update_state()

        
    def update_state(self):
        if self.states:
            path, explored = self.states[self.state_index]
            self.visualize_maze_with_path_and_explored(path, explored)

        
    def visualize_maze_with_path_and_explored(self, path, explored):
        self.ax.clear()
        self.ax.imshow(self.maze, cmap='Greys', interpolation='nearest')
        
        explored_matrix = np.zeros_like(self.maze)
        for node in explored:
            explored_matrix[node[0]][node[1]] = 1
        self.ax.imshow(explored_matrix, cmap='YlOrBr', alpha=0.5, interpolation='nearest')

        path_matrix = np.zeros_like(self.maze)
        for i, node in enumerate(path):
            path_matrix[node[0]][node[1]] = 1 if i < len(path) - 1 else 2
        self.ax.imshow(path_matrix, cmap='Reds', alpha=0.5, interpolation='nearest')
        
        wall1 = np.ma.masked_where(self.maze != 1, self.maze)
        wall2 = np.ma.masked_where(self.maze != 2, self.maze)
        self.ax.imshow(wall1, cmap='Blues', alpha=0.5, interpolation='nearest')
        self.ax.imshow(wall2, cmap='Greens', alpha=0.5, interpolation='nearest')
        
        self.ax.set_xticks(range(len(self.maze[0])))
        self.ax.set_yticks(range(len(self.maze)))
        self.ax.set_xticks([x - 0.5 for x in range(1, len(self.maze[0]))], minor=True)
        self.ax.set_yticks([y - 0.5 for y in range(1, len(self.maze))], minor=True)
        self.ax.grid(which="minor", color="black", linestyle='-', linewidth=2)
        
        self.ax.axis('on')
        plt.pause(0.01)

        if self.state_index == len(self.states) - 1:
            self.btn_next.set_active(False)
        else:
            self.btn_next.set_active(True)
        
        if self.state_index == 0:
            self.btn_prev.set_active(False)
        else:
            self.btn_prev.set_active(True)

    def solve_maze_astar(self):
        start = (0, 0)
        goal = (self.n - 1, self.m - 1)
        
        heapq.heappush(self.queue, (0 + self.heuristic(start, goal), 0, start, []))  # (f_score, g_score, node, path)
        
        while self.queue:
            f_score, g_score, node, path = heapq.heappop(self.queue)
            
            if node == goal:
                path += [node]
                self.states.append((path, self.visited.copy()))
                self.visualize_maze_with_path_and_explored(path, self.visited)
                break
            
            for dx, dy in self.directions:
                nx, ny = node[0] + dx, node[1] + dy
                
                if 0 <= nx < self.n and 0 <= ny < self.m and self.maze[nx][ny] in [0, 2]:
                    new_g_score = g_score + (2 if self.maze[nx][ny] == 2 else 1)
                    
                    if (nx, ny) not in self.visited or new_g_score < g_score:
                        self.visited.add((nx, ny))
                        new_path = path + [node]
                        heapq.heappush(self.queue, (new_g_score + self.heuristic((nx, ny), goal), new_g_score, (nx, ny), new_path))
                        self.states.append((new_path, self.visited.copy()))
                        self.visualize_maze_with_path_and_explored(new_path, self.visited)

        plt.show()

    
    def heuristic(self, node, goal):
        return abs(node[0] - goal[0]) + abs(node[1] - goal[1])

n, m = map(int, input().split())
maze = []
for _ in range(n):
    row = list(map(int, input().split()))
    maze.append(row)

solver = MazeSolver(maze)
solver.solve_maze_astar()
