import numpy as np
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
        
        self.stack = deque([(0, 0, 0, [])])
        self.visited = set()
        self.visited.add((0, 0))
        
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
        plt.pause(0.1)
        
        if self.state_index == len(self.states) - 1:
            self.btn_next.set_active(False)
        else:
            self.btn_next.set_active(True)
        
        if self.state_index == 0:
            self.btn_prev.set_active(False)
        else:
            self.btn_prev.set_active(True)
    
    def solve_maze_dfs(self):
        shortest_path = None 
        while self.stack:
            x, y, moves, path = self.stack.pop()  
            if x == self.n - 1 and y == self.m - 1:
                path += [(x, y)]
                if shortest_path is None or len(path) < len(shortest_path[0]):
                    shortest_path = [(path, self.visited.copy())]
                    self.states.append((path, self.visited.copy()))  
                    self.visualize_maze_with_path_and_explored(path, self.visited)
                continue 

            for dx, dy in self.directions:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.n and 0 <= ny < self.m and (self.maze[nx][ny] == 0 or self.maze[nx][ny] == 2) and (nx, ny) not in self.visited:
                    new_path = path + [(nx, ny)]
                    if self.maze[nx][ny] == 2:
                        self.stack.append((nx, ny, moves + 2, new_path))
                    else:
                        self.stack.append((nx, ny, moves + 1, new_path))
                    self.visited.add((nx, ny))
                    self.states.append((new_path, self.visited.copy()))
                    self.visualize_maze_with_path_and_explored(new_path, self.visited)

        self.visualize_maze_with_path_and_explored(shortest_path[0][0], shortest_path[0][1])
        plt.show()



n, m = map(int, input().split())
maze = []
for _ in range(n):
    row = list(map(int, input().split()))
    maze.append(row)

solver = MazeSolver(maze)
solver.solve_maze_dfs()
