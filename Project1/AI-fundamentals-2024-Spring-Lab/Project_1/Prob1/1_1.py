from collections import deque

def bfs(graph, start, end):
    n = len(graph)
    visited = [False] * n
    visited[start - 1] = True
    distance = [0] * n
    queue = deque([start])

    while queue:
        u = queue.popleft()
        if u == end:
            return distance[u - 1]
        for v in graph[u]:
            if not visited[v - 1]:
                visited[v - 1] = True
                distance[v - 1] = distance[u - 1] + 1
                queue.append(v)

    return -1

n, m = map(int, input().split())
graph = {i: [] for i in range(1, n + 1)}

for _ in range(m):
    u, v = map(int, input().split())
    graph[u].append(v)

print(bfs(graph, 1, n))
