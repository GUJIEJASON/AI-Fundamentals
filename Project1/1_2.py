def dijkstra(graph, start, end):
    n = len(graph)
    distance = [float('inf')] * (n + 1)
    distance[start] = 0
    visited = set()
    while len(visited) < n:
        min_dist = float('inf')
        u = -1
        for v in range(1, n + 1):
            if v not in visited and distance[v] < min_dist:
                min_dist = distance[v]
                u = v
        visited.add(u)
        for v in range(1, n + 1):
            if v not in visited and graph[u][v] != 0:
                distance[v] = min(distance[v], distance[u] + graph[u][v])
    return distance[end] if distance[end] != float('inf') else -1

n, m = map(int, input().split())
# 初始化图
graph = [[0] * (n+1) for _ in range(1, n+1)]

for _ in range(m):
    x, y, z = map(int, input().split())
    # 考虑到重边的情况，只保存权重最小的边
    if graph[x][y] == 0 or z < graph[x][y]:
        graph[x][y] = z

print(dijkstra(graph, 1, n))
