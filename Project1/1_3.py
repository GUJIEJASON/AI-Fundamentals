import heapq

def heap_dijkstra(graph, start, end):
    n = len(graph)
    distance = [float('inf')] * (n + 1)
    distance[start] = 0
    
    pq = [(0, start)]  
    
    while pq:
        dist_u, u = heapq.heappop(pq) 
        if dist_u > distance[u]:
            continue
        
        for v, weight in graph[u]:
            if u == v:  # 自环
                continue
            if distance[u] + weight < distance[v]:
                distance[v] = distance[u] + weight
                heapq.heappush(pq, (distance[v], v)) 
    
    return distance[end] if distance[end] != float('inf') else -1

n, m = map(int, input().split())

graph = [[] for _ in range(n + 1)]

for _ in range(m):
    x, y, z = map(int, input().split())
    graph[x].append((y, z))

print(heap_dijkstra(graph, 1, n))
