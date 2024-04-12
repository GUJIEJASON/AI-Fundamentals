def dijkstra(n, start, edges):
    dist = [float('inf')] * (n + 1)
    dist[start] = 0
    st = [False] * (n + 1)

    for _ in range(n):
        t = -1
        for j in range(1, n + 1):
            if not st[j] and (t == -1 or dist[t] > dist[j]):
                t = j

        st[t] = True

        for j in range(1, n + 1):
            dist[j] = min(dist[j], dist[t] + edges[t][j])

    if dist[n] == float('inf'):
        return -1
    return dist[n]


n, m = map(int, input().split())
edges = [[float('inf')] * (n + 1) for _ in range(n + 1)]

for _ in range(m):
    a, b, c = map(int, input().split())
    edges[a][b] = min(edges[a][b], c)

print(dijkstra(n, 1, edges))
