from heapq import heappush, heappop

def parse_input(input_str):
    board = tuple(int(x) if x != 'x' else 0 for x in input_str.split())
    return board

def is_valid_move(zero_index, move):
    if move == 'up':
        return zero_index >= 3
    elif move == 'down':
        return zero_index < 6
    elif move == 'left':
        return zero_index % 3 != 0
    elif move == 'right':
        return zero_index % 3 != 2

def move_zero(board, zero_index, move):
    new_board = list(board)
    if move == 'up':
        new_board[zero_index], new_board[zero_index - 3] = new_board[zero_index - 3], new_board[zero_index]
    elif move == 'down':
        new_board[zero_index], new_board[zero_index + 3] = new_board[zero_index + 3], new_board[zero_index]
    elif move == 'left':
        new_board[zero_index], new_board[zero_index - 1] = new_board[zero_index - 1], new_board[zero_index]
    elif move == 'right':
        new_board[zero_index], new_board[zero_index + 1] = new_board[zero_index + 1], new_board[zero_index]
    return tuple(new_board)

def dijkstra(initial_board, target_board):
    visited = set()
    heap = [(0, initial_board)]  # 用于保存节点的优先级队列，每个节点由 (权重, 状态) 组成
    while heap:
        swaps, current_board = heappop(heap)
        if current_board == target_board:
            return swaps
        zero_index = current_board.index(0)
        if current_board in visited:
            continue
        visited.add(current_board)
        for move in ['up', 'down', 'left', 'right']:
            if is_valid_move(zero_index, move):
                new_board = move_zero(current_board, zero_index, move)
                if new_board not in visited:
                    heappush(heap, (swaps + 1, new_board))
    return -1  # 无解

initial_board_str = input()
target_board = (1, 2, 3, 4, 5, 6, 7, 8, 0)

initial_board = parse_input(initial_board_str)
print(dijkstra(initial_board, target_board))
