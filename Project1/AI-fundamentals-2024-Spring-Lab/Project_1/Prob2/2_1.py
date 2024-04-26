def parse_input(input_str):
    board = [int(x) if x != 'x' else 0 for x in input_str.split()]
    return tuple(board)

def is_valid_move(board, move):
    zero_index = board.index(0)
    if move == 'up':
        return zero_index >= 3
    elif move == 'down':
        return zero_index < 6
    elif move == 'left':
        return zero_index % 3 != 0
    elif move == 'right':
        return zero_index % 3 != 2

def move_zero(board, move):
    new_board = list(board)
    zero_index = new_board.index(0)
    if move == 'up':
        new_board[zero_index], new_board[zero_index - 3] = new_board[zero_index - 3], new_board[zero_index]
    elif move == 'down':
        new_board[zero_index], new_board[zero_index + 3] = new_board[zero_index + 3], new_board[zero_index]
    elif move == 'left':
        new_board[zero_index], new_board[zero_index - 1] = new_board[zero_index - 1], new_board[zero_index]
    elif move == 'right':
        new_board[zero_index], new_board[zero_index + 1] = new_board[zero_index + 1], new_board[zero_index]
    return tuple(new_board)

def dfs(board, target, max_depth):
    visited = set()
    stack = [(board, [board])]

    while stack:
        current_board, path = stack.pop()
        if current_board == target:
            return 1
        if len(path) >= max_depth:
            continue
        visited.add(current_board)

        for move in ['up', 'down', 'left', 'right']:
            if is_valid_move(current_board, move):
                new_board = move_zero(current_board, move)
                if new_board not in visited:
                    stack.append((new_board, path + [new_board]))

    return 0

initial_board_str = input()
target_board = (1, 2, 3, 4, 5, 6, 7, 8, 0)

initial_board = parse_input(initial_board_str)
print(dfs(initial_board, target_board, 1000))




