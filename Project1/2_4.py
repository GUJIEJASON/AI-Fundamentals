from heapq import heappush, heappop

def parse_input(input_str):
    board = tuple(int(x) if x != 'x' else 0 for x in input_str.split())
    return board

def is_valid_move(zero_index, move):
    if move == 'u': # 上
        return zero_index >= 3
    elif move == 'd': # 下
        return zero_index < 6
    elif move == 'l': # 左
        return zero_index % 3 != 0
    elif move == 'r': # 右
        return zero_index % 3 != 2

def move_zero(board, zero_index, move):
    new_board = list(board)
    if move == 'u':
        new_board[zero_index], new_board[zero_index - 3] = new_board[zero_index - 3], new_board[zero_index]
    elif move == 'd':
        new_board[zero_index], new_board[zero_index + 3] = new_board[zero_index + 3], new_board[zero_index]
    elif move == 'l':
        new_board[zero_index], new_board[zero_index - 1] = new_board[zero_index - 1], new_board[zero_index]
    elif move == 'r':
        new_board[zero_index], new_board[zero_index + 1] = new_board[zero_index + 1], new_board[zero_index]
    return tuple(new_board)

def manhattan_distance(state1, state2):
    distance = 0
    for i in range(3):
        for j in range(3):
            if state1[i*3+j] != state2[i*3+j] and state1[i*3+j] != 0:
                x1, y1 = divmod(state1[i*3+j], 3)
                x2, y2 = divmod(state2[i*3+j], 3)
                distance += abs(x1 - x2) + abs(y1 - y2)
    return distance

def get_next_states(state):
    zero_index = state.index(0)
    next_states = []
    for move in ['u', 'd', 'l', 'r']:
        if is_valid_move(zero_index, move):
            next_states.append((move, move_zero(state, zero_index, move)))
    return next_states

def count_inversions(board):
    inversions = 0
    for i in range(len(board)):
        for j in range(i + 1, len(board)):
            if board[i] != 0 and board[j] != 0 and board[i] > board[j]:
                inversions += 1
    return inversions

def solve_puzzle(initial_state, goal_state):
    open_list = []
    closed_set = set()
    action_record = {}  # 用于记录每个状态的移动动作

    # 初始状态加入 open 列表
    heappush(open_list, (manhattan_distance(initial_state, goal_state), 0, initial_state))
    action_record[initial_state] = ""
    
    while open_list:
        _, cost, current_state = heappop(open_list)

        if current_state == goal_state:
            return action_record[current_state]
        
        if current_state in closed_set:
            continue
        
        closed_set.add(current_state)

        for action, next_state in get_next_states(current_state):
            if next_state not in closed_set:
                heappush(open_list, (manhattan_distance(next_state, goal_state) + cost + 1, cost + 1, next_state))
                action_record[next_state] = action_record[current_state] + action
    
    return "unsolvable"

try:
    initial_board_str = input()
    target_board = (1, 2, 3, 4, 5, 6, 7, 8, 0)

    initial_board = parse_input(initial_board_str)
    inversions = count_inversions(initial_board)
    if inversions % 2 == 1:
        print("unsolvable")
    else:
        print(solve_puzzle(initial_board, target_board))
except Exception as e:
    print("Error:", e)
