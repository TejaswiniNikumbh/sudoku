M = 9
def print_puzzle(a):
    for i in range(M):
        for j in range(M):
            print(a[i][j],end = " ")
        print()
    return a

def isvalid(puzzle, row, col, num):
    for x in range(9):
        if puzzle[row][x] == num:
            return False

    for x in range(9):
        if puzzle[x][col] == num:
            return False

    startRow = row - row % 3
    startCol = col - col % 3
    for i in range(3):
        for j in range(3):
            if puzzle[i + startRow][j + startCol] == num:
                return False
    return True

def solve(puzzle):
    for row in range(9):
        for col in range(9):
            if puzzle[row][col] == 0:
                for num in range(1, 10):
                    if isvalid(puzzle, row, col, num):
                        puzzle[row][col] = num

                        if solve(puzzle):
                            return True
                        else:
                            puzzle[row][col] = 0
                return False
    return True

def solve_sudoku(puzzle):
    solve(puzzle)
    return puzzle



# if solve(puzzle):
#     print_puzzle(puzzle)
# else:
#     print("Solution does not exist.")
