from PIL import Image, ImageDraw

# Define the size of the board and the size of each cell in pixels
def plot_board(board,path):
    board_size = len(board[0])
    cell_size = 50

    # Create a new image with a white background
    image = Image.new("RGB", (board_size*cell_size, board_size*cell_size), "white")

    # Draw the cells for the board
    draw = ImageDraw.Draw(image)
    for row in range(board_size):
        for col in range(board_size):
    #         if (row ==0 or row ==1) and (col ==0 or col ==1):
    #             continue
    #         if (row == 0 or row == 1) and (col == 5 or col == 6):
    #             continue
    #         if (row ==5 or row ==6) and (col ==0 or col ==1):
    #             continue
    #         if (row ==5 or row ==6) and (col ==5 or col ==6):
    #             continue

            x0 = col*cell_size
            y0 = row*cell_size
            x1 = x0 + cell_size
            y1 = y0 + cell_size
            draw.rectangle((x0, y0, x1, y1), outline="black")

    # Draw the pegs for the starting position
    peg_positions = []
    for row in range(board_size):
        for col in range(board_size):
            if board[row, col] == 1:
                peg_positions.append((row,col))

    for row, col in peg_positions:
        x = col*cell_size + cell_size/2
        y = row*cell_size + cell_size/2
        radius = cell_size/3
        draw.ellipse((x-radius, y-radius, x+radius, y+radius), fill="red", outline="red")

    # Save the image to a file
    image.save(path)
