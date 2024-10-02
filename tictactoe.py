# to run this code, run: play_tic_tac_toe()

##_______________________________________________

import pandas as pd

# Step 1 - Initalize gameboard as a dataframe, with a-c as columns and 1-3 as rows.

def create_board():
    

    game_board_dict = {"A":["empty","empty","empty"],
                        "B":["empty","empty","empty"],
                        "C":["empty","empty","empty"],
                        }

    game_board = pd.DataFrame(game_board_dict, index = [1,2,3])

    return game_board


def player_x_turn(df):
    while True:
        print("Current game board:")
        print(df)

        # Get a list of the empty squares
        empty_cells = df[df == "empty"].stack().index.tolist()
        moves_left = [f"{col}{row}" for row, col in empty_cells]

        # Input function for a player's move
        x_choice = input("Player X: Type your move on the game board (e.g., A1): ").upper()

        # Check length and index
        if len(x_choice) == 2 and x_choice[0] in df.columns and int(x_choice[1]) in df.index:
            
            if x_choice in moves_left:
                # change data type to int
                df.loc[int(x_choice[1]), x_choice[0]] = "X"
                print("Updated game board:")
                print(df)
                break
            else:
                print("Choice invalid. Pick a valid remaining move on the game board.")
        else:
            print("Invalid move format. Enter in the format 'ColumnRow' (e.g., A1)")

    # Update after move
    empty_cells = df[df == "empty"].stack().index.tolist()
    moves_left = [f"{col}{row}" for row, col in empty_cells]

    return df, moves_left


def player_o_turn(df):
    while True:
        print("Current game board:")
        print(df)

        # Get a list of the empty squares
        empty_cells = df[df == "empty"].stack().index.tolist()
        moves_left = [f"{col}{row}" for row, col in empty_cells]

        # Input function for a player's move
        o_choice = input("Player X: Type your move on the game board (e.g., A1): ").upper()

        # Check length and index
        if len(o_choice) == 2 and o_choice[0] in df.columns and int(o_choice[1]) in df.index:
            
            if o_choice in moves_left:
                # change data type to int
                df.loc[int(o_choice[1]), o_choice[0]] = "O"
                print("Updated game board:")
                print(df)
                break
            else:
                print("Choice invalid. Pick a valid remaining move on the game board.")
        else:
            print("Invalid move format. Enter in the format 'ColumnRow' (e.g., A1)")

    # Update after move
    empty_cells = df[df == "empty"].stack().index.tolist()
    moves_left = [f"{col}{row}" for row, col in empty_cells]

    return df, moves_left


# Step 4: In-between moves, evaluate if the game has been won (3-in-a-row), or if there is a tie (all squares full without 3-in-a-row)
# v2

def check_for_wins_2(df, moves_left):
    #check for row wise wins
    for row in df.index:
        if (df.loc[row,'A'] == df.loc[row,'B'] == df.loc[row,'B']) and (df.loc[row,'A'] != 'empty'):
            print(f"Row {row} has all the same values, {df.loc[row,'A']}")
            return df.loc[row,'A']
    
    #check for col wise wins
    for col in df.columns:
        if (df.loc[1,col] == df.loc[2,col] == df.loc[3,col]) and (df.loc[1,col] != 'empty'):
            print(f"Column {col} has all the same values, {df.loc[1,col]}")
            return df.loc[1,col]
    
    #check for diagonal wins

    if (df.loc[1,'A'] == df.loc[2,'B'] == df.loc[3,'C']) and (df.loc[1,'A'] != 'empty'):
        print(f"Diagonal columns are all {df.loc[1,'A']}")
        return df.loc[1,'A']
    
    if (df.loc[3,'A'] == df.loc[2,'B']) == df.loc[1,'C'] and (df.loc[3,'A'] != 'empty'):
        print(f"Diagonal columns are all {df.loc[3,'A']}")
        return df.loc[1,'A']

    # check for a tie
    if len(moves_left) == 0:
        print("No remaining moves, it's a tie")
        return "Tie"

    print("No winner yet")

    return None


# Main Game loop
def play_tic_tac_toe():

    df = create_board()
    player = "X"

    while True:
        if player == "X":
            df, moves_left = player_x_turn(df)
        else:
            df, moves_left = player_o_turn(df)

        result = check_for_wins_2(df,moves_left)

        if result:
            print("Final board")
            print(df)
            if result == "Tie":
                print("Tie")
            else:
                print(f"{result} wins!")
            break

        player = "O" if player == "X" else "X"


#______________________________________________________
        
play_tic_tac_toe()
