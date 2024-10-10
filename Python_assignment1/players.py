from __future__ import annotations
from abc import abstractmethod
import numpy as np
from typing import TYPE_CHECKING
from tree import TreeNode
if TYPE_CHECKING:
    from heuristics import Heuristic
    from board import Board



class PlayerController:
    """Abstract class defining a player
    """
    def __init__(self, player_id: int, game_n: int, heuristic: Heuristic) -> None:
        """
        Args:
            player_id (int): id of a player, can take values 1 or 2 (0 = empty)
            game_n (int): n in a row required to win
            heuristic (Heuristic): heuristic used by the player
        """
        self.player_id = player_id
        self.game_n = game_n
        self.heuristic = heuristic


    def get_eval_count(self) -> int:
        """
        Returns:
            int: The amount of times the heuristic was used to evaluate a board state
        """
        return self.heuristic.eval_count
    

    def __str__(self) -> str:
        """
        Returns:
            str: representation for representing the player on the board
        """
        if self.player_id == 1:
            return 'X'
        return 'O'
        

    @abstractmethod
    def make_move(self, board: Board) -> int:
        """Gets the column for the player to play in

        Args:
            board (Board): the current board

        Returns:
            int: column to play in
        """
        pass


class MinMaxPlayer(PlayerController):
    """Class for the minmax player using the minmax algorithm
    Inherits from Playercontroller
    """
    def __init__(self, player_id: int, game_n: int, depth: int, heuristic: Heuristic) -> None:
        """
        Args:
            player_id (int): id of a player, can take values 1 or 2 (0 = empty)
            game_n (int): n in a row required to win
            depth (int): the max search depth
            heuristic (Heuristic): heuristic used by the player
        """
        super().__init__(player_id, game_n, heuristic)
        self.depth: int = depth


    def make_move(self, board: Board) -> int:
        """Gets the column for the player to play in

        Args:
            board (Board): the current board

        Returns:
            int: column to play in
        """
              
        root = TreeNode(state=board)
        
        max_value = -np.inf
        best_move = None
        
        root.generate_children(self.player_id)
        
        for child in root.children:
            value = self.minimax(child, self.depth - 1, False)
            
            if value >max_value:
                max_value = value
                best_move = child.move
                
            
        return best_move
    
    def minimax(self, node: TreeNode, depth: int, maximizing_player: bool) -> int:
            if depth == 0 or self.heuristic.winning(node.state.get_board_state(), self.game_n) != 0:
                # Base case: if depth is 0 or there's a winner, return heuristic evaluation
                return self.heuristic.evaluate_board(self.player_id, node.state)
            
            if maximizing_player:
                #this executes when it is the minimax player's turn (trying to maximize)
                
                max_eval = -np.inf
                node.generate_children(self.player_id)
                for child in node.children:
                    eval = self.minimax(child, depth - 1, False)
                    max_eval = max(max_eval, eval)
                
                node.utility = max_eval
                return max_eval
            
            else:
                #this exexcutes when oponents turn
                
                opponent_id = 3 - self.player_id
                min_eval = np.inf
                node.generate_children(opponent_id)
                
                for child in node.children:
                    eval = self.minimax(child, depth - 1, True)
                    min_eval = min(min_eval, eval)
                    
                node.utility = min_eval 
                return min_eval

class AlphaBetaPlayer(PlayerController):
    """Class for the minmax player using the minmax algorithm with alpha-beta pruning
    Inherits from Playercontroller
    """
    def __init__(self, player_id: int, game_n: int, depth: int, heuristic: Heuristic) -> None:
        """
        Args:
            player_id (int): id of a player, can take values 1 or 2 (0 = empty)
            game_n (int): n in a row required to win
            depth (int): the max search depth
            heuristic (Heuristic): heuristic used by the player
        """
        super().__init__(player_id, game_n, heuristic)
        self.depth: int = depth


    def make_move(self, board: Board) -> int:
        """Gets the column for the player to play in

        Args:
            board (Board): the current board

        Returns:
            int: column to play in
        """
        
        root = TreeNode(state=board)
        
        max_value = -np.inf
        best_move = None
        
        root.generate_children(self.player_id)
        
        for child in root.children:
            value = self.alpha_beta(child, self.depth - 1, -np.inf, np.inf, False)

            if value > max_value:
                max_value = value
                best_move = child.move

        return best_move

        
    
    
    def alpha_beta(self, node: TreeNode, depth: int, alpha: float, beta: float, maximizing_player: bool):
        
        if depth == 0 or self.heuristic.winning(node.state.get_board_state(), self.game_n) != 0:
            return self.heuristic.evaluate_board(self.player_id, node.state)
        
        
        if maximizing_player:
                #this executes when it is the minimax player's turn (trying to maximize)
                
            max_eval = -np.inf
            node.generate_children(self.player_id)
            
            for child in node.children:
                eval = self.alpha_beta(child, depth - 1, alpha, beta, False)
                max_eval = max(max_eval, eval)
                alpha = max(alpha, eval)
                    
                if beta <= alpha:
                    break
                
            node.utility = max_eval
            return max_eval
        
        
        else:
                #this exexcutes when oponents turn
                
                opponent_id = 3 - self.player_id
                min_eval = np.inf
                node.generate_children(opponent_id)
                
                for child in node.children:
                    eval = self.alpha_beta(child, depth - 1, alpha, beta, True)
                    min_eval = min(min_eval, eval)
                    beta = min(beta, eval)
                    
                    if beta <= alpha:
                        break
                    
                node.utility = min_eval 
                return min_eval
            
        
            
            
       
        


class HumanPlayer(PlayerController):
    """Class for the human player
    Inherits from Playercontroller
    """
    def __init__(self, player_id: int, game_n: int, heuristic: Heuristic) -> None:
        """
        Args:
            player_id (int): id of a player, can take values 1 or 2 (0 = empty)
            game_n (int): n in a row required to win
            heuristic (Heuristic): heuristic used by the player
        """
        super().__init__(player_id, game_n, heuristic)

    
    def make_move(self, board: Board) -> int:
        """Gets the column for the player to play in

        Args:
            board (Board): the current board

        Returns:
            int: column to play in
        """
        print(board)

        if self.heuristic is not None:
            print(f'Heuristic {self.heuristic} calculated the best move is:', end=' ')
            print(self.heuristic.get_best_action(self.player_id, board) + 1, end='\n\n')

        col: int = self.ask_input(board)

        print(f'Selected column: {col}')
        return col - 1
    

    def ask_input(self, board: Board) -> int:
        """Gets the input from the user

        Args:
            board (Board): the current board

        Returns:
            int: column to play in
        """
        try:
            col: int = int(input(f'Player {self}\nWhich column would you like to play in?\n'))
            assert 0 < col <= board.width
            assert board.is_valid(col - 1)
            return col
        except ValueError: # If the input can't be converted to an integer
            print('Please enter a number that corresponds to a column.', end='\n\n')
            return self.ask_input(board)
        except AssertionError: # If the input matches a full or non-existing column
            print('Please enter a valid column.\nThis column is either full or doesn\'t exist!', end='\n\n')
            return self.ask_input(board)
        