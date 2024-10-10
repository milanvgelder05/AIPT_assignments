class TreeNode:
    def __init__(self, state, move=None, parent=None):
        """Constructor for the TreeNode class.

        Args:
            state : Current state of the game.
            move : The move that led to this game state.
            parent : The parent node representing the previous game state.
            children : A list of child nodes representing the future states.
            utility: An evaluated value for this state.
        """
        self.state = state
        self.move = move
        self.parent = parent
        self.children = []
        self.utility = None
        
    def add_child(self, child_node):
        self.children.append(child_node)
        
    def generate_children(self, player_id: int):
       
        
        for col in range(self.state.width):
            if self.state.is_valid(col):
                
                new_board = self.state.get_new_board(col, player_id)
                child_node = TreeNode(state=new_board, move=col, parent=self)
                self.add_child(child_node)
                