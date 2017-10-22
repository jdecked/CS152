"""
Design a python class named PuzzleNode with appropriate attributes
and methods to capture the state of the 8-puzzle and its generalizations,
as well as the elements needed to implement an A* search as seen in Lesson 3.1.
Document the design of your class, indicating clearly the purpose of each of its elements.
Also indicate what data structure you are using to represent the state of the tiles.
"""

import re

class PuzzleNode(object):
    def __init__(self, size):
        self.board = [[101]*size for count in range(size)]
        self.parent = None


    def __str__(self):
        """
        Prints 8-puzzle in following format:
        +-----------+
        | - | - | - |
        | - | - | - |
        | - | - | - |
        +-----------+
        Square puzzles of other sizes are printed analogously.

        @return: Puzzle board formatted as described above.

        I can come up with a better way that handles multiple digit numbers
        and single digit numbers at the same time.
        """
        size = len(self.board)

        # Fun tidbit: I've separated the bits which use re.sub and those which
        # use str.replace because re.sub tends to be slower. Also, just because
        # I don't need it if I'm not actually using a non-static expression!
        # More fun discussion here:
        # https://stackoverflow.com/questions/452104/is-it-worth-using-pythons-re-compile
        cells_with_spaces = re.sub('([1-9]+0*[1-9]*)', r' \1 |', str(self.board)[1:-1])
        with_dashes = re.sub('(None)|\b0\b', ' - |', cells_with_spaces)

        return '+' + '-' * (4 * size - 1) + '+\n' + \
               with_dashes.replace('[', '|') \
                          .replace(']', '\n') \
                          .replace(', ', '') + \
               '+' + '-' * (4 * size - 1) + '+'

        # Yes, okay, I didn't *need* to use regex... but I've gone with it
        # since it communicates more than the below one-liner.
        #
        # flattened_board = [cell for row in self.board for cell in row]
        # return '+' + '-' * (4 * size - 1) + '+\n' + \
        #        (('|' + ' {} |' * size + '\n') * size) \
        #             .format(*flattened_board) \
        #             .replace('None', '-') \
        #             .replace('0', '-') + \
        #        '+' + '-' * (4 * size - 1) + '+'

for i in range(11):
    test = PuzzleNode(i)
    print test
