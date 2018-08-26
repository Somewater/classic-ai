from typing import Iterator
from my import Word

class TrieNode:
    def __init__(self):
        self.word = None
        self.children = None

    def insert( self, word: Word ):
        node = self
        for letter in self.word_to_text(word):
            if node.children is None:
                node.children = dict()
            if letter not in node.children:
                node.children[letter] = TrieNode()
            node = node.children[letter]
        node.word = word

    def search(self, word, maxCost) -> Iterator[str]:
        currentRow = range(len(word) + 1 )

        results = []

        # recursively search each branch of the trie
        for letter in self.children:
            if self.children:
                self._searchRecursive( self.children[letter], letter, word, currentRow, results, maxCost )

        return results

    def word_to_text(self, word: Word) -> str:
        return word.phonetic().lower()

    # This recursive helper is used by the search function above. It assumes that
    # the previousRow has been filled in already.
    def _searchRecursive(self, node, letter, word, previousRow, results, maxCost ):

        columns = len( word ) + 1
        currentRow = [ previousRow[0] + 1 ]

        # Build one row for the letter, with a column for each letter in the target
        # word, plus one for the empty string at column 0
        for column in range( 1, columns ):

            insertCost = currentRow[column - 1] + 1
            deleteCost = previousRow[column] + 1

            if word[column - 1] != letter:
                replaceCost = previousRow[ column - 1 ] + 1
            else:
                replaceCost = previousRow[ column - 1 ]

            currentRow.append( min( insertCost, deleteCost, replaceCost ) )

        # if the last entry in the row indicates the optimal cost is less than the
        # maximum cost, and there is a word in this trie node, then add it.
        if currentRow[-1] <= maxCost and node.word is not None:
            results.append( (node.word, currentRow[-1] ) )

        # if any entries in the row are less than the maximum cost, then
        # recursively search each branch of the trie
        if min( currentRow ) <= maxCost:
            if node.children:
                for letter in node.children:
                    self._searchRecursive( node.children[letter], letter, word, currentRow, results, maxCost )