import numpy as np 
import os

class DeckParser():

    def __init__(self):
        return

    def keyword_search(self, deck_path:str, keyword:str="METRIC"):
        """
        This function parses through the file specified in deck_path and
        look for the specified keyword.

        Args:
            deck_path   : (str) path to the .DATA file
            keyword     : (str) keyword to search

        Returns: 
            hit         : (bool) whether keyword is found
            content     : (list) content of the keyword 
        """

        hit = False
        content = []

        hit, content = self._read_file(deck_path, hit, content, keyword)

        return hit, content

    def _read_file(self, file_path, hit: bool, content:list, keyword:str):
        
        """
        This function parses through the file specified in deck_path and
        look for the specified keyword. Recursively goes through the file
        inside the INCLUDE keyword, too. 

        Args:
            file_path   : (str) path to the .DATA file
            hit         : (bool) whether keyword is found
            content     : (list) content of the keyword 
            keyword     : (str) keyword to search

        Returns: 
            hit         : (bool) whether keyword is found
            content     : (list) content of the keyword 
        """

        # Whether the line is inside the specified keyword
        is_in = False

        if not os.path.isfile(file_path):
            raise Exception("%s not found" %file_path)

        with open(file_path, 'r') as f:
            lines = f.readlines()
       
        for line in lines:

            l = line.rstrip().split()
            
            # Skip empty lines
            if len(l) == 0:
                continue

            # Skip commented line
            if l[0][0] == "-":
                continue
            
            # Check keyword
            if l[0] == keyword:
                is_in = True
                hit = True
                continue

            if is_in:    
                # Check if the next line hit another keyword
                if l[0].isalpha():
                    is_in = False
                    continue

                # Add the content
                content.append(l)

                # If it hits include keyword,
                # read the include file
                if keyword == 'INCLUDE':
                    file_path = "%s/%s" %(file_path, l[0])
                    hit, content = _read_file(file_path, hit, content, keyword)
        
        return hit, content
