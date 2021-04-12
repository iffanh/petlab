import numpy as np 
import os

class DeckParser():

    def __init__(self):
        return


    ### Reading ...

    def get_all_keywords(self, file_path, content=[]):

        data_folder = os.path.dirname(file_path)
        data_file = os.path.basename(file_path)

        is_include = False
        _content = []

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

            # If INCLUDE keyword is hit, this is for the keyword, 
            # while the block just below this is for the content
            if l[0] == 'INCLUDE':
                is_include = True
                try:
                    content.append((keyword, _content))
                except:
                    # first pass
                    pass
                    
                keyword = l[0]
                _content = []

                continue

            # Only gets into this block if and only if it's inside include keyword. 
            # the line that goes here is the content instead of the include keyword
            # itself
            if is_include:
                _content.append(l[0])

                try:
                    content.append((keyword, _content))
                except:
                    # first pass
                    pass

                if l[0][0] == "'":
                    file_path = os.path.join(data_folder, l[0][1:-1])
                else:
                    file_path = os.path.join(data_folder, l[0])

                content = self.get_all_keywords(file_path, content)

                is_include = False
                continue

            # When it hits the next keyword *besides INCLUDE)
            if l[0].isalpha():  
                
                # Append the previous keyword + content
                try:
                    # We do not INCLUDE keyword since it's handled separately
                    if keyword != 'INCLUDE':  
                        content.append((keyword, _content))
                except:
                    # first pass
                    pass
                
                #  Restart
                keyword = l[0]
                _content = []
                continue

            _content.append(l)

        # Last keyword+content, if any
        try:
            content.append((keyword, _content))
        except:
            # first pass
            pass

        return content


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
        is_include = False

        data_folder = os.path.dirname(file_path)
        data_file = os.path.basename(file_path)
        
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

            # If it hits include keyword,
            # read the include file
            if l[0] == 'INCLUDE':
                is_include = True
                continue

            if is_include:
                if l[0][0] == "'":
                    file_path = os.path.join(data_folder, l[0][1:-1])
                else:
                    file_path = os.path.join(data_folder, l[0])

                hit, content = self._read_file(file_path, hit, content, keyword)
                is_include = False
                
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

                    if l[0][0] == "'":
                        file_path = os.path.join(data_folder, l[0][1:-1])
                    else:
                        file_path = os.path.join(data_folder, l[0])

                    hit, content = self._read_file(file_path, hit, content, keyword)

                    
        return hit, content


    ### Writing ...