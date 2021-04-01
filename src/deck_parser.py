import numpy as np 
import os

class DeckParser():

    def __init__(self):
        return 

    def keyword_search(self, deck_path:str, keyword:str="METRIC"):
        
        if not os.path.isfile(deck_path):
            raise Exception("%s not found" %deck_path)


        with open(deck_path, 'r') as f:
            lines = f.readlines()

        for line in lines:

            l = line.rstrip().split()
            
            # Skip empty lines
            if len(l) == 0:
                continue

            # Skip commented line
            if l[0] == "--":
                continue
            

            print(line.rstrip().split())

        # for line in lines:
        #     print(line.rstrip('\t').splitlines())



        return