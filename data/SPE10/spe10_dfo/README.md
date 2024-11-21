DO NOT DELETE

This folder containts config files and .DATA files for the Computational Geoscience journal paper

Description:
1. For each well, 4 (or 3) well control and 2 (x and y) position, total of (4 + 2)*5 = 30 decision variables
2. Output constraints are: 
    - Minimal distance between two wells < 5 cells
    - Water production
3. The objective function is the NPV
4. Tested using different optimizers