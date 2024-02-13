import numpy as np
import src.utils.utilities as u

import unittest

SIMULATOR_PATH = f"/usr/bin/flow"

class SimulationTest(unittest.TestCase):

    # def test_broken_run(self):
        
    #     # Simulation will NOT run
        
    #     real_path = f"./tests/test_run_folder/ECL_5SPOT_BROKEN.DATA"
    #     command = [SIMULATOR_PATH, '--enable-terminal-output=false', real_path]
    #     is_success = u.run_bash_commands_in_parallel([command], max_tries=1, n_parallel=1)
        
    #     self.assertFalse(is_success[0])
    
    # def test_successful_run(self):
        
    #     # Simualtion will run and successfully reaches the final timesteps 
        
    #     real_path = f"./tests/test_run_folder/ECL_5SPOT_SUCCESS.DATA"
    #     command = [SIMULATOR_PATH, '--enable-terminal-output=false', real_path]
    #     is_success = u.run_bash_commands_in_parallel([command], max_tries=1, n_parallel=1)
        
    #     self.assertTrue(is_success[0])
        
    # def test_failed_run(self):
        
    #     # Simulation will run but failed before it reaches the final timesteps
        
    #     real_path = f"./tests/test_run_folder/ECL_5SPOT_FAILED.DATA"
    #     command = [SIMULATOR_PATH, '--enable-terminal-output=true', real_path]
    #     is_success = u.run_bash_commands_in_parallel([command], max_tries=1, n_parallel=1)
        
    #     self.assertFalse(is_success[0])
        
    def test_run(self):
        
        real_path = f"./tests/test_run_folder/ECL_5SPOT_TEST.DATA"
        command = [SIMULATOR_PATH, real_path]
        is_success = u.run_bash_commands_in_parallel([command], max_tries=1, n_parallel=1)
        
        self.assertFalse(is_success[0])
        
if __name__ == '__main__':
    # begin the unittest.main()
    unittest.main()