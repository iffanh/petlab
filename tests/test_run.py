import shutil
import unittest

from petlab.io import run_bash_commands_in_parallel

SIMULATOR_PATH = "/usr/bin/flow"


@unittest.skipUnless(shutil.which(SIMULATOR_PATH) or shutil.which("flow"), "no OPM flow install found")
class SimulationTest(unittest.TestCase):

    def test_run(self):
        real_path = "./tests/test_run_folder/ECL_5SPOT_TEST.DATA"
        command = [SIMULATOR_PATH, real_path]
        is_success = run_bash_commands_in_parallel([command], max_tries=1, n_parallel=1)

        self.assertFalse(is_success[0])


if __name__ == "__main__":
    unittest.main()
