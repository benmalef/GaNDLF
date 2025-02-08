import json

from GANDLF.config_manager import ConfigManager
from pathlib import Path

if __name__ == "__main__":
    testingDir = Path(__file__).parent.absolute().__str__()
    parameters = ConfigManager(
        testingDir + "/config_all_options.yaml", version_check_flag=False
    )
    print(json.dumps(parameters,indent=4))
