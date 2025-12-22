import utils.utilities as u

import sys
import os

def get_control(study):
    
    controls = study["extension"]["iterations"]["x"][-1]

    return controls

def change_parameters(config, controls, idx, to_replace):
    "change parameters based on the clrm settings"
    
    assert len(idx) == len(to_replace)
    
    ii = 0
    
    parameters = config["parameters"]
    for i, parameter in enumerate(parameters):
        if parameter["Name"] in to_replace:
            config["parameters"][i]["Distribution"]["parameters"]["value"] = controls[idx[ii]]
            ii += 1

    return config

def copy_control(config, study):
    
    if "copy_control" in config["clrm"].keys():
        controls_to_copy = config["clrm"]["copy_control"]
    else:
        return config
    
    ii = 0
    parameters = config["parameters"]
    for i, parameter in enumerate(parameters):
        if parameter["Name"] in controls_to_copy:
            params = study["creation"]["config"]["parameters"]
            for j, param in enumerate(params):
                if param["Name"] == parameter["Name"]:
                    config["parameters"][i]["Distribution"]["parameters"]["value"] = param["Distribution"]["parameters"]["value"]*1
            ii += 1
            
    return config

def main(argv):
    
    study_path = argv[0]
    config_path = argv[1]
    
    if not os.path.isfile(study_path):
        raise ValueError("%s cannot be found" %study_path)
    
    if not os.path.isfile(config_path):
        raise ValueError("%s cannot be found" %config_path)
    
    study = u.read_json(study_path)
    config = u.read_json(config_path)
    
    clrm_settings = config["clrm"]
    idx = clrm_settings["idx"]
    to_replace = clrm_settings["to_replace"]
    
    controls = get_control(study)
    
    # replace the old control with the optimal control 
    config = change_parameters(config, controls, idx, to_replace)
    
    # copy control from other, if any
    config = copy_control(config, study)
    
    
    u.save_to_json(config_path, config)
    
    print(f"The file {config_path} has been updated based on the optimal control which replaces {to_replace}. Indices = {idx}!")

if __name__ == "__main__":
    """Apply the control from study.json to the config.json based on the first control on the optimization.
    config.json must have the "clrm" keys with the following form:
    
    "clrm": 
            {
                  "idx" : [0, 1],
                  "to_replace": 
                              [
                                 "$PRODRATE1",
                                 "$INJRATE1" 
                              ]
            }
    
    Ex: "python3 src/apply_control.py study.json config.json"
    """

    main(sys.argv[1:])