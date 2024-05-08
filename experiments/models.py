import numpy as np
import os, sys
import json
import bridgestan as bs
BRIDGESTAN = "/mnt/home/cmodi/Research/Projects/bridgestan/"
bs.set_bridgestan_path(BRIDGESTAN)
CURR_DIR = "/mnt/home/cmodi/Research/Projects/ADSampler/" #os.getcwd()


def write_tmpdata(D, datapath):
    # Data to be written
    dictionary = {
        "D": D,
    }
    # Serializing json
    json_object = json.dumps(dictionary, indent=4)
    print(dictionary)
    # Writing to sample.json
    with open(f"{datapath}", "w") as outfile:
        outfile.write(json_object)

    return f"{datapath}"


##### Setup the models
def setup_model(name, D=0):
    stanfile = f"{CURR_DIR}/stan/{name}.stan" 
    if D != 0:
        print(f"{CURR_DIR}/stan/{name}.data.json")
        datafile = write_tmpdata(D, f"{CURR_DIR}/stan/{name}.data.json")
    else:
        if os.path.isfile(f"{CURR_DIR}/stan/{name}.data.json"):
            datafile = f"{CURR_DIR}/stan/{name}.data.json"
        elif os.path.isfile(f"{CURR_DIR}/stan/{name}.json"):
            datafile = f"{CURR_DIR}/stan/{name}.json"
    bsmodel = bs.StanModel.from_stan_file(stanfile, datafile)

    D = bsmodel.param_num()
    lp = lambda x: bsmodel.log_density(x)
    lp_g = lambda x: bsmodel.log_density_gradient(x)[1]
    if name == "normal":
        ref_samples = np.random.normal(0, 1, 10000*D).reshape(1, 10000, D)
    else:
        ref_samples = np.load(f'/mnt/ceph/users/cmodi/PosteriorDB/{name}/samples.npy')
    return bsmodel, D, lp, lp_g, ref_samples, [stanfile, datafile]
