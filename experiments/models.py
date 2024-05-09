import numpy as np
import os, sys
import json
import bridgestan as bs
from gen_reference_samples import ReferenceSamples

BRIDGESTAN = "/mnt/home/cmodi/Research/Projects/bridgestan/"
MODELDIR = '../'



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
def stan_model(name, D=0, 
               bridgestan_path=BRIDGESTAN, 
               model_directory=MODELDIR, 
               reference_samples_path=None, 
               run_nuts=True):
    
    bs.set_bridgestan_path(bridgestan_path)
    stanfile = f"{model_directory}/stan/{name}.stan" 
    if D != 0:
        print(f"{model_directory}/stan/{name}.data.json")
        datafile = write_tmpdata(D, f"{model_directory}/stan/{name}.data.json")
    else:
        if os.path.isfile(f"{model_directory}/stan/{name}.data.json"):
            datafile = f"{model_directory}/stan/{name}.data.json"
        elif os.path.isfile(f"{model_directory}/stan/{name}.json"):
            datafile = f"{model_directory}/stan/{name}.json"
    bsmodel = bs.StanModel.from_stan_file(stanfile, datafile)

    D = bsmodel.param_num()
    lp = lambda x: bsmodel.log_density(x)
    lp_g = lambda x: bsmodel.log_density_gradient(x)[1]

    if reference_samples_path is not None:
        ref_samples = np.load(reference_samples_path)
    else:
        try:
            ref_samples = ReferenceSamples().generate_samples(name, D, run_nuts=run_nuts)
        except Exception as e:
            print("Exception in generating reference samples : ", e)
            ref_samples = None

    return bsmodel, D, lp, lp_g, ref_samples, [stanfile, datafile]
