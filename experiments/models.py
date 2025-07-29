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
def stan_model(exp, n=0, 
               bridgestan_path=BRIDGESTAN, 
               model_directory=MODELDIR, 
               reference_samples_path=None, 
               run_nuts=True):
    
    bs.set_bridgestan_path(bridgestan_path)
    stanfile = f"{model_directory}/stan/{exp}.stan" 
    if n != 0:
        print(f"{model_directory}/stan/{exp}.data.json")
        datafile = write_tmpdata(n, f"{model_directory}/stan/{exp}.data.json")
    else:
        datafile = f"{model_directory}/stan/{exp}.data.json"
        if os.path.isfile(datafile): pass
        else:
            datafile = f"{model_directory}/stan/{exp}.json"
            if os.path.isfile(datafile): pass
            else : raise FileNotFoundError
    bsmodel = bs.StanModel.from_stan_file(stanfile, datafile)

    D = bsmodel.param_unc_num()
    lp = lambda x: bsmodel.log_density(x)
    lp_g = lambda x: bsmodel.log_density_gradient(x)[1]

    # load reference samples. Run NUTS if not found.
    if n == 0 : reference_samples_path =  f'{reference_samples_path}/{exp}/'
    else: reference_samples_path =  f'{reference_samples_path}/{exp}-{D}/'
    try:
        ref_samples = np.load(reference_samples_path + '/samples.npy')
        if len(ref_samples.shape) == 2:
            ref_samples = np.expand_dims(ref_samples, axis=0)

    except Exception as e:
        print("Exception in loading reference samples: ", e)
        try:
            ref_samples_object = ReferenceSamples(stanfile = stanfile,
                                                  datafile = datafile,
                                                  model_directory = model_directory,
                                                  savefolder = reference_samples_path)
            ref_samples = ref_samples_object.generate_samples(exp, D, run_nuts=run_nuts)
        except Exception as e:
            print("Exception in generating reference samples : ", e)
            ref_samples = None

    return bsmodel, D, lp, lp_g, ref_samples, [stanfile, datafile]
