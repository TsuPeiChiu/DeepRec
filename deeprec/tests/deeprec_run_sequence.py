import sys
import deeprec.encoders as ec
import deeprec.models as dm
import deeprec.ensembler as eb
import deeprec.explainers as ep
import deeprec.utils.file_selex as fs
from deeprec.argument import ArgumentReader

def main(args):
    ar = ArgumentReader(args)
    mode = ar.mode #'012345'

    if '0' in mode and '1' in mode:
        print('mode 0 and 1 cannot coexist!')
        sys.exit(0)

    if '3' in mode and '4' in mode:
        print('mode 3 and 4 cannot coexist!')
        sys.exit(0)

    # 0: data preparation for SELEX tool's output 
    if '0' in mode:
        onestrand_selex = fs.remove_redundancy(ar.input_data)
        deeprec_encoder = ec.Encoders()
        deeprec_encoder.prepare_data(infile=onestrand_selex,
                                     testfile=ar.target_seq,
                                     config=ar.config,
                                     test_size=ar.valid_size,
                                     random_state=ar.random_state)
    
    # 1: data preparation for two-columns format data    
    if '1' in mode:
        deeprec_encoder = ec.Encoders()
        deeprec_encoder.prepare_data(infile=ar.input_data,
                                 testfile=ar.target_seq,
                                 config=ar.config,
                                 test_size=ar.valid_size,
                                 random_state=ar.random_state)
        
    # 2: model tuning
    if '2' in mode:
        deeprec_model = dm.DeepRecModel(config=ar.config, 
                                        random_state=ar.random_state,
                                        mode='seq')
        config_tuned = deeprec_model.tune(config_tune=ar.config_tune, 
                                        nb_params=ar.nb_params, 
                                        verbose=False)

    # 3: ensemble modeling
    if '3' in mode:
        config_tuned = ar.config.replace('.yaml','.tuned.yaml')
        deeprec_ensembler = eb.DeepRecEnsembler(config=config_tuned, 
                                                nb_models=ar.nb_models, 
                                                quantile=ar.quantile, 
                                                random_state=ar.random_state,
                                                mode='seq')  
        deeprec_models = deeprec_ensembler.fit(verbose=False,
                                                is_shuffle=ar.shuffle)
        deeprec_ensembler.predict()
        
    # 4: prediction from trained models
    if '4' in mode:
        config_tuned = ar.config.replace('.yaml','.tuned.yaml')
        deeprec_ensembler = eb.DeepRecEnsembler(config=config_tuned,
                                                nb_models=ar.nb_models,
                                                quantile=ar.quantile,
                                                random_state=ar.random_state,
                                                mode='seq')


        deeprec_models = deeprec_ensembler.load_models()
        deeprec_ensembler.predict()

    # 5. model interpreting
    if '5' in mode:
        deeprec_explainer = ep.DeepRecExplainer(config=config_tuned, 
                            models=deeprec_models,                        
                            xs=deeprec_models[0].test_x,
                            seqs=deeprec_models[0].test_seqs,
                            random_state=ar.random_state,
                            y_lim=ar.y_lim, 
                            mode='seq')
        deeprec_explainer.plot_logos()
    
    
if __name__=='__main__':
    main(sys.argv)
