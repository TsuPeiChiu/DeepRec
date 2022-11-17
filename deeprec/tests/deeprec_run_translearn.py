import sys
import deeprec.encoders as ec
import deeprec.models as dm
import deeprec.ensembler as eb
import deeprec.explainers as ep
import deeprec.utils.file_selex as fs
from deeprec.argument import ArgumentReader

def main(args):
    ar = ArgumentReader(args)

    # 0: data preparation
    deeprec_encoder = ec.Encoders()
    deeprec_encoder.prepare_data(infile=ar.input_data,
                                 testfile=ar.target_seq,
                                 config=ar.config,
                                 test_size=ar.valid_size,
                                 random_state=ar.random_state)

    # 1. transfer learning
    deeprec_ensembler = eb.DeepRecEnsembler(config=ar.config, 
                                            nb_models=ar.nb_models, 
                                            quantile=ar.quantile, 
                                            random_state=ar.random_state)
    deeprec_ensembler.transfer_models()
    deeprec_ensembler.refit(verbose=False) 
    deeprec_ensembler.predict()

    # 2. model interpreting
    deeprec_explainer = ep.DeepRecExplainer(config=ar.config, 
                            models=deeprec_ensembler.selected_models,                        
                            xs=deeprec_ensembler.test_x,
                            seqs=deeprec_ensembler.test_seqs,
                            random_state=ar.random_state,
                            y_lim=ar.y_lim)
    deeprec_explainer.plot_logos()
    
    
if __name__=='__main__':
    main(sys.argv)
