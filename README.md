# DeepRec (<ins>Deep</ins> <ins>Rec</ins>ognition for TF–DNA binding)
DNA-binding proteins selectively bind to their genomic binding sites and trigger various cellular processes. This selective binding occurs when the DNA-binding domain of the protein recognizes its binding site by reading physicochemical signatures on the base-pair edges.


![alt text](https://github.com/TsuPeiChiu/deeprec/blob/main/deeprec/imgs/figure1.jpg)



DeepRec is a deep-learning-based method that integrates two CNN modules for extracting important physicochemical signatures in the major and minor grooves of DNA. Each CNN module extracts nonlinear spatial information between functional groups in DNA base pairs to mine potential insights beyond DNA sequence.


![alt text](https://github.com/TsuPeiChiu/deeprec/blob/main/deeprec/imgs/figure2.jpg)


# Run DeepRec

## Installation

- Download DeepRec: `git clone https://github.com/TsuPeiChiu/deeprec.git`

- DeepRec is Linux/Unix-based system compatible. We strongly recommend users to use Anaconda to setup the virtual environment to run DeepRec. Install Anaconda [Individual](https://www.anaconda.com/products/individual) Edition first.

- Create an environment `deeprec` using the command `conda create -n deeprec python=2.7.13`

- Activate the environment: `source activate deeprec`

- Run command: `conda install pyyaml pandas keras=2.1.3 scikit-learn tensorflow Pillow matplotlib seaborn pandas`

- Add `deeprec` directory in PYTHONPATH: `export PYTHONPATH=$PYTHONPATH:../../../deeprec`

## Main command

<i>(Take TF MAX as an example)</i>

- Move to the `tests` directory: `cd deeprec/tests/`

`python ./deeprec_run.py -c ./max/config/config.yaml -t ./max/config/tune_params.yaml -p 100 -s 0 -e 100 -q 0.5 -d ./max/data/test_seq.txt -i ./max/data/r0_r1_max_selex_seq_10mer_150.txt -v 0.1 -f 0 -m 0235 -y 0.6`


### Parameters
`-c`: config file for training a model

`-t`: config file for hyperparameter search

`-p`: number of parameter sets for hyperparameter search

`-s`: seed for reproducing the result

`-e`: number of models for ensemble learning

`-q`: quantile of models selected for analysis

`-d`: target sequences for interpretation

`-i`: input file

`-v`: proportion of data for validation

`-f`: flag for shuffling y

`-m`: mode for running specific DeepRec functions (for example, -m 0235)

> 0: data encoding

> 1: data encoding (from SELEX-seq). Mode 0 and 1 cannot coexist

> 2: model tuning

> 3: prediction with ensemble modeling (from scratch)

> 4: prediction with exisiting models. Mode 3 and 4 cnnot coexist 

> 5: model interpreting

`-y`: y-lim for visualization


## Systems tested
- MAX (SELEX-seq)

`python ./deeprec_run.py -c ./max/config/config.yaml -t ./max/config/tune_params.yaml -p 100 -s 0 -e 100 -q 0 -d ./max/data/test_seq.txt -i ./max/data/r0_r1_max_selex_seq_10mer_150.txt -v 0.1 -f 0 -m 0235 -y 0.8`

`python ./deeprec_run.py -c ./max/config/config.yaml -t ./max/config/tune_params.yaml -p 100 -s 0 -e 100 -q 0 -d ./max/data/test_seq.txt -i ./max/data/r0_r1_max_selex_seq_10mer_150.txt -v 0.1 -f 1 -m 35 -y 0.8` (for control)


- MAX (SMiLE-seq)

`python ./deeprec_run.py -c ./max_smile/config/config.yaml -t ./max_smile/config/tune_params.yaml -p 100 -s 0 -e 100 -q 0.5 -d ./max_smile/data/test_seq.txt -i ./max_smile/data/r0_r1_max_smile_seq_10mer_50.txt -v 0.1 -f 0 -m 0235 -y 0.8`

`python ./deeprec_run.py -c ./max_smile/config/config.yaml -t ./max_smile/config/tune_params.yaml -p 100 -s 0 -e 100 -q 0.5 -d ./max_smile/data/test_seq.txt -i ./max_smile/data/r0_r1_max_smile_seq_10mer_50.txt -v 0.1 -f 1 -m 35 -y 0.8` (for control)

- MEF2B (SELEX-seq)

`python ./deeprec_run.py -c ./mef2b/config/config.yaml -t ./mef2b/config/tune_params.yaml -p 100 -s 0 -e 100 -q 0.5 -d ./mef2b/data/test_seq.txt -i ./mef2b/data/r0_r1_mef2b_selex_seq_10mer_100.txt -v 0.1 -f 0 -m 0235 -y 0.6`

`python ./deeprec_run.py -c ./mef2b/config/config.yaml -t ./mef2b/config/tune_params.yaml -p 100 -s 0 -e 100 -q 0.5 -d ./mef2b/data/test_seq.txt -i ./mef2b/data/r0_r1_mef2b_selex_seq_10mer_100.txt -v 0.1 -f 1 -m 35 -y 0.6` (for control)


- p53 (SELEX-seq)

`python ./deeprec_run.py -c ./p53/config/config.yaml -t ./p53/config/tune_params.yaml -p 100 -s 0 -e 100 -q 0.5 -d ./p53/data/test_seq.txt -i ./p53/data/r0_r1_p53_selex_seq_10mer_300.txt -v 0.1 -f 0 -m 0235 -y 0.6`

`python ./deeprec_run.py -c ./p53/config/config.yaml -t ./p53/config/tune_params.yaml -p 100 -s 0 -e 100 -q 0.5 -d ./p53/data/test_seq.txt -i ./p53/data/r0_r1_p53_selex_seq_10mer_300.txt -v 0.1 -f 1 -m 0235 -y 0.6` (for control)


- ATF4 (EpiSELEX)

`python ./deeprec_run.py -c ./atf4_episelex/config/config.yaml -t ./atf4_episelex/config/tune_params.yaml -p 100 -s 0 -e 100 -q 0.5 -d ./atf4_episelex/data/test_seq.txt -i ./atf4_episelex/data/r0_r1_atf4_episelex_seq_10mer_75_U_M_nor.txt -v 0.1 -f 0 -m 0235 -y 0.8`


- CEBPB (EpiSELEX)

`python ./deeprec_run.py -c ./cebpb_episelex/config/config.yaml -t ./cebpb_episelex/config/tune_params.yaml -p 100 -s 0 -e 100 -q 0.5 -d ./cebpb_episelex/data/test_seq.txt -i ./cebpb_episelex/data/r0_r1_cebpb_episelex_seq_10mer_25_U_M_nor.txt -v 0.1 -f 0 -m 0235 -y 0.8`


## Onehot sequence learning

- MAX (SELEX-seq)

`python ./deeprec_run_sequence.py -c ./max_sequence/config/config.yaml -t ./max_sequence/config/tune_params.yaml -p 100 -s 0 -e 100 -q 0.5 -d ./max_sequence/data/test_seq.txt -i ./max_sequence/data/r0_r1_max_selex_seq_10mer_150.txt -v 0.1 -f 0 -m 0235 -y 0.8`

- MAX (SMiLE-seq)

`python ./deeprec_run_sequence.py -c ./max_smile_sequence/config/config.yaml -t ./max_smile_sequence/config/tune_params.yaml -p 100 -s 0 -e 100 -q 0.5 -d ./max_smile_sequence/data/test_seq.txt -i ./max_smile_sequence/data/r0_r1_max_smile_seq_10mer_50.txt -v 0.1 -f 0 -m 0235 -y 0.8`

- MEF2B (SELEX-seq)

`python ./deeprec_run_sequence.py -c ./mef2b_sequence/config/config.yaml -t ./mef2b_sequence/config/tune_params.yaml -p 100 -s 0 -e 100 -q 0.5 -d ./mef2b_sequence/data/test_seq.txt -i ./mef2b_sequence/data/r0_r1_mef2b_selex_seq_10mer_100.txt -v 0.1 -f 0 -m 0235 -y 0.6`

- p53 (SELEX-seq)

`python ./deeprec_run_sequence.py -c ./p53_sequence/config/config.yaml -t ./p53_sequence/config/tune_params.yaml -p 100 -s 0 -e 100 -q 0.5 -d ./p53_sequence/data/test_seq.txt -i ./p53_sequence/data/r0_r1_p53_selex_seq_10mer_300.txt -v 0.1 -f 0 -m 0235 -y 0.6`


## Mismatched DNA

### DeepRec

python ./deeprec_run.py -c ./max_mismatch/config/config.yaml -t ./max_mismatch/config/tune_params.yaml -p 100 -s 0 -e 100 -q 0.5 -d ./max_mismatch/data/test_seq.txt -i ./max_mismatch/data/342434_0_source_data_3131787_pkvn7f.format.txt -v 0.1 -f 0 -m 035 -y 0.8

### DeepRec + Transfer learning

- max

run base model

`python ./deeprec_run.py -c ./max_translearn_base/config/config.yaml -t ./max_translearn_base/config/tune_params.yaml -p 100 -s 0 -e 100 -q 0.5 -d ./max_translearn_base/data/test_seq.txt -i ./max_translearn_base/data/r0_r1_max_selex_seq_10mer_150.txt -v 0.1 -f 0 -m 035 -y 0.8`

run transfer learning

`python ./deeprec_run_translearn.py -c ./max_translearn/config/config.tuned.yaml -t ./max_translearn/config/tune_params.yaml -s 0 -q 0.5 -d ./max_translearn/data/test_seq.txt -i ./max_translearn/data/342434_0_source_data_3131787_pkvn7f.format.txt -v 0.1 -f 0 -m 35 -y 0.8`

