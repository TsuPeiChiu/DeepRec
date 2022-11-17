import os
import random as ra
import numpy as np
import deeprec.names as na
import deeprec.utils.file_hd5 as hf
from sklearn.model_selection import train_test_split

class Encoders(object):
    """ """
    def __init__(self):
        pass
    
    def prepare_data(self, infile, testfile, config, test_size=0.1, sep='\t', 
                     random_state=None):
        """ """
        print('preparing data ...')
        if random_state != None:
            os.environ['PYTHONHASHSEED'] = str(random_state)
            np.random.seed(random_state)
            ra.seed(random_state)          
        seqs, resps = [], []    
        with open(infile) as f:
            for line in f:
#                items = line.strip().upper().split(sep)
                items = line.strip().split(sep)
                seq, resp = items[0], items[1]
                seq = seq + self.__revcompl(seq)
                seqs.append(seq)
                resps.append(resp)
        seqs_test, resps_test = [], []
        with open(testfile) as f:
            for line in f:
                items = line.strip().split(sep)
                if len(items)>1:
                    seq, resp = items[0], items[1]                    
                    seq = seq + self.__revcompl(seq)
                    seqs_test.append(seq)
                    resps_test.append(resp)
                else:
                    seq = items[0]
                    seq = seq + self.__revcompl(seq)
                    seqs_test.append(seq)
                    resps_test.append(0)
        seq_len = len(seqs[0])/2     
                          
        seqs_train, seqs_val, resps_train, resps_val = train_test_split(
                seqs, resps, test_size=test_size, random_state=random_state)      
        encode_seqs = {'train': seqs_train, 'val': seqs_val, 'test': seqs_test}
        encode_resps = {'train': resps_train, 'val': resps_val, 
                        'test': resps_test}
        encode_maps = {'hbond_major': hbond_major_encode,
                       'hbond_minor': hbond_minor_encode,
                       'seq_onehot': dna_onehot_encode}
        for k1, v1 in encode_seqs.items():
            for k2, v2 in encode_maps.items():
                outfile = infile.replace('.txt', '.'.join(['',k1,k2,'tmp']))
                with open(outfile, 'w') as f:
                    for seq in encode_seqs[k1]:
                        encode = self.__encode_sequence(seq, v2)
                        encode_string = ','.join(str(e) for e in encode)
                        f.write(encode_string + '\n')
            seqs_trim = [item[0:seq_len] for item in encode_seqs[k1]]    
            infile_prefix = infile.replace('.txt', '.'.join(['',k1]))
            hf.ascii_to_hd5(infile_prefix, seqs_trim, encode_resps[k1])           
        
    def __revcompl(self, s):
        """ """
        rev_s = ''.join([revcompl_map[B] for B in s][::-1])
        return rev_s                
        
    def __encode_sequence(self, sequence, encode_map, reshape_length=1):
        """ """
        encode_array  = np.asarray([], dtype = int)
        for c in sequence:
            encode_array = np.append(encode_array, encode_map[c])
        return encode_array

dna_onehot_encode = na.dna_onehot_encode
hbond_major_encode = na.hbond_major_encode
hbond_minor_encode = na.hbond_minor_encode 
revcompl_map = na.revcompl_map
