import os
import numpy as np
import random as ra
import pandas as pd
import math as ma
import deeprec.names as na
import deeprec.params as pr
import deeprec.visualizers as vi

class DeepRecExplainer(object):
    """"""
    def __init__(self, config, models, xs, seqs, random_state=None, 
                y_lim=1.0, mode='pc'):        
        """"""
        if random_state != None:
            os.environ['PYTHONHASHSEED'] = str(random_state)
            np.random.seed(random_state)
            ra.seed(random_state)            
        self.models = models
        self.xs = xs
        self.seqs = seqs
        self.random_state = random_state
        self.mode = mode
        self.params = pr.Params(config, self.random_state, self.mode)
        if self.mode=='seq':
            self.seq_len = int(len(self.xs[0])/4) # with padding
            self.results_column = na.results_column_seq
        else:
            self.seq_len = int(len(self.xs[0])/28) # with padding
            self.results_column = na.results_column
        self.pad_len = self.seq_len-2*len(self.seqs[0])
        self.names_1d = na.generate_names_1d(self.seq_len)
        self.names_1d_seq = na.generate_names_1d_seq(self.seq_len)
        self.groove_map = na.groove_map
        self.groove_names = na.groove_names
        self.channel_map = na.channel_map
        self.channel_map_seq = na.channel_map_seq
        self.dna_onehot_encode = na.dna_onehot_encode
        self.seq_map = na.seq_map
        self.dispaly_map = na.dispaly_map
        self.seq_letters = na.seq_letters
        self.seq_letters_rev = na.seq_letters_rev
        self.pc_letters = na.pc_letters
        self.groove_map = na.groove_map
        self.y_lim = y_lim

    def plot_logos(self):
        """"""
        if len(self.seqs)>50:
            print ('too many testing sequences:' + str(self.seq_len) + 
                   'for plotting')
            exit()
            
        test_predictions_file = os.path.join(self.params.output_path, 
                                             self.params.test_predictions)
        with open(test_predictions_file, 'w') as tpf:        
            for self.x, self.seq in zip(self.xs, self.seqs):
                self.samples, self.samples_name = self.__perturb()
                self.ys = self.__predict()
                model_logos_results_file = \
                    self.params.model_logos_results.replace('.tsv','.'+ \
                                                            self.seq+'.tsv')
                model_logos_results_file = os.path.join(self.params.output_path, 
                                             model_logos_results_file)
                results = self.__calculate_logos(model_logos_results_file)
                logos_file = self.params.model_logos.replace('.png',
                                                        '.'+self.seq+'.png')
                outfile = os.path.join(self.params.output_path, logos_file)  
                vi.plot_logos(outfile, self.seq, results, self.y_lim, self.mode)
                tpf.write(self.samples_name[0] + '\t' + 
                          str(np.mean(self.ys[0])) + '\n')        
        
    def __perturb(self):
        """"""
        samples, samples_name = [], []
        p = self.x.copy()
        s = self.seq
        s_padded = ''.join([s,'N'*self.pad_len,na.get_reverse_complement(s)])
        
        samples.append(p)
        samples_name.append(s)

        if self.mode == 'seq':
            for i in range(len(self.seq)):
                i_rev = self.seq_len-i-1

                # nullify the the bp
                p_null = self.x.copy()
                s_null = self.seq + '_'.join(['',str(i+1),'null'])

                for n in na.seq_letters_short:
                    key = '_'.join([str(i+1),n])
                    idx = self.names_1d_seq.index(key)
                    key_rev = '_'.join([str(i_rev+1),n])
                    idx_rev = self.names_1d_seq.index(key_rev)
                    p_null[idx] = 0.25
                    p_null[idx_rev] = 0.25
                    samples.append(p_null)
                    samples_name.append(s_null)

                # add A,C,T,G to null
                for c in na.seq_letters_short:
                    p_add = p_null.copy()
                    s_add = '_'.join([s_null,c])
                    key = '_'.join([str(i+1),c])
                    idx = self.names_1d_seq.index(key)
                    key_rev = '_'.join([str(i_rev+1),na.revcompl_map[c]])
                    idx_rev = self.names_1d_seq.index(key_rev)
                    p_add[idx] = 1
                    p_add[idx_rev] = 1
                    
                    for n in na.seq_letters_short:
                        if n!=c:
                            key = '_'.join([str(i+1),n])
                            idx = self.names_1d_seq.index(key)
                            key_rev = '_'.join([str(i_rev+1),na.revcompl_map[n]])
                            idx_rev = self.names_1d_seq.index(key_rev)
                            p_add[idx] = 0
                            p_add[idx_rev] = 0
                    samples.append(p_add)
                    samples_name.append(s_add)

        else:
            for i in range(len(self.seq)):
                i_rev = self.seq_len-i-1
                for j, seq_letter in enumerate(self.seq_letters):                  
                    for g_type, nb_pos in self.groove_map.items():                        
                        for l in range(nb_pos):                            
                            # nullify the pc
                            p_null = self.x.copy()
                            s_null = self.seq + '_'.join(['',str(i+1),
                                                self.groove_names[g_type],
                                                str(l+1),'null'])
                            for pc_letter in self.pc_letters:
                                key = '_'.join([str(i+1),
                                            self.groove_names[g_type],
                                            str(l+1),pc_letter])
                                idx = self.names_1d.index(key)                   
                                key_rev = '_'.join([str(i_rev+1),
                                            self.groove_names[g_type],
                                            str(nb_pos-l),pc_letter])
                                idx_rev = self.names_1d.index(key_rev)
                                p_null[idx] = 0.25
                                p_null[idx_rev] = 0.25
                            samples.append(p_null)
                            samples_name.append(s_null)

                            # add A,D,M,N to null
                            for c in na.pc_letters:
                                p_add = p_null.copy()
                                s_add = '_'.join([s_null,c])
                                key = '_'.join([str(i+1),
                                        self.groove_names[g_type],
                                        str(l+1),c])
                                idx = self.names_1d.index(key)
                                key_rev = '_'.join([str(i_rev+1),
                                                self.groove_names[g_type],
                                                str(nb_pos-l),c])
                                idx_rev = self.names_1d.index(key_rev)
                                p_add[idx] = 1
                                p_add[idx_rev] = 1

                                for n in na.pc_letters:
                                    if n!=c:
                                        key = key = '_'.join([str(i+1),
                                                self.groove_names[g_type],
                                                str(l+1),n])
                                        idx = self.names_1d.index(key)
                                        key_rev = '_'.join([str(i_rev+1),
                                                self.groove_names[g_type],
                                                str(nb_pos-l),n])
                                        idx_rev = self.names_1d.index(key_rev)
                                        p_add[idx] = 0
                                        p_add[idx_rev] = 0

                                samples.append(p_add)
                                samples_name.append(s_add)

        return samples, samples_name

    def __predict(self):
        """"""
        ys = []
        for model in self.models:
            y = model.predict(np.array(self.samples))
            ys.append(y)
        return ys

    def __calculate_logos(self, outfile):
        """"""
        results = pd.DataFrame(columns=self.results_column)
        if self.mode=='seq':
            for s_pos in range(len(self.seq)):
                #for c in na.seq_letters_short:
                for c in [self.seq[s_pos]]:
                #c = self.seq[s_pos]


                    diffs_means, diffs_sems = [], []
                    key_ref = '_'.join([self.seq,str(s_pos+1),'null',c])
                    key_null = '_'.join([self.seq,str(s_pos+1),'null'])
        
                    diffs_mean, diffs_sem = self.__calculate_diffs(key_ref, key_null)
                    diffs_means.append(diffs_mean)
                    diffs_sems.append(diffs_sem)
        
                    pc_mean, pc_sem = self.__calculate_mean_sem(
                                          diffs_means, diffs_sems)
                    results = results.append({'seq': self.seq,
                                    's_pos': s_pos,
                                    'channel': self.channel_map_seq[c],
                                    'delta': pc_mean,
                                    'sem': pc_sem}, ignore_index=True)
            results.to_csv(outfile, sep='\t')    
        else:
            for s_pos in range(len(self.seq)):
                for g_type, nb_pos in self.groove_map.items():
                    for h_pos in range(nb_pos):
                        for pc_type, pc_code in self.channel_map.items():                                                                     
                            if pc_type in self.dispaly_map[g_type][h_pos] \
                                                        [self.seq[s_pos]]:                           
                                if pc_type in self.seq_map[g_type] \
                                                        [self.seq[s_pos]] \
                                                        [h_pos]:                                                                                                                        
                                    diffs_means, diffs_sems = [], []
                                    key_ref = '_'.join([self.seq, 
                                                    str(s_pos+1),
                                                    self.groove_names[g_type],
                                                    str(h_pos+1),'null',
                                                    pc_type])
                                    key_null = '_'.join([self.seq, 
                                                     str(s_pos+1),
                                                     self.groove_names[g_type],
                                                     str(h_pos+1),'null'])                                
                                else:
                                    diffs_means, diffs_sems = [], []
                                    seq_pc_type = self.seq_map[g_type] \
                                                [self.seq[s_pos]] \
                                                [h_pos]                                
                                    key_ref = '_'.join([self.seq, 
                                                     str(s_pos+1),
                                                     self.groove_names[g_type],
                                                     str(h_pos+1), 'null'])
#                                   key_ref = '_'.join([self.seq, 
#                                                     str(s_pos+1),
#                                                     self.groove_names[g_type],
#                                                     str(h_pos+1), 'null',
#                                                     pc_type])
    
                                    key_null = '_'.join([self.seq, 
                                                    str(s_pos+1),
                                                    self.groove_names[g_type],
                                                    str(h_pos+1), 'null',
                                                    seq_pc_type])                                                            
                                
                                diffs_mean, diffs_sem = \
                                    self.__calculate_diffs(key_ref, key_null)                                                                        
                                diffs_means.append(diffs_mean)
                                diffs_sems.append(diffs_sem)                                                                                                                                                   
                                pc_mean, pc_sem = self.__calculate_mean_sem(
                                        diffs_means, diffs_sems)
                                results = results.append({'seq': self.seq,
                                    'type': g_type, 
                                    'h_pos': h_pos, 
                                    's_pos': s_pos, 
                                    'channel': self.channel_map[pc_type], 
                                    'delta': pc_mean,
                                    'sem': pc_sem}, ignore_index=True)                                                                                                                                                   
                results.to_csv(outfile, sep='\t')
        return results
    
    def __calculate_diffs(self, key_ref, key_null):
        """
        difference between models
        """
        if self.mode=='seq':
            idx_ref = self.samples_name.index(key_ref)
            idx_null = self.samples_name.index(key_null)
        else:
            idx_ref = self.samples_name.index(key_ref)
            idx_null = self.samples_name.index(key_null)     
        diffs = []
        for i in range(len(self.ys)):
            val_ref = self.ys[i][idx_ref]
            val_null = self.ys[i][idx_null]            
            if val_null!=0:
                dddG = ma.log(val_ref)-ma.log(val_null)
            else:
                print(val_ref)
                dddG = ma.log(val_ref)-ma.log(0.000001)            
            diffs.append(dddG)
        
        """
        if key_null.find('_m_2_')!=-1:
            print(key_ref)
            print(key_null)
            print(self.ys[0][idx_ref])
            print(self.ys[0][idx_null])
            print('')
        """
            
        diffs_mean = np.mean(diffs)
        diffs_std = np.std(diffs)
        diffs_sem = diffs_std/np.sqrt(len(diffs))        
        return diffs_mean, diffs_sem
    
    def __calculate_mean_sem(self, diffs_means, diffs_sems):
        pc_mean = np.mean(diffs_means)        
        pc_sem = np.sqrt(np.sum(np.square(diffs_sems)))
        return pc_mean, pc_sem
            
