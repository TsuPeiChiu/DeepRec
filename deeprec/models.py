import os, gc, math
import numpy as np
import random as ra
import tensorflow as tf
import tensorflow.keras.callbacks as cb
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
import deeprec.params as pr
import deeprec.nets as ne
import deeprec.names as na
import deeprec.metrics as me
import deeprec.utils.file_utils as fut
from keras import backend as K

class DeepRecModel(object):
    """ """
    def __init__(self, config, h5_file=None, 
                 random_state=None, input_data=None, mode='pc'):
        """ """
        if random_state != None:
            os.environ['PYTHONHASHSEED'] = str(random_state)
            np.random.seed(random_state)
            ra.seed(random_state)

        self.random_state = random_state
        self.config = config
        self.mode = mode

        if self.mode=='seq':
            self.params = pr.Params(self.config, self.random_state, 'seq')
        else:
            self.params = pr.Params(self.config, self.random_state)
        
        if not os.path.exists(self.params.output_path):
            os.mkdir(self.params.output_path)
        
        if input_data==None:
            self.__prepare_input(self.mode)
        else:
            self.train_x = input_data['train_x']
            self.train_y = input_data['train_y']
            self.train_seqs = input_data['train_seqs']
            self.seq_len = input_data['seq_len']        
            self.val_x = input_data['val_x']
            self.val_y  = input_data['val_y']
            self.val_seqs =input_data['val_seqs'] 
            self.test_x = input_data['test_x']
            self.test_y  = input_data['test_y']
            self.test_seqs =input_data['test_seqs']
        
        self.__build(h5_file)
                    
    def tune(self, config_tune, nb_params=50, verbose=True):
        """ """
        self.nb_params = nb_params
        self.params_tune = self.params.get_grid(config_tune, nb_params)      
        p = pd.DataFrame(columns=na.results_tune_column)
        p.to_csv(os.path.join(self.params.output_path, 
                               self.params.model_tune), 
                        sep='\t', header='column_names', index=False)        
        for i in range(len(self.params_tune)):
            self.params.update(self.params_tune[i])
            self.__build(None)
            self.fit(is_tune=True, idx_params_tune=i, verbose=verbose)
            
        p = pd.read_csv(os.path.join(self.params.output_path,
                                      self.params.model_tune), sep='\t')
        idx_best = p.groupby('trail_idx')['val_r_squared'].mean().idxmax()-1
        print("the best setting:" + str(idx_best+1) + ":" + 
              str(self.params_tune[idx_best]))                
        self.params.update(self.params_tune[idx_best])
        
        config_tuned = '.'.join([self.config.replace('.yaml',''),
                                 'tuned','yaml'])        
        self.params.save(config_tuned)                
        self.__build(None)
        
        return config_tuned
                        
    def fit(self, is_tune=False, idx_params_tune=None, verbose=True):
        """ """
        callback_early_stopping = cb.EarlyStopping(monitor='loss', patience=5)
        def lr_exp_decay(epoch, lr):
            return self.params.optimizer_params['lr']*math.exp(-0.01*epoch)
        
        if is_tune:
            kf = KFold(n_splits=3)
            idx_cv = 0
            for idx_train, idx_val in kf.split(self.train_x):                
                print('tuning parameters ' + str(idx_params_tune+1) + '/' + 
                      str(self.nb_params) + ':' + str(idx_cv+1) + '/3 ...')
                
                history = self.model.fit(self.train_x[idx_train],
                                         self.train_y[idx_train],
                                         validation_data=(self.train_x[idx_val]
                                             ,self.train_y[idx_val]), 
                                         epochs=self.params.nb_epoch,
                                         shuffle='batch',
                                         verbose=verbose,
                                         batch_size=self.params.batch_size, 
                                         callbacks=[callback_early_stopping])
                p = pd.DataFrame({'trail_idx': idx_params_tune+1,
                                  'cross_val_idx': idx_cv+1,
                                  'params': 
                                      str(self.params_tune[idx_params_tune]),
                                  'loss': history.history['loss'][-1],
                                  'val_loss': history.history['val_loss'][-1],
                                  'r_squared': 
                                      history.history['r_squared'][-1],
                                  'val_r_squared': 
                                      history.history['val_r_squared'][-1]},
                                  index = [0], columns=na.results_tune_column)
                idx_cv+=1                                              
                with open(os.path.join(self.params.output_path,
                                       self.params.model_tune), mode='a') as f: 
                    p.to_csv(f, sep='\t', header=False, index=False)                
                del history, p
                gc.collect()                
        else:             
            history = self.model.fit(self.train_x,
                                     self.train_y, 
                                     validation_data=(self.val_x, self.val_y), 
                                     epochs=self.params.nb_epoch, 
                                     shuffle='batch',
                                     verbose=verbose,
                                     batch_size=self.params.batch_size, 
                                     callbacks=[callback_early_stopping])
#                                     cb.LearningRateScheduler(lr_exp_decay)])
            if verbose==True:
                self.__plot(history)
            
            return history.history['val_r_squared'][-1]            
            del history
            gc.collect()
    
    def predict(self, x):
        """ """
        return self.model.predict(x)
    
    def save(self, path):
        """ """
        self.model.save(path)

    def load_model(self, path):
        """ """
        self.model.load_model(path)

    def transfer_model(self):
        """ """
        self.model = ne.build_transfer_model(self.params, self.model)

    def get_x(self, target_seq):
        x, target_type, target_index = None, None, None
        data_type = ['train', 'val']
        for t in data_type:
            if t=='train':
                for i in range(len(self.train_seqs)):                                                                            
                    if self.train_seqs[i].astype(str)==target_seq:                                               
                        x, target_type, target_index = self.train_x[i], t, i
                        break
            elif t=='val':
                for i in range(len(self.val_seqs)):
                    if self.val_seqs[i].astype(str)==target_seq:                    
                        x, target_type, target_index = self.val_x[i], t, i
                        break
        print('target_seq: ' + target_seq)
        print('target_type: ' + target_type)
        print('target_index: ' + str(target_index))
        return x
    
    def __build(self, h5_file=None):
        """ """
        if h5_file is None:
            self.model = ne.build_deeprec_model(self.params, 
                                                self.seq_len,
                                                self.random_state,
                                                self.mode)
        else:
            K.clear_session()
            self.model = tf.keras.models.load_model(h5_file, 
                          custom_objects={'tf':tf, 'r_squared':me.r_squared})        

    def __plot(self, history):
        """ """
        # summarize history for loss
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()
        
        # summarize history for r_squared
        plt.plot(history.history['r_squared'])
        plt.plot(history.history['val_r_squared'])
        plt.title('model r_squared')
        plt.ylabel('r_squared')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()
                
    def __prepare_input(self, mode='pc'):
        """ """
        # train data
        train_file, train_data = fut.read_hdf(self.params.train, 1024)        
        train_x_major = np.array(train_data['hbond_major_x']) 
        train_x_minor = np.array(train_data['hbond_minor_x'])
        train_x_seq = np.array(train_data['seq_onehot_x'])
        if mode=='seq':
            train_x_con = train_x_seq
        else:
            train_x_con = np.concatenate([train_x_major, train_x_minor], axis=2)

#        patch_len = self.params.hbond_major['filter_len']
        patch_len = 10
        patch_col = np.zeros(train_x_con.shape[0]*
                             train_x_con.shape[1]*
                             train_x_con.shape[2]*
                             patch_len)
        patch_col = patch_col.reshape(train_x_con.shape[0],
                             train_x_con.shape[1],
                             train_x_con.shape[2],
                             patch_len)
        seq_len = int(train_x_con.shape[-1]/2)
        train_x_patched = np.concatenate((train_x_con[:,:,:,:seq_len], 
                                          patch_col, 
                                          train_x_con[:,:,:,seq_len:]), axis=3)
        self.train_x = train_x_patched.reshape(train_x_patched.shape[0], 
                                     train_x_patched.shape[1]*
                                     train_x_patched.shape[2]*
                                     train_x_patched.shape[3],
                                     order='C')

        self.train_y = np.array(train_data['c0_y'])
        self.train_seqs = train_data['probe_seq']
        self.seq_len = seq_len

        # val data
        val_file, val_data = fut.read_hdf(self.params.val, 1024)
        val_x_major = np.array(val_data['hbond_major_x'])
        val_x_minor = np.array(val_data['hbond_minor_x'])
        val_x_seq = np.array(val_data['seq_onehot_x'])
        if mode=='seq':
            val_x_con = val_x_seq
        else:
            val_x_con = np.concatenate([val_x_major, val_x_minor], axis=2)

#        patch_len = self.params.hbond_minor['filter_len']
        patch_len = 10
        patch_col = np.zeros(val_x_con.shape[0]*
                             val_x_con.shape[1]*
                             val_x_con.shape[2]*
                             patch_len)
        patch_col = patch_col.reshape(val_x_con.shape[0],
                             val_x_con.shape[1],
                             val_x_con.shape[2],
                             patch_len)
        seq_len = int(val_x_con.shape[-1]/2)
        val_x_patched = np.concatenate((val_x_con[:,:,:,:seq_len], 
                                        patch_col, 
                                        val_x_con[:,:,:,seq_len:]), axis=3)              
        self.val_x = val_x_patched.reshape(val_x_patched.shape[0],
                                 val_x_patched.shape[1]*
                                 val_x_patched.shape[2]*
                                 val_x_patched.shape[3], 
                                 order='C')
        self.val_y  = np.array(val_data['c0_y'])
        self.val_seqs = val_data['probe_seq']
        
        # test data
        test_file, test_data = fut.read_hdf(self.params.test, 1024)
        test_x_major = np.array(test_data['hbond_major_x'])
        test_x_minor = np.array(test_data['hbond_minor_x'])
        test_x_seq = np.array(test_data['seq_onehot_x'])
        
        if mode=='seq':
            test_x_con = test_x_seq
        else:
            test_x_con = np.concatenate([test_x_major, test_x_minor], axis=2)

#        patch_len = self.params.hbond_minor['filter_len']
        patch_len = 10
        patch_col = np.zeros(test_x_con.shape[0]*
                             test_x_con.shape[1]*
                             test_x_con.shape[2]*
                             patch_len)
        patch_col = patch_col.reshape(test_x_con.shape[0],
                             test_x_con.shape[1],
                             test_x_con.shape[2],
                             patch_len)
        seq_len = int(test_x_con.shape[-1]/2)
        test_x_patched = np.concatenate((test_x_con[:,:,:,:seq_len], 
                                        patch_col, 
                                        test_x_con[:,:,:,seq_len:]), axis=3)              
        self.test_x = test_x_patched.reshape(test_x_patched.shape[0],
                                 test_x_patched.shape[1]*
                                 test_x_patched.shape[2]*
                                 test_x_patched.shape[3], 
                                 order='C')
        
        self.test_y  = np.array(test_data['c0_y'])
        self.test_seqs = test_data['probe_seq']
        
