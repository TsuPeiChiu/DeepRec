import os
import yaml
import itertools
import random as ra
import numpy as np

class Params(object):
    """ """    
    def __init__(self, config, random_state=None, mode='pc'):
        if random_state != None:
            os.environ['PYTHONHASHSEED']=str(random_state)
            np.random.seed(random_state)
            ra.seed(random_state)
        self.random_state = random_state
        self.config = config
        self.mode = mode
        
        with open(self.config) as file:
            c = yaml.full_load(file)            
            self.train = c['input']['train']
            self.val = c['input']['val']
            self.test = c['input']['test']
            self.output_path = c['output']['path']
            self.model_tune = c['output']['model_tune']
            self.model_selected = c['output']['model_selected']
            self.model_logos = c['output']['model_logos']
            self.model_logos_results = c['output']['model_logos_results']
            self.model_performances = c['output']['model_performances']
            self.test_predictions = c['output']['test_predictions']  
            self.optimizer = c['optimizer']
            self.optimizer_params = {
                'lr': float(c['optimizer_params']['lr'])
            }
            self.loss = c['loss']
            self.nb_epoch = c['nb_epoch']
            self.batch_size = c['batch_size']  
            if self.mode=='seq':
                self.onehot_seq = {
                    'nb_filter_1': int(c['cartridges']['onehot_seq']['nb_filter'][0]),
                    'filter_len_1': int(c['cartridges']['onehot_seq']['filter_len'][0]),
                    'filter_hei_1': int(c['cartridges']['onehot_seq']['filter_hei'][0]),
                    'pool_len_1': int(c['cartridges']['onehot_seq']['pool_len'][0]),
                    'pool_hei_1': int(c['cartridges']['onehot_seq']['pool_hei'][0]),
                    'activation_1': str(c['cartridges']['onehot_seq']['activation'][0]),
                    'alpha_1': float(c['cartridges']['onehot_seq']['alpha'][0]),
                    'l1_ratio_1': float(c['cartridges']['onehot_seq']['l1_ratio'][0])
                }
            else:
                self.hbond_major = {
                    'nb_filter_1': int(c['cartridges']['hbond_major']['nb_filter'][0]),
                    'filter_len_1': int(c['cartridges']['hbond_major']['filter_len'][0]),
                    'filter_hei_1': int(c['cartridges']['hbond_major']['filter_hei'][0]),
                    'pool_len_1': int(c['cartridges']['hbond_major']['pool_len'][0]),
                    'pool_hei_1': int(c['cartridges']['hbond_major']['pool_hei'][0]),
                    'activation_1': str(c['cartridges']['hbond_major']['activation'][0]),                                
                    'alpha_1': float(c['cartridges']['hbond_major']['alpha'][0]),
                    'l1_ratio_1': float(c['cartridges']['hbond_major']['l1_ratio'][0]),

                    'nb_filter_2': int(c['cartridges']['hbond_major']['nb_filter'][1]),
                    'filter_len_2': int(c['cartridges']['hbond_major']['filter_len'][1]),
                    'filter_hei_2': int(c['cartridges']['hbond_major']['filter_hei'][1]),
                    'pool_len_2': int(c['cartridges']['hbond_major']['pool_len'][1]),
                    'pool_hei_2': int(c['cartridges']['hbond_major']['pool_hei'][1]),
                    'activation_2': str(c['cartridges']['hbond_major']['activation'][1]),                                
                    'alpha_2': float(c['cartridges']['hbond_major']['alpha'][1]),
                    'l1_ratio_2': float(c['cartridges']['hbond_major']['l1_ratio'][1])          
                }            
                self.hbond_minor = {
                    'nb_filter_1': int(c['cartridges']['hbond_minor']['nb_filter'][0]),
                    'filter_len_1': int(c['cartridges']['hbond_minor']['filter_len'][0]),
                    'filter_hei_1': int(c['cartridges']['hbond_minor']['filter_hei'][0]),
                    'pool_len_1': int(c['cartridges']['hbond_minor']['pool_len'][0]),
                    'pool_hei_1': int(c['cartridges']['hbond_minor']['pool_hei'][0]),
                    'activation_1': str(c['cartridges']['hbond_minor']['activation'][0]),                                
                    'alpha_1': float(c['cartridges']['hbond_minor']['alpha'][0]),
                    'l1_ratio_1': float(c['cartridges']['hbond_minor']['l1_ratio'][0]),
                
                    'nb_filter_2': int(c['cartridges']['hbond_minor']['nb_filter'][1]),
                    'filter_len_2': int(c['cartridges']['hbond_minor']['filter_len'][1]),
                    'filter_hei_2': int(c['cartridges']['hbond_minor']['filter_hei'][1]),
                    'pool_len_2': int(c['cartridges']['hbond_minor']['pool_len'][1]),
                    'pool_hei_2': int(c['cartridges']['hbond_minor']['pool_hei'][1]),
                    'activation_2': str(c['cartridges']['hbond_minor']['activation'][1]),                                
                    'alpha_2': float(c['cartridges']['hbond_minor']['alpha'][1]),
                    'l1_ratio_2': float(c['cartridges']['hbond_minor']['l1_ratio'][1])
                }
            
            self.joint = {
                'nb_hidden': int(c['joint']['nb_hidden']),
                'activation': str(c['joint']['activation']),
                'alpha': float(c['joint']['alpha']),
                'l1_ratio': float(c['joint']['l1_ratio']),
                'drop_out': int(c['joint']['drop_out'])
            }            
            self.target = {
                'activation': str(c['target']['activation'])
            }

    def get_grid(self, config_tune, nb_params=50):
        """ """
        with open(config_tune) as f:
            c = yaml.full_load(f)            
            g, lr, nb_epoch, batch_size = [], [], [], []
            cartridges_alpha_1, cartridges_l1_ratio_1 = [], []
            cartridges_alpha_2, cartridges_l1_ratio_2 = [], []
            cartridges_filter_len_1, cartridges_filter_len_2 = [], []
            cartridges_nb_filter_1, cartridges_nb_filter_2 = [], []
            joint_nb_hidden, joint_alpha, joint_l1_ratio, joint_drop_out = [], [], [], []
            
            for i in c['lr']: lr.append({'lr':i})
            for i in c['nb_epoch']: nb_epoch.append({'nb_epoch':i})
            for i in c['batch_size']: batch_size.append({'batch_size':i})

            for i in c['cartridges_alpha_1']:
                cartridges_alpha_1.append({'cartridges_alpha_1':i})
            for i in c['cartridges_alpha_2']: 
                cartridges_alpha_2.append({'cartridges_alpha_2':i})
            
            for i in c['cartridges_l1_ratio_1']: 
                cartridges_l1_ratio_1.append({'cartridges_l1_ratio_1':i})
            for i in c['cartridges_l1_ratio_2']: 
                cartridges_l1_ratio_2.append({'cartridges_l1_ratio_2':i})

            for i in c['cartridges_filter_len_1']:    
                cartridges_filter_len_1.append({'cartridges_filter_len_1':i})  
            for i in c['cartridges_filter_len_2']:    
                cartridges_filter_len_2.append({'cartridges_filter_len_2':i})
            for i in c['cartridges_nb_filter_1']:    
                cartridges_nb_filter_1.append({'cartridges_nb_filter_1':i}) 
            for i in c['cartridges_nb_filter_2']:    
                cartridges_nb_filter_2.append({'cartridges_nb_filter_2':i})                               
            for i in c['joint_nb_hidden']: 
                joint_nb_hidden.append({'joint_nb_hidden':i})
            for i in c['joint_alpha']: 
                joint_alpha.append({'joint_alpha':i})
            for i in c['joint_l1_ratio']:
                joint_l1_ratio.append({'joint_r1_ratio':i})
            for i in c['joint_drop_out']: 
                joint_drop_out.append({'joint_drop_out':i})
                                
            if c['lr']!=[]: g+=[lr]
            if c['nb_epoch']!=[]: g+=[nb_epoch]
            if c['batch_size']!=[]: g+=[batch_size]
            if c['cartridges_alpha_1']!=[]: g+=[cartridges_alpha_1]
            if c['cartridges_alpha_2']!=[]: g+=[cartridges_alpha_2]                                                
            if c['cartridges_l1_ratio_1']!=[]: g+=[cartridges_l1_ratio_1]
            if c['cartridges_l1_ratio_2']!=[]: g+=[cartridges_l1_ratio_2]            
            if c['cartridges_filter_len_1']!=[]: g+=[cartridges_filter_len_1]
            if c['cartridges_filter_len_2']!=[]: g+=[cartridges_filter_len_2]            
            if c['cartridges_nb_filter_1']!=[]: g+=[cartridges_nb_filter_1]
            if c['cartridges_nb_filter_2']!=[]: g+=[cartridges_nb_filter_2]
            if c['joint_alpha']!=[]: g+=[joint_alpha]
            if c['joint_l1_ratio']!=[]: g+=[joint_l1_ratio]
            if c['joint_drop_out']!=[]: g+=[joint_drop_out]
            if c['joint_nb_hidden']!=[]: g+=[joint_nb_hidden]
            
            all_comb = list(itertools.product(*g))
            selected_comb = []
            idx = ra.sample(range(len(all_comb)), nb_params)
            for i in idx: 
                selected_comb.append(all_comb[i])
        return selected_comb       
        
    def update(self, item):
        """ """
        for i in item:
            for k, v in i.items():
                if k=='lr': 
                    self.optimizer_params['lr']=float(i['lr'])
                if k=='nb_epoch': 
                    self.nb_epoch=int(i['nb_epoch'])
                if k=='batch_size': 
                    self.batch_size=int(i['batch_size'])
                if self.mode=='seq':
                    if k=='cartridges_alpha_1':
                        self.onehot_seq['alpha_1']=float(i['cartridges_alpha_1'])
                    if k=='cartridges_l1_ratio_1':
                        self.onehot_seq['l1_ratio_1']=float(i['cartridges_l1_ratio_1'])
                    if k=='cartridges_filter_len_1':
                        self.onehot_seq['filter_len_1']= \
                            int(i['cartridges_filter_len_1'])
                    if k=='cartridges_nb_filter_1':
                        self.onehot_seq['nb_filter_1']= \
                            int(i['cartridges_nb_filter_1'])
                else:
                    if k=='cartridges_alpha_1': 
                        self.hbond_major['alpha_1']=float(i['cartridges_alpha_1'])
                        self.hbond_minor['alpha_1']=float(i['cartridges_alpha_1'])
                    if k=='cartridges_alpha_2': 
                        self.hbond_major['alpha_2']=float(i['cartridges_alpha_2'])
                        self.hbond_minor['alpha_2']=float(i['cartridges_alpha_2'])
                    if k=='cartridges_l1_ratio_1': 
                        self.hbond_major['l1_ratio_1']=float(i['cartridges_l1_ratio_1'])
                        self.hbond_minor['l1_ratio_1']=float(i['cartridges_l1_ratio_1'])
                    if k=='cartridges_l1_ratio_2': 
                        self.hbond_major['l1_ratio_2']=float(i['cartridges_l1_ratio_2'])
                        self.hbond_minor['l1_ratio_2']=float(i['cartridges_l1_ratio_2'])                                                                               
                    if k=='cartridges_filter_len_1': 
                        self.hbond_major['filter_len_1']= \
                            int(i['cartridges_filter_len_1'])
                        self.hbond_minor['filter_len_1']= \
                            int(i['cartridges_filter_len_1'])                              
                    if k=='cartridges_filter_len_2': 
                        self.hbond_major['filter_len_2']= \
                            int(i['cartridges_filter_len_2'])
                        self.hbond_minor['filter_len_2']= \
                            int(i['cartridges_filter_len_2'])
                    if k=='cartridges_nb_filter_1': 
                        self.hbond_major['nb_filter_1']= \
                            int(i['cartridges_nb_filter_1'])
                        self.hbond_minor['nb_filter_1']= \
                            int(i['cartridges_nb_filter_1'])                       
                    if k=='cartridges_nb_filter_2': 
                        self.hbond_major['nb_filter_2']= \
                            int(i['cartridges_nb_filter_2'])
                        self.hbond_minor['nb_filter_2']= \
                            int(i['cartridges_nb_filter_2'])      
                if k=='joint_nb_hidden': 
                    self.joint['nb_hidden']=int(i['joint_nb_hidden'])
                if k=='joint_alpha': 
                    self.joint['alpha']=float(i['joint_alpha'])
                if k=='joint_l1_ratio':
                    self.joint['l1_ratio']=float(i['joint_l1_ratio'])
                if k=='joint_drop_out': 
                    self.joint['drop_out']=float(i['joint_drop_out'])
    
    def save(self, outfile):
        """ """
        c = {}
        c['input'] = {'train': self.train, 'val': self.val, 'test': self.test}
        c['output'] = {'path': self.output_path,
                         'model_tune': self.model_tune,
                         'model_selected': self.model_selected, 
                         'model_logos': self.model_logos,
                         'model_logos_results': self.model_logos_results,
                         'model_performances': self.model_performances,
                         'test_predictions': self.test_predictions}
        c['optimizer'] = self.optimizer
        c['optimizer_params'] = {'lr': self.optimizer_params['lr']}
        c['loss'] = self.loss
        c['nb_epoch'] = self.nb_epoch
        c['batch_size'] = self.batch_size 
        if self.mode=='seq':
            c['cartridges'] = {
                    'onehot_seq':{
                    'nb_filter': [self.onehot_seq['nb_filter_1']],
                    'filter_len': [self.onehot_seq['filter_len_1']],
                    'filter_hei': [self.onehot_seq['filter_hei_1']],
                    'pool_len': [self.onehot_seq['pool_len_1']],
                    'pool_hei': [self.onehot_seq['pool_hei_1']],
                    'activation': [self.onehot_seq['activation_1']],
                    'alpha': [self.onehot_seq['alpha_1']],
                    'l1_ratio': [self.onehot_seq['l1_ratio_1']]}}

        else:
            c['cartridges'] = {
                'hbond_major':{
                        'nb_filter': [self.hbond_major['nb_filter_1'],
                                      self.hbond_major['nb_filter_2']],
                        'filter_len': [self.hbond_major['filter_len_1'],
                                       self.hbond_major['filter_len_2']],                       
                        'filter_hei': [self.hbond_major['filter_hei_1'],
                                         self.hbond_major['filter_hei_2']],
                        'pool_len': [self.hbond_major['pool_len_1'],
                                     self.hbond_major['pool_len_2']],
                        'pool_hei': [self.hbond_major['pool_hei_1'],
                                       self.hbond_major['pool_hei_2']],
                        'activation': [self.hbond_major['activation_1'],
                                         self.hbond_major['activation_2']],
                        'alpha': [self.hbond_major['alpha_1'],
                               self.hbond_major['alpha_2']],
                        'l1_ratio': [self.hbond_major['l1_ratio_1'],
                               self.hbond_major['l1_ratio_2']]},
                                                                        
                'hbond_minor':{
                        'nb_filter': [self.hbond_minor['nb_filter_1'],
                                      self.hbond_minor['nb_filter_2']],
                        'filter_len': [self.hbond_minor['filter_len_1'],
                                       self.hbond_minor['filter_len_2']],
                        'filter_hei': [self.hbond_minor['filter_hei_1'],
                                       self.hbond_minor['filter_hei_2']],
                        'pool_len': [self.hbond_minor['pool_len_1'],
                                     self.hbond_minor['pool_len_2']],
                        'pool_hei': [self.hbond_minor['pool_hei_1'],
                                     self.hbond_minor['pool_hei_2']],
                        'activation': [self.hbond_minor['activation_1'],
                                       self.hbond_minor['activation_2']],
                        'alpha': [self.hbond_minor['alpha_1'],
                               self.hbond_minor['alpha_2']],
                        'l1_ratio': [self.hbond_minor['l1_ratio_1'],
                               self.hbond_minor['l1_ratio_2']]}}
                       
        c['joint'] = {'nb_hidden': self.joint['nb_hidden'],
                         'activation': self.joint['activation'],
                         'alpha': self.joint['alpha'],
                         'l1_ratio': self.joint['l1_ratio'],
                         'drop_out': self.joint['drop_out']}
        c['target'] = {'activation': self.target['activation']}
        
        with open(outfile, 'w') as f:
            yaml.dump(c, f)
          
