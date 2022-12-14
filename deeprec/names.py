groove_map = {'major':4, 'minor':3}
groove_names = {'major':'M', 'minor':'m'}
channel_map = {
    'A': '[0, 0, 0, 1]',
    'D': '[0, 0, 1, 0]',
    'M': '[0, 1, 0, 0]',
    'N': '[1, 0, 0, 0]'
}
channel_map_seq = {
    'A': '[1, 0, 0, 0]',
    'C': '[0, 1, 0, 0]',
    'G': '[0, 0, 1, 0]',
    'T': '[0, 0, 0, 1]'
}


seq_map = {
    'major':{'A':{0:'A',1:'D',2:'A',3:'M'},
             'T':{0:'M',1:'A',2:'D',3:'A'},
             'C':{0:'N',1:'D',2:'A',3:'A'},        
             'G':{0:'A',1:'A',2:'D',3:'N'},
             # methylation
             'M':{0:'M',1:'D',2:'A',3:'A'},
             'g':{0:'A',1:'A',2:'D',3:'M'},
             # mismatch
             'B':{0:'A',1:'D',2:'D',3:'A'}, # AA
             'D':{0:'A',1:'D',2:'D',3:'N'}, # AC
             'E':{0:'A',1:'D',2:'A',3:'A'}, # AG
             'F':{0:'N',1:'D',2:'D',3:'A'}, # CA
             'H':{0:'N',1:'D',2:'D',3:'N'}, # CC
             'I':{0:'N',1:'D',2:'A',3:'M'}, # CT
             'J':{0:'A',1:'A',2:'D',3:'A'}, # GA
             'K':{0:'A',1:'A',2:'D',3:'D'}, # GG # or 'K':{0:'A',1:'A',2:'A',3:'D'} or reverse,
             'R':{0:'D',1:'D',2:'A',3:'A'}, # GG
             'L':{0:'A',1:'A',2:'A',3:'M'}, # GT reverse
             'O':{0:'M',1:'A',2:'D',3:'N'}, # TC
             'P':{0:'M',1:'A',2:'A',3:'A'}, # TG
             'Q':{0:'M',1:'A',2:'A',3:'M'}  # TT
    },
    'minor':{'A':{0:'A',1:'N',2:'A'},
             'T':{0:'A',1:'N',2:'A'},
             'C':{0:'A',1:'D',2:'A'},
             'G':{0:'A',1:'D',2:'A'},
             # methylation
             'M':{0:'A',1:'D',2:'A'},
             'g':{0:'A',1:'D',2:'A'},
             # mismatch
             'B':{0:'A',1:'N',2:'A'}, # AA
             'D':{0:'A',1:'N',2:'A'}, # AC # or 'D':{0:'A',1:'D',2:'A'},
             'E':{0:'A',1:'D',2:'A'}, # AG # or 'E':{0:'N',1:'D':2:'A'}, 
             'F':{0:'A',1:'N',2:'A'}, # CA # or 'F':{0:'A',1:'D',2:'A'},
             'H':{0:'A',1:'N',2:'A'}, # CC # or 'H':{0:'A',1:'D',2:'A'},
             'I':{0:'A',1:'N',2:'A'}, # CT
             'J':{0:'A',1:'D',2:'A'}, # GA # or 'J':{0:'A',1:'D',2:'N'},
             'K':{0:'A',1:'D',2:'N'}, # GG
             'R':{0:'N',1:'D',2:'A'}, # GG reverse
             'L':{0:'A',1:'D',2:'A'}, # GT
             'O':{0:'A',1:'N',2:'A'}, # TC
             'P':{0:'A',1:'D',2:'A'}, # TG
             'Q':{0:'A',1:'N',2:'A'}  # TT
    }
}
    
dispaly_map = {
    'major':{
        0:{'A':['A'], 'C':['N','M'], 'G':['A'], 'T':['M'], 'N':[], 
            # methylation
            'M':['M'], 'g':['A'], 
            # mismatch
            'B':['A'], 'D':['A'], 'E':['A'], 'F':['N'], 'H':['N'], 'I':['N'], 
            'J':['A'], 'K':['A'], 'R':['D'], 'L':['A'], 'O':['M'], 'P':['M'], 'Q':['M']},
        1:{'A':['D'], 'C':['D'], 'G':['A'], 'T':['A'], 'N':[],
            # methylation
            'M':['D'], 'g':['A'],
            # mismatch
            'B':['D'], 'D':['D'], 'E':['D'], 'F':['D'], 'H':['D'], 'I':['D'], 
            'J':['A'], 'K':['A'], 'R':['D'],'L':['A'], 'O':['A'], 'P':['A'], 'Q':['A']},
        2:{'A':['A'], 'C':['A'], 'G':['D'], 'T':['D'], 'N':[],
            # methylation
            'M':['A'], 'g':['D'],
            # mismatch
            'B':['D'], 'D':['D'], 'E':['A'], 'F':['D'], 'H':['D'], 'I':['A'], 
            'J':['D'], 'K':['D'], 'R':['A'], 'L':['A'], 'O':['D'], 'P':['A'], 'Q':['A']},
        3:{'A':['M'], 'C':['A'], 'G':['N','M'], 'T':['A'], 'N':[],
            # methylation
            'M':['A'], 'g':['M'],
            # mismatch
            'B':['A'], 'D':['N'], 'E':['A'], 'F':['A'], 'H':['N'], 'I':['M'], 
            'J':['A'], 'K':['D'], 'R':['A'], 'L':['M'], 'O':['N'], 'P':['A'], 'Q':['M']}
    },  
    'minor':{
        0:{'A':['A'], 'C':['A'], 'G':['A'], 'T':['A'], 'N':[],
            # methylation
            'M':['A'], 'g':['A'],
            # mismatch
            'B':['A'], 'D':['A'], 'E':['A'], 'F':['A'], 'H':['A'], 'I':['A'], 
            'J':['A'], 'K':['A'], 'R':['N'], 'L':['A'], 'O':['A'], 'P':['A'], 'Q':['A']},
        1:{'A':['N','D'], 'C':['D','N'], 'G':['D','N'], 'T':['N','D'], 'N':[], 
            # methylation
            'M':['D','N'], 'g':['D','N'], 
            'B':['N'], 'D':['N'], 'E':['D'], 'F':['N'], 'H':['N'], 'I':['N'], 
            # mismatch
            'J':['D'], 'K':['D'], 'R':['D'], 'L':['D'], 'O':['N'], 'P':['D'], 'Q':['N']},
        2:{'A':['A'], 'C':['A'], 'G':['A'], 'T':['A'], 'N':[],
            # methylation
            'M':['A'], 'g':['A'], 
            # mismatch 
            'B':['A'], 'D':['A'], 'E':['A'], 'F':['A'], 'H':['A'], 'I':['A'], 
            'J':['A'], 'K':['N'], 'R':['A'], 'L':['A'], 'O':['A'], 'P':['A'], 'Q':['A']}
    }
}

dna_onehot_encode = {
    'A': [1,0,0,0],
    'C': [0,1,0,0],
    'G': [0,0,1,0],
    'T': [0,0,0,1],
    'N': [0.25,0.25,0.25,0.25],
    'a': [1,0,0,0],
    'c': [0,1,0,0],
    'g': [0,0,1,0],
    't': [0,0,0,1],
    'n': [0.25,0.25,0.25,0.25],
    
    # others (not supported, making it 'N'
    'B': [0.25,0.25,0.25,0.25], # AA
    'D': [0.25,0.25,0.25,0.25], # AC
    'E': [0.25,0.25,0.25,0.25], # AG
    'F': [0.25,0.25,0.25,0.25], # CA
    'H': [0.25,0.25,0.25,0.25], # CC
    'I': [0.25,0.25,0.25,0.25], # CT
    'J': [0.25,0.25,0.25,0.25], # GA
    'K': [0.25,0.25,0.25,0.25], # GG [AADD]
    'R': [0.25,0.25,0.25,0.25], # GG [DDAA]
    'L': [0.25,0.25,0.25,0.25], # GT
    'O': [0.25,0.25,0.25,0.25], # TC
    'P': [0.25,0.25,0.25,0.25], # TG
    'Q': [0.25,0.25,0.25,0.25]  # TT
}

hbond_major_encode = {
    'A': [0,0,0,1,0,0,1,0,0,0,0,1,0,1,0,0],
    'C': [1,0,0,0,0,0,1,0,0,0,0,1,0,0,0,1],
    'G': [0,0,0,1,0,0,0,1,0,0,1,0,1,0,0,0],
    'T': [0,1,0,0,0,0,0,1,0,0,1,0,0,0,0,1],
    'N': [0.25,0.25,0.25,0.25,0.25,0.25,0.25,0.25,0.25,0.25,0.25,0.25,0.25,0.25,0.25,0.25],
    'a': [0,0,0,1,0,0,1,0,0,0,0,1,0,1,0,0],
    'c': [1,0,0,0,0,0,1,0,0,0,0,1,0,0,0,1],
    't': [0,1,0,0,0,0,0,1,0,0,1,0,0,0,0,1],
    'n': [0.25,0.25,0.25,0.25,0.25,0.25,0.25,0.25,0.25,0.25,0.25,0.25,0.25,0.25,0.25,0.25],
    # methylation    
    'M': [0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,1],
    'g': [0,0,0,1,0,0,0,1,0,0,1,0,0,1,0,0],
    # mismatch
    'B': [0,0,0,1,0,0,1,0,0,0,1,0,0,0,0,1], # AA
    'D': [0,0,0,1,0,0,1,0,0,0,1,0,1,0,0,0], # AC
    'E': [0,0,0,1,0,0,1,0,0,0,0,1,0,0,0,1], # AG
    'F': [1,0,0,0,0,0,1,0,0,0,1,0,0,0,0,1], # CA
    'H': [1,0,0,0,0,0,1,0,0,0,1,0,1,0,0,0], # CC
    'I': [1,0,0,0,0,0,1,0,0,0,0,1,0,1,0,0], # CT
    'J': [0,0,0,1,0,0,0,1,0,0,1,0,0,0,0,1], # GA
    'K': [0,0,0,1,0,0,0,1,0,0,1,0,0,0,1,0], # GG [AADD]
    'R':[0,0,1,0,0,0,1,0,0,0,0,1,0,0,0,1], # GG [DDAA]
    'L': [0,0,0,1,0,0,0,1,0,0,0,1,0,1,0,0], # GT
    'O': [0,1,0,0,0,0,0,1,0,0,1,0,1,0,0,0], # TC
    'P': [0,1,0,0,0,0,0,1,0,0,0,1,0,0,0,1], # TG
    'Q': [0,1,0,0,0,0,0,1,0,0,0,1,0,1,0,0]  # TT
}

hbond_minor_encode = {
    'A': [0,0,0,1,1,0,0,0,0,0,0,1],
    'C': [0,0,0,1,0,0,1,0,0,0,0,1],
    'G': [0,0,0,1,0,0,1,0,0,0,0,1],
    'T': [0,0,0,1,1,0,0,0,0,0,0,1],
    'N': [0.25,0.25,0.25,0.25,0.25,0.25,0.25,0.25,0.25,0.25,0.25,0.25],
    'a': [0,0,0,1,1,0,0,0,0,0,0,1],
    'c': [0,0,0,1,0,0,1,0,0,0,0,1],
    't': [0,0,0,1,1,0,0,0,0,0,0,1],
    'n': [0.25,0.25,0.25,0.25,0.25,0.25,0.25,0.25,0.25,0.25,0.25,0.25],
    # methylation
    'M': [0,0,0,1,0,0,1,0,0,0,0,1],
    'g': [0,0,0,1,0,0,1,0,0,0,0,1],
    # mismatch
    'B': [0,0,0,1,1,0,0,0,0,0,0,1], # AA
    'D': [0,0,0,1,1,0,0,0,0,0,0,1], # AC
    'E': [0,0,0,1,0,0,1,0,0,0,0,1], # AG
    'F': [0,0,0,1,1,0,0,0,0,0,0,1], # CA
    'H': [0,0,0,1,1,0,0,0,0,0,0,1], # CC
    'I': [0,0,0,1,1,0,0,0,0,0,0,1], # CT
    'J': [0,0,0,1,0,0,1,0,0,0,0,1], # GA
    'K': [0,0,0,1,0,0,1,0,1,0,0,0], # GG [ADN]
    'R':[1,0,0,0,0,0,1,0,0,0,0,1], # GG [NDA]
    'L': [0,0,0,1,0,0,1,0,0,0,0,1], # GT
    'O': [0,0,0,1,1,0,0,0,0,0,0,1], # TC
    'P': [0,0,0,1,0,0,1,0,0,0,0,1], # TG
    'Q': [0,0,0,1,1,0,0,0,0,0,0,1]  # TT
}

revcompl_map = {
    'A':'T',
    'C':'G',
    'G':'C',
    'T':'A',
    'N':'N',
    # methylation
    'M':'g',
    'g':'M',
    # mismatch    
    'B':'B', # AA:AA
    'D':'F', # CA:AC
    'E':'J', # GA:AG
    'F':'D', # AC:CA
    'H':'H', # CC:CC
    'I':'O', # CT:TC
    'J':'E', # AG:GA
    'K':'R', # GG:GG
    'R':'K', # GG:GG
    'L':'P', # GT:TG
    'O':'I', # TC:CT
    'P':'L', # TG:GT
    'Q':'Q', # TT:TT
}
    
results_column = ['seq', 'type', 'h_pos',
                    's_pos', 'channel', 'delta', 'sem']
results_column_seq = ['seq', 's_pos', 'channel', 'delta', 'sem']
seq_letters     = ['A','C','G','T','N','M','g','B','D','E','F','H','I','J','K','R','L','O','P','Q']
seq_letters_rev = ['T','G','C','A','N','g','M','B','F','J','D','H','O','E','R','K','P','I','L','Q']
seq_letters_short = ['A','C','G','T']
pc_letters = ['N','M','D','A']
results_tune_column = ['trail_idx','cross_val_idx','params',
                       'loss','val_loss','r_squared','val_r_squared']

def generate_names_1d(seq_len):
    """"""
    names = []
    seq_idx = range(1,seq_len+1)
    sig_idx = ['N', 'M', 'D', 'A']
    for i in sig_idx:
        for k in range(1,5):
                for l in seq_idx:
                    name = str(l)
                    names.append(name+"_M_"+str(k)+"_"+i)
        for k in range(1,4):
                for l in seq_idx:
                    name = str(l)
                    names.append(name+"_m_"+str(k)+"_"+i)
    return names

def generate_names_1d_seq(seq_len):
    names = []
    seq_idx = range(1,seq_len+1)
    bp_idx = ['A', 'C', 'G', 'T']
    for i in bp_idx:
        for l in seq_idx:
            name = str(l)
            names.append(name+'_'+str(i))
    return names

def get_reverse_complement(seq):
    reverse_complement = "".join(revcompl_map.get(base, base) for base in reversed(seq))
    return reverse_complement
