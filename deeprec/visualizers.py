"""
Functions for generating physicochemical logos
"""
import os
import numpy as np
import matplotlib as mpl
#mpl.use('agg')
import matplotlib.pyplot as plt
#plt.switch_backend('agg')
import seaborn as sns
from matplotlib.text import TextPath
from matplotlib.patches import PathPatch
from matplotlib.font_manager import FontProperties
from PIL import Image


def plot_logos(outfile, seq, results, y_lim=1.0, mode='pc'):
    if mode=='seq':
        plot_logos_seq(outfile, seq, results, y_lim=y_lim, mode='seq')
    else:
        plot_logos_pc(outfile, seq, results, y_lim=y_lim, mode='pc')


def plot_logos_seq(outfile, seq, results, y_lim=1.8, mode='seq'):
    """
    """
    seq_idx = 0
    sns.set(style='ticks')
    f, ax = plt.subplots(1, 1, figsize=(10,4))
    max_y = results['delta'].astype(float).abs().max()
    max_y = max_y*1.2
    
    __plot_details_seq(results, seq_idx, seq, ax, seq, max_y, y_lim)
    #__plot_details_seq(results, seq_idx, seq, ax, seq, max_y, y_lim)


    ax.set_xticklabels(labels= [z.replace ('M', 'm')  \
                    for z in list(__reverse(seq))], size=23)


    seq_len=len(seq) ######
    ax_t = ax.twiny()
    ax_t.set_xticks(np.arange(seq_len+2))
    ax_t.set_xticklabels(labels=list(' '+seq+' '), size=23)
    
   # ax_t2 = ax.twiny()
   # ax_t2.set_xticks(np.arange(seq_len+2))
   # ax_t2.set_xticklabels(labels=list(' '+seq+' '), size=23)
    
    
    #for axis in ['top','bottom','left','right']:
    #    ax_t.spines[axis].set_linewidth(0.5)
    #    ax_t2.spines[axis].set_linewidth(0.5)
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.savefig(outfile, bbox_inches='tight')
    background = Image.open(outfile)
    foreground = Image.open("../imgs/keys_logo.png")
    background.paste(foreground, (0, 0), foreground)
    background.save(outfile+".overlay.png","PNG")
    os.remove(outfile)
    os.rename(outfile+".overlay.png", outfile)
    print("model logos: " + outfile)



def plot_logos_pc(outfile, seq, results, y_lim=1.0, mode='pc'): 
    """
    The format of results is ['seq', 'type', 'h_pos', 's_pos', 
                             'channel', 'delta', 'sem'])
    """
    seq_idx = 0
    max_y = 0.8
    sns.set(style="ticks")
    f, ax = plt.subplots(4, 2, figsize=(20,16))
    max_y = results['delta'].astype(float).abs().max()
    max_y = max_y*1.2    
    for i in np.arange(4): # major
        __plot_details(results, seq_idx, seq, 'major', i, ax[i,0], seq, 
                       max_y, y_lim)
            
    for i in np.arange(3): # minor
        __plot_details(results, seq_idx, seq, 'minor', i, ax[i,1], seq, 
                       max_y, y_lim)            
    x = []
    y = []
    ax[3,1].scatter(x,y)
    ax[3,1].set_axis_off()
    ax[3,0].set_xticklabels(labels= [z.replace ('M', 'm')  \
                    for z in list(__reverse(seq))], size=23) 
    ax[2,1].set_xticklabels(labels= [z.replace ('M', 'm')  \
                    for z in list(__reverse(seq))], size=23)
    ax[0,1].set_yticklabels(labels=[])
    ax[1,1].set_yticklabels(labels=[])
    ax[2,1].set_yticklabels(labels=[])
    ax[0,1].set_ylabel('')
    ax[1,1].set_ylabel('')
    ax[2,1].set_ylabel('')

    seq_len=len(seq) ######
    
    ax_t = ax[0,0].twiny()
    ax_t.set_xticks(np.arange(seq_len+2))
    ax_t.set_xticklabels(labels=list(' '+seq+' '), size=23)                
    ax_t2 = ax[0,1].twiny()
    ax_t2.set_xticks(np.arange(seq_len+2))
    ax_t2.set_xticklabels(labels=list(' '+seq+' '), size=23)
    for axis in ['top','bottom','left','right']:
        ax_t.spines[axis].set_linewidth(0.5)            
        ax_t2.spines[axis].set_linewidth(0.5)                
    f.text(0.25, 0.93, 'Major groove', fontsize=30)
    f.text(0.64, 0.93, 'Minor groove', fontsize=30)        
    f.text(0.05, 0.80, 'Pos 1', fontsize=25, rotation=90)
    f.text(0.05, 0.61, 'Pos 2', fontsize=25, rotation=90)
    f.text(0.05, 0.41, 'Pos 3', fontsize=25, rotation=90)
    f.text(0.05, 0.22, 'Pos 4', fontsize=25, rotation=90)    
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.savefig(outfile, bbox_inches='tight')    
    background = Image.open(outfile)
    foreground = Image.open("../imgs/keys_logo.png")
    background.paste(foreground, (0, 0), foreground)
    background.save(outfile+".overlay.png","PNG")
    os.remove(outfile)
    os.rename(outfile+".overlay.png", outfile)
    print("model logos: " + outfile)
        
def __plot_details(df, seq_idx, seq, side, h_pos, ax, base, max_y, y_lim):
    """ """
    idx_h_pos = df['h_pos'] == h_pos
    idx_side = df['type'] == side    
    for i in np.arange(len(seq)):
        idx_s_pos = df['s_pos'] == i
        tmp = df[idx_h_pos & idx_side & idx_s_pos]
        tmp = tmp.sort_values(by='delta')
        # calculate the baseline
        y = 0        
        for j in np.arange(len(tmp)):
            if float(tmp.iloc[j]['delta']) < 0:
                y += float(tmp.iloc[j]['delta'])                
        for j in np.arange(len(tmp)):
            x = i+1            
            score = float(tmp.iloc[j]['delta'])
            score_sd = float(tmp.iloc[j]['sem'])
            channel = None
            if tmp.iloc[j]['channel'] == '[0, 0, 0, 1]': channel = 'A'
            elif tmp.iloc[j]['channel'] == '[0, 0, 1, 0]': channel = 'D'  
            elif tmp.iloc[j]['channel'] == '[0, 1, 0, 0]': channel = 'M'
            elif tmp.iloc[j]['channel'] == '[1, 0, 0, 0]': channel = 'N'         
            _letterAt(channel, x, y, abs(score), score_sd, ax)
            y += abs(score)
    
    lim_y_min, lim_y_max = (-1*y_lim, y_lim)
    values = range(int(lim_y_min*10), int(lim_y_max*10), 2)
    labels = [str(round(float(v)/10,2)) for v in values]    
    labels[0] = ''
    labels.append('')

#    lim_y_min, lim_y_max = (-1.0, 1.0)
#    labels = ['', '-0.8', '-0.6', '-0.4', '-0.2', '0.0', 
#              '0.2', '0.4', '0.6', '0.8', '']    
    
#    if max_y*1.5 < 0.8:
#        lim_y_min, lim_y_max = (-0.8, 0.8)
#        labels = ['', '-0.6', '-0.4', '-0.2', '0.0', '0.2', '0.4', '0.6', '']        
#    if max_y*1.5 < 0.6:
#        lim_y_min, lim_y_max = (-0.6, 0.6)
#        labels = ['', '-0.4', '-0.2', '0.0', '0.2', '0.4', '']

#    if max_y*1.5 < 0.4:
#        lim_y_min, lim_y_max = (-0.4, 0.4)
#        labels = ['', '-0.2', '0.0', '0.2', '']
#    if max_y*1.5 < 0.2:
#        lim_y_min, lim_y_max = (-0.2, 0.2)
#        labels = ['', '0.0', '']
#    if max_y*1.5 < 0.1:
#        lim_y_min, lim_y_max = (-0.1, 0.1)
#        labels = ['', '-0.08', '-0.06', '-0.04', '-0.02', '0.00', 
#              '0.02', '0.04', '0.06', '0.08', '']
    
    ax.tick_params(labelsize=20)
    ax.set_xlim(0,len(seq)+1)
    ax.set_ylim(lim_y_min,lim_y_max)
    ax.set_ylabel(r'$\mathit{-\Delta\Delta\Delta G/RT}$', fontsize=20)    
    base = 0       
    ax.axhline(y=base, linestyle='--', alpha=0.1, color='black')
    ax.set_xticklabels(labels=[])
    ax.set(xticks=np.arange(1,len(seq)+1,1))    
    ax.set_yticklabels(labels)
    ax.set_yticks(np.arange(lim_y_min, lim_y_max, 0.2))



def __plot_details_seq(df, seq_idx, seq, ax, base, max_y, y_lim):
    """ """
    for i in np.arange(len(seq)):
        idx_s_pos = df['s_pos'] == i
        tmp = df[idx_s_pos]
        tmp = tmp.sort_values(by='delta')
        # calculate the baseline
        y = 0
        for j in np.arange(len(tmp)):
            if float(tmp.iloc[j]['delta']) < 0:
                y += float(tmp.iloc[j]['delta'])
        for j in np.arange(len(tmp)):
            x = i+1
            score = float(tmp.iloc[j]['delta'])
            score_sd = float(tmp.iloc[j]['sem'])
            channel = None

            #if tmp.iloc[j]['channel'] == '[1, 0, 0, 0]': channel = 'A'
            #elif tmp.iloc[j]['channel'] == '[0, 1, 0, 0]': channel = 'C'
            #elif tmp.iloc[j]['channel'] == '[0, 0, 1, 0]': channel = 'G'
            #elif tmp.iloc[j]['channel'] == '[0, 0, 0, 1]': channel = 'T'
            
            if tmp.iloc[j]['channel'] == '[1, 0, 0, 0]': channel = 'A'
            elif tmp.iloc[j]['channel'] == '[0, 1, 0, 0]': channel = 'C'
            elif tmp.iloc[j]['channel'] == '[0, 0, 1, 0]': channel = 'G'
            elif tmp.iloc[j]['channel'] == '[0, 0, 0, 1]': channel = 'T'
            
            
            _letterAt_seq(channel, x, y, abs(score), score_sd, ax)
            y += abs(score)

    lim_y_min, lim_y_max = (-1*y_lim, y_lim)
    #if lim_y_min<=0.2:
    #    values = range(int(lim_y_min*10), int(lim_y_max*10), 1)
    #else: values = range(int(lim_y_min*10), int(lim_y_max*10), 4)
    values = range(int(lim_y_min*10), int(lim_y_max*10), 4)

    labels = [str(round(float(v)/10,2)) for v in values]
    labels[0] = ''
    labels.append('')


    ax.tick_params(labelsize=20)
    ax.set_xlim(0,len(seq)+1)
    ax.set_ylim(lim_y_min,lim_y_max)
    ax.set_ylabel(r'$\mathit{-\Delta\Delta\Delta G/RT}$', fontsize=20)
    base = 0
    ax.axhline(y=base, linestyle='--', alpha=0.1, color='black')
    ax.set_xticklabels(labels=[])
    ax.set(xticks=np.arange(1,len(seq)+1,1))
    ax.set_yticklabels(labels)
    #if lim_y_max<0.4:
    #    ax.set_yticks(np.arange(lim_y_min, lim_y_max, 0.1))
    #else: ax.set_yticks(np.arange(lim_y_min, lim_y_max, 0.4))
    ax.set_yticks(np.arange(lim_y_min, lim_y_max, 0.4))
    
def _letterAt(letter, x, y, yscale=1, yerror=0, ax=None):        
    fp = FontProperties(family="Arial", weight="medium")
    globscale = 1.35
    LETTERS = { "A" : TextPath((-0.34, 0), "A", size=1, prop=fp),
            "D" : TextPath((-0.384, 0), "D", size=1, prop=fp),
            "M" : TextPath((-0.43, 0), "M", size=1, prop=fp),
            "N" : TextPath((-0.366, 0), "N", size=1, prop=fp) }
    COLOR_SCHEME = {'A': 'red',
                'D': 'blue',
                'M': 'gold',
                'N': 'grey'}
           
    text = LETTERS[letter]
    t = mpl.transforms.Affine2D().scale(1*globscale, yscale*globscale) + \
        mpl.transforms.Affine2D().translate(x,y) + ax.transData
    if letter=='P':    
        p = PathPatch(text, lw=0, fc=COLOR_SCHEME[letter], transform=t,
                      alpha=0.5)
    else: 
        p = PathPatch(text, lw=0, fc=COLOR_SCHEME[letter],  transform=t)
    
    if ax != None:
        ax.add_artist(p)
        ax.errorbar(x, y+yscale, yerr=yerror, ecolor=COLOR_SCHEME[letter], 
                    capsize=0, fmt=' ', alpha=0.5)        
    return p



def _letterAt_seq(letter, x, y, yscale=1, yerror=0, ax=None):
    fp = FontProperties(family="Roboto", weight="medium")
    globscale = 1.35
    LETTERS = { "A" : TextPath((-0.34, 0), "A", size=1, prop=fp),
            "C" : TextPath((-0.384, 0), "C", size=1, prop=fp),
            "G" : TextPath((-0.43, 0), "G", size=1, prop=fp),
            "T" : TextPath((-0.366, 0), "T", size=1, prop=fp) }
    COLOR_SCHEME = {'A': '#109648',
            'C':'#255C99',
            'G':'#F7B32B',
            'T':'#D62839'}

    text = LETTERS[letter]
    t = mpl.transforms.Affine2D().scale(1*globscale, yscale*globscale) + \
        mpl.transforms.Affine2D().translate(x,y) + ax.transData
    if letter=='P':
        p = PathPatch(text, lw=0, fc=COLOR_SCHEME[letter], transform=t,
                      alpha=0.5)
    else:
        p = PathPatch(text, lw=0, fc=COLOR_SCHEME[letter],  transform=t)

    if ax != None:
        ax.add_artist(p)
        ax.errorbar(x, y+yscale, yerr=yerror, ecolor=COLOR_SCHEME[letter],
                    capsize=0, fmt=' ', alpha=0.5)
    return p



def __reverse(seq):
    """ """
    alt_map = {'ins':'0'}
    complement = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A', 'M':'g', 'g':'M'} 
    for k,v in alt_map.items():
        seq = seq.replace(k,v)
    bases = list(seq) 
    bases = [complement.get(base,base) for base in bases]
    bases = ''.join(bases)
    for k,v in alt_map.items():
        bases = bases.replace(v,k)
    return bases 
