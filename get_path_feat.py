######################
import librosa
import torch,yaml
import numpy as np
import os,argparse
from tqdm import tqdm
from model import JDCNet
from meldataset import  MelDataset


def _split_full_s_e_( f0_pre_c ):
    '''
    f0_pre_c : numpy  array  L
    '''
    res_0 = []
    speech_biao = (f0_pre_c[0] == 1)
    start = 0
    end = 0
    for ind , oo in enumerate(f0_pre_c):
        if speech_biao:
            if end>start:
                res_0.append([start , ind])
    #             print (f0_pre_c[start :ind-1 ])
            start = ind
        else:
            end = ind
        speech_biao = (oo == 1)
    if start !=ind:
        res_0.append([start , ind])
    ###
    if len(res_0) > 0:
        pass
        # _mm_ = max(res_0[0][0]-2 , 0)
        # res_0[0][0] = _mm_
    else:
        return []
    
    #######  合并
    res_0_he = []
    for oo in res_0:
        _temp_ = oo.copy()
        if len(res_0_he) ==0:
            res_0_he.append(_temp_)
            continue
        if _temp_[0] - res_0_he[-1][1] < 21:
            res_0_he[-1][1] = _temp_[1]
        else:
            res_0_he.append(_temp_)
    return res_0_he

def _pl_(audio , oo , _jis):
    '''
    分片
    '''
    _mm_ = max(oo[0]-4 , 0)
    sta = _mm_*_jis
    end = (oo[1]+4 )  *_jis

    return audio[ sta:end ]


class funs_vad_f0():
    def __init__(self,config_path,model_path ):
        self.configs = yaml.safe_load(open( config_path ))
        self.device = self.configs.get('device', 'cpu')
        self.sample_rate = self.configs['dataset_params'].get('sample_rate',24000)
        self._dataset = MelDataset( []  ,
                                validation=True, 
                                **self.configs['dataset_params'] )
        _sys_model = JDCNet(num_class=1) # num_class = 1 means regression

        cpk = torch.load(model_path,map_location='cpu')['model']
        _sys_model.load_state_dict(cpk)
        _sys_model.to(self.device)
        _sys_model.eval()
        self._sys_model = _sys_model
        self.f0_shai = self.configs.get('f0_shai',45)

    def audio_f0(self,audio_path):
        audio_data,sr = librosa.core.load( audio_path , sr=None)
        _jis = int(sr / 1000 * 10)
        inp_au = librosa.resample( audio_data,sr,self.sample_rate )
        wave_tensor = torch.FloatTensor([inp_au])
        _feat = self._dataset.to_melspec(wave_tensor.cpu() )
        _feat = (torch.log(1e-5 + _feat) - self._dataset.mean ) / self._dataset.std
        _feat = _feat.to(self.device)
        _feat = torch.unsqueeze(_feat , 1)
        with torch.no_grad():
            f0_pre = self._sys_model.get_feature_GAN(_feat)
            f0_pre = torch.squeeze(f0_pre , -1)
        return audio_data , sr, f0_pre,_jis
    def clean_vad(self,audio_path):

        audio_data , sr, f0_pre,_jis = self.audio_f0(audio_path)

        f0_pre = (f0_pre * (f0_pre >self.f0_shai))
        non_zero = (f0_pre > 0) 
        f0_pre = f0_pre * (non_zero*1) + (1-non_zero*1)
        f0_pre_c = f0_pre[0].cpu().data.numpy()
        ### full
        res_0 = _split_full_s_e_(f0_pre_c)
        if len(res_0)==0:
            print ('path is 0000 {}'.format( audio_path ) )
            return None
        con_a = [ _pl_(audio_data , oo , _jis) for oo in res_0]
        con_a_0 = np.concatenate(con_a)

        return  con_a_0,sr

if __name__ == "__main__":
    config_path = './config.yml'
    model_path = './pretrained_model/model.pth'
    audio_path = './singing-07-013.flac'
    save_path = './output.wav'
    _v_a_ = funs_vad_f0( config_path , model_path )
    res , sr = _v_a_.clean_vad(audio_path)
    librosa.output.write_wav(save_path, res , sr)




