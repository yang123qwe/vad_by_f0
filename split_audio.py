import warnings
warnings.filterwarnings("ignore")
import os
import utils,librosa
import numpy as np
from my_dataset_e2e import _audio_pre_
def _split_s_e(f0_pre_c):
    '''
    收尾截取
    '''
    start = None
    for ii in range(len(f0_pre_c)):
        if f0_pre_c[ii] !=1:
            start = ii
            break
    for ii in range(len(f0_pre_c)-1,0,-1):
        if f0_pre_c[ii] !=1:
            end = ii
            break
    if start is None:
        return []
    else:
        return [[start,end]]

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

def _path_split_audio_(D_C,audio,_max_t_ , sr=None):
    '''
    分音频
    '''
    if isinstance(audio , str):
        audio_data,sr = librosa.core.load(audio , sr=None)
    else:
        if  sr is None:
            print ('音频 数据 需要 sr')
            exit()
        audio_data = audio
    audio_data = audio_data[:sr*_max_t_]
    _jis = int(sr / 1000 * 10)
    if sr == 24000:
        conv = False
    else:
        conv = True
    f0_pre = D_C._path_sys_(audio=audio_data , _sr_=sr , conv=conv )
    f0_pre_c = f0_pre[0].cpu().data.numpy()
    ### full
    res_0 = _split_full_s_e_(f0_pre_c)
    # ### start end  split
    # res_0 = _split_s_e(f0_pre_c)
    if len(res_0):
        pass
    else:
        if isinstance(audio , str):
            print ('path is 0000 {}'.format(audio) )
        else:
            print ('data is 0000 {}'.format(audio.shape) )
        return None
    con_a = [ _pl_(audio_data , oo , _jis) for oo in res_0]
    con_a_0 = np.concatenate(con_a)
    
    return  con_a_0,sr


if  __name__ == '__main__':
    import os,argparse,glob
    from tqdm import tqdm
    os.chdir('/home/nr/my_vc_st_vi')
    parser = argparse.ArgumentParser()
    parser.add_argument("--fen_s", default=1, type=int)
    parser.add_argument("--start_i", default=0, type=int)
    parser.add_argument("--end_i", default=1, type=int)
    parser.add_argument("--device", default='cuda', type=str)
    parser.add_argument("--save_file_root", default='/home/nr/datas/_tts_datas_/englishs/_lens', type=str)
    args = parser.parse_args()

    # with open('/home/nr/voice_zhuan/_code_/spkemb/ssss_c.txt') as file:
    #     paths = file.read().split('\n')[:-1]
        # paths = [oo.split('|')[0] for oo in paths ]

    # paths = glob.glob('/home/nr/datas/biao_bei_datas/emo_datas/*.wav')
    # paths = [oo for oo in paths if '.clean_f0.wav' not in oo]

    # ### jp data  ko data
    # jp_paths = glob.glob('/home/nr/datas/_tts_datas_/jp_data/*.wav')
    # ko_paths = glob.glob('/home/nr/datas/_tts_datas_/h_wavs_kg_0/*.wav')
    # paths = jp_paths + ko_paths
    # paths = [oo for oo in paths if '.clean_f0.wav' not in oo]

    # ### vock english
    # with open('/home/nr/datas/_tts_datas_/vcdk_tts_eng/labs.txt') as file:
    #     _temp_eng_ = file.read().split('\n')
    #     paths = [oo.split('|')[1] for oo in _temp_eng_]
        
    ### hifi english
    with open('/home/nr/datas/_tts_datas_/englishs/lab.txt') as file:
        _temp_eng_ = file.read().split('\n')
        paths = [oo.split('|')[1] for oo in _temp_eng_]
        

    _len_ = len(paths) // args.fen_s
    paths = paths[ _len_*args.start_i : _len_*(args.start_i+1)   ]
    print (  '\n'.join(paths[:2])   )
    if len(paths) == 0:
        exit()
    device = 'cuda:1'
    _min_t_ = 700 ## ms
    _max_t_ = 20 ## s
    save_file_root = args.save_file_root
    os.makedirs( save_file_root ,exist_ok=1)
    writ_path = os.path.join(  save_file_root, 'text_{}_{}.txt'.format(args.fen_s , args.start_i) )
    file_ww = open(writ_path , 'a+')
    hps = utils.get_hparams_from_yaml_file('./_my.yaml')
    D_C = _audio_pre_(**hps.data , device=device)

    res_2_0 = []
    save_ppp_rroot = '/home/nr/datas/_tts_datas_/englishs/hifi_spker_6097'
    os.makedirs(save_ppp_rroot , exist_ok=1)
    for  ind_i , path in  enumerate(tqdm(paths)):
        # save_path_w = '{}.clean_f0.wav'.format(path)
        save_path_w = os.path.join( save_ppp_rroot, '{}.clean_f0.wav'.format(os.path.basename(path)))
        if os.path.isfile(save_path_w):
            res_2_0.append('{}|{}|{}'.format(_temp_eng_[ind_i].split('|')[0]  , 
                                                                            save_path_w,
                                                                            _temp_eng_[ind_i].split('|')[2] ))
            continue
        ##################################
        _rts_ = _path_split_audio_(D_C,path,_max_t_ )
        if _rts_ is None:
            continue
        con_a_0,sr = _rts_
        #####################################
        # res_0 = _split_s_e(f0_pre_c)
        if  (con_a_0.shape[0]/sr * 1000) <  _min_t_ :
            print ('data len min {}  len {} ms'.format(path , con_a_0.shape[0]/sr * 1000 ))
            continue
        else:
            # pass
            res_2_0.append('{}|{}|{}'.format(_temp_eng_[ind_i].split('|')[0]  , 
                                                                            save_path_w,
                                                                            _temp_eng_[ind_i].split('|')[2] ))
            librosa.output.write_wav(save_path_w, con_a_0 , sr, norm=False)
        
        www_ = '{}|{}\n'.format(save_path_w , round(  con_a_0.shape[0]/sr, 5 ))
        file_ww.write(www_)
    file_ww.close()
    print ( '{}  {} is  okkk '.format(args.fen_s , args.start_i) )

    with open('/home/nr/datas/_tts_datas_/englishs/clean_hifi_6097.txt','w') as file:
        file.write('\n'.join(res_2_0))












