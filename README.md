# vad_by_f0
音频vad处理 和 提取f0

# 致谢：
https://github.com/yl4579/PitchExtractor

## 说明
1. 按照Data的例子准备好数据
2. python train.py (训练)
3. vad 和 f0的提取使用get_path_feat.py

主要环境：python 3.6.13  
torch==1.8.0   
torchaudio==0.8.0  
librosa==0.7.2  
numpy==1.19.2  
pyworld==0.3.0

使用get_path_feat.py可以去除音频非人声片段