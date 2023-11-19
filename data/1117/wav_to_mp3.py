import os
import subprocess
import torchaudio
def ffmpeg_WavToMP3(input_path, output_path):
    cmd = "ffmpeg -y -i " + input_path + " " + output_path  #将input_path转为.mp3文件
    subprocess.call(cmd, shell=True)

def wavdir_to_mp3dir(wavdir,mp3dir):
    wavs=os.listdir(wavdir)
    for file in wavs:
        wavpath=wavdir+'/'+file
        mp3path=mp3dir+'/'+file.replace('.wav','.mp3')
        ffmpeg_WavToMP3(wavpath, mp3path)
        try:
            wav,sr=torchaudio.load(mp3path,normalize=True)    
            duration=wav.shape[-1]/sr
            if duration>0.2:
                os.remove(wavpath)
        except Exception as e:
            print("An error occurred:", e)
            os.remove(mp3path)
            print("remove:", mp3path)
wavdir='/home/huangrm/audio/whisper_music2lyric_finetune/data/1117/wav'
mp3dir='/home/huangrm/audio/whisper_music2lyric_finetune/data/1117/mp3'
wavdir_to_mp3dir(wavdir,mp3dir)
print('convert wav to mp3 over')