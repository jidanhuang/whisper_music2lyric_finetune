import yaml
from pathlib import Path 
import whisper
import torch
from tqdm import tqdm
import evaluate

from pytorch_lightning import Trainer

import sys
sys.path.append(str(Path(__file__).resolve().absolute().parents[2]))
from whisper_finetune.dataset import WhisperASRDataset, load_data_list, WhisperASRDataCollator
from whisper_finetune.model import WhisperModelModule
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
""" beamsearch:(1)input
    best of sampling:   (1)_main_loop:input(audio_features,initial tokens,best_of);output(sum_logprobs,tokens)
                        (2)rank:return sum_logprobs,tokens
    top-k:  (1)_main_loop:input(audio_features,initial tokens,k);output(sum_logprobs,tokens,logits)
    rank logits    keep topk->mask    log(softmax(logits))->logit_logprobs    
    top-p
    rank logits    softmax(logits)->keep topp->mask      log(softmax(logits/t))->logit_logprobs    
    目前策略：5个beam 对每个beam(seq)，在top-p个token中采样，最后获得5个seq选取其中sum_logprob最大的
            common best of N,5个beam，对每个beam(seq),选logits最大的token，最后获得5个seq选取其中sum_logprob最大的
            top-p：每次在top-p个token中采样
            top-p+best of N-search:目前策略
            beam search:5个beam 对每个beam(seq)，选取topk个new_tokens（k=beam_size+1），最后从k*beam_size个seq中选取选取其中sum_logprob最大的5个seq
            原策略：random sampling+beam search:
            best of N search: N个seq候选，每个时间步使用P(Xi|x0~xi-1)分布随机采样，最终选取累计得分(sum_logprob)最高的seq
            beam search: beam_size个seq候选，每个时间步根据P(x1~xi|x0)选取最大的beam_size个token，最终选取累计得分(sum_logprob)最高的seq
            top-p/top-k:独立采样前把概率分布p(xi|x0~xi-1)修改
            可选修改：1.best_of_N:最终根据sum_logprob采样选取——等效于ramdom sampling；N=1；因此最终的sum_logprob选取避免了多样化引起的不常见seq
                     2.top-p本来是连环采样（已经有效避免了边缘词出现）；加上best_of_N后多样性受到限制（容易出现重复无营养的话），也许单独用效果更好
            

    """
def inference():
    # load config 
    config_path = Path("config.yaml")
    # whisper-asr-finetune-main/recipes/music/config.yaml
    # whisper-asr-finetune-main/recipes/music/config.yaml
    files=os.listdir("./")
    print(files)
    config = yaml.load(open(config_path, "r"), Loader=yaml.FullLoader)

    # dirs and paths
    in_data_dir = Path(config["path"]["preprocessed"])#'data/processed_1117'
    checkpoint_dir = Path(config["path"]["checkpoint"])#checkpoint
    with_timestamps = bool(config["data"]["timestamps"])#false,无时间戳
    # device = "gpu" if torch.cuda.is_available() else "cpu"

    # tools，指定语言zh，无时间戳
    whisper_options = whisper.DecodingOptions(
        language=config["data"]["lang"], without_timestamps=not with_timestamps,task=config["inference"]["task"],top_p=config["inference"]["top_p"]
    )
    #mel分词器，指定语言，任务：转录
    whisper_tokenizer = whisper.tokenizer.get_tokenizer(
        True, language=config["data"]["lang"], task=whisper_options.task
    )

    # list[文件名，mel相对路径，歌词]
    test_list = load_data_list(in_data_dir / "test.txt")
    test_list=test_list[0:20]
    test_list=test_list*5
    test_list.sort()
    #生成迭代器loader
    dataset = WhisperASRDataset(test_list, whisper_tokenizer)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=1,
        collate_fn=WhisperASRDataCollator()
    )

    # load models
    epoch = config["inference"]["epoch_index"]#7
    checkpoint_path = checkpoint_dir / "checkpoint" / f"checkpoint-epoch={epoch:04d}.ckpt"#checkpoint/checkpoint/checkpoint-epoch=0007.ckpt
    state_dict = torch.load(checkpoint_path)#状态加载
    state_dict = state_dict['state_dict']
    whisper_model = WhisperModelModule(config["train"],whisper_options)
    whisper_model.load_state_dict(state_dict)#模型加载

    # inference
    ref, hyp = [], []
    for b in tqdm(loader):#迭代生成b包含一个样本信息，来自音频mel
        input_id = b["input_ids"].half().cuda()#mel，生成hypothesis；shape:[1, 80, 3000]；1：；80通道数；3000每个样本的mel_token数
        label = b["labels"].long().cuda()#生成ref；shape[1,39];[1，seq_len+1（<eot>）]
        with torch.no_grad():
            #[DecodingResult1,DecodingResult2...]
            hypothesis = whisper_model.model.decode(input_id, whisper_options)
            #h :DecodingResult(audio_features=(shape[1500,512]),tokens=(shape=[43]),text='我会在这里陪着你 像一个演员 把悲伤当作是排练 白天被驯服 夜晚却更剧烈')
            #audio_features:[token0_size,emb_size]
            for h in hypothesis:
                hyp.append(h.text)
            
            for l in label:
                l[l == -100] = whisper_tokenizer.eot
                r = whisper_tokenizer.decode(l, skip_special_tokens=True)
                ref.append(r)
    i=0
    f= open('inf.txt','a') 
    f.writelines(f"{config['inference']}") 
    for r, h in zip(ref, hyp):

        if i%5==0:
            print("-"*10)
            f.writelines(f"\n {r}")
            print(f"reference:  {r}")
        print(f"|hypothesis: {h}")
        f.writelines(f"{h}")
        i+=1

    # compute CER
    cer_metrics = evaluate.load("cer")
    cer = cer_metrics.compute(references=ref, predictions=hyp)
    print(f"CER: {cer}")
    #做一个查重指标r=sum(长度为i的字符串重复字数*wi)/hypothesis总字数
    #算法：遍历hyp中所有长度为l的子串，如果能与ref匹配上,r+=l*wi/hypothesis总字数
    #l从maxlen(h)开始递减，每匹配成功一次就把子串删除，直到l=4或3

if __name__ == "__main__":
    inference()