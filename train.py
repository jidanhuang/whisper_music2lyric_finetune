import yaml
from pathlib import Path 
import whisper
import torch

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

import sys
sys.path.append(str(Path(__file__).resolve().absolute().parents[2]))
from whisper_finetune.dataset import load_data_list
from whisper_finetune.model import WhisperModelModule
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
def train():
    # load config 
    config_path = Path("config.yaml")
    config = yaml.load(open(config_path, "r"), Loader=yaml.FullLoader)
    
    # dirs and paths
    in_data_dir = Path(config["path"]["preprocessed"])
    #"./preprocessed/jsut_ver1.1"=data/processed_1117/
    out_log_dir = Path(config["path"]["log"])
    #"./log"{}
    checkpoint_dir = Path(config["path"]["checkpoint"])
    #"./checkpoint" # dir to save model
    with_timestamps = bool(config["data"]["timestamps"])
    #False无时间戳
    device = "gpu" if torch.cuda.is_available() else "cpu"
    #创建模型文件夹
    out_log_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # tools
    #指定生成任务，参数：语言“ja”,是否需要时间戳
    whisper_options = whisper.DecodingOptions(
        language=config["data"]["lang"], without_timestamps=not with_timestamps,task=config["data"]["task"]
    )
        #mel分词器，指定语言，任务：转录
    whisper_tokenizer = whisper.tokenizer.get_tokenizer(
        True, language=config["data"]["lang"], task=whisper_options.task
    )
    #
    # list：训练集路径
    train_list = load_data_list(in_data_dir / "train.txt")
    val_list = load_data_list(in_data_dir / "val.txt")

    # logger日志记录？参数：日志名whisper，版本1
    tflogger = TensorBoardLogger(
        save_dir=out_log_dir,
        name=config["train_name"],
        version=config["train_id"]
    )

    # callback回调函数，参数：模型路径checkpoint/checkpoint/checkpoint-0003
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir / "checkpoint",
        filename="checkpoint-{epoch:04d}",
        save_top_k=-1 # all model save
    )
    callback_list = [
        checkpoint_callback, LearningRateMonitor(logging_interval="epoch")]
    #模型
    model = WhisperModelModule(config["train"],whisper_options,config["model_name"],train_list,val_list)

    trainer = Trainer(
        precision=16,
        accelerator=device,
        max_epochs=config["train"]["num_train_epochs"],
        accumulate_grad_batches=config["train"]["gradient_accumulation_steps"],
        logger=tflogger,
        callbacks=callback_list,
        devices=1,
        auto_select_gpus=True
    )

    trainer.fit(model)

if __name__ == "__main__":
    train()