import whisper
from pytorch_lightning import LightningModule
import evaluate
import torch
from torch import nn
from transformers import (
    AdamW,
    get_linear_schedule_with_warmup
)

from .dataset import WhisperASRDataset, WhisperASRDataCollator

class WhisperModelModule(LightningModule):
    def __init__(self, config, options,model_name="base", train_dataset=[], eval_dataset=[]) -> None:
        super().__init__()
        self.options = options
        self.model = whisper.load_model(model_name)
        self.tokenizer = whisper.tokenizer.get_tokenizer(True, language=self.options.language, task=self.options.task)

        # both encode decode train
        for p in self.model.encoder.parameters():
            p.requires_grad = True
        
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
        self.metrics_wer = evaluate.load("wer")
        self.metrics_cer = evaluate.load("cer")

        self.config = config
        self.__train_dataset = train_dataset
        self.__eval_dataset = eval_dataset
    
    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_id):
        input_ids = batch["input_ids"]#[16, 80, 3000];mel 80通道 3000个窗口
        labels = batch["labels"].long()#[16, 125];[batch_size,seq_len]，eot，-100。。
        dec_input_ids = batch["dec_input_ids"].long()#eot,eot


        audio_features = self.model.encoder(input_ids)#encoder输出，[16, 1500, 512];[batch_size=sample_nums,token_size,emb_size]
        #out.shape:[16, 125, 51865];[batch_size,seq_len,vocab_num(每个seq的每个token的每个词的score)]
        out = self.model.decoder(dec_input_ids, audio_features)#dec_input_ids的预测右移的logits[16, 125, 51865]输入加上sot且扩展eot的token_id;和encoder输出
        loss = self.loss_fn(out.view(-1, out.size(-1)), labels.view(-1))#交叉熵
        self.log("train/loss", loss, on_step=True, prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, batch, batch_id):#5.每训练完一批后，使用验证集获得Loss,cer,wer,计入日志
        input_ids = batch["input_ids"]#[16, 80, 3000],每批的mel
        labels = batch["labels"].long()#[16,90],每批的token_id_seq,单个eot_id,90是seq长度
        dec_input_ids = batch["dec_input_ids"].long()#[16,90]每批的token_id_seq,多个eot_id


        audio_features = self.model.encoder(input_ids)#[16, 1500, 512]
        out = self.model.decoder(dec_input_ids, audio_features)#[16, 90, 51865]，log_probs?

        loss = self.loss_fn(out.view(-1, out.size(-1)), labels.view(-1))#交叉熵[1440, 51865];[1440]

        out[out == -100] = self.tokenizer.eot#扩展eot_id
        labels[labels == -100] = self.tokenizer.eot
        #o:<lang>,<>,<notimestamps>
        o_list, l_list = [], []
        for o, l in zip(out, labels):#循环16次
            o = torch.argmax(o, dim=1)#贪婪法预测的token_id_seq
            o_list.append(self.tokenizer.decode(o, skip_special_tokens=True))#生成text
            l_list.append(self.tokenizer.decode(l, skip_special_tokens=True))
        cer = self.metrics_cer.compute(references=l_list, predictions=o_list)
        wer = self.metrics_wer.compute(references=l_list, predictions=o_list)

        self.log("val/loss", loss, on_step=True, prog_bar=True, logger=True)
        self.log("val/cer", cer, on_step=True, prog_bar=True, logger=True)
        self.log("val/wer", wer, on_step=True, prog_bar=True, logger=True)#记录日志

        return {
            "cer": cer,
            "wer": wer,
            "loss": loss
        }

    def configure_optimizers(self):#2.设置优化
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() 
                            if not any(nd in n for nd in no_decay)],
                "weight_decay": self.config["weight_decay"],
            },
            {
                "params": [p for n, p in model.named_parameters() 
                            if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]#优化的参数
        optimizer = AdamW(optimizer_grouped_parameters, 
                        lr=self.config["learning_rate"], 
                        eps=self.config["adam_epsilon"]
                    )#学习率，优化参数，种类
        self.optimizer = optimizer

        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=self.config["warmup_steps"], 
            num_training_steps=self.t_total
        )
        self.scheduler = scheduler

        return [optimizer], [{"scheduler": scheduler, "interval": "step", "frequency": 1}]
    
    def setup(self, stage=None):#1

        if stage == 'fit' or stage is None:
            self.t_total = (
                (len(self.__train_dataset) // (self.config["batch_size"]))
                // self.config["gradient_accumulation_steps"]
                * float(self.config["num_train_epochs"])
            )#2000/16/epoch_num
    
    def train_dataloader(self):
        dataset = WhisperASRDataset(self.__train_dataset, self.tokenizer)
        return torch.utils.data.DataLoader(dataset, 
                    batch_size=self.config["batch_size"], 
                    drop_last=True, shuffle=True, num_workers=self.config["num_worker"],
                    collate_fn=WhisperASRDataCollator()
                )

    def val_dataloader(self):#3.val
        dataset = WhisperASRDataset(self.__eval_dataset, self.tokenizer)#g
        return torch.utils.data.DataLoader(dataset, 
                    batch_size=self.config["batch_size"], 
                    num_workers=self.config["num_worker"],
                    collate_fn=WhisperASRDataCollator()
                )
    
