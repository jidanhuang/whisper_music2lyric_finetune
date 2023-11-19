from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Iterable, Optional, Sequence, Union, TYPE_CHECKING
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.distributions import Categorical

from .audio import CHUNK_LENGTH
from .tokenizer import Tokenizer, get_tokenizer
from .utils import compression_ratio

if TYPE_CHECKING:
    from .model import Whisper


@torch.no_grad()
def detect_language(model: "Whisper", mel: Tensor, tokenizer: Tokenizer = None) -> Tuple[Tensor, List[dict]]:
    """
    Detect the spoken language in the audio, and return them as list of strings, along with the ids
    of the most probable language tokens and the probability distribution over all language tokens.
    This is performed outside the main decode loop in order to not interfere with kv-caching.

    Returns
    -------
    language_tokens : Tensor, shape = (n_audio,)
       x, ids of the most probable language tokens, which appears after the startoftranscript token.
    language_probs : List[Dict[str, float]], length = n_audio
        list of dictionaries containing the probability distribution over all languages.
    """
    if tokenizer is None:
        tokenizer = get_tokenizer(model.is_multilingual)
    if tokenizer.language is None or tokenizer.language_token not in tokenizer.sot_sequence:
        raise ValueError(f"This model doesn't have language tokens so it can't perform lang id")

    single = mel.ndim == 2
    if single:
        mel = mel.unsqueeze(0)

    # skip encoder forward pass if already-encoded audio features were given
    if mel.shape[-2:] != (model.dims.n_audio_ctx, model.dims.n_audio_state):
        mel = model.encoder(mel)

    # forward pass using a single token, startoftranscript
    n_audio = mel.shape[0]
    x = torch.tensor([[tokenizer.sot]] * n_audio).to(mel.device)  # [n_audio, 1]
    logits = model.logits(x, mel)[:, 0]

    # collect detected languages; suppress all non-language tokens
    mask = torch.ones(logits.shape[-1], dtype=torch.bool)
    mask[list(tokenizer.all_language_tokens)] = False
    logits[:, mask] = -np.inf
    language_tokens = logits.argmax(dim=-1)
    language_token_probs = logits.softmax(dim=-1).cpu()
    language_probs = [
        {
            c: language_token_probs[i, j].item()
            for j, c in zip(tokenizer.all_language_tokens, tokenizer.all_language_codes)
        }
        for i in range(n_audio)
    ]

    if single:
        language_tokens = language_tokens[0]
        language_probs = language_probs[0]

    return language_tokens, language_probs


@dataclass(frozen=True)
class DecodingOptions:#decode参数表
    task: str = "transcribe"  # whether to perform X->X "transcribe" or X->English "translate"
    language: Optional[str] = None  # language that the audio is in; uses detected language if None

    # sampling-related options
    temperature: float = 1.
    sample_len: Optional[int] =  None #生成seq的最大长度，maximum number of tokens to sample
    best_of: Optional[int] = None    #收集的独立样本 number of independent samples to collect, when t > 0
    beam_size: Optional[int] = None   #光束大小 number of beams in beam search, when t == 0
    patience: Optional[float] = None  #光束搜索耐心 patience in beam search (https://arxiv.org/abs/2204.05424)
    top_p: Optional[float] = None
    # options for ranking generations (either beams or best-of-N samples)
    length_penalty: Optional[float] = None   #长度惩罚中的a因子 "alpha" in Google NMT, None defaults to length norm

    # prompt, prefix, and token suppression
    prompt: Optional[Union[str, List[int]]] = None   #前一个上下文的文本或标记 text or tokens for the previous context
    prefix: Optional[Union[str, List[int]]] = None   #文本或标记作为当前上下文的前缀 text or tokens to prefix the current context
    suppress_blank: bool = True                      #这将抑制空白输出 this will suppress blank outputs

    #要屏蔽的令牌id(或逗号分隔的令牌id)列表 list of tokens ids (or comma-separated token ids) to suppress
    #"-1"将抑制在' tokenizer.non_speech_tokens() '中定义的一组符号集合 "-1" will suppress a set of symbols as defined in `tokenizer.non_speech_tokens()` 
    suppress_tokens: Optional[Union[str, Iterable[int]]] = "-1"

    #时间戳采样选项 timestamp sampling options
    without_timestamps: bool = False              # 仅使用<|notimestamps|>对文本标记进行采样，use <|notimestamps|> to sample text tokens only
    max_initial_timestamp: Optional[float] = 1.0  # 初始时间戳不能晚于此 the initial timestamp cannot be later than this

    # implementation details
    fp16: bool = True  # use fp16 for most of the calculation


@dataclass(frozen=True)#DecodingResult=dataclass(DecodingResult)#也就是将DecodingResult小盒子打包进dataclass大盒子中定义的中函数，返回中函数，此时DecodingResult就是这个中函数，其中打包了DecodingResult，同时包含别的部分
class DecodingResult:#decode结果集合
    audio_features: Tensor
    language: str
    language_probs: Optional[Dict[str, float]] = None#Optional使得language_probs除了是Dict[str, float]外默认值还可是none
    tokens: List[int] = field(default_factory=list)
    text: str = ""
    avg_logprob: float = np.nan
    no_speech_prob: float = np.nan
    temperature: float = np.nan
    compression_ratio: float = np.nan


class Inference:
    def logits(self, tokens: Tensor, audio_features: Tensor) -> Tensor:
        """Perform a forward pass on the decoder and return per-token logits"""
        #在解码器上执行forward并返回每个令牌的logits
        raise NotImplementedError

    def rearrange_kv_cache(self, source_indices) -> None:
        """Update the key-value cache according to the updated beams"""
        #根据更新的光束更新键值缓存
        raise NotImplementedError

    def cleanup_caching(self) -> None:
        """Clean up any resources or hooks after decoding is finished"""
        pass


class PyTorchInference(Inference):
    def __init__(self, model: "Whisper", initial_token_length: int):
        self.model: "Whisper" = model
        self.initial_token_length = initial_token_length#4
        self.kv_cache = {}
        self.hooks = []

    def logits(self, tokens: Tensor, audio_features: Tensor) -> Tensor:
        if not self.kv_cache:
            self.kv_cache, self.hooks = self.model.install_kv_cache_hooks()

        if tokens.shape[-1] > self.initial_token_length:#如果不是预测第一个词
            # only need to use the last token except in the first forward pass只需要使用最后一个令牌
            tokens = tokens[:, -1:]#torch.Size([5, 1])

        return self.model.decoder(tokens, audio_features, kv_cache=self.kv_cache)#return logits torch.Size([5, 1, 51865])

    def cleanup_caching(self):
        for hook in self.hooks:
            hook.remove()

        self.kv_cache = {}

        self.hooks = []

    def rearrange_kv_cache(self, source_indices):
        for module, tensor in self.kv_cache.items():
            # update the key/value cache to contain the selected sequences
            self.kv_cache[module] = tensor[source_indices].detach()


class SequenceRanker:
    def rank(self, tokens: List[List[Tensor]], sum_logprobs: List[List[float]]) -> List[int]:
        """
        Given a list of groups of samples and their cumulative log probabilities,
        return the indices of the samples in each group to select as the final result
        """
        #给定一组样本及其累积对数概率的列表，返回每组样本的索引作为最终结果，#返回每个样本的预测标签？
        raise NotImplementedError

#最大似然
class MaximumLikelihoodRanker(SequenceRanker):
    """
    Select the sample with the highest log probabilities, penalized using either
    a simple length normalization or Google NMT paper's length penalty
    """
    #长度处罚因子
    def __init__(self, length_penalty: Optional[float]):
        self.length_penalty = length_penalty

    #输入token：每个tensor一个token[[tensor11,tensor12],[tensor21,tensor22]]=[sample1,sample2]?
    #输入sumlogprobs[[float11,float12],[float21,float22]]
    def rank(self, tokens: List[List[Tensor]], sum_logprobs: List[List[float]]):
        #对logprobs进行长度处罚#tokens[[tensor1,tensor2,..,tensor5]];sum_logprobs:[[f1,f2,...,f5]]
        def scores(logprobs, lengths):
            result = []
            for logprob, length in zip(logprobs, lengths):
                if self.length_penalty is None:
                    penalty = length
                else:
                    # from the Google NMT paper
                    penalty = ((5 + length) / 6) ** self.length_penalty
                result.append(logprob / penalty)
            return result#[logprob1 / penalty,logprob2 / penalty,...,logprob5 / penalty]
        #得到beams中得分最高的序列 get the sequence with the highest score
        lengths = [[len(t) for t in s] for s in tokens]#[[91, 122, 65, 94, 135]]
        return [np.argmax(scores(p, l)) for p, l in zip(sum_logprobs, lengths)]#不是随机，而是选最大概率的
        #p[tenseri1,tensori2,...]->logprobs
        #s[floati1,floati2,...]->lengths
        #length[[len11,len12],[len21,len22]]
        #l[leni1,leni2,...]


class TokenDecoder:
    def reset(self):
        """Initialize any stateful variables for decoding a new sequence"""
    #初始化任何有状态变量以解码新序列

    def update(self, tokens: Tensor, logits: Tensor, sum_logprobs: Tensor) -> Tuple[Tensor, bool]:
        """Specify how to select the next token, based on the current trace and logits
                    指定如何根据当前trace and logits选择下一个token
        Parameters
        ----------
        tokens : Tensor, shape = (n_batch, current_sequence_length)
            all tokens in the context so far, including the prefix and sot_sequence tokens
            到目前为止上下文中的所有令牌，包括prefix和sot_sequence令牌
        logits : Tensor, shape = (n_batch, vocab_size)
            per-token logits of the probability distribution at the current step
            当前步骤每个令牌的概率分布logits
        sum_logprobs : Tensor, shape = (n_batch)
            cumulative log probabilities for each sequence
            每个序列的累积对数概率
        Returns
        -------
        tokens : Tensor, shape = (n_batch, current_sequence_length + 1)
            the tokens, appended with the selected next token
            #与所选的下一个标记一起追加
        completed : bool
            True if all sequences has reached the end of text

        """
        raise NotImplementedError

    def finalize(
        self, tokens: Tensor, sum_logprobs: Tensor
    ) -> Tuple[Sequence[Sequence[Tensor]], List[List[float]]]:
        """Finalize search and return the final candidate sequences
            完成搜索并返回最终候选序列
        Parameters
        ----------
        tokens : Tensor, shape = (n_audio, n_group, current_sequence_length)
            all tokens in the context so far, including the prefix and sot_sequence
            
        sum_logprobs : Tensor, shape = (n_audio, n_group)
            cumulative log probabilities for each sequence

        Returns
        -------
        tokens : Sequence[Sequence[Tensor]], length = n_audio
            sequence of Tensors containing candidate token sequences, for each audio input

        sum_logprobs : List[List[float]], length = n_audio
            sequence of cumulative log probabilities corresponding to the above

        """
        raise NotImplementedError
#x149zA40
#top-k sampling:logits选出topk;keep certain logits->softmax（with temperature）(/T,T>1减小概率差距，结果多样；T<1两极分化，结果更单一，T=1)
#top-p sampling:logits softmax选出topp;keep certain logits->softmax to rescale（with temperature）
# 直接选择分布中概率最大的token当作解码出来的词，但是该GreedyDecoder问题在于，总是选择概率最大的词，将会生成很多重复的句子（get stuck in loops）。
# 输入logits，输出top-p外被mask的logits
def top_k_top_p_filtering(logits, top_k=0, top_p=0.6, filter_value=-float("Inf"), min_tokens_to_keep=1):
    """
    根据top_k, top_p的值，将不满足的值置为filter_value的值

    :param torch.Tensor logits: bsz x vocab_size
    :param int top_k: 如果大于0，则只保留最top_k的词汇的概率，剩下的位置被置为filter_value
    :param int top_p: 根据(http://arxiv.org/abs/1904.09751)设置的筛选方式
    :param float filter_value:
    :param int min_tokens_to_keep: 每个sample返回的分布中有概率的词不会低于这个值#top-p留下的最小k值
    :return:
    """
    if top_k > 0:
        top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))  # Safety check
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p <= 1.0:
        #sorted_logits：每个logits按照大小获得新index,element=0/1
        #sorted_indices：element=每个logits的原index
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)#logits递减
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)#softmax获得prob后获得cum_prob

        # Remove tokens with cumulative probability above the threshold (token with 0 are kept)
        sorted_indices_to_remove = cumulative_probs > top_p#[0,0,0,...,0,1,1,...] eg[0,0,0,1,1]
        if min_tokens_to_keep > 1:
            # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
            sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
        # Shift the indices to the right to keep also the first token above the threshold
        #将索引向右移动，以保持第一个令牌高于阈值
        #eg [0,0,0,0,1]，右移用0补齐shape不变，这样0位置的token比原来多一个，0位置的id就留下来，其cum_prob>=p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # scatter sorted tensors to original indexing
        #将已排序的张量分散到原始索引：恢复每个logits的原index
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value
    return logits
class GreedyDecoder(TokenDecoder):
    """input:温度，<eot>;
            logits
            sum_logprobs"""
    def __init__(self, temperature: float, eot: int):
        self.temperature = temperature
        self.eot = eot

    def update(self, tokens: Tensor, logits: Tensor, sum_logprobs: Tensor) -> Tuple[Tensor, bool]:
        temperature = self.temperature
        if temperature == 0:#贪婪search
            next_tokens = logits.argmax(dim=-1)#dim=0行被压缩，变为所有行里的最大值的行索引（每一列的最大值的行索引）
            #最后一个维度logits[...,vocab_size]是所有word（token）中概率最大的id
        else:#
            next_tokens = Categorical(logits=logits / temperature).sample()#random采样过程torch.Size([5])
        #按照softmax(logits/t)的概率分布随机采样返回索引也就是next_tokens
        #torch.Size([5, 51865])
        logprobs = F.log_softmax(logits.float(), dim=-1)#softmax+log=log(pi)log防止梯度消失
        current_logprobs = logprobs[torch.arange(logprobs.shape[0]), next_tokens]#torch.Size([5]) [[0,1,2,...,batch_size],]
        sum_logprobs += current_logprobs * (tokens[:, -1] != self.eot)#torch.Size([5])

        next_tokens[tokens[:, -1] == self.eot] = self.eot#最后
        tokens = torch.cat([tokens, next_tokens[:, None]], dim=-1)#torch.Size([5, 5])加上预测的token

        completed = (tokens[:, -1] == self.eot).all()#空张量
        return tokens, completed

    def finalize(self, tokens: Tensor, sum_logprobs: Tensor):
        # make sure each sequence has at least one EOT token at the end
        tokens = F.pad(tokens, (0, 1), value=self.eot)
        return tokens, sum_logprobs.tolist()


class BeamSearchDecoder(TokenDecoder):
    def __init__(self, beam_size: int, eot: int, inference: Inference, patience: Optional[float] = None):
        self.beam_size = beam_size
        self.eot = eot
        self.inference = inference
        self.patience = patience or 1.0
        self.max_candidates: int = round(beam_size * self.patience)#5*1
        self.finished_sequences = None

        assert self.max_candidates > 0, f"Invalid beam size ({beam_size}) or patience ({patience})"

    def reset(self):
        self.finished_sequences = None

    def update(self, tokens: Tensor, logits: Tensor, sum_logprobs: Tensor) -> Tuple[Tensor, bool]:
        if tokens.shape[0] % self.beam_size != 0:
            raise ValueError(f"{tokens.shape}[0] % {self.beam_size} != 0")

        n_audio = tokens.shape[0] // self.beam_size
        if self.finished_sequences is None:  # for the first update
            self.finished_sequences = [{} for _ in range(n_audio)]#[{}]
        #logits,logprob:torch.Size([5, 51865])
        logprobs = F.log_softmax(logits.float(), dim=-1)#获得每个beam新token的logprobs;torch.Size([5, 51865])
        next_tokens, source_indices, finished_sequences = [], [], []
        for i in range(n_audio):
            scores, sources, finished = {}, {}, {}

            # STEP 1: calculate the cumulative log probabilities for possible candidates
            for j in range(self.beam_size):
                idx = i * self.beam_size + j#第i个audio;第j个beam
                prefix = tokens[idx].tolist()#第i个audio;第j个beam的tokens [50258, 50260, 50359, 50363]
                for logprob, token in zip(*logprobs[idx].topk(self.beam_size + 1)):#对第i个audio;第j个beam的token选取其中logprob最大的6（beam_size+1）个
                    new_logprob = (sum_logprobs[idx] + logprob).item()#获得第j个beam的新sum_logprobs
                    sequence = tuple(prefix + [token.item()])#第j个beam的新tokens
                    scores[sequence] = new_logprob#字典记载所有seq的累计得分 {seq:sum_logprob}6个key
                    sources[sequence] = idx#记载seq对应的audio_id和beam_id，在预测第一个token时会发生id覆盖
            #由于预测第一个token时输入的beams相同，因此每个beam预测到的next_token相同，因此候选只有6个，否则候选有beam_size*6个
            # STEP 2: rank the candidates and keep the top beam_size sequences for each audio#对候选音频进行排序，并保留每个音频的最高beam_size序列
            saved = 0
            for sequence in sorted(scores, key=scores.get, reverse=True):#按sum_logprob从大到小排列seq
                if sequence[-1] == self.eot:#判断seq有没有结束；如果有放进finished字典
                    finished[sequence] = scores[sequence]
                else:
                    sum_logprobs[len(next_tokens)] = scores[sequence]#更新sum_logprobs
                    next_tokens.append(sequence)#存放所有候选成功的seq
                    source_indices.append(sources[sequence])#存放所有成功的seq的id

                    saved += 1
                    if saved == self.beam_size:
                        break

            finished_sequences.append(finished)
        #获得[5,4][beam_size,seq_len of output]
        tokens = torch.tensor(next_tokens, device=tokens.device)#更新tokens
        self.inference.rearrange_kv_cache(source_indices)

        # add newly finished sequences to self.finished_sequences
        assert len(self.finished_sequences) == len(finished_sequences)
        for previously_finished, newly_finished in zip(self.finished_sequences, finished_sequences):
            for seq in sorted(newly_finished, key=newly_finished.get, reverse=True):
                if len(previously_finished) >= self.max_candidates:
                    break  # the candidate list is full
                previously_finished[seq] = newly_finished[seq]

        # mark as completed if all audio has enough number of samples
        completed = all(
            len(sequences) >= self.max_candidates for sequences in self.finished_sequences
        )
        return tokens, completed

    def finalize(self, preceding_tokens: Tensor, sum_logprobs: Tensor):
        # collect all finished sequences, including patience, and add unfinished ones if not enough
        sum_logprobs = sum_logprobs.cpu()
        for i, sequences in enumerate(self.finished_sequences):
            if len(sequences) < self.beam_size:  # when not enough sequences are finished
                for j in list(np.argsort(sum_logprobs[i]))[::-1]:
                    sequence = preceding_tokens[i, j].tolist() + [self.eot]
                    sequences[tuple(sequence)] = sum_logprobs[i][j].item()
                    if len(sequences) >= self.beam_size:
                        break

        tokens: List[List[Tensor]] = [
            [torch.tensor(seq) for seq in sequences.keys()] for sequences in self.finished_sequences
        ]
        sum_logprobs: List[List[float]] = [
            list(sequences.values()) for sequences in self.finished_sequences
        ]
        return tokens, sum_logprobs


class LogitFilter:
    def apply(self, logits: Tensor, tokens: Tensor) -> None:
        """Apply any filtering or masking to logits in-place

        Parameters
        ----------
        logits : Tensor, shape = (n_batch, vocab_size)
            per-token logits of the probability distribution at the current step

        tokens : Tensor, shape = (n_batch, current_sequence_length)
            all tokens in the context so far, including the prefix and sot_sequence tokens

        """
        raise NotImplementedError


class SuppressBlank(LogitFilter):
    def __init__(self, tokenizer: Tokenizer, sample_begin: int):
        self.tokenizer = tokenizer
        self.sample_begin = sample_begin

    def apply(self, logits: Tensor, tokens: Tensor):#torch.Size([5, 51865]),torch.Size([5, 4])
        if tokens.shape[1] == self.sample_begin:#如果预测第一个词
            logits[:, self.tokenizer.encode(" ") + [self.tokenizer.eot]] = -np.inf#抑制第一个预测不能是“ ”或eot将其logits mask


class SuppressTokens(LogitFilter):
    def __init__(self, suppress_tokens: Sequence[int]):
        self.suppress_tokens = list(suppress_tokens)

    def apply(self, logits: Tensor, tokens: Tensor):
        logits[:, self.suppress_tokens] = -np.inf#抑制特殊字符


class ApplyTimestampRules(LogitFilter):
    def __init__(
        self, tokenizer: Tokenizer, sample_begin: int, max_initial_timestamp_index: Optional[int]
    ):
        self.tokenizer = tokenizer
        self.sample_begin = sample_begin
        self.max_initial_timestamp_index = max_initial_timestamp_index

    def apply(self, logits: Tensor, tokens: Tensor):
        # suppress <|notimestamps|> which is handled by without_timestamps
        if self.tokenizer.no_timestamps is not None:
            logits[:, self.tokenizer.no_timestamps] = -np.inf

        # timestamps have to appear in pairs, except directly before EOT; mask logits accordingly
        for k in range(tokens.shape[0]):
            seq = [t for t in tokens[k, self.sample_begin :].tolist()]
            last_was_timestamp = len(seq) >= 1 and seq[-1] >= self.tokenizer.timestamp_begin
            penultimate_was_timestamp = len(seq) < 2 or seq[-2] >= self.tokenizer.timestamp_begin

            if last_was_timestamp:
                if penultimate_was_timestamp:  # has to be non-timestamp
                    logits[k, self.tokenizer.timestamp_begin :] = -np.inf
                else:  # cannot be normal text tokens
                    logits[k, : self.tokenizer.eot] = -np.inf

        if tokens.shape[1] == self.sample_begin:
            # suppress generating non-timestamp tokens at the beginning
            logits[:, : self.tokenizer.timestamp_begin] = -np.inf

            # apply the `max_initial_timestamp` option
            if self.max_initial_timestamp_index is not None:
                last_allowed = self.tokenizer.timestamp_begin + self.max_initial_timestamp_index
                logits[:, last_allowed + 1 :] = -np.inf

        # if sum of probability over timestamps is above any other token, sample timestamp
        logprobs = F.log_softmax(logits.float(), dim=-1)
        for k in range(tokens.shape[0]):
            timestamp_logprob = logprobs[k, self.tokenizer.timestamp_begin :].logsumexp(dim=-1)
            max_text_token_logprob = logprobs[k, : self.tokenizer.timestamp_begin].max()
            if timestamp_logprob > max_text_token_logprob:
                logits[k, : self.tokenizer.timestamp_begin] = -np.inf


class DecodingTask:
    inference: Inference
    sequence_ranker: SequenceRanker
    decoder: TokenDecoder
    logit_filters: List[LogitFilter]

    def __init__(self, model: "Whisper", options: DecodingOptions):
        self.model = model

        language = options.language or "en"#默认为en
        #is_multilingual：true ；task:transscibe;language="zh"
        tokenizer = get_tokenizer(model.is_multilingual, language=language, task=options.task)#默认语言为en
        self.tokenizer: Tokenizer = tokenizer
        self.options: DecodingOptions = self._verify_options(options)
        #B:n_group=5
        self.n_group: int = options.beam_size or options.best_of or 1
        self.n_ctx: int = model.dims.n_text_ctx#448
        self.sample_len: int = options.sample_len or model.dims.n_text_ctx // 2
        #(50258, 50260, 50359)(sot,transcribe,lang)
        self.sot_sequence: Tuple[int] = tokenizer.sot_sequence
        if self.options.without_timestamps:
            self.sot_sequence = tokenizer.sot_sequence_including_notimestamps
        #(50258, 50260, 50359, 50363)(sot,transcribe,lang，notimestamp)
        self.initial_tokens: Tuple[int] = self._get_initial_tokens()#(50258, 50260, 50359, 50363)
        self.sample_begin: int = len(self.initial_tokens)#4
        self.sot_index: int = self.initial_tokens.index(tokenizer.sot)#0

        # inference: implements the forward pass through the decoder, including kv caching
        self.inference = PyTorchInference(model, len(self.initial_tokens))

        # sequence ranker: implements how to rank a group of sampled sequences
        self.sequence_ranker = MaximumLikelihoodRanker(options.length_penalty)

        #指定decoder decoder: implements how to select the next tokens, given the autoregressive distribution
        if options.beam_size is not None:
            self.decoder = BeamSearchDecoder(
                options.beam_size, tokenizer.eot, self.inference, options.patience
            )
        else:
            self.decoder = GreedyDecoder(options.temperature, tokenizer.eot)#50257

        # logit filters: applies various rules to suppress or penalize certain tokens
        self.logit_filters = []
        if self.options.suppress_blank:
            self.logit_filters.append(SuppressBlank(self.tokenizer, self.sample_begin))
        if self.options.suppress_tokens:
            self.logit_filters.append(SuppressTokens(self._get_suppress_tokens()))
        if not options.without_timestamps:
            precision = CHUNK_LENGTH / model.dims.n_audio_ctx  # usually 0.02 seconds
            max_initial_timestamp_index = None
            if options.max_initial_timestamp:
                max_initial_timestamp_index = round(self.options.max_initial_timestamp / precision)
            self.logit_filters.append(
                ApplyTimestampRules(tokenizer, self.sample_begin, max_initial_timestamp_index)
            )

    def _verify_options(self, options: DecodingOptions) -> DecodingOptions:
        #beam_size和best_of不能同时有数字
        if options.beam_size is not None and options.best_of is not None:
            raise ValueError("beam_size and best_of can't be given together")
        #T=0是贪婪搜索，但best_of不是，因此T=best_of一定要=none
        if options.temperature == 0:
            if options.best_of is not None:
                raise ValueError("best_of with greedy sampling (T=0) is not compatible")
        if options.patience is not None and options.beam_size is None:
            raise ValueError("patience requires beam_size to be given")
        if options.length_penalty is not None and not (0 <= options.length_penalty <= 1):
            raise ValueError("length_penalty (alpha) should be a value between 0 and 1")

        return options

    def _get_initial_tokens(self) -> Tuple[int]:
        tokens = list(self.sot_sequence)
        prefix = self.options.prefix
        prompt = self.options.prompt

        if prefix:
            prefix_tokens = (
                self.tokenizer.encode(" " + prefix.strip()) if isinstance(prefix, str) else prefix
            )
            if self.sample_len is not None:
                max_prefix_len = self.n_ctx // 2 - self.sample_len
                prefix_tokens = prefix_tokens[-max_prefix_len:]
            tokens = tokens + prefix_tokens

        if prompt:
            prompt_tokens = (
                self.tokenizer.encode(" " + prompt.strip()) if isinstance(prompt, str) else prompt
            )
            tokens = [self.tokenizer.sot_prev] + prompt_tokens[-(self.n_ctx // 2 - 1) :] + tokens

        return tuple(tokens)

    def _get_suppress_tokens(self) -> Tuple[int]:
        suppress_tokens = self.options.suppress_tokens

        if isinstance(suppress_tokens, str):
            suppress_tokens = [int(t) for t in suppress_tokens.split(",")]

        if -1 in suppress_tokens:
            suppress_tokens = [t for t in suppress_tokens if t >= 0]
            suppress_tokens.extend(self.tokenizer.non_speech_tokens)
        elif suppress_tokens is None or len(suppress_tokens) == 0:
            suppress_tokens = []  # interpret empty string as an empty list
        else:
            assert isinstance(suppress_tokens, list), "suppress_tokens must be a list"

        suppress_tokens.extend(
            [self.tokenizer.sot, self.tokenizer.sot_prev, self.tokenizer.sot_lm]
        )
        if self.tokenizer.no_speech is not None:
            # no-speech probability is collected separately
            suppress_tokens.append(self.tokenizer.no_speech)

        return tuple(sorted(set(suppress_tokens)))

    def _get_audio_features(self, mel: Tensor):
        if self.options.fp16:#true
            mel = mel.half()
        #mel [1, 80, 3000]
        if mel.shape[-2:] == (self.model.dims.n_audio_ctx, self.model.dims.n_audio_state):
            # encoded audio features are given; skip audio encoding
            audio_features = mel
        else:
            audio_features = self.model.encoder(mel)#torch.Size([1, 1500, 512])

        if audio_features.dtype != (torch.float16 if self.options.fp16 else torch.float32):
            return TypeError(f"audio_features has an incorrect dtype: {audio_features.dtype}")

        return audio_features

    def _detect_language(self, audio_features: Tensor, tokens: Tensor):
        languages = [self.options.language] * audio_features.shape[0]
        lang_probs = None

        if self.options.language is None or self.options.task == "lang_id":
            lang_tokens, lang_probs = self.model.detect_language(audio_features, self.tokenizer)
            languages = [max(probs, key=probs.get) for probs in lang_probs]
            if self.options.language is None:
                tokens[:, self.sot_index + 1] = lang_tokens  # overwrite language tokens

        return languages, lang_probs

    def _main_loop(self, audio_features: Tensor, tokens: Tensor):
        assert audio_features.shape[0] == tokens.shape[0]
        n_batch = tokens.shape[0]#best_of/beam_size 5
        sum_logprobs: Tensor = torch.zeros(n_batch, device=audio_features.device)#[5]
        no_speech_probs = [np.nan] * n_batch

        try:
            for i in range(self.sample_len):
                logits = self.inference.logits(tokens, audio_features)#获得新词logits；torch.Size([5, 4, 51865])表示右移一格的得分
                #非第一个词：torch.Size([5, 4, 51865]) 5个seq的新tokne得分
                if i == 0 and self.tokenizer.no_speech is not None:  # save no_speech_probs
                    probs_at_sot = logits[:, self.sot_index].float().softmax(dim=-1)
                    no_speech_probs = probs_at_sot[:, self.tokenizer.no_speech].tolist()

                # now we need to consider the logits at the last token only
                logits = logits[:, -1]#只取新token的得分 torch.Size([5, 51865])5个seq
                if self.options.top_p!=None:
                    logits=top_k_top_p_filtering(logits,top_p=self.options.top_p)#top-p，留下top-p的token-logits
                # apply the logit filters, e.g. for suppressing or applying penalty to
                # 应用logit过滤器，例如抑制或应用惩罚；生成第一个时抑制空格，此外还抑制特殊token字符
                # for logit_filter in self.logit_filters[1:]:
                #     logit_filter.apply(logits, tokens)
                # expand the tokens tensor with the selected next tokens#返回五个候选的完整token
                tokens, completed = self.decoder.update(tokens, logits, sum_logprobs)#best_of和beam在此处不同；输入5个seq新词的logits和k*seq_len的token和5个候选的累计得分；返回torch.Size([5, 5])，
                #如果预测完了（eot）或者token长度超过n_ctx，停止生成
                if completed or tokens.shape[-1] > self.n_ctx:#tensor(True, device='cuda:0')
                    break
        finally:
            self.inference.cleanup_caching()
        #最后输出tokens[5,33][beam_size,seq_len of finaloutput]
        #sum_logprobs:[5]五个candidates的log_probs
        return tokens, sum_logprobs, no_speech_probs#torch.Size([5, 140])；torch.Size([5])

    @torch.no_grad()
    def run(self, mel: Tensor) -> List[DecodingResult]:
        self.decoder.reset()
        tokenizer: Tokenizer = self.tokenizer
        n_audio: int = mel.shape[0]#1
        #torch.Size([1, 1500, 512])
        audio_features: Tensor = self._get_audio_features(mel)  # encoder forward pass
        tokens: Tensor = torch.tensor([self.initial_tokens]).repeat(n_audio, 1)
        #torch.Size([1, 4])(n_audio,token_nums)

        # detect language if requested, overwriting the language token
        languages, language_probs = self._detect_language(audio_features, tokens)#检测语言并重写token
        if self.options.task == "lang_id":
            return [
                DecodingResult(audio_features=features, language=language, language_probs=probs)
                for features, language, probs in zip(audio_features, languages, language_probs)
            ]#如果只需要判断language就可以直接返回DecodingResult结果了

        # repeat the audio & text tensors by the group size, for beam search or best-of-n sampling
        # 重复音频和文本张量的组大小，用于波束搜索或n的最佳抽样;batch_size扩大五倍
        # audio_features before[1,1500,768] after[5,1500,768]
        # beam_size=5
        audio_features = audio_features.repeat_interleave(self.n_group, dim=0)#torch.Size([5, 1500, 512])
        tokens = tokens.repeat_interleave(self.n_group, dim=0).to(audio_features.device)#[5,4]
        # call the main sampling loop
        tokens, sum_logprobs, no_speech_probs = self._main_loop(audio_features, tokens)
        #torch.Size([5, 140]);torch.Size([5]);
        # reshape the tensors to have (n_audio, n_group) as the first two dimensions
        audio_features = audio_features[:: self.n_group]#torch.Size([1, 1500, 512])
        no_speech_probs = no_speech_probs[:: self.n_group]#[float]
        assert audio_features.shape[0] == len(no_speech_probs) == n_audio

        tokens = tokens.reshape(n_audio, self.n_group, -1)#torch.Size([1, 5, 140])
        sum_logprobs = sum_logprobs.reshape(n_audio, self.n_group)#torch.Size([1, 5])

        # get the final candidates for each group, and slice between the first sampled token and EOT
        # 获取每个组的最终候选对象，并在第一个采样令牌和EOT之间进行切片
        tokens, sum_logprobs = self.decoder.finalize(tokens, sum_logprobs)#torch.Size([1, 5, 141]),list:[[score1,...,score5]]
        tokens: List[List[Tensor]] = [
            [t[self.sample_begin : (t == tokenizer.eot).nonzero()[0, 0]] for t in s] for s in tokens
        ]#[[seq1_tokesid_tensor,seq2_tokens,...,seq5_tokens]]#纯文本的token_id
        # beam search 根据sum_logprobs选取top-k tokens
        # top-p search 根据sum_probs采样top-p tokens
        # select the top-ranked sample in each group
        # 在每组中选择排名靠前的样本

        selected = self.sequence_ranker.rank(tokens, sum_logprobs)#[beam_id]
        tokens: List[List[int]] = [t[i].tolist() for i, t in zip(selected, tokens)]#选取sum_logprobs最大的[[token_id1,token_id2,...]]
        texts: List[str] = [tokenizer.decode(t).strip() for t in tokens]

        sum_logprobs: List[float] = [lp[i] for i, lp in zip(selected, sum_logprobs)]#eg [-50.72019577026367]
        avg_logprobs: List[float] = [lp / (len(t) + 1) for t, lp in zip(tokens, sum_logprobs)]

        fields = (texts, languages, tokens, audio_features, avg_logprobs, no_speech_probs)
        if len(set(map(len, fields))) != 1:
            raise RuntimeError(f"inconsistent result lengths: {list(map(len, fields))}")

        return [
            DecodingResult(
                audio_features=features,
                language=language,
                tokens=tokens,
                text=text,
                avg_logprob=avg_logprob,
                no_speech_prob=no_speech_prob,
                temperature=self.options.temperature,
                compression_ratio=compression_ratio(text),
            )
            for text, language, tokens, features, avg_logprob, no_speech_prob in zip(*fields)
        ]


@torch.no_grad()
def decode(model: "Whisper", mel: Tensor, options: DecodingOptions = DecodingOptions()) -> Union[DecodingResult, List[DecodingResult]]:
    """
    Performs decoding of 30-second audio segment(s), provided as Mel spectrogram(s).

    Parameters
    ----------
    model: Whisper
        the Whisper model instance

    mel: torch.Tensor, shape = (80, 3000) or (*, 80, 3000)
        A tensor containing the Mel spectrogram(s)

    options: DecodingOptions
        A dataclass that contains all necessary options for decoding 30-second segments

    Returns
    -------
    result: Union[DecodingResult, List[DecodingResult]]
        The result(s) of decoding contained in `DecodingResult` dataclass instance(s)
    """
    #false
    single = mel.ndim == 2
    if single:
        mel = mel.unsqueeze(0)
    #mel[1,80,3000],80bands,3000time
    result = DecodingTask(model, options).run(mel)
    
    if single:
        result = result[0]

    return result
