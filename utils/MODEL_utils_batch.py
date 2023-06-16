
## 전처리 필요 라이브러리
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import numpy as np
import pandas as pd
import re
import random
from tqdm import tqdm, tqdm_notebook

# 자칫하면 오류날 수도 있는 부분
import gluonnlp as nlp

# Model
from kobert_tokenizer import KoBERTTokenizer
from transformers import BertModel
from transformers import AdamW
from transformers.optimization import get_cosine_schedule_with_warmup
from transformers import ElectraForSequenceClassification, ElectraTokenizer


seed = 2021
deterministic = True

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# GPU 있으면 할당
device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')

######################### KoBERT #########################
bert = BertModel.from_pretrained('skt/kobert-base-v1', return_dict=False)
tokenizer = KoBERTTokenizer.from_pretrained('skt/kobert-base-v1')
vocab = nlp.vocab.BERTVocab.from_sentencepiece(tokenizer.vocab_file, padding_token='[PAD]')

######################### KoELECTRA #########################
electramodel = ElectraForSequenceClassification.from_pretrained("monologg/koelectra-base-v3-discriminator")  # KoELECTRA-Small-v3
tokenizer_electra = ElectraTokenizer.from_pretrained("monologg/koelectra-base-v3-discriminator")


# KoBERT 모델 입력 전처리
class BERTSentenceTransform:
    def __init__(self, tokenizer, max_seq_length,vocab, pad=True, pair=True):
        self._tokenizer = tokenizer
        self._max_seq_length = max_seq_length
        self._pad = pad
        self._pair = pair
        self._vocab = vocab 

    def __call__(self, line):
        # convert to unicode
        text_a = line[0]
        if self._pair:
            assert len(line) == 2
            text_b = line[1]

        ##### 여기 수정!! #####
        tokens_a = self._tokenizer.tokenize(text_a)
        tokens_b = None

        if self._pair:
            tokens_b = self._tokenizer(text_b)

        if tokens_b:
            self._truncate_seq_pair(tokens_a, tokens_b,
                                    self._max_seq_length - 3)
        else:
            if len(tokens_a) > self._max_seq_length - 2:
                tokens_a = tokens_a[0:(self._max_seq_length - 2)]
        vocab = self._vocab
        tokens = []
        tokens.append(vocab.cls_token)
        tokens.extend(tokens_a)
        tokens.append(vocab.sep_token)
        segment_ids = [0] * len(tokens)

        if tokens_b:
            tokens.extend(tokens_b)
            tokens.append(vocab.sep_token)
            segment_ids.extend([1] * (len(tokens) - len(segment_ids)))

        input_ids = self._tokenizer.convert_tokens_to_ids(tokens)

        # The valid length of sentences. Only real  tokens are attended to.
        valid_length = len(input_ids)

        if self._pad:
            # Zero-pad up to the sequence length.
            padding_length = self._max_seq_length - valid_length
            # use padding tokens for the rest
            input_ids.extend([vocab[vocab.padding_token]] * padding_length)
            segment_ids.extend([0] * padding_length)

        return np.array(input_ids, dtype='int32'), np.array(valid_length, dtype='int32'),\
            np.array(segment_ids, dtype='int32')

class BERTDataset(Dataset):
    def __init__(self, dataset, sent_idx, label_idx, bert_tokenizer, vocab, max_len, pad, pair):
        transform = BERTSentenceTransform(bert_tokenizer, max_seq_length=max_len, vocab=vocab, pad=pad, pair=pair)

        self.sentences = [transform([i[sent_idx]]) for i in dataset]
        self.labels = [np.int32(i[label_idx]) for i in dataset]

    def __getitem__(self, i):
        return (self.sentences[i] + (self.labels[i], ))

    def __len__(self):
        return (len(self.labels))


# KoBERT 모델 구조
class KoBERTClassifier(nn.Module):
  def __init__(self, bert, hidden_size=768, num_classes=2, dr_rate=None, params=None):
    super(KoBERTClassifier, self).__init__()
    self.bert = bert
    self.dr_rate = dr_rate

    self.classifier = nn.Linear(hidden_size, num_classes)
    if dr_rate:
      self.dropout = nn.Dropout(p=dr_rate)

  def forward(self, token_ids, valid_length, segment_ids, attention_mask):
    _, pooler = self.bert(input_ids=token_ids, token_type_ids=segment_ids.long(), attention_mask=attention_mask.float().to(token_ids.device))
    if self.dr_rate:
      out = self.dropout(pooler)
    out = self.classifier(out)
    return out


# KoBERT 추론
def Custom_KoBERT_Predict(test, max_len=146, dr_rate=0.3):
  def sen_label(sentences):
    new_sen = []
    for i in sentences:
      if '"' in i:
        i = re.sub('"', '', i)
      new_sen.append([i, '0'])
    return new_sen

  test = sen_label(test) # 입력 포맷

  test_data = BERTDataset(test, 0, 1, tokenizer, vocab, max_len, True, False)
  test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=32)

  # model 불러오기
  model = KoBERTClassifier(bert, dr_rate=0.3).to(device)

  model_state_dict = torch.load("./models/KoBERT_state_dict.pt", map_location=device)
  model.load_state_dict(model_state_dict)

  def gen_attention_mask(token_ids, valid_length):
    attention_mask = torch.zeros_like(token_ids)
    for i, v in enumerate(valid_length):
      attention_mask[i][:v] = 1
    return attention_mask.float()

  # 테스트
  model.eval()
  with torch.no_grad():
    pred = []
    for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(test_dataloader):
      token_ids = token_ids.long().to(device)
      segment_ids = segment_ids.long().to(device)
      label = label.long().to(device)
      valid_length = valid_length

      attention_mask = gen_attention_mask(token_ids, valid_length)

      out = model(token_ids, valid_length, segment_ids, attention_mask)
      
      softamx_out = nn.functional.softmax(out, dim=1)
      pred_labels = softamx_out.argmax(dim=1)

      pred_list = list(pred_labels.detach().cpu().numpy())
      pred += pred_list
  return pred

######################### KoELECTRA #########################
# KoELECTRA 모델 입력 전처리
class ElectraClassificationDataset(Dataset):
  def __init__(self, test, tokenizer, max_len):
    self.input_data = list(test)
    self.tokenizer = tokenizer
    self.max_len = max_len

  def __len__(self):
    return len(self.input_data)

  def __getitem__(self, idx):
    inputs = self.tokenizer(self.input_data[idx], 
                            return_tensors='pt', 
                            truncation=True, 
                            max_length=self.max_len, 
                            padding='max_length',
                            add_special_tokens=True)

    return {
        'input_ids': inputs['input_ids'][0],
        'attention_mask': inputs['attention_mask'][0],
        'token_type_ids': inputs['token_type_ids'][0]
          }


# KoELECTRA 추론
def Custom_KoELECTRA_Predict(test, max_len=146, dr_rate=0.3):
  def sen_label(sentences):
    new_sen = []
    for i in sentences:
      if '"' in i:
        i = re.sub('"', '', i)
      new_sen.append([i, '0'])
    return new_sen

  test = sen_label(test) # 입력 포맷

  test_data = ElectraClassificationDataset(test, tokenizer_electra, max_len)
  test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=32)

  # model 불러오기
  model = electramodel.to(device)

  model_state_dict = torch.load("./models/KoELECTRA_state_dict.pt", map_location=device)
  model.load_state_dict(model_state_dict)

  # 테스트
  model.eval()
  with torch.no_grad():
    pred = []
    for batch_id, data in enumerate(test_dataloader):
      token_ids = data['input_ids'].long().to(device)
      token_type_ids = data['token_type_ids'].long().to(device)
      attention_mask = data['attention_mask'].long().to(device)

    out = model(input_ids=token_ids, 
                token_type_ids=token_type_ids, 
                attention_mask=attention_mask)
    
    softmax_out = nn.functional.softmax(out.logits, dim=1)
    pred_labels = softmax_out.argmax(dim=1)
    pred += pred_labels

  return pred
