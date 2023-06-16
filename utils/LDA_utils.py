
## 전처리 필요 라이브러리
import re
import time
import datetime
import numpy as np
import pandas as pd

# Tokenize
# from pykospacing import Spacing # 띄어쓰기
from gensim import corpora # 단어 빈도수 계산 패키지
import pyLDAvis.gensim_models # LDA 시각화용 패키지?

# 한국어 형태소 분석기 중 성능이 가장 우수한 Mecab 사용
from konlpy.tag import *
mecab = Mecab()


## 토픽모델링 필요 라이브러리
import gensim
from gensim.corpora import Dictionary
from gensim.models import ldaseqmodel
from gensim.models import CoherenceModel
from gensim.models.callbacks import CoherenceMetric
from gensim.models.callbacks import PerplexityMetric

# 기타
from tqdm import tqdm
import warnings # 경고 메시지 무시
import matplotlib.pyplot as plt
warnings.filterwarnings(action='ignore')

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)



## --------------- 데이터 전처리 --------------- ##
def date_extract(dataset, start_day:str, printing=False):
  """
  Input: OTT별 전체 데이터프레임
  Output: 최근 n년의 기간에 해당하는 데이터 추출, 결측처리 완료

  start_day : yyyy-mm-dd
  """

  dataset['at'] = pd.to_datetime(dataset['at'])
  dataset = dataset[dataset['at'] >= start_day]
  dataset = dataset.dropna(subset=['content'])
  dataset['year'] = dataset['at'].dt.year
  dataset = dataset.sort_values(by='at', ascending=True)
  
  if printing:
    for y in dataset['year'].unique():
      tmp = dataset[dataset['year'] == y]
      print(f'{y}: {len(tmp)} rows\n')
  return dataset


class Data_processing:
  '''
  LDA 적용을 위한 모든 전처리
  '''
  def __init__(self, dataset):
    self.dataset = dataset
    self.replace_list = pd.read_excel('./data/replace_list.xlsx')
    self.stopwords_list = list(pd.read_excel('./data/stopword_list.xlsx')['stopword'])

  def ko_language(self, text):
    # 한글 외 문자 제거 & 띄어쓰기 맞추기
    hangul = re.compile('[^가-힣 ]')
    result = hangul.sub('', text)
    return result

  def replace_word(self, text):
    # 단어 치환
    for i, word in enumerate(self.replace_list['before_replacement']):
      if word in text:
        result = re.sub(word, self.replace_list['after_replacement'][i], text)
        return result
    return text

  def tokenize(self, text):
    # 토큰화
    return mecab.nouns(text)

  def remove_stopwords(self, text, add_stopwords=None):
    # 불용어제거
    if add_stopwords:
      self.stopwords_list += add_stopwords
    result = [x for x in text if x not in self.stopwords_list]
    return result
  
  def select_review(self, data, min_token_n:int, max_token_n:int):
    # 특정 토큰 개수의 리뷰 선별
    remove_idx_list = []
    for i in range(len(data)):
      if min_token_n <= len(data.iloc[i]['review_prep']) <= max_token_n:
        continue
      else:
        remove_idx_list.append(i)

    return data.drop(remove_idx_list, axis=0)

  def get_token(self, add_stopwords=None, min_token_n:int=3, max_token_n:int=1000):
    self.dataset['review_prep'] = self.dataset['content'].apply(lambda x:self.ko_language(x))
    self.dataset = self.dataset.reset_index(drop=True)
    print('ko_language done..')
    self.dataset['review_prep'] = self.dataset['review_prep'].apply(lambda x:self.replace_word(x))
    self.dataset = self.dataset.reset_index(drop=True)
    print('replace_word done..')
    self.dataset['review_prep'] = self.dataset['review_prep'].apply(lambda x:self.tokenize(x))
    self.dataset = self.dataset.reset_index(drop=True)
    print('tokenize done..')
    self.dataset['review_prep'] = self.dataset['review_prep'].apply(lambda x:self.remove_stopwords(x, add_stopwords))
    self.dataset = self.dataset.reset_index(drop=True)
    print('remove_stopwords done..')
    self.dataset = self.select_review(self.dataset, min_token_n, max_token_n)
    return list(self.dataset['review_prep']), self.dataset



## ------------------ 모델링 ------------------ ##
class Model:
  '''
  LDA 모델

  no_below = 분석에 사용할 단어의 최소 빈도 수 제약 (ex) 2이면, 빈도가 최소 2이상 넘어간 단어만 취급)
  no_above = 전체의 몇 %로 이상 차지하는 단어를 필터링 할 것인지?
  '''
  def __init__(self, inputs, num_topics:int, no_below:int=2):
    self.dictionary = corpora.Dictionary(inputs)
    self.dictionary.filter_extremes(no_below=no_below)
    self.corpus = [self.dictionary.doc2bow(x) for x in inputs]
    self.inputs = inputs
    self.num_topics = num_topics

  def LDAseqmodel(self, timeslices, chunksize=2000, passes=10):
    '''
    CTM 함수
    '''

    self.model = ldaseqmodel.LdaSeqModel(
        corpus=self.corpus,
        id2word=self.dictionary,
        time_slice=timeslices,
        num_topics=self.num_topics,
        chunksize=chunksize,
        passes=passes)
    
  
  def LDA_model(self, chunksize=2000, passes=20, iterations=400, eval_every=None):
    '''
    num_topics: 생성될 토픽의 개수
    chunksize: 한번의 트레이닝에 처리될 문서의 개수
    passes: 딥러닝에서 Epoch와 같은 개념으로, 전체 corpus로 모델 학습 횟수 결정
    interations: 문서 당 반복 횟수
    '''
    temp = self.dictionary[0]
    id2word = self.dictionary.id2token

    self.model = LdaModel(
      corpus=self.corpus,
      id2word=id2word,
      chunksize=chunksize,
      alpha='auto',
      eta='auto',
      iterations=iterations,
      num_topics=self.num_topics,
      passes=passes,
      eval_every=eval_every)


  def print_topic_prop(self, topn=10, num_words=20):
    self.LDA_model()
    topics = self.model.print_topics(num_words=num_words)


    # 토픽별 포함 단어 추출
    topic_words = {}
    for idx, words in topics:
      topic_words[idx] = words.split('+')

    topic_table = pd.DataFrame(topic_words)
    topic_table.columns = [f'topic_{t+1}' for t in range(len(topics))]

    # coherence
    coherence_model_lda = CoherenceModel(model=self.model, texts=self.inputs, dictionary = self.dictionary, topn=10)
    coherence_lda = coherence_model_lda.get_coherence()
    print('LDA done..')

    return topic_table, coherence_lda, topics
