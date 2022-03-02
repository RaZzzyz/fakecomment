import re
import jieba
from gensim.models import Word2Vec,TfidfModel
import numpy as np
from gensim import corpora

def deal_data():
    stop=[]
    with open('stopwords.txt','r',encoding='utf-8')as f:
        for word in f:
            stop.append(word.replace('\n',''))
    with open('./data/review.txt','r',encoding='utf-8')as f1:
        tem=[st for st in f1.readlines()]
    with open('./data/comments.txt','r',encoding='utf-8')as f1:
        tem+=[st for st in f1.readlines()]
    with open('./data/mengniu.txt','r',encoding='utf-8')as f1:
        tem+=[st for st in f1.readlines()]
    with open('corpus.txt','w',encoding='utf-8')as f2:  #用于生成同时存储数据集和标签的文件
        for s in tem:
            line=(''.join(s.split()[:-1])).replace('～','')
            label=s.split()[-1]
            line1 = re.sub(r"[\s+\\\/_,$%^*()-【】+\"\']+|[+——~@#￥%……&*]+", "", line)
            wordlist = list(jieba.cut(line1))  # 用结巴分词，对每行内容进行分词  
            outstr = ''
            for word in wordlist:
                if word not in stop:
                    outstr+=word
                    outstr+=' '
            if outstr!='':
                f2.write(outstr+'\t'+label+'\n')
    with open('corpus1.txt','w',encoding='utf-8')as f2:  #用于生成使用glove标记的数据
            for s in tem:
                line=(''.join(s.split()[:-1])).replace('～','')
                label=s.split()[-1]
                line1 = re.sub(r"[\s+\\\/_,$%^*()-【】+\"\']+|[+——~@#￥%……&*]+", "", line)
                wordlist = list(jieba.cut(line1))  # 用结巴分词，对每行内容进行分词  
                outstr = ''
                for word in wordlist:
                    if word not in stop:
                        outstr+=word
                        outstr+=' '
                if outstr!='':
                    f2.write(outstr+'\n')                
if __name__ == '__main__':
    # deal_data()
    #构建tfidf模型
    
    #测试tfidf的使用
    # with open('corpus.txt','r',encoding='utf-8')as f:
    #    s= [line.split('\t')[0].split() for line in f.readlines()]
    # dic = corpora.Dictionary(s)
    # new_corpus = [dic.doc2bow(text) for text in s]   
    # tfidf=TfidfModel(new_corpus)
    # a=list(tfidf[new_corpus[0]])
    
    #根据vectors生成wv文件
    # from gensim.scripts.glove2word2vec import glove2word2vec
    # from gensim.models import KeyedVectors
    # glove_input_file = 'vectors.txt'
    # word2vec_output_file = 'wv1.txt'
    # (count, dimensions) = glove2word2vec(glove_input_file, word2vec_output_file)