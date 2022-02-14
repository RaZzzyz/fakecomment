import re
import jieba
from gensim.models import Word2Vec,TfidfModel
import numpy as np
from gensim import corpora

def deal_data():
    stop=[]
    with open('stopwords.txt','r',encoding='utf-8')as f:
        print(f)
        for word in f:
            stop.append(word.replace('\n',''))
    with open('./data/dev.txt','r',encoding='utf-8')as f1:
        tem=[st for st in f1.readlines()]
    with open('./data/test.txt','r',encoding='utf-8')as f1:
        tem+=[st for st in f1.readlines()]
    with open('./data/train.txt','r',encoding='utf-8')as f1:
        tem+=[st for st in f1.readlines()]
    with open('corpus.txt','w',encoding='utf-8')as f2:  
        for s in tem:
            line=s.split('\t')[0].replace('～','')
            label=s.split('\t')[1]
            line1 = re.sub(r"[\s+\\\/_,$%^*()-【】+\"\']+|[+——~@#￥%……&*]+", "", line)
            wordlist = list(jieba.cut(line1))  # 用结巴分词，对每行内容进行分词  
            outstr = ''
            for word in wordlist:
                if word not in stop:
                    outstr+=word
                    outstr+=' '
            if outstr!='':
                f2.write(outstr+'\t'+label)
if __name__ == '__main__':
    # deal_data()
    
    #构建tfidf模型
    with open('corpus.txt','r',encoding='utf-8')as f:
       s= [line.split('\t')[0].split() for line in f.readlines()]
    dic = corpora.Dictionary(s)
    new_corpus = [dic.doc2bow(text) for text in s]   
    tfidf=TfidfModel(new_corpus)
    tfidf.save('tfidf.m')
    tfidf=TfidfModel.load('tfidf.m')
    a=list(tfidf[new_corpus[0]])
    print(a)