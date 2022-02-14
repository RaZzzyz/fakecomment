from torch.utils.data import Dataset
from gensim import corpora
from gensim.models import TfidfModel
class corpus(Dataset):
    def __init__(self,data,wv):
        self.data=data
        self.wv=wv
        s=[line.split('\t')[0].split() for line in data]
        self.dic = corpora.Dictionary(s)
        self.new_corpus = [self.dic.doc2bow(text) for text in s]   
        self.tfidf=TfidfModel(self.new_corpus)
    def __getitem__(self, index) :
        line=self.data[index]
        s=line.split('\t')[0].split()
        label=float(line.split('\t')[1][0])
        c=0
        vec=[0]*128
        ind=self.tfidf[self.new_corpus[index]]
        for i in s:
            e=1
            if i in self.dic:
                id=self.dic[i]
                for n in ind:
                    id1=[x for x in n][0]
                    v=[x for x in n][1]
                    if id==id1:
                        e=v
                        break
                    elif id<id1:
                        break  
            vec+=self.wv[i]*e
            c+=1
        return vec/c,label
    def __len__(self):
        return len(self.data)