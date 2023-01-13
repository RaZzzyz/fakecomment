# fakecomment
基于nlp的虚假评论分析

本次使用LSTM进行分析词语大小，不同的卷积核大小代表了不同的分析程度。

使用glove和tfidf两种方法来生成word embedding,glove的效果会好一点，最高的准确率维持在75左右。

也尝试了使用bert模型生成word embedding，但最终没什么大的提升。

