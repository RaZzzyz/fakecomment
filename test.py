from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import KeyedVectors
glove_input_file = 'vectors.txt'
word2vec_output_file = 'wv1.txt'
(count, dimensions) = glove2word2vec(glove_input_file, word2vec_output_file)
# print(count, '\n', dimensions)
wv = KeyedVectors.load_word2vec_format('wv1.txt', binary=False)
print(wv['蒙牛'])