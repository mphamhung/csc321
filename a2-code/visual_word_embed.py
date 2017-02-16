import numpy as np
import cPickle as pickle
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
 
 
def main():    
    embeddings_file = 'data/word_embeddings.p'
    embed_dict = load_embeddings(embeddings_file)
    
    vocabulary = embed_dict.keys()
    word_vec = np.array(embed_dict.values())

    ############################################################################
    # You should modify this part by selecting a subset of word embeddings 
    # for better visualization
    ############################################################################
    
    word_vec = word_vec[300:400, :]    
    vocabulary = vocabulary[300:400]

    ############################################################################

    tsne = TSNE(n_components=2, random_state=0)
    np.set_printoptions(suppress=True)    
    Y = tsne.fit_transform(word_vec)
 
    plt.scatter(Y[:, 0], Y[:, 1])
    for label, x, y in zip(vocabulary, Y[:, 0], Y[:, 1]):
        plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points')
    plt.show()


def load_embeddings(file_name):
    """ Load in the embeddings """
    return pickle.load(open(file_name, 'rb'))


if __name__ == '__main__':    
    main()
