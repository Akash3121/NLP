import numpy as np
from gensim.models import Word2Vec
from nltk.corpus import gutenberg
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import gensim.downloader as api

 

def reduce_dimensions(wv):
    from sklearn.manifold import TSNE                   # for dimensionality reduction
    num_dimensions = 2  # final num dimensions (2D, 3D, etc)
    # extract the words & their vectors, as numpy arrays
    vectors = np.asarray(wv.vectors)
    labels = wv.index_to_key  # list
    # reduce using t-SNE
    tsne = TSNE(n_components=num_dimensions, random_state=0)
    vectors = tsne.fit_transform(vectors) 

    x_vals = [v[0] for v in vectors]
    y_vals = [v[1] for v in vectors]
    return x_vals, y_vals, labels

 

def plot_with_matplotlib(x_vals, y_vals, labels, wordsToBePlotted):
    plt.figure(figsize=(12, 12))
    plt.scatter(x_vals, y_vals)
    for w in wordsToBePlotted:
        if w in labels :
            i = labels.index(w)
            print("Plotting",w,"at",x_vals[i], y_vals[i])
            plt.annotate(w, (x_vals[i], y_vals[i]))
        else :
            print(w,"cannot be plotted because its word embedding is not given.")
    plt.show()

 

#Training and Pre-processing

sentences=gutenberg.sents()
print("Length of the sentences: ",len(list(sentences)))
sentences_list=list(sentences)
sentences_list_lower = [[''.join([w.lower() for w in s]) for s in b] for b in sentences_list]
print(sentences_list_lower)

 

#Training CBOW
cbowModels = Word2Vec(sentences, vector_size=120, window=5, min_count=1, workers=4, sg=0)
 

#Training Skipgram
skipgramModels = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4, sg=1)

 
#Pretrained Wordembeddings
wv = api.load("word2vec-google-news-300")

 

#Saving the Embeddings
cbowModels.wv.save_word2vec_format("cbowstep2.txt")
skipgramModels.wv.save_word2vec_format("skipgramstep2.txt")

 

#Visualisation
wordsToBePlotted=['discouraged' , 'encouraged' , 'full' , 'enough' , 'woman' , 'man' , 'happy' , 'sorrowful' , 'present', 'former' , 'father' , 'mother' , 'contrasted' , 'valuable' , 'remarkably' , 'object' , 'subject' , 'intrest' , 'trust' , 'betray']
cbow_x_vals, cbow_y_vals, cbow_labels = reduce_dimensions(cbowModels.wv)

 

print("CBOW Model Word Embeddings")
plot_with_matplotlib(cbow_x_vals, cbow_y_vals, cbow_labels, wordsToBePlotted)

 

skipgram_x_vals,skipgram_y_vals, skipgram_labels = reduce_dimensions(skipgramModels.wv)

 

print("SkipGram Model Word Embeddings")
plot_with_matplotlib(skipgram_x_vals, skipgram_y_vals, skipgram_labels, wordsToBePlotted)

 

 

#Evaluation of Similarity scores

 

cbowEvaluation=cbowModels.wv.evaluate_word_pairs("10pairs.txt")
SkipgramEvaluation=skipgramModels.wv.evaluate_word_pairs("10pairs.txt")
googlenewsEvaluation=wv.evaluate_word_pairs("10pairs.txt")

 

print("CBOW Evaluation:\n", cbowEvaluation)
print("SkipGram Evaluation:\n", SkipgramEvaluation)
print("Goggle News Evaluation:\n", googlenewsEvaluation)

 

#Finding out similar words
wordsToTest=['discouraged', 'managed', 'full', 'enough', 'man']
for words in wordsToTest:

 

    print(f"Topmost 5 similar words to {words} using CBOW:")

 

    for similarWord, similarityScore in cbowModels.wv.most_similar(words, topn=5):

 

        print(f"{similarWord} ({similarityScore:.2f})")   

 

    print(f"\nTopmost 5 similar words to {words} using SkipGram:")

 

    for similarWord, similarityScore in skipgramModels.wv.most_similar(words, topn=5):

 

        print(f"{similarWord} ({similarityScore:.2f})")

 

    print(f"\nTopmost 5 similar words to {words} using Google News embeddings:")

 

    for similarWord, similarityScore in wv.most_similar(words, topn=5):

 

        print(f"{similarWord} ({similarityScore:.2f})")

 

    print("\n" + "***"*15 + "\n")
