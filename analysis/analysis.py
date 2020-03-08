import matplotlib.pyplot as plt
import numpy as np
import math 
import pandas as pd
import tqdm

import matplotlib.pyplot as plt
import seaborn as sns

from collections import Counter



plt.style.use('ggplot')

import spacy; nlp = spacy.load('en_core_web_sm')

with open('wiki.valid.log_probs.knn.txt') as infile:
    knn_log_probs = np.array([float(log_prob) for log_prob in infile.read().split()])

with open('wiki.valid.log_probs.orig.txt') as infile:
    orig_log_probs = np.array([float(log_prob) for log_prob in infile.read().split()])

with open('wiki.valid.tokens') as infile:
    tokens = infile.read().split()

print("Loading training tokens...")
with open('wiki.train.tokens') as infile:
    train_tokens = infile.read().split()

print("TODO, CHECK ME!!! Skipping first 1536 training tokens...")
train_tokens = train_tokens[1536:] # TODO, this isn't needed anymore if you rerun eval_lm

def compare_and_plot_knnlm_parametric():
    print("knnlm prob > parametric prob: {:.2%}".format((knn_log_probs > orig_log_probs).mean()))
    print("knnlm didn't find correct word: {:.2%}".format((knn_log_probs == -10000.0).mean()))
    oracle_log_probs = np.maximum(knn_log_probs, orig_log_probs)
    
    avg_oracle_nll = -oracle_log_probs.sum() / len(oracle_log_probs) / math.log(2)  # convert to base 2
    print("oracle perplexity: {}".format(2**avg_oracle_nll))

    # TODO convert the log probabilities to perplexities
    filtered_orig_log_probs = orig_log_probs[knn_log_probs != -10000.0]
    filtered_knn_log_probs = knn_log_probs[knn_log_probs != -10000.0]

    avg_filtered_knn_nll = -filtered_knn_log_probs.sum() / len(filtered_knn_log_probs) / math.log(2)  # convert to base 2
    print("filtered knn perplexity: {}".format(2**avg_filtered_knn_nll))

    avg_filtered_orig_nll = -filtered_orig_log_probs.sum() / len(filtered_orig_log_probs) / math.log(2)  # convert to base 2
    print("filtered orig perplexity: {}".format(2**avg_filtered_orig_nll))

    plt.scatter(filtered_orig_log_probs, filtered_knn_log_probs, alpha=0.5, s=0.1, c='red')
    min_filtered_log_prob = min(filtered_orig_log_probs.min(), filtered_knn_log_probs.min())
    plt.plot(
        [min_filtered_log_prob, 0],
        [min_filtered_log_prob, 0])
    plt.xlabel('parametric perplexity')
    plt.ylabel('knn perplexity')
    plt.savefig('perplexity_scatterplot.png')
    # plt.show()

def compare_knn_parametric_word_frequency():
    filtered_orig_log_probs = orig_log_probs[knn_log_probs != -10000.0]
    filtered_knn_log_probs = knn_log_probs[knn_log_probs != -10000.0]
    frequencies = Counter(tokens)
    filtered_frequencies = [frequencies[token] \
                            for knn_found, token in zip(knn_log_probs != -10000.0, tokens) if knn_found]
    
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle('knn vs original')

    ax1.scatter(filtered_frequencies, filtered_knn_log_probs, alpha=0.1, s=5, c='blue', marker="o",label='knn log probs')
    ax1.set_xlabel('word frequency')
    ax1.set_ylabel('knn log prob')
    ax1.set_ylim(-25, 0) 

    ax2.scatter(filtered_frequencies, filtered_orig_log_probs, alpha=0.1, s=5, c='red', marker="s", label='original log probs')
    ax2.set_xlabel('word frequency')
    ax2.set_ylabel('original log prob')
    ax2.set_ylim(-25, 0)
    plt.savefig('word_frequency_scatterplot.png')
    plt.show()

def compare_knn_parametric_word_frequency_difference():
    filtered_orig_log_probs = orig_log_probs[knn_log_probs != -10000.0]
    filtered_knn_log_probs = knn_log_probs[knn_log_probs != -10000.0]
    frequencies = Counter(tokens)
    filtered_frequencies = [frequencies[token] \
                            for knn_found, token in zip(knn_log_probs != -10000.0, tokens) if knn_found]
    
    difference = filtered_knn_log_probs - filtered_orig_log_probs
    plt.scatter(filtered_frequencies, difference, alpha=0.1, s=5)
    plt.xlabel('word frequency')
    plt.ylabel('knn log prob - original log prob')
    plt.savefig('difference_word_frequency_scatterplot.png')
    plt.show()

def compare_knn_parametric_difference():
    filtered_orig_log_probs = orig_log_probs[knn_log_probs != -10000.0]
    filtered_knn_log_probs = knn_log_probs[knn_log_probs != -10000.0]
    difference = filtered_knn_log_probs - filtered_orig_log_probs
    plt.hist(difference, bins=100)
    plt.ylabel('knn log prob - original log prob')
    plt.savefig('difference_knn_original.png')
    plt.show()



def compare_knn_parametric_spacy():
    doc = spacy.tokens.doc.Doc(
        nlp.vocab,
        words=tokens,
        spaces=[True for token in tokens])
    for name, proc in nlp.pipeline:
        doc = proc(doc)

    # KNN perplexity on entities
    entity_mask = get_mask(doc)

    filtered_orig_log_probs = orig_log_probs[knn_log_probs != -10000.0]
    filtered_knn_log_probs = knn_log_probs[knn_log_probs != -10000.0]

    # TODO these need to be filtered
    entity_knn_log_probs = knn_log_probs * entity_mask
    avg_entity_knn_nll = -entity_knn_log_probs.sum() / entity_mask.sum() / math.log(2)
    print("knn perplexity on entities {}".format(avg_entity_knn_nll))

    entity_orig_log_probs = orig_log_probs * entity_mask
    avg_entity_orig_nll = -entity_orig_log_probs.sum() / entity_mask.sum() / math.log(2)
    print("orig perplexity on entities {}".format(avg_entity_orig_nll))


    perplexity_dataframe = pd.DataFrame(dict(knn_log_probs=filtered_knn_log_probs,
                                             orig_log_probs=filtered_orig_log_probs,
                                             color=[token.ent_type_ for knn_found, token in zip(knn_log_probs != -10000.0, doc) if knn_found]))
    sns.lmplot('knn_log_probs', 'orig_log_probs', data=perplexity_dataframe, hue='color', fit_reg=False, scatter_kws={"s": 15})
    min_filtered_log_prob = min(filtered_orig_log_probs.min(), filtered_knn_log_probs.min())
    plt.plot(
        [min_filtered_log_prob, 0],
        [min_filtered_log_prob, 0])
    plt.savefig('entities_scatterplot.png')
    plt.show()


def get_mask(doc, entity_type=None):
    if entity_type:
        return np.array([1 if token.ent_type_ == entity_type else 0 for token in doc])
    else:
        return np.array([1 if token.ent_type_ else 0 for token in doc])

def print_examples():
    filtered_orig_log_probs = orig_log_probs[knn_log_probs != -10000.0]
    filtered_knn_log_probs = knn_log_probs[knn_log_probs != -10000.0]
    difference = filtered_knn_log_probs - filtered_orig_log_probs
    filtered_tokens = [token for knn_found, token in zip(knn_log_probs != -10000.0, tokens) if knn_found]

    context_length = 30
    difference_threshold = 10
    absolute_threshold = 0.1

    # Indices where knn is good and differece is large
    indices_knn_better = [idx for idx, (diff, knn_log_prob) \
                          in enumerate(zip(difference, filtered_knn_log_probs)) \
                          if abs(knn_log_prob) < absolute_threshold and abs(diff) > difference_threshold]
    for idx in indices_knn_better:
        print(' '.join(filtered_tokens[idx - context_length:idx+1]))
    print('\n\n')
    print('{} examples where KNN is less than {} and difference greater than {}'.format(len(indices_knn_better), absolute_threshold, difference_threshold))
    print('\n\n')
    print('=' * 80)
    print('\n\n')
    # Indices where knn is good and differece is large
    indices_orig_better = [idx for idx, (diff, orig_log_prob) \
                          in enumerate(zip(difference, filtered_orig_log_probs)) \
                          if abs(orig_log_prob) < absolute_threshold and abs(diff) > difference_threshold]
    for idx in indices_orig_better:
        print(' '.join(filtered_tokens[idx - context_length:idx+1]))
    print('\n\n')
    print('{} examples where orig is less than {} and difference greater than {}'.format(len(indices_orig_better), absolute_threshold, difference_threshold))
    print('\n\n')
    print('=' * 80)
    print('\n\n')

def print_validation_examples():
    print("Loading validation distances...")
    with open('wiki.valid.dists.txt') as infile:
        lines = infile.readlines()[:5000]
        dists = []
        for line in tqdm.tqdm(lines):
            dists.append(eval(line.strip()))

    print("Loading validation KNN indices...")
    with open('wiki.valid.knn_indices.txt') as infile:
        lines = infile.readlines()[:5000]
        knn_indices = []
        for line in tqdm.tqdm(lines):
            knn_indices.append(eval(line.strip()))

    for i, (token, dists_for_token, knn_indices_for_token) in enumerate(zip(tokens, dists, knn_indices)):
        context_size = 20
        num_neighbors = 10
        print("Example:", " ".join(tokens[max(i - context_size, 0):i]), "[[", token, "]]")
        best_dist_indices = np.argsort(dists_for_token)[-num_neighbors:][::-1]
        for j, neighbor_index in enumerate(best_dist_indices):
            distance = dists_for_token[neighbor_index]
            knn_index = knn_indices_for_token[neighbor_index]
            print("Best neighbor {} (distance {:.2f}):".format(j, distance), " ".join(train_tokens[knn_index - context_size:knn_index]), "[[", train_tokens[knn_index], "]]")
        print("Original log prob:", orig_log_probs[i])
        print("KNN log prob:", knn_log_probs[i])
        print()
        input()

if __name__ == "__main__":
    # compare_and_plot_knnlm_parametric()
    # compare_knn_parametric_spacy()
    # compare_knn_parametric_word_frequency()
    # compare_knn_parametric_word_frequency_difference()
    # print_examples()
    print_validation_examples()

