import pandas as pd
import pandas as pd
import numpy as np

from plotnine import *
from plotnine.data import *

with open('bleu.txt') as bleu_score_file:
    bleu_scores = bleu_score_file.readlines()
bleu_scores = [float(bleu_score.lstrip('BLEU4 = ')) for bleu_score in bleu_scores]
lambdas = np.linspace(0, 1, 9).tolist()
df = pd.DataFrame(list(zip(lambdas, bleu_scores)), 
               columns =['lambdas', 'bleu_scores']) 
p = ggplot(aes(x='lambdas', y='bleu_scores'), df)
p += geom_point(aes(lambdas, bleu_scores))
p.save('bleu.pdf', height=6, width=8)


