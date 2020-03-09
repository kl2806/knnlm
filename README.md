# Nearest Neighbor Language Models

This repository is a fork of the [Fairseq](https://github.com/pytorch/fairseq) repository and the exact commit that this code is based on can be found [here](https://github.com/pytorch/fairseq/tree/6a5181509aa1fa7d260985157e77211753da544b). Please use the exact commit page to determine software requirements for using this code. This code builds off the ICLR 2020 paper: [Generalization through Memorization: Nearest Neighbor Language Models](https://arxiv.org/pdf/1911.00172.pdf). 

Before starting, make sure you install Fairseq (after pulling the code, from the project directory) and [FAISS](https://github.com/facebookresearch/faiss/wiki):
```bash
pip install --editable .

pip install faiss

pip install sacremoses
```

TODO, I have to do `pip install faiss-cpu` to get it to run. TODO, check out the GPU version.

### A Note about Hardware

Experiments for this paper were conducted on machines that contain 500GB of RAM, NVIDIA V100 32GB GPUs and flash storage (SSDs). Saving the Wikitext-103 datastore requires 400GB of disk space. The speed of saving the datastore, building the FAISS index and evaluating the nearest neighbors language model heavily depends on the amount of RAM available for each job. Some of these steps can be sped up by parallelizing, which we leave for users to do in order to best cater to their setup.

If you are working with a remote cluster, please note that we use [memmaps](https://numpy.org/doc/1.18/reference/generated/numpy.memmap.html) for saving the datastore. This allows us to keep the data on disk while accessing it by loading small chunks into memory, depending on the available RAM. This means there are a large number of disk seeks. In order to prevent slowing down your entire cluster, we suggest always reading/writing this data to/from local disks (as opposed to NFS directories), and flash storage is best for faster access.

### Preparing the data

We share Fairseq's instructions on how to prepare the data here.

For language modeling on Wikitext-103:
```bash
cd examples/language_model/
bash prepare-wikitext-103.sh
cd ../..


TEXT=examples/language_model/wikitext-103
python preprocess.py \
    --only-source \
    --trainpref $TEXT/wiki.train.tokens \
    --validpref $TEXT/wiki.valid.tokens \
    --testpref $TEXT/wiki.test.tokens \
    --destdir data-bin/wikitext-103 \
    --workers 20
```

For machine translation on IWSLT14 German-English:
```bash
cd examples/translation/
bash prepare-iwslt14.sh
cd ../..

# Preprocess/binarize the data
TEXT=examples/translation/iwslt14.tokenized.de-en
fairseq-preprocess --source-lang de --target-lang en \
    --trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test \
    --destdir data-bin/iwslt14.tokenized.de-en \
    --workers 20
```

### Training the Model

We share Fairseq's instructions on how to train the model here. Alternatively, you can download the checkpoints used for language modeling, [model one](https://nlp.stanford.edu/projects/knnlm/wt103_checkpoint_best.pt) and [model two](https://dl.fbaipublicfiles.com/fairseq/models/lm/adaptive_lm_wiki103.tar.bz2). You can download with the following command:
```bash
mkdir -p lm-checkpoints
wget https://nlp.stanford.edu/projects/knnlm/wt103_checkpoint_best.pt
mv wt103_checkpoint_best.pt lm-checkpoints/checkpoint_best.pt
```

A checkpoint for machine translation that gets 34.94 BLEU on the validation set is [here](https://drive.google.com/file/d/1GySMVIpOH4GkmPfQhOF8lC_t-gghCHIW/view?usp=sharing), which can be downloaded using the following command (got the command from [this website](https://gdrive-wget.glitch.me/)):
```bash
mkdir -p mt-checkpoints
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1GySMVIpOH4GkmPfQhOF8lC_t-gghCHIW' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1GySMVIpOH4GkmPfQhOF8lC_t-gghCHIW" -O mt-checkpoints/checkpoint_best.pt && rm -rf /tmp/cookies.txt
```

For language modeling on Wikitext-103:
```bash
python train.py --task language_modeling \
    data-bin/wikitext-103 \
    --save-dir lm-checkpoints/ \
    --arch transformer_lm_wiki103 \
    --max-update 286000 --max-lr 1.0 --t-mult 2 --lr-period-updates 270000 --lr-scheduler cosine --lr-shrink 0.75 \
    --warmup-updates 16000 --warmup-init-lr 1e-07 --min-lr 1e-09 --optimizer nag --lr 0.0001 --clip-norm 0.1 \
    --criterion adaptive_loss --max-tokens 3072 --update-freq 3 --tokens-per-sample 3072 --seed 1 --fp16 \
    --sample-break-mode none --skip-invalid-size-inputs-valid-test --ddp-backend=no_c10d
```

For machine translation on IWSLT:
```bash
python train.py --task translation \
    data-bin/iwslt14.tokenized.de-en \
    --save-dir mt-checkpoints/ \
    --arch transformer_iwslt_de_en --share-decoder-input-output-embed \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 1e-3 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --dropout 0.3 --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens 4096 \
    --eval-bleu \
    --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
    --eval-bleu-detok moses \
    --eval-bleu-remove-bpe \
    --eval-bleu-print-samples \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric
```

Both models are trained on 8 gpus.

### Evaluating the Model

To evaluate the model on the validation set:

For language modeling on Wikitext-103:
```bash
python eval_lm.py data-bin/wikitext-103 \
    --path lm-checkpoints/checkpoint_best.pt \
    --sample-break-mode complete --max-tokens 3072 \
    --context-window 2560 --softmax-batch 1024 \
    --gen-subset valid
```
You should get a perplexity near 18.0 for a well-trained adaptive input transformer model.

For machine translation on IWSLT:
```bash
python generate.py data-bin/iwslt14.tokenized.de-en \
    --path mt-checkpoints/checkpoint_best.pt \
    --batch-size 128 --beam 5 --remove-bpe \
    --gen-subset valid    
```
You should get a BLEU score near 33-34 for a well-trained transformer model.

### Saving the keys and values for the datastore

In order to save keys and values for the datastore, we must run model evaluation over the entire training set. 

**Caution**: Running this step requires a large amount of disk space (400GB!). Please read the note about hardware above, before running this! 

For language modeling on Wikitext-103:
```bash
python eval_lm.py data-bin/wikitext-103 \
    --path lm-checkpoints/checkpoint_best.pt \
    --sample-break-mode none --max-tokens 3072 \
    --softmax-batch 1024 --gen-subset train \
    --context-window 1536 --tokens-per-sample 1536 \
    --dstore-mmap lm-checkpoints/dstore --knn-keytype 'last_ffn_input' \
    --dstore-size 103225485 --model-overrides "{'knn_keytype': 'last_ffn_input'}" \
    --save-knnlm-dstore --fp16 \
    --output-tokens-file wiki.train.tokens
```

For machine translation on IWSLT:
```bash
python generate.py data-bin/iwslt14.tokenized.de-en  \
    --path mt-checkpoints/checkpoint_best.pt \
    --gen-subset train --batch-size 128 \
    --dstore-mmap mt-checkpoints/dstore --knn-keytype 'last_ffn_input' \
    --dstore-size TODO --model-overrides "{'knn_keytype': 'last_ffn_input'}" \
    --score-reference --quiet \
    --save-knnlm-dstore --fp16 \    
    --output-tokens-file iwslt.train.tokens
```

The total number of tokens in the Wikitext-103 training set is `103227021`. The total number of tokens in the IWSLT training set is `TODO`. The dstore size for Wikitext-103 `103225485` is `1536` tokens less than the total because we want each key to be constructed using a minimum amount of prior context. 

The tokens for the entire training set will be dumped into `wiki.train.tokens` or `iwslt.train.tokens`. These tokens used for printing out the retrieved training tokens in later commands.

For MT, `--score-reference` activates teacher forcing at generation time, so you should see a BLEU score near 100.0.

If you would prefer to save the keys and values in float16, use the `--dstore-fp16` flag and remember to use it during the index building and evaluation steps as well.


### Building the FAISS index

The FAISS index requires a training stage where it learns a set of clusters for the keys. Once this is completed, the keys must all be added to the index. The speed of adding keys to the index depends on the hardware, particularly the amount of RAM available. Please check the paper for more details on our use of FAISS.

Note that the following command runs on CPU.

For language modeling on Wikitext-103:
```bash
python build_dstore.py \
    --dstore_mmap lm-checkpoints/dstore \
    --dstore_size 103225485 \
    --faiss_index lm-checkpoints/knn.index \
    --num_keys_to_add_at_a_time 500000 \
    --starting_point 0
```

For machine translation on IWSLT:
```bash
python build_dstore.py \
    --dstore_mmap mt-checkpoints/dstore \
    --dstore_size TODO \
    --faiss_index mt-checkpoints/knn.index \
    --num_keys_to_add_at_a_time 500000 \
    --dimension 512 \
    --starting_point 0
```


### Evaluating the KNN-Augmented Model

To evaluate the KNN language model on the Wikitext-103 validation set:

```bash
python eval_lm.py data-bin/wikitext-103 \
    --path lm-checkpoints/checkpoint_best.pt \
    --sample-break-mode complete --max-tokens 3072 \
    --context-window 2560 --softmax-batch 1024 \
    --gen-subset valid --dstore-filename lm-checkpoints/dstore \
    --indexfile lm-checkpoints/knn.index  \
    --model-overrides "{'knn_keytype': 'last_ffn_input'}" \
    --k 1024 --lmbda 0.25 --dstore-size 103225485 --knn-keytype last_ffn_input \
    --probe 32 --knnlm --fp16 \
    --output-probs-file-prefix wiki.valid \
    --output-tokens-file wiki.valid
```

To evaluate the KNN MT on the IWSLT validation set:

```bash
python generate.py data-bin/iwslt14.tokenized.de-en \
    --path mt-checkpoints/checkpoint_best.pt \
    --batch-size 1 --beam 1 --remove-bpe \
    --gen-subset valid \
    --remove-bpe \
    --dstore-filename mt-checkpoints/dstore \
    --indexfile mt-checkpoints/knn.index  \
    --model-overrides "{'knn_keytype': 'last_ffn_input'}" \
    --k 1024 --lmbda 0.25 --dstore-size TODO --knn-keytype last_ffn_input \
    --probe 32 --knnlm --fp16
```

TODO, make beam search and batched mode work.

If your hardware constraints make this too slow, you can run it without using full precision keys by adding two flags: `--no-load-keys` and `--knn-sim-func "do_not_recomp_l2"`. This uses the quantized versions of keys stored within the FAISS index. You can make things faster by reducing the value of the `probe` (the number of clusters FAISS checks for neighbors) at the cost of performance. You can also try reducing the number of neighbors `k`.


### Interactive Model Generations

To interactively sample from the KNN language model, and also print the retrieved neighbors, use the following:

```bash
python interactive.py data-bin/wikitext-103 \
	--path lm-checkpoints/checkpoint_best.pt \
    --dstore-filename lm-checkpoints/dstore \
    --indexfile lm-checkpoints/knn.index  \
    --model-overrides "{'knn_keytype': 'last_ffn_input'}" \
    --k 1024 --lmbda 1.0 --dstore-size 103225485 --knn-keytype last_ffn_input \
    --probe 32 --knnlm --fp16 \
    --task language_modeling --beam 1 --nbest 1 \
    --input-tokens-file wiki.train.tokens \
    --max-len-a 1 --max-len-b 10
```

`--input-tokens-file wiki.train.tokens` is the file created from `--output-tokens-file` in the `eval_lm.py` run above. The length of the generation is given by ax + b, where `--max-len-a` is a and `--max-len-b` is b. This command will do greedy decoding from the model, to run sampling, use options like `--sampling --sampling_topk 10`. 

- TODO, get MT numbers.
- TODO, get interactive printing for multiple steps.
- TODO, MT interactive working. It will probably print only the target side now.
