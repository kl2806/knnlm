import torch
import faiss
import math
import numpy as np
from fairseq import utils
import time
from fairseq.data import Dictionary

class KNN_Dstore(object):
    def __init__(self, args):
        self.half = args.fp16
        self.dimension = args.decoder_embed_dim
        self.k = args.k
        self.dstore_size = args.dstore_size
        self.metric_type = args.faiss_metric_type
        self.sim_func = args.knn_sim_func
        self.dstore_fp16 = args.dstore_fp16
        self.index = self.setup_faiss(args)
        self.vocab_size = args.vocab_size


    def setup_faiss(self, args):
        if not args.dstore_filename:
            raise ValueError('Cannot build a datastore without the data.')

        start = time.time()
        index = faiss.read_index(args.indexfile, faiss.IO_FLAG_ONDISK_SAME_DIR)
        print('Reading datastore took {} s'.format(time.time() - start))
        index.nprobe = args.probe

        if args.dstore_fp16:
            print('Keys are fp16 and vals are int16')
            if not args.no_load_keys:
                self.keys = np.memmap(args.dstore_filename+'_keys.npy', dtype=np.float16, mode='r', shape=(self.dstore_size, self.dimension))
            self.vals = np.memmap(args.dstore_filename+'_vals.npy', dtype=np.int16, mode='r', shape=(self.dstore_size, 1))
        else:
            print('Keys are fp32 and vals are int64')
            if not args.no_load_keys:
                self.keys = np.memmap(args.dstore_filename+'_keys.npy', dtype=np.float32, mode='r', shape=(self.dstore_size, self.dimension))
            self.vals = np.memmap(args.dstore_filename+'_vals.npy', dtype=np.int, mode='r', shape=(self.dstore_size, 1))

        # If you wish to load all the keys into memory
        # CAUTION: Only do this if your RAM can handle it!
        if args.move_dstore_to_mem:
            print('Loading to memory...')
            start = time.time()

            if not args.no_load_keys:
                del self.keys
                self.keys_from_memmap = np.memmap(args.dstore_filename+'_keys.npy', dtype=np.float16 if args.dstore_fp16 else np.float32, mode='r', shape=(self.dstore_size, self.dimension))
                self.keys = np.zeros((self.dstore_size, self.dimension), dtype=np.float16 if args.dstore_fp16 else np.float32)
                self.keys = self.keys_from_memmap[:]
                self.keys = self.keys.astype(np.float16 if args.dstore_fp16 else np.float32)

            del self.vals
            self.vals_from_memmap = np.memmap(args.dstore_filename+'_vals.npy', dtype=np.int16 if args.dstore_fp16 else np.int, mode='r', shape=(self.dstore_size, 1))
            self.vals = np.zeros((self.dstore_size, 1), dtype=np.int16 if args.dstore_fp16 else np.int)
            self.vals = self.vals_from_memmap[:]
            self.vals = self.vals.astype(np.int16 if args.dstore_fp16 else np.int)
            print('Loading to memory took {} s'.format(time.time() - start))

        return index


    def get_knns(self, queries):
        start = time.time()
        dists, knns = self.index.search(queries.detach().cpu().float().numpy(), self.k)
        return dists, knns


    def get_knn_log_prob(self, queries, tgt, pad_idx):
        def dist_func(d, k, q, function=None):
            d = d.cpu()
            q = q.cpu()
            if not function:
                # Default behavior for L2 metric is to recompute distances.
                # Default behavior for IP metric is to return faiss distances.
                qsize = q.shape
                if self.metric_type == 'l2':
                    start = time.time()
                    knns_vecs = torch.from_numpy(self.keys[k]).view(qsize[0], self.k, -1)
                    if self.half:
                        knns_vecs = knns_vecs.half()
                    query_vecs = q.view(qsize[0], 1, qsize[1]).repeat(1, self.k, 1)
                    l2 = torch.sum((query_vecs - knns_vecs.detach())**2, dim=2)
                    return -1 * l2
                d = d.cuda(); q = q.cuda()
                return d

            if function == 'dot':
                qsize = q.shape
                d = d.cuda(); q = q.cuda()
                return (torch.from_numpy(self.keys[k]) * q.view(qsize[0], 1, qsize[1])).sum(dim=-1).cuda()

            if function == 'do_not_recomp_l2':
                d = d.cuda(); q = q.cuda()
                return -1 * d.cuda()

            raise ValueError("Invalid knn similarity function!")

        # queries  are TxBxC
        # reshape: (TxB)xC
        qshape = queries.shape
        queries = queries.view(-1, qshape[-1])
        if tgt is not None:
            tgt = tgt.contiguous().view(-1)
            dists, knns = self.get_knns(queries[tgt != pad_idx])
        else:
            dists, knns = self.get_knns(queries)

         
        # (T_reducedxB)xK
        dists = torch.from_numpy(dists).cuda()
        start = time.time()
        if tgt is not None:
            dists = dist_func(dists, knns, queries[tgt != pad_idx, :], function=self.sim_func).cuda()
        else:
            dists = dist_func(dists, knns, queries, function=self.sim_func).cuda()
        probs = utils.softmax(dists, dim=-1).cuda()
        if tgt is not None:
            index_mask = torch.eq(torch.from_numpy(self.vals[knns]).long().cuda().squeeze(-1), tgt[tgt != pad_idx].unsqueeze(-1)).float()
            index_mask[index_mask == 0] = 0 # for stability
            index_mask[index_mask == 1] = 1

            # (T_reducedxB)
            yhat_knn_prob = torch.sum(probs * index_mask, dim=-1).clone()
            for i, val in enumerate(yhat_knn_prob):
                if val < 1e-6:
                    yhat_knn_prob[i] = torch.FloatTensor([-10000]).squeeze()
                else:
                    yhat_knn_prob[i] = torch.log(yhat_knn_prob[i])
            full_yhat_knn_prob = torch.full([qshape[0]*qshape[1]], -10000).cuda()
            full_yhat_knn_prob[tgt != pad_idx] = yhat_knn_prob

	# TODO, this won't work for batched or beam > 1
        else:
            idx = torch.from_numpy(self.vals[knns]).long().cuda().squeeze(-1)
            idx_unique = idx.unique(sorted=True).cuda()
            yhat_knn_prob_retrieved_tokens = torch.zeros(len(idx_unique)).cuda()
            for enumerate_idx, idx_unique_curr in enumerate(idx_unique):
                yhat_knn_prob_retrieved_tokens[enumerate_idx] = torch.sum((probs * (idx == idx_unique_curr)), dim=-1).clone()

            # yhat_knn_prob_retrieved_tokens = torch.zeros(len(idx_unique)) #.cuda().scatter_add(0, idx.squeeze(), probs.squeeze()).cuda()

            full_yhat_knn_prob = torch.full((qshape[0]*qshape[1], self.vocab_size), -10000).cuda()
            full_yhat_knn_prob[:,idx_unique] = torch.log(yhat_knn_prob_retrieved_tokens)

        dists_full = torch.full((qshape[0]*qshape[1], dists.shape[-1]), 10000.0, dtype=dists.dtype).cuda()
        if tgt is not None:
            dists_full[tgt != pad_idx] = dists 
        else:
            dists_full = dists
        
        knns = torch.from_numpy(knns).cuda()
        knns_full = torch.full((qshape[0]*qshape[1], knns.shape[-1]), -1, dtype=knns.dtype).cuda()
        if tgt is not None:
            knns_full[tgt != pad_idx] = knns 
        else:
            knns_full = knns

        assert dists.size() == knns.size()

        # TxBx1
        if tgt is not None:
            return full_yhat_knn_prob.view(qshape[0], qshape[1], 1), dists_full.view(qshape[0], qshape[1], -1), knns_full.view(qshape[0], qshape[1], -1)
        else:
            return full_yhat_knn_prob.view(qshape[0], qshape[1], self.vocab_size), dists_full.view(qshape[0], qshape[1], -1), knns_full.view(qshape[0], qshape[1], -1)
