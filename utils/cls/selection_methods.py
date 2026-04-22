
from torch.utils.data import DataLoader
from data.sampler import SubsetSequentialSampler
import random
import torch
import torch.nn.functional as F
import logging
import numpy as np
import copy
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# TMFAL temporal uncertainty
# NOTE: this function mutates global_model weights — caller must save/restore
# ---------------------------------------------------------------------------

def al_timeu(global_model, dataloader, model_pools, decision_model):
    global_model.eval()
    decision_model.eval()
    g_u_data_list = torch.tensor([]).cuda()
    data_label     = torch.tensor([]).cuda()

    with torch.no_grad():
        for _, (_, data) in enumerate(dataloader):
            image = data['image'].cuda()

            logits_list, outputs_list = [], []
            for param in model_pools:
                global_model.load_state_dict(param)
                _, logits, outputs, _ = global_model(image)
                logits_list.append(logits)
                outputs_list.append(outputs)

            logits_tensor = torch.stack(logits_list, dim=1)
            mean_logits   = logits_tensor.mean(dim=1)
            prob, _       = torch.max(mean_logits, dim=1)

            outputs_tensor = torch.stack(outputs_list, dim=1)
            u = torch.std(outputs_tensor, dim=1).mean(dim=1)
            u = u / (prob + 1e-14)

            _, deci_logits, _, _ = decision_model(image)
            deci_label = deci_logits.argmax(dim=1)

            g_u_data_list = torch.cat((g_u_data_list, u))
            data_label    = torch.cat((data_label, deci_label))

    return g_u_data_list, data_label.long()


# ---------------------------------------------------------------------------
# Prototype-guided pseudo-labeling  (FairFAL §5.2)
# ---------------------------------------------------------------------------

def build_class_prototypes(global_model, data_unlabeled, labeled_set,
                           num_classes, batch_size):
    """Build L2-normalised per-class feature prototypes from labeled samples."""
    global_model.eval()
    feat_dim   = None
    feat_sums  = None
    feat_counts = torch.zeros(num_classes).cuda()

    loader = DataLoader(
        dataset=data_unlabeled,
        batch_size=batch_size,
        sampler=SubsetSequentialSampler(labeled_set),
        num_workers=1,
        pin_memory=True,
    )

    with torch.no_grad():
        for _, (_, data) in enumerate(loader):
            image = data['image'].cuda()
            label = data['label'].cuda()
            _, _, feat, _ = global_model(image)
            feat = F.normalize(feat, dim=1)

            if feat_dim is None:
                feat_dim  = feat.shape[1]
                feat_sums = torch.zeros(num_classes, feat_dim).cuda()

            for c in range(num_classes):
                mask = (label == c)
                if mask.sum() > 0:
                    feat_sums[c]  += feat[mask].sum(dim=0)
                    feat_counts[c] += mask.sum().float()

    prototypes = torch.zeros(num_classes, feat_dim).cuda()
    for c in range(num_classes):
        if feat_counts[c] > 0:
            prototypes[c] = F.normalize(feat_sums[c] / feat_counts[c], dim=0)

    return prototypes, feat_counts


def prototype_pseudo_labels(global_model, dataloader, prototypes, feat_counts):
    """Assign pseudo-labels by cosine similarity to class prototypes."""
    global_model.eval()
    pseudo_labels = torch.tensor([], dtype=torch.long).cuda()
    seen_classes  = (feat_counts > 0)

    with torch.no_grad():
        for _, (_, data) in enumerate(dataloader):
            image = data['image'].cuda()
            _, _, feat, _ = global_model(image)
            feat = F.normalize(feat, dim=1)
            sim  = feat @ prototypes.T
            sim[:, ~seen_classes] = -1e9
            pl   = sim.argmax(dim=1)
            pseudo_labels = torch.cat((pseudo_labels, pl))

    return pseudo_labels


# ---------------------------------------------------------------------------
# Greedy k-center diversity selection  (FairFAL §5.3, no external anchors)
# ---------------------------------------------------------------------------

def greedy_kcenter(features: torch.Tensor, budget: int) -> List[int]:
    """
    Greedy k-center on L2-normalised `features` (N, D).
    Picks `budget` indices that maximise coverage (minimax cosine distance).
    No external anchors — starts from the point with highest mean distance.
    """
    N = features.shape[0]
    budget = min(budget, N)
    if budget <= 0:
        return []

    # Seed: pick the sample farthest from the centroid
    centroid = features.mean(dim=0, keepdim=True)
    centroid = F.normalize(centroid, dim=1)
    init_dist = 1.0 - (features @ centroid.T).squeeze(1)  # (N,)
    min_dist  = init_dist.clone()

    selected: List[int] = []
    for _ in range(budget):
        idx = int(min_dist.argmax().item())
        selected.append(idx)
        d = 1.0 - (features @ features[idx])   # (N,)
        min_dist = torch.minimum(min_dist, d)
        min_dist[idx] = -1.0   # never re-select

    return selected


# ---------------------------------------------------------------------------
# FEAL helpers (kept for FEAL baseline)
# ---------------------------------------------------------------------------

def fl_duc(global_model, local_model, dataloader, client_idx=0, round_idx=0):
    import torch.distributions as dist
    from torchmetrics.functional.pairwise import pairwise_cosine_similarity

    global_model.eval()
    local_model.eval()
    g_u_data_list  = torch.tensor([]).cuda()
    l_u_data_list  = torch.tensor([]).cuda()
    g_u_dis_list   = torch.tensor([]).cuda()
    l_feature_list = torch.tensor([]).cuda()

    with torch.no_grad():
        for _, (_, data) in enumerate(dataloader):
            image = data['image'].cuda()
            g_logit, _, _, _ = global_model(image)
            alpha = F.relu(g_logit) + 1
            total_alpha = torch.sum(alpha, dim=1, keepdim=True)
            dirichlet   = dist.Dirichlet(alpha)
            g_u_data    = torch.sum(
                (alpha / total_alpha) *
                (torch.digamma(total_alpha + 1) - torch.digamma(alpha + 1)), dim=1)
            g_u_dis     = dirichlet.entropy()

            l_logit, _, _, block_features = local_model(image)
            l_feature = F.adaptive_avg_pool2d(block_features[-1], 3).flatten(start_dim=1)
            l_feature_list = torch.cat((l_feature_list, l_feature))
            alpha = F.relu(l_logit) + 1
            total_alpha = torch.sum(alpha, dim=1, keepdim=True)
            l_u_data = torch.sum(
                (alpha / total_alpha) *
                (torch.digamma(total_alpha + 1) - torch.digamma(alpha + 1)), dim=1)

            g_u_data_list = torch.cat((g_u_data_list, g_u_data))
            l_u_data_list = torch.cat((l_u_data_list, l_u_data))
            g_u_dis_list  = torch.cat((g_u_dis_list,  g_u_dis))

    return g_u_data_list, l_u_data_list, g_u_dis_list, l_feature_list


def relaxation(u_rank_arg, l_feature_list, neighbor_num, query_num,
               unlabeled_len, cosine=0.85):
    from torchmetrics.functional.pairwise import pairwise_cosine_similarity
    query_flag = torch.zeros(unlabeled_len).cuda()
    chosen_idx = []
    for i in u_rank_arg:
        if len(chosen_idx) == query_num:
            break
        cos_sim     = pairwise_cosine_similarity(l_feature_list[i:i+1, :], l_feature_list)[0]
        neighbor_arg = torch.argsort(-cos_sim)
        neighbor_arg = neighbor_arg[cos_sim[neighbor_arg] > cosine][1:1+neighbor_num]
        if query_flag[neighbor_arg].sum() == 0 or len(neighbor_arg) < neighbor_num:
            query_flag[i] = 1
            chosen_idx.append(i.item())
    remain_idx = list(set(range(unlabeled_len)) - set(chosen_idx))
    return remain_idx + chosen_idx


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def query_samples(
    al_method,
    global_model,
    local_model,
    data_unlabeled,
    data_labeled=None,       # kept for API compatibility
    unlabeled_set=None,
    labeled_set=None,
    query_num=500,
    num_per_class=None,
    client_idx=0,
    round_idx=0,
    args=None,
    model_pools=None,
    save_dir=None,
    zero_model=None,
    cluster_size=10,
    s_step=1,
    num_classes=8,
    kappa=4,
):
    unlabeled_len = len(unlabeled_set)

    # ------------------------------------------------------------------
    # Random
    # ------------------------------------------------------------------
    if al_method == 'Random':
        rank_arg = list(range(unlabeled_len))
        random.shuffle(rank_arg)
        return rank_arg

    # ------------------------------------------------------------------
    # FEAL
    # ------------------------------------------------------------------
    elif al_method == 'FEAL':
        unlabeled_loader = DataLoader(
            dataset=data_unlabeled,
            batch_size=args.batch_size,
            sampler=SubsetSequentialSampler(unlabeled_set),
            num_workers=1, pin_memory=True,
        )
        g_data_list, l_data_list, u_dis_list, l_feature_list = fl_duc(
            global_model, local_model, unlabeled_loader, client_idx, round_idx)
        u_dis_norm  = (u_dis_list - u_dis_list.min()) / (u_dis_list.max() - u_dis_list.min() + 1e-9)
        uncertainty = u_dis_norm * (g_data_list + l_data_list)
        u_rank_arg  = torch.argsort(-uncertainty).cpu().numpy()
        rank_arg    = relaxation(
            u_rank_arg=u_rank_arg, l_feature_list=l_feature_list,
            neighbor_num=args.n_neighbor, query_num=query_num,
            unlabeled_len=unlabeled_len, cosine=args.cosine,
        )
        return rank_arg

    # ------------------------------------------------------------------
    # TMFAL original  (ours) — with BUG 1 fix: save/restore global_model
    # ------------------------------------------------------------------
    elif al_method == 'ours':
        unlabeled_loader = DataLoader(
            dataset=data_unlabeled,
            batch_size=args.batch_size,
            sampler=SubsetSequentialSampler(unlabeled_set),
            num_workers=1, pin_memory=True,
        )

        saved_state = copy.deepcopy(global_model.state_dict())   # BUG 1 FIX

        g_data_list, labels = al_timeu(
            global_model, unlabeled_loader,
            model_pools=model_pools,
            decision_model=copy.deepcopy(global_model),
        )

        global_model.load_state_dict(saved_state)                # BUG 1 FIX

        rank      = torch.argsort(-g_data_list)
        rank_list = rank.cpu().numpy()

        sort_labels  = labels[rank].cpu().numpy()
        class_labels = np.unique(sort_labels)
        result       = []
        label_indexs = {}
        max_length   = 0
        for i in class_labels:
            label_indexs[i] = np.where(sort_labels == i)[0]
            max_length = max(max_length, len(label_indexs[i]))

        for i in range(max_length):
            for j in class_labels:
                if len(label_indexs[j]) > i:
                    result.append(rank_list[label_indexs[j][i]])

        rank_arg = np.flip(np.array(result)).copy()
        return rank_arg

    # ------------------------------------------------------------------
    # TMFAL+  (ours2) — all 3 bugs fixed + prototype PL + k-center diversity
    # ------------------------------------------------------------------
    elif al_method == 'ours2':
        batch_size = args.batch_size

        unlabeled_loader = DataLoader(
            dataset=data_unlabeled,
            batch_size=batch_size,
            sampler=SubsetSequentialSampler(unlabeled_set),
            num_workers=1, pin_memory=True,
        )

        # ---- Step 1: Temporal uncertainty (TMFAL) ----
        # BUG 1 FIX: save state — al_timeu overwrites global_model weights
        saved_state = copy.deepcopy(global_model.state_dict())

        g_data_list, _ = al_timeu(
            global_model, unlabeled_loader,
            model_pools=model_pools,
            decision_model=copy.deepcopy(global_model),
        )

        # BUG 1 FIX: restore correct final trained weights
        global_model.load_state_dict(saved_state)

        # ---- Step 2: Prototype pseudo-labels (on correct model state) ----
        prototypes, feat_counts = build_class_prototypes(
            global_model=global_model,
            data_unlabeled=data_unlabeled,
            labeled_set=labeled_set,
            num_classes=num_classes,
            batch_size=batch_size,
        )

        pseudo_labels = prototype_pseudo_labels(
            global_model=global_model,
            dataloader=unlabeled_loader,
            prototypes=prototypes,
            feat_counts=feat_counts,
        )  # LongTensor (unlabeled_len,)

        # ---- Step 3: Equal budget per class ----
        # BUG 2 FIX: equal budget maximises balanced accuracy (mean per-class recall)
        class_ids_present = torch.unique(pseudo_labels).cpu().numpy()
        n_present         = len(class_ids_present)
        base_per_class    = query_num // n_present
        remainder         = query_num - base_per_class * n_present
        class_budget: Dict[int, int] = {
            int(c): base_per_class + (1 if i < remainder else 0)
            for i, c in enumerate(class_ids_present)
        }

        # ---- Step 4: Collect normalised features for the unlabeled subset ----
        global_model.eval()
        all_feats_list = []
        with torch.no_grad():
            for _, (_, data) in enumerate(unlabeled_loader):
                image = data['image'].cuda()
                _, _, feat, _ = global_model(image)
                all_feats_list.append(F.normalize(feat, dim=1))
        all_feats = torch.cat(all_feats_list, dim=0)   # (unlabeled_len, D)

        unc = g_data_list.cpu()   # (unlabeled_len,)

        # ---- Step 5: Two-stage selection per class ----
        # Stage A — top-κ uncertain candidates within class
        # Stage B — greedy k-center within candidates (no external anchors, BUG 3 FIX)
        selected_global_indices: List[int] = []

        for c in class_ids_present:
            c_int = int(c)
            b_c   = class_budget[c_int]
            if b_c == 0:
                continue

            class_local_idx = np.where((pseudo_labels == c_int).cpu().numpy())[0]
            if len(class_local_idx) == 0:
                continue

            # Stage A
            candidate_count = min(kappa * b_c, len(class_local_idx))
            unc_class       = unc[class_local_idx]
            topk_pos        = torch.argsort(-unc_class)[:candidate_count].numpy()
            candidate_local_idx = class_local_idx[topk_pos]

            # Stage B — k-center on candidate features only (BUG 3 FIX)
            cand_feats  = all_feats[candidate_local_idx]   # (candidate_count, D)
            picked_pos  = greedy_kcenter(cand_feats, budget=b_c)
            picked_local_idx = candidate_local_idx[picked_pos]

            selected_global_indices.extend(picked_local_idx.tolist())

        # rank_arg: unselected first, selected last
        # main loop picks the LAST query_num entries via rank_arg[-query_num:]
        selected_set = set(selected_global_indices)
        remain_idx   = [i for i in range(unlabeled_len) if i not in selected_set]
        rank_arg     = remain_idx + selected_global_indices
        return rank_arg

    # ------------------------------------------------------------------
    # Novel idea: FEAL gating -> TMFAL pool temporal ranking -> Coreset diversity
    # ------------------------------------------------------------------
    elif al_method == 'novel':
        unlabeled_loader = DataLoader(
            dataset=data_unlabeled,
            batch_size=args.batch_size,
            sampler=SubsetSequentialSampler(unlabeled_set),
            num_workers=1, pin_memory=True,
        )
        
        # --- Stage 1: Evidential Gating (EDL) ---
        g_data_list, l_data_list, u_dis_list, l_feature_list = fl_duc(
            global_model, local_model, unlabeled_loader, client_idx, round_idx)
        
        # Epistemic = u_dis_list, Aleatoric = g_data_list + l_data_list
        u_dis_norm  = (u_dis_list - u_dis_list.min()) / (u_dis_list.max() - u_dis_list.min() + 1e-9)
        aleatoric_unc = g_data_list + l_data_list
        epistemic_unc = u_dis_norm 
        
        median_aleatoric = torch.median(aleatoric_unc)
        valid_mask = aleatoric_unc <= median_aleatoric
        
        score = epistemic_unc.clone()
        score[~valid_mask] = -1e9 # Discard aleatoric-dominant
        
        M_candidates = min(args.kappa * 2 * query_num, valid_mask.sum().item())
        if M_candidates < query_num:
            M_candidates = min(unlabeled_len, args.kappa * 2 * query_num)
            score = epistemic_unc.clone()
            
        stage1_topM_idx_tensor = torch.argsort(-score)[:M_candidates]
        stage1_topM_idx = stage1_topM_idx_tensor.cpu().numpy().tolist()
        
        # --- Stage 2: TMFAL Temporal Ranking on Candidate Pool ---
        saved_state = copy.deepcopy(global_model.state_dict())
        subset_candidates = [unlabeled_set[i] for i in stage1_topM_idx]
        candidate_loader = DataLoader(
            dataset=data_unlabeled,
            batch_size=args.batch_size,
            sampler=SubsetSequentialSampler(subset_candidates),
            num_workers=1, pin_memory=True,
        )
        
        temporal_unc_list, _ = al_timeu(
            global_model, candidate_loader,
            model_pools=model_pools,
            decision_model=copy.deepcopy(global_model),
        )
        global_model.load_state_dict(saved_state)
        
        K_candidates = min(args.kappa * query_num, len(subset_candidates))
        stage2_topK_local_idx = torch.argsort(-temporal_unc_list)[:K_candidates]
        
        stage2_indices_in_unlabeled = [stage1_topM_idx[i] for i in stage2_topK_local_idx]
        stage2_feats = l_feature_list[stage2_indices_in_unlabeled]
        stage2_feats = F.normalize(stage2_feats, dim=1)
        
        # --- Stage 3: Coreset Diversity Filter ---
        picked_local_idx = greedy_kcenter(stage2_feats, budget=query_num)
        selected_global_indices = [stage2_indices_in_unlabeled[i] for i in picked_local_idx]
        
        selected_set = set(selected_global_indices)
        remain_idx   = [i for i in range(unlabeled_len) if i not in selected_set]
        rank_arg     = remain_idx + selected_global_indices
        return rank_arg

    else:
        raise ValueError(f"Unknown al_method: {al_method}")
