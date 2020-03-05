"""
Compared with XML, we replace 
[video/subtitle cross-encoder repr. fused with video/subtitle query vector]
with 
[video+subtitle self-encoder repr. fused with joint query vector]
to perform single-video moment retrieval.

Note we maintain the VR branch unchanged.

So the cross-modality information change happens in the middle, resuling in the 
name of "XML_MIDDLE".
"""
import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from easydict import EasyDict as edict
from baselines.crossmodal_moment_localization.model_components import \
    (BertAttention, PositionEncoding, LinearLayer, BertSelfAttention,
     TrainablePositionalEncoding, ConvEncoder)
from utils.model_utils import RNNEncoder

base_bert_layer_config = dict(
    hidden_size=768,
    intermediate_size=768,
    hidden_dropout_prob=0.1,
    attention_probs_dropout_prob=0.1,
    num_attention_heads=4,
)

xml_config = edict(
    merge_two_stream=True,  # merge only the scores
    cross_att=True,  # cross-attention for video and subtitles
    span_predictor_type="conv",
    encoder_type="transformer",  # cnn, transformer, lstm, gru
    add_pe_rnn=False,  # add positional encoding for RNNs, (LSTM and GRU)
    visual_input_size=2048,  # changes based on visual input type
    query_input_size=768,
    sub_input_size=768,
    hidden_size=500,  #
    conv_kernel_size=5,  # conv kernel_size for st_ed predictor
    stack_conv_predictor_conv_kernel_sizes=-1,  # Do not use
    conv_stride=1,  #
    max_ctx_l=100,
    max_desc_l=30,
    input_drop=0.1,  # dropout for input
    drop=0.1,  # dropout for other layers
    n_heads=4,  # self attention heads
    ctx_mode="video_sub",  # which context are used. 'video', 'sub' or 'video_sub'
    margin=0.1,  # margin for ranking loss
    ranking_loss_type="hinge",  # loss type, 'hinge' or 'lse'
    lw_neg_q=1,  # loss weight for neg. query and pos. context
    lw_neg_ctx=1,  # loss weight for pos. query and neg. context
    lw_st_ed=1,  # loss weight for st ed prediction
    use_hard_negative=False,  # use hard negative at video level, we may change it during training.
    hard_pool_size=20,
    use_self_attention=True,
    no_modular=False,
    pe_type="none",  # no positional encoding
    initializer_range=0.02,
    fuse_mode="concat"  # support add, mul, concat
)


class XML_MIDDLE(nn.Module):
    def __init__(self, config):
        super(XML_MIDDLE, self).__init__()
        self.config = config
        self.query_pos_embed = TrainablePositionalEncoding(
            max_position_embeddings=config.max_desc_l,
            hidden_size=config.hidden_size, dropout=config.input_drop)
        self.cxt_pos_embed = TrainablePositionalEncoding(
            max_position_embeddings=config.max_ctx_l,
            hidden_size=config.hidden_size, dropout=config.input_drop)
        self.query_input_proj = LinearLayer(config.query_input_size,
                                            config.hidden_size,
                                            layer_norm=True,
                                            dropout=config.input_drop,
                                            relu=True)
        if self.config.encoder_type == "transformer":
            # self-att encoder
            self.query_encoder = BertAttention(edict(
                hidden_size=config.hidden_size,
                intermediate_size=config.hidden_size,
                hidden_dropout_prob=config.drop,
                attention_probs_dropout_prob=config.drop,
                num_attention_heads=config.n_heads,
            ))
        else:
            raise NotImplementedError

        conv_cfg = dict(in_channels=1,
                        out_channels=1,
                        kernel_size=config.conv_kernel_size,
                        stride=config.conv_stride,
                        padding=config.conv_kernel_size // 2,
                        bias=False)

        # we only consider inputs of both video and subtitle
        assert "video" in config.ctx_mode and "sub" in config.ctx_mode, \
                config.ctx_mode

        # video branch
        self.video_input_proj = LinearLayer(config.visual_input_size,
                                            config.hidden_size,
                                            layer_norm=True,
                                            dropout=config.input_drop,
                                            relu=True)
        self.video_encoder = copy.deepcopy(self.query_encoder)

        # subtitle branch        
        self.sub_input_proj = LinearLayer(config.sub_input_size,
                                            config.hidden_size,
                                            layer_norm=True,
                                            dropout=config.input_drop,
                                            relu=True)
        self.sub_encoder = copy.deepcopy(self.query_encoder)

        # middle fusion
        # For now, we will simply use the addition of two modalities
        # self.middle_fuse = BertSelfAttention(cross_att_cfg)
        # self.middle_fuse_layernorm = nn.LayerNorm(config.hidden_size)
        if config.fuse_mode == "add":
            self.joint_fuse = lambda x, y: x + y
        elif config.fuse_mode == "mul":
            self.joint_fuse = lambda x, y: x * y
        elif config.fuse_mode == "concat":
            self.middle_fuse = nn.Sequential(
                    nn.Linear(config.hidden_size+config.hidden_size, 
                              config.hidden_size),
                    nn.Dropout(config.drop)
            )
            self.joint_fuse = lambda x, y: self.middle_fuse(
                                            torch.cat([x, y], -1))
        else:
            raise NotImplementedError
        self.joint_encoder = copy.deepcopy(self.query_encoder)
        self.joint_pos_emb = copy.deepcopy(self.cxt_pos_embed)

        # modular query vectors
        # [query_v, query_s, query_joint]
        self.modular_vector_mapping = nn.Linear(
                            in_features=config.hidden_size,
                            out_features=3)
        self.joint_query_linear = nn.Linear(config.hidden_size, 
                                            config.hidden_size)

        self.temporal_criterion = nn.CrossEntropyLoss(reduction="mean")

        if config.merge_two_stream and config.span_predictor_type == "conv":
            if self.config.stack_conv_predictor_conv_kernel_sizes == -1:
                self.merged_st_predictor = nn.Conv1d(**conv_cfg)
                self.merged_ed_predictor = nn.Conv1d(**conv_cfg)
            else:
                raise NotImplementedError
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """ Initialize the weights."""

        def re_init(module):
            if isinstance(module, (nn.Linear, nn.Embedding)):
                # Slightly different from the TF version which uses truncated_normal for initialization
                # cf https://github.com/pytorch/pytorch/pull/5617
                module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            elif isinstance(module, nn.LayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
            elif isinstance(module, nn.Conv1d):
                module.reset_parameters()
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()

        self.apply(re_init)

    def set_hard_negative(self, use_hard_negative, hard_pool_size):
        """use_hard_negative: bool; hard_pool_size: int, """
        self.config.use_hard_negative = use_hard_negative
        self.config.hard_pool_size = hard_pool_size

    def set_train_st_ed(self, lw_st_ed):
        """pre-train video retrieval then span prediction"""
        self.config.lw_st_ed = lw_st_ed
    
    def encode_context(self, video_feat, video_mask, sub_feat, sub_mask):
        encoded_video_feat = self.encode_input(video_feat, video_mask, 
                                               self.video_input_proj, 
                                               self.video_encoder,
                                               self.cxt_pos_embed)
        encoded_sub_feat = self.encode_input(sub_feat, sub_mask,
                                             self.sub_input_proj,
                                             self.sub_encoder,
                                             self.cxt_pos_embed)
        # we simply use addition of two modalities to the middle fusion layer
        # Note, video_mask == sub_mask
        encoded_x_feat = self.joint_fuse(encoded_video_feat, encoded_sub_feat)
        encoded_x_feat = self.joint_encoder(self.joint_pos_emb(encoded_x_feat), 
                                            video_mask.unsqueeze(1))
        return encoded_video_feat, encoded_sub_feat, encoded_x_feat

    def forward(self, query_feat, query_mask, video_feat, video_mask, 
                sub_feat, sub_mask, tef_feat, tef_mask, st_ed_indices):
        """
        Args:
            query_feat: (N, Lq, Dq)
            query_mask: (N, Lq)
            video_feat: (N, Lv, Dv) or None
            video_mask: (N, Lv) or None
            sub_feat: (N, Lv, Ds) or None
            sub_mask: (N, Lv) or None
            tef_feat: (N, Lv, 2) or None,
            tef_mask: (N, Lv) or None,
            st_ed_indices: (N, 2), torch.LongTensor, 1st, 2nd columns are st, ed 
                            labels respectively.
        """
        encoded_video_feat, encoded_sub_feat, encoded_x_feat = \
            self.encode_context(video_feat, video_mask, sub_feat, sub_mask) 

        query_context_scores, st_prob, ed_prob = \
            self.get_pred_from_raw_query(query_feat, query_mask,
                                         encoded_video_feat, video_mask, 
                                         encoded_sub_feat, sub_mask,
                                         encoded_x_feat, cross=False)

        loss_st_ed = 0
        if self.config.lw_st_ed != 0:
            loss_st = self.temporal_criterion(st_prob, st_ed_indices[:, 0])
            loss_ed = self.temporal_criterion(ed_prob, st_ed_indices[:, 1])
            loss_st_ed = loss_st + loss_ed

        loss_neg_ctx, loss_neg_q = 0, 0
        if self.config.lw_neg_ctx != 0 or self.config.lw_neg_q != 0:
            loss_neg_ctx, loss_neg_q = self.get_video_level_loss(query_context_scores)

        loss_st_ed = self.config.lw_st_ed * loss_st_ed
        loss_neg_ctx = self.config.lw_neg_ctx * loss_neg_ctx
        loss_neg_q = self.config.lw_neg_q * loss_neg_q
        loss = loss_st_ed + loss_neg_ctx + loss_neg_q
        return loss, {"loss_st_ed": float(loss_st_ed),
                      "loss_neg_ctx": float(loss_neg_ctx),
                      "loss_neg_q": float(loss_neg_q),
                      "loss_overall": float(loss)}

    def get_pred_from_raw_query(self, query_feat, query_mask, 
                                video_feat, video_mask, sub_feat, sub_mask,
                                x_feat, cross=False):
        """
        Args:
            query_feat: (N, Lq, Dq)
            query_mask: (N, Lq)
            video_feat: (N, Lv, D)
            video_mask: (N, Lv)
            sub_feat  : (N, Lv, D)
            sub_mask  : (N, Lv)
            x_feat    : (N, Lv, D)
            cross     : bool
        """
        video_query, sub_query, joint_query = \
            self.encode_query(query_feat, query_mask)
        
        # get video-level retrieval scores
        video_q2ctx_scores = self.get_video_level_scores(
                                    video_query, video_feat, video_mask)
        sub_q2ctx_scores = self.get_video_level_scores(
                                    sub_query, sub_feat, sub_mask)
        q2ctx_scores = 0.5 * (video_q2ctx_scores + sub_q2ctx_scores)  # (N, N)

        # get st_prob and ed_prob
        st_prob, ed_prob = self.get_merged_st_ed_prob(
            joint_query, x_feat, video_mask, cross=cross)
        return q2ctx_scores, st_prob, ed_prob

    def get_merged_st_ed_prob(self, joint_query, cxt_feat, cxt_mask, 
                              cross=False, return_similarity=False):
        joint_query = self.joint_query_linear(joint_query)
        if cross:
            # (Nq, Nv, L) from query to all videos.
            similarity = torch.einsum("md,nld->mnl", joint_query, cxt_feat)  
            n_q, n_c, l = similarity.shape
            similarity = similarity.view(n_q * n_c, 1, l)
            st_prob = self.merged_st_predictor(similarity).view(n_q, n_c, l)
            ed_prob = self.merged_ed_predictor(similarity).view(n_q, n_c, l)
        else:
            # (N, L)
            similarity = torch.einsum("bd,bld->bl", joint_query, cxt_feat)
            st_prob = self.merged_st_predictor(similarity.unsqueeze(1)).squeeze()
            ed_prob = self.merged_ed_predictor(similarity.unsqueeze(1)).squeeze()
        # mask
        st_prob = mask_logits(st_prob, cxt_mask)
        ed_prob = mask_logits(ed_prob, cxt_mask)
        if return_similarity:
            assert not cross
            return st_prob, ed_prob, similarity
        else:
            return st_prob, ed_prob
        
    def encode_input(self, feat, mask, input_proj_layer, encoder_layer, pos_embed_layer):
        """
        Args:
            feat: (N, L, D_input), torch.float32
            mask: (N, L), torch.float32, with 1 indicates valid query, 0 indicates mask
            input_proj_layer: down project input
            encoder_layer: encoder layer
            # add_pe: bool, whether to add positional encoding
            pos_embed_layer
        """
        feat = input_proj_layer(feat)

        if self.config.encoder_type in ["cnn", "transformer"]:
            feat = pos_embed_layer(feat)
            mask = mask.unsqueeze(1)  # (N, 1, L), torch.FloatTensor
            return encoder_layer(feat, mask)  # (N, L, D_hidden)
        elif self.config.encoder_type in ["gru", "lstm"]:
            if self.config.add_pe_rnn:
                feat = pos_embed_layer(feat)
            mask = mask.sum(1).long()  # (N, ), torch.LongTensor
            return encoder_layer(feat, mask)[0]  # (N, L, D_hidden)

    def encode_query(self, query_feat, query_mask):
        encoded_query = self.encode_input(query_feat, query_mask,
                            self.query_input_proj, self.query_encoder, 
                            self.query_pos_embed)  # (N, Lq, D)
        video_query, sub_query, joint_query = \
            self.get_modularized_queries(encoded_query, query_mask)  # (N, D) * 3
        return video_query, sub_query, joint_query
            
    def get_modularized_queries(self, encoded_query, query_mask, 
                                return_modular_att=False):
        """
        Args:
            encoded_query : (N, L, D)
            query_mask    : (N, L)
            return_modular_att: bool
        Return:
            modular_query1: (N, D)
            modular_query2: (N, D)
            joint_query   : (N, D)
        """
        modular_attention_scores = self.modular_vector_mapping(encoded_query)
        modular_attention_scores = F.softmax(
            mask_logits(modular_attention_scores, query_mask.unsqueeze(2)),
            dim=1)
        modular_queries = torch.einsum("blm,bld->bmd",
                                       modular_attention_scores, 
                                       encoded_query)  # (N, 3, D)
        return modular_queries[:, 0], modular_queries[:, 1], modular_queries[:, 2]

    def get_video_level_scores(self, modularied_query, 
                               context_feat1, context_mask):
        """ Calculate video2query scores for each pair of video and query inside 
            the batch.
        Args:
            modularied_query: (N, D)
            context_feat1: (N, L, D), output of the first transformer encoder
            context_mask: (N, L)
        Returns:
            context_query_scores: (N, N)  score of each query w.r.t. each video 
            inside the batch, diagonal positions are positive. used to get 
            negative samples.
        """
        modularied_query = F.normalize(modularied_query, dim=-1)
        context_feat1 = F.normalize(context_feat1, dim=-1)
        query_context_scores = torch.einsum("md,nld->mln", 
                                modularied_query, context_feat1)  # (N, L, N)
        context_mask = context_mask.transpose(0, 1).unsqueeze(0)  # (1, L, N)
        query_context_scores = mask_logits(
                                query_context_scores, context_mask)  # (N, L, N)
        # (N, N) diagonal positions are positive pairs.
        query_context_scores, _ = torch.max(query_context_scores,
                                            dim=1)  
        return query_context_scores

    def get_video_level_loss(self, query_context_scores):
        """ ranking loss between (pos. query + pos. video) and 
            (pos. query + neg. video) or (neg. query + pos. video)
        Args:
            query_context_scores: (N, N), cosine similarity [-1, 1].
            Each row contains the scores between the query to each of the 
            videos inside the batch.
        """
        bsz = len(query_context_scores)
        diagonal_indices = torch.arange(bsz).to(query_context_scores.device)
        pos_scores = query_context_scores[diagonal_indices, diagonal_indices]  # (N, )
        query_context_scores_masked = copy.deepcopy(query_context_scores.data)
        # impossibly large for cosine similarity, the copy is created as modifying the original will cause error
        query_context_scores_masked[diagonal_indices, diagonal_indices] = 999
        pos_query_neg_context_scores = self.get_neg_scores(query_context_scores,
                                                           query_context_scores_masked)
        neg_query_pos_context_scores = self.get_neg_scores(query_context_scores.transpose(0, 1),
                                                           query_context_scores_masked.transpose(0, 1))
        loss_neg_ctx = self.get_ranking_loss(pos_scores, pos_query_neg_context_scores)
        loss_neg_q = self.get_ranking_loss(pos_scores, neg_query_pos_context_scores)
        return loss_neg_ctx, loss_neg_q

    def get_neg_scores(self, scores, scores_masked):
        """
        scores: (N, N), cosine similarity [-1, 1],
            Each row are scores: query --> all videos. Transposed version: video --> all queries.
        scores_masked: (N, N) the same as scores, except that the diagonal (positive) positions
            are masked with a large value.
        """
        bsz = len(scores)
        batch_indices = torch.arange(bsz).to(scores.device)
        _, sorted_scores_indices = torch.sort(scores_masked, descending=True, dim=1)
        sample_min_idx = 1  # skip the masked positive
        sample_max_idx = min(sample_min_idx + self.config.hard_pool_size, bsz) \
            if self.config.use_hard_negative else bsz
        sampled_neg_score_indices = sorted_scores_indices[
            batch_indices, torch.randint(sample_min_idx, sample_max_idx, size=(bsz,)).to(scores.device)]  # (N, )
        sampled_neg_scores = scores[batch_indices, sampled_neg_score_indices]  # (N, )
        return sampled_neg_scores

    def get_ranking_loss(self, pos_score, neg_score):
        """ Note here we encourage positive scores to be larger than negative scores.
        Args:
            pos_score: (N, ), torch.float32
            neg_score: (N, ), torch.float32
        """
        if self.config.ranking_loss_type == "hinge":  # max(0, m + S_neg - S_pos)
            return torch.clamp(self.config.margin + neg_score - pos_score, min=0).sum() / len(pos_score)
        elif self.config.ranking_loss_type == "lse":  # log[1 + exp(S_neg - S_pos)]
            return torch.log1p(torch.exp(neg_score - pos_score)).sum() / len(pos_score)
        else:
            raise NotImplementedError("Only support 'hinge' and 'lse'")


def mask_logits(target, mask):
    return target * mask + (1 - mask) * (-1e10)
