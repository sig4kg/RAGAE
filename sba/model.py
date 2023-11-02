import tokenize

from transformers.models.roberta import RobertaModel, RobertaTokenizer, RobertaForCausalLM, RobertaConfig
from transformers.models.roberta.modeling_roberta import RobertaEncoder, RobertaLayer, \
    RobertaAttention
import torch.nn as nn
import torch
from typing import List, Optional, Tuple, Union
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
import math

class SBASelfAttention(nn.Module):
    def __init__(self, config, position_embedding_type=None):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.position_embedding_type = position_embedding_type or getattr(
            config, "position_embedding_type", "absolute"
        )
        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            self.max_position_embeddings = config.max_position_embeddings
            self.distance_embedding = nn.Embedding(2 * config.max_position_embeddings - 1, self.attention_head_size)

        self.is_decoder = config.is_decoder
        self.G_q = nn.Linear(config.hidden_size, 1, bias=False)
        self.G_z = nn.Linear(config.hidden_size, 1, bias=False)
        nn.init.xavier_normal_(self.G_q.weight)
        nn.init.xavier_normal_(self.G_z.weight)


    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.FloatTensor] = None,
            head_mask: Optional[torch.FloatTensor] = None,
            encoder_hidden_states: Optional[torch.FloatTensor] = None,
            encoder_attention_mask: Optional[torch.FloatTensor] = None,
            past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
            output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        mixed_query_layer = self.query(hidden_states)

        # gq = self.G_q(hidden_states)
        # gz = self.G_z(encoder_hidden_states)
        # gate = torch.sigmoid(gq + gz)

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        is_cross_attention = encoder_hidden_states is not None

        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_layer = past_key_value[0]
            value_layer = past_key_value[1]
            attention_mask = encoder_attention_mask
        elif is_cross_attention:
            key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
            value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
            attention_mask = encoder_attention_mask
        elif past_key_value is not None:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
            key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
            value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
        else:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))

        query_layer = self.transpose_for_scores(mixed_query_layer)

        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_layer, value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            seq_length = hidden_states.size()[1]
            position_ids_l = torch.arange(seq_length, dtype=torch.long, device=hidden_states.device).view(-1, 1)
            position_ids_r = torch.arange(seq_length, dtype=torch.long, device=hidden_states.device).view(1, -1)
            distance = position_ids_l - position_ids_r
            positional_embedding = self.distance_embedding(distance + self.max_position_embeddings - 1)
            positional_embedding = positional_embedding.to(dtype=query_layer.dtype)  # fp16 compatibility

            if self.position_embedding_type == "relative_key":
                relative_position_scores = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores
            elif self.position_embedding_type == "relative_key_query":
                relative_position_scores_query = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                relative_position_scores_key = torch.einsum("bhrd,lrd->bhlr", key_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores_query + relative_position_scores_key

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in RobertaModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        # context_layer = torch.matmul(attention_probs, value_layer)
        query_layer = query_layer.permute(0, 2, 1, 3).contiguous()
        new_query_layer_shape = query_layer.size()[:-2] + (self.all_head_size,)
        query_layer = query_layer.view(new_query_layer_shape)

        key_layer = key_layer.permute(0, 2, 1, 3).contiguous()
        new_key_layer_shape = key_layer.size()[:-2] + (self.all_head_size,)
        key_layer = key_layer.view(new_key_layer_shape)

        gq = self.G_q(query_layer)
        gz = self.G_z(key_layer)
        gate = torch.sigmoid(gq + gz)

        value_layer = value_layer.permute(0, 2, 1, 3).contiguous()
        new_value_layer_shape = value_layer.size()[:-2] + (self.all_head_size,)
        value_layer = value_layer.view(new_value_layer_shape)
        context_layer = gate * value_layer.repeat(1, gate.shape[1], 1)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        if self.is_decoder:
            outputs = outputs + (past_key_value,)
        return outputs


class SBAAttention(RobertaAttention):
    def __init__(self, config, position_embedding_type=None):
        super().__init__(config)
        self.self = SBASelfAttention(config, position_embedding_type=position_embedding_type)


class SBALayer(RobertaLayer):
    def __init__(self, config):
        super().__init__(config)
        self.crossattention = SBAAttention(config, position_embedding_type="absolute")


class SBAEncoder(RobertaEncoder):
    def __init__(self, config):
        super().__init__(config)
        self.layer = nn.ModuleList([SBALayer(config) for _ in range(config.num_hidden_layers)])


class SBARoberta(RobertaModel):
    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config, add_pooling_layer)
        self.encoder = SBAEncoder(config)
        self.post_init()


class SBADecoder(RobertaForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        self.roberta = SBARoberta(config, add_pooling_layer=False)
        self.post_init()


class SBA(nn.Module):
    def __init__(self, config):
        super(SBA, self).__init__()
        self.bottleneck_attn = RobertaAttention(config)
        self.decoder = SBADecoder(config)
        self.pretrained_LM = RobertaModel.from_pretrained('roberta-base')
        for param in self.pretrained_LM.parameters():
            param.requires_grad = False

    def forward(self, input_ids, attention_mask, run_decode=True, is_train=False):
        '''
        Run sentence bottleneck auto-encoder
        :param input_text:
        :param hidden_states: hidden_states from pre-trained RoBERTa
        :param input_ids: input_ids of labels
        :param run_decode: whether to run decode using SBA decoder
        :return: [sba_representation, word_logits]
        '''
        assert (is_train and run_decode or not is_train)
        input_ids = input_ids.squeeze(1)
        attention_mask = attention_mask.squeeze(1)
        hidden_states = self.pretrained_LM(input_ids=input_ids, attention_mask=attention_mask)[0].detach().clone()
        q = hidden_states[:, 0, :].unsqueeze(1)
        z = self.bottleneck_attn(hidden_states=q, encoder_hidden_states=hidden_states, attention_mask=attention_mask)[0]
        outputs = (z,)
        if run_decode:
            if is_train:
                dec_output = self.decoder(input_ids, attention_mask, encoder_hidden_states=z, labels=input_ids)
            else:
                dec_output = self.decoder(input_ids, attention_mask, encoder_hidden_states=z)
            outputs = outputs + (dec_output,)

        return outputs
