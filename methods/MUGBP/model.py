import torch.nn.functional as F
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
from transformers import BertPreTrainedModel
from transformers.models.bert.modeling_bert import BertEmbeddings, BertEncoder, BertPooler
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions
import torchvision.models as models
import torchaudio.models as audio_models
from .SubNets.transformers_encoder.transformer import TransformerEncoder
from .SubNets.dynamicfc import DynamicLayer
from .AlignNets import AlignSubNet
from .PeepholeLSTM import BiPeepholeLSTMLayer



class BiLSTMModule(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, batch_first=True, dropout_rate=0.5):
        super(BiLSTMModule, self).__init__()
        self.dropout_rate = dropout_rate
        self.dropout_input = nn.Dropout(p=0.0)  
        self.bilstm = nn.LSTM(input_size=input_dim,
                              hidden_size=hidden_dim,
                              num_layers=num_layers,
                              batch_first=batch_first,
                              bidirectional=True)
        self.dropout_output = nn.Dropout(p=self.dropout_rate) 
                                  
    def forward(self, x):
        output, (hn, cn) = self.bilstm(x)
        output = self.dropout_output(output)  
        return output
    

    
class PeepholeLSTMModule(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=1, batch_first=True, dropout_rate=0.5):
        super(PeepholeLSTMModule, self).__init__()
        self.dropout_rate = dropout_rate
        self.peepholelstm = BiPeepholeLSTMLayer(input_size=input_dim,
                                        hidden_size=hidden_dim,)
        self.dropout_output = nn.Dropout(p=self.dropout_rate)  
                                  
    def forward(self, x):
        output = self.peepholelstm(x)
        output = self.dropout_output(output)  
        return output



class DAF(nn.Module): 
    def __init__(self, config, args):
        super(DAF, self).__init__()
        self.args = args

        text_feat_dim, audio_feat_dim, video_feat_dim = args.text_feat_dim, args.audio_feat_dim, args.video_feat_dim
        
        self.extra_encoder = args.extra_encoder
        if self.extra_encoder:
            self.visual_attn =  nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=video_feat_dim + text_feat_dim, nhead=8, dim_feedforward=1024),
            num_layers=6
            )
            self.acoustic_attn =  nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=audio_feat_dim + text_feat_dim, nhead=8, dim_feedforward=1024),
            num_layers=6
            )

        self.visual_dyn = DynamicLayer(video_feat_dim + text_feat_dim, text_feat_dim, max_depth=args.max_depth) 
        self.acoustic_dyn = DynamicLayer(audio_feat_dim + text_feat_dim, text_feat_dim, max_depth=args.max_depth)


        self.visual_reshape = nn.Linear(video_feat_dim, text_feat_dim)
        self.acoustic_reshape = nn.Linear(audio_feat_dim, text_feat_dim)


        self.attn_v = nn.Sequential(
            BiLSTMModule(input_dim=text_feat_dim, hidden_dim = video_feat_dim // 2, num_layers=1, dropout_rate=0.5),
            nn.Linear(video_feat_dim, 1)
        )
        self.attn_a = nn.Sequential(
            BiLSTMModule(input_dim=text_feat_dim, hidden_dim = audio_feat_dim // 2, num_layers=1, dropout_rate=0.5),
            nn.Linear(audio_feat_dim, 1)
        )

        self.LayerNorm = nn.LayerNorm(config.hidden_size)
        self.dropout = nn.Dropout(args.output_droupout_prob)

        self.prelu_weight_v = nn.Parameter(torch.tensor(0.25))
        self.prelu_weight_a = nn.Parameter(torch.tensor(0.25)) 

    def forward(self, text_embedding, visual, acoustic):
        eps = 1e-6

        if self.extra_encoder:
            visual_text_pair = self.visual_attn(torch.cat((visual, text_embedding), dim=-1))
            acoustic_text_pair = self.acoustic_attn(torch.cat((acoustic, text_embedding), dim=-1))
        else:
            visual_text_pair = torch.cat((visual, text_embedding), dim=-1)
            acoustic_text_pair = torch.cat((acoustic, text_embedding), dim=-1)
        weight_v = F.prelu(self.visual_dyn(visual_text_pair), self.prelu_weight_v)  
        weight_a = F.prelu(self.acoustic_dyn(acoustic_text_pair), self.prelu_weight_a)

        visual_transformed = self.visual_reshape(visual)
        acoustic_transformed = self.acoustic_reshape(acoustic)

        # Compute intermediate modality-specific features
        weighted_v = weight_v * visual_transformed
        weighted_a = weight_a * acoustic_transformed


        attn_scores_v = torch.sigmoid(self.attn_v(weighted_v))
        attn_scores_a = torch.sigmoid(self.attn_a(weighted_a))

        # Normalize attention scores across modalities
        total_attn = attn_scores_v + attn_scores_a  + eps 
        attn_scores_v = attn_scores_v / total_attn
        attn_scores_a = attn_scores_a / total_attn

        weighted_v = attn_scores_v * weighted_v
        weighted_a = attn_scores_a * weighted_a

        fusion = weighted_v + weighted_a + text_embedding

        # Normalize and apply dropout
        output_fusion = self.dropout(self.LayerNorm(fusion)) 
        return output_fusion


class Anchor(BertPreTrainedModel):
    def __init__(self, config, args, add_pooling_layer=True):
        super().__init__(config)
        self.config = config

        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)

        self.pooler = BertPooler(config) if add_pooling_layer else None
        self.args = args
        # Initialize weights and apply final processing
        self.post_init()

    
    def forward(
        self, 
        condition_idx,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict


        if self.config.is_decoder:
            use_cache = use_cache if use_cache is not None else self.config.use_cache
        else:
            use_cache = False

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length + past_key_values_length)), device=device)

        if token_type_ids is None:
            if hasattr(self.embeddings, "token_type_ids"):
                buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape, device)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)        

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
        )      
        # print('embedding_output.shape', embedding_output.shape)

        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )


class Positive(BertPreTrainedModel):
    def __init__(self, config, args):
        super().__init__(config)
        self.args = args
        self.config = config


        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)

        # Visual Encoder 
        self.visual_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=args.video_feat_dim, nhead=8, dim_feedforward=1024),
            num_layers=6
        )
        self.visual_reshape = nn.Linear(args.video_feat_dim, args.text_feat_dim)

        # Acoustic Encoder 
        self.acoustic_encoder = PeepholeLSTMModule(args.audio_feat_dim, args.audio_feat_dim//2, num_layers=1, dropout_rate=0.0)
        self.acoustic_reshape = nn.Linear(args.audio_feat_dim, args.text_feat_dim)

        self.DAF = DAF(config, args)
        self.alignNet = AlignSubNet(args, args.aligned_method)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.init_weights()
    
    def forward(
        self,
        input_ids,
        visual,
        acoustic,
        condition_idx,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,        
    ):
        
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time"
            )
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError(
                "You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device
        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(
                input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(
            attention_mask, input_shape, device
        )

        # If a 2D ou 3D attention mask is provided for the cross-attention
        # we need to make broadcastabe to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            (
                encoder_batch_size,
                encoder_sequence_length,
                _,
            ) = encoder_hidden_states.size()
            encoder_hidden_shape = (
                encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(
                    encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(
                encoder_attention_mask
            )
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(
            head_mask, self.config.num_hidden_layers)
        
        # get embeddings of normal samples
        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
        )

        text_embedding, visual, acoustic  = self.alignNet(embedding_output, visual, acoustic)


        # text_encoder
        encoder_outputs = self.encoder(
            text_embedding,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        text_feat = encoder_outputs[0]
        text_feat = self.dropout(text_feat) # -> DAF
        text_view = text_feat

        # Visual Encoder
        visual_feat = self.visual_encoder(visual)
        visual_view = self.visual_reshape(visual_feat)


        # Acoustic Encoder
        acoustic_feat = self.acoustic_encoder(acoustic)
        acoustic_view = self.acoustic_reshape(acoustic_feat)

        fused_embedding = self.DAF(text_feat, visual_feat, acoustic_feat)

        pooled_output = self.pooler(fused_embedding)

        return fused_embedding, pooled_output, text_view, visual_view, acoustic_view
    
class Positive_Model(BertPreTrainedModel):
    def __init__(self, config, args):
        super().__init__(config)
        self.num_labels = args.num_labels
        self.label_len = args.label_len

        self.bert = Positive(config, args)


        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, args.num_labels)
        self.init_weights()

    def forward(
        self, 
        text,
        visual,
        acoustic,
        condition_idx,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,        
    ):
        input_ids, attention_mask, token_type_ids = text[:, 0], text[:, 1], text[:, 2]

        outputs, pooled, text_view, visual_view, acoustic_view \
            = self.bert(
            input_ids,
            visual,
            acoustic,
            condition_idx,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )


        text_condition_tuple = tuple(text_view[torch.arange(text_view.shape[0]), condition_idx.view(-1) + i, :].unsqueeze(1) for i in range(self.label_len))
        text_condition = torch.cat(text_condition_tuple, dim=1)

        visual_condition_tuple = tuple(visual_view[torch.arange(visual_view.shape[0]), condition_idx.view(-1) + i, :].unsqueeze(1) for i in range(self.label_len))
        visual_condition = torch.cat(visual_condition_tuple, dim=1)

        acoustic_condition_tuple = tuple(acoustic_view[torch.arange(acoustic_view.shape[0]), condition_idx.view(-1) + i, :].unsqueeze(1) for i in range(self.label_len))
        acoustic_condition = torch.cat(acoustic_condition_tuple, dim=1)

        condition_tuple = tuple(outputs[torch.arange(outputs.shape[0]), condition_idx.view(-1) + i, :].unsqueeze(1) for i in range(self.label_len))
        condition = torch.cat(condition_tuple, dim=1)

        pooled_output = pooled
        outputs = self.classifier(pooled_output)
        
        return outputs, pooled_output, condition, text_condition, visual_condition,acoustic_condition
    
class MUGBP(nn.Module):
    def __init__(self, args):
        super().__init__()    

        self.positive = Positive_Model.from_pretrained(args.cache_path, local_files_only=True, args = args)
        self.anchor = Anchor.from_pretrained(args.cache_path, local_files_only=True, args=args)

        self.label_len = args.label_len
        args.feat_size = args.text_feat_dim
        args.video_feat_size = args.video_feat_dim
        args.audio_feat_size = args.audio_feat_dim
    
    def forward(
        self,
        text_feats,
        video_feats,
        audio_feats,
        cons_text_feats,
        condition_idx
    ):
        video_feats = video_feats.float()
        audio_feats = audio_feats.float() 

        outputs_map, pooled_output_map, condition, text_condition, visual_condition,acoustic_condition\
            = self.positive(
            text = text_feats,
            visual = video_feats,
            acoustic = audio_feats,
            condition_idx=condition_idx, 
        )

        outputs = outputs_map 
        pooled_output = pooled_output_map 

        cons_input_ids, cons_input_mask, cons_segment_ids = cons_text_feats[:, 0], cons_text_feats[:, 1], cons_text_feats[:, 2]
        cons_outputs = self.anchor(
            input_ids = cons_input_ids, 
            condition_idx=condition_idx,
            token_type_ids = cons_segment_ids, 
            attention_mask = cons_input_mask
        )
        last_hidden_state = cons_outputs.last_hidden_state

        cons_condition_tuple = tuple(last_hidden_state[torch.arange(last_hidden_state.shape[0]), condition_idx.view(-1) + i, :].unsqueeze(1) for i in range(self.label_len))
        cons_condition = torch.cat(cons_condition_tuple, dim=1)

        return outputs, pooled_output, condition.mean(dim=1), cons_condition.mean(dim=1), text_condition.mean(dim=1), visual_condition.mean(dim=1), acoustic_condition.mean(dim=1)
