import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel, BertConfig, AutoTokenizer, BertForMaskedLM

class BiGRU(nn.Module):
    """
    decotor 使用 BiGRU
    """
    def __init__(self, embedding_size, hidden, n_layers, dropout=0.0):
        super(BiGRU, self).__init__()
        self.rnn = nn.GRU(embedding_size, hidden, num_layers=n_layers,
                          bidirectional=True, dropout=dropout, batch_first=True)
        self.sigmoid = nn.Sigmoid()
        self.linear = nn.Linear(hidden*2, 1)

    def forward(self, x):
        out, _ = self.rnn(x)
        prob = self.sigmoid(self.linear(out))
        return prob
    

class SoftMaskedBert(nn.Module):
    """
    Soft-Masked Bert
    论文：https://arxiv.org/pdf/2005.07421.pdf
    https://github.com/hiyoung123/SoftMaskedBert
    """
    def __init__(self, cfg):   # bert, tokenizer, hidden, layer_n, device
        super(SoftMaskedBert, self).__init__()
        self.bert=BertModel.from_pretrained(cfg.bert_name)
        self.tokenizer=AutoTokenizer.from_pretrained(cfg.bert_name)
        self.embedding = self.bert.embeddings.to(cfg.device)
        self.config = self.bert.config
        self.device=cfg.device
        embedding_size = self.config.to_dict()['hidden_size']

        self.detector = BiGRU(embedding_size, cfg.hidden, cfg.layer_n)
        self.corrector = self.bert.encoder
        mask_token_id = torch.tensor([[self.tokenizer.mask_token_id]]).to(cfg.device)
        self.mask_e = self.embedding(mask_token_id).detach()
        self.linear = nn.Linear(embedding_size, self.config.vocab_size)
        self.softmax = nn.LogSoftmax(dim=-1)

        """
        loss
        """
        self.criterion_d = nn.BCELoss().to(self.device)
        self.criterion_c = nn.NLLLoss().to(self.device)
        self.gama = 0.9
        self.to(self.device)
    # def forward(self, input_ids, input_mask, segment_ids):
    def forward(self, batch):    
        
        input_ids=batch["error_ids"].to(self.device)     #(bs, max_len)
        input_mask=batch["error_masks"].to(self.device)
        segment_ids=batch["error_segment_ids"].to(self.device)
        detection_labels=batch["detection_labels"].to(self.device)
        correct_ids=batch["correct_ids"].to(self.device)
        
        e = self.embedding(input_ids=input_ids, token_type_ids=segment_ids) #(bs, max_len, emb_dim)
        p = self.detector(e)        #(bs, max_len, 1)
        e_ = p * self.mask_e + (1-p) * e
        _, _, _, _, \
        _, \
        head_mask, \
        encoder_hidden_states, \
        encoder_extended_attention_mask= self._init_inputs(input_ids, input_mask)
        h = self.corrector(e_,
                           attention_mask=encoder_extended_attention_mask,
                           head_mask=head_mask,
                           encoder_hidden_states=encoder_hidden_states,
                           encoder_attention_mask=encoder_extended_attention_mask)
        """
        我感觉这里有问题，这里的residual没有什么道理呀！
        """
        h = h[0] #+ e        #(bs, max_len, emb_dim)
        
        
        
        
        logits = self.softmax(self.linear(h))   #(bs, max_len, vocab_size)
        p=p.squeeze(-1) #(bs, max_len)
        loss=self._loss(logits, p, detection_labels, correct_ids)
        
        if self.training:
            strings_error=None
            strings_predict=None
            strings_correct=None
        else:
            strings_error=self.ids_2_str(input_ids)
            strings_predict=self.ids_prob_2_str(logits)
            strings_correct=self.ids_2_str(correct_ids)
        outputs={
            "loss": loss,
            "logits": logits,
            "strings_error": strings_error,
            "strings_predict": strings_predict,
            "strings_correct": strings_correct,
        }
        return outputs
        return out, p, loss
    def _loss(self, out, prob, detection_label, correct_ids):
        """
        correct_ids: (bs, max_len)
        """
        # detection_label=torch.LongTensor(detection_label)
        loss_d = self.criterion_d.forward(prob, detection_label.to(torch.float32))
        loss_c = self.criterion_c(out.transpose(1, 2), correct_ids) # num_classes 放在 dim1
        loss = self.gama * loss_c + (1-self.gama) * loss_d
        return loss
        
    def ids_prob_2_str(self, ids:torch.Tensor):
        ids=torch.argmax(ids, dim=-1).detach().cpu().tolist()   #(bs, max_len)
        strings=["".join(self.tokenizer.convert_ids_to_tokens(x)) for x in ids]
        return strings

    def ids_2_str(self, ids:torch.Tensor):
        strings=["".join(self.tokenizer.convert_ids_to_tokens(x)) for x in ids]
        return strings

    def _init_inputs(self,
                    input_ids=None,
                    attention_mask=None,
                    token_type_ids=None,
                    position_ids=None,
                    head_mask=None,
                    inputs_embeds=None,
                    encoder_hidden_states=None,
                    encoder_attention_mask=None,
                    ):
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            # Provided a padding mask of dimensions [batch_size, seq_length]
            # - if the model is a decoder, apply a causal mask in addition to the padding mask
            # - if the model is an encoder, make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
            if self.config.is_decoder:
                batch_size, seq_length = input_shape
                seq_ids = torch.arange(seq_length, device=device)
                causal_mask = seq_ids[None, None, :].repeat(batch_size, seq_length, 1) <= seq_ids[None, :, None]
                causal_mask = causal_mask.to(
                    attention_mask.dtype
                )  # causal and attention masks must have same type with pytorch version < 1.3
                extended_attention_mask = causal_mask[:, None, :, :] * attention_mask[:, None, None, :]
            else:
                extended_attention_mask = attention_mask[:, None, None, :]
        else:
            raise ValueError(
                "Wrong shape for input_ids (shape {}) or attention_mask (shape {})".format(
                    input_shape, attention_mask.shape
                )
            )

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        # If a 2D ou 3D attention mask is provided for the cross-attention
        # we need to make broadcastabe to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)

            if encoder_attention_mask.dim() == 3:
                encoder_extended_attention_mask = encoder_attention_mask[:, None, :, :]
            elif encoder_attention_mask.dim() == 2:
                encoder_extended_attention_mask = encoder_attention_mask[:, None, None, :]
            else:
                raise ValueError(
                    "Wrong shape for encoder_hidden_shape (shape {}) or encoder_attention_mask (shape {})".format(
                        encoder_hidden_shape, encoder_attention_mask.shape
                    )
                )

            encoder_extended_attention_mask = encoder_extended_attention_mask.to(
                dtype=next(self.parameters()).dtype
            )  # fp16 compatibility
            encoder_extended_attention_mask = (1.0 - encoder_extended_attention_mask) * -10000.0
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.expand(self.config.num_hidden_layers, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = (
                    head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
                )  # We can specify head_mask for each layer
            head_mask = head_mask.to(
                dtype=next(self.parameters()).dtype
            )  # switch to fload if need + fp16 compatibility
        else:
            head_mask = [None] * self.config.num_hidden_layers

        return input_ids, position_ids, token_type_ids, inputs_embeds, \
               extended_attention_mask, head_mask, encoder_hidden_states, encoder_extended_attention_mask