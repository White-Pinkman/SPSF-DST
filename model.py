"""
SOM-DST
Copyright (c) 2020-present NAVER Corp.
MIT license
"""

import torch
import torch.nn as nn
from pytorch_transformers.modeling_bert import BertPreTrainedModel, BertModel
from utils.data_utils import O_INDEX,PAD_INDEX

class SomDST(BertPreTrainedModel):
    def __init__(self, config, n_op,n_slot, n_domain, update_id, exclude_domain=False):
        super(SomDST, self).__init__(config)
        self.hidden_size = config.hidden_size
        #config:dir n_op:slot operate class n_domain:5  update_id:3  exclude_domain:no
        self.encoder = Encoder(config, n_op,n_slot, n_domain, update_id, exclude_domain)
        #
        self.decoder = Decoder(config, self.encoder.bert.embeddings.word_embeddings.weight)
        #init weight

        #self.apply(self.init_weights) # mj注释掉了

    def forward(self, input_ids, token_type_ids,
                state_positions, attention_mask,
                max_value, slot_labels_id,slot_labels_pos,op_ids=None, max_update=None, teacher=None):
        # input_ids : B * L
        # token_type_ids: B * L  在转移槽的位置为1
        # state_positions: B * 30 槽的位置
        # attention_mask : B * L
        # op_ids : B * 30 状态转移标签

        #print(slot_label_ids)
        #import sys
        #sys.exit()

        enc_outputs = self.encoder(input_ids=input_ids,
                                   token_type_ids=token_type_ids,
                                   state_positions=state_positions,
                                   attention_mask=attention_mask,
                                   slot_labels_id=slot_labels_id,
                                   slot_labels_pos=slot_labels_pos,
                                   op_ids=op_ids,
                                   max_update=max_update)

        domain_scores, slot_scores,state_scores, decoder_inputs, sequence_output, pooled_output = enc_outputs
        gen_scores = self.decoder(input_ids, decoder_inputs, sequence_output,
                                  pooled_output, max_value, teacher)

        return domain_scores, slot_scores,state_scores, gen_scores


class Encoder(nn.Module):
    def __init__(self, config, n_op,n_slot, n_domain, update_id, exclude_domain=False):
        super(Encoder, self).__init__()
        self.hidden_size = config.hidden_size
        self.exclude_domain = exclude_domain
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.dropout)
        self.action_cls = nn.Linear(config.hidden_size, n_op)
        self.slot_cls = nn.Linear(config.hidden_size,n_slot)
        self.n_slot = n_slot
        if self.exclude_domain is not True:
            self.domain_cls = nn.Linear(config.hidden_size, n_domain)
        self.n_op = n_op
        self.n_domain = n_domain
        self.update_id = update_id

    def forward(self, input_ids, token_type_ids,
                state_positions, attention_mask,slot_labels_id,slot_labels_pos,
                op_ids=None, max_update=None,max_slot=None):

        bert_outputs = self.bert(input_ids, token_type_ids, attention_mask)
        #
        sequence_output, pooled_output = bert_outputs[:2]
        # print(sequence_output.size,pooled_output.size)
        state_pos = state_positions[:, :, None].expand(-1, -1, sequence_output.size(-1))
        slot_labels_pos = slot_labels_pos[:,:,None].expand(-1,-1,sequence_output.size(-1))

        #
        state_output = torch.gather(sequence_output, 1, state_pos)
        #
        slot_output = torch.gather(sequence_output,1,slot_labels_pos)

        # slot_output : B * 42 * H
        # state_positions : B * 30
        # state_pos : B * 30 * H
        # state_output : B * 30 * H
        # sequence_output : B * L * H

        # 此处加上entity_slot 对 state_scores的 attention

        slot_scores = self.slot_cls(self.dropout(slot_output))
        if slot_labels_id is None:
            slot_labels_id = slot_scores.view(-1,self.n_slot)
        if max_slot is None:
            max_slot = (slot_labels_id.ne(O_INDEX)& slot_labels_id.ne(PAD_INDEX)).sum(-1).max().item()

        gathered_slot = []
        masks = []
        # entity_slot只用来做attention 因此补的0改成 -inf
        for b, a in zip(slot_output, slot_labels_id.ne(O_INDEX) & slot_labels_id.ne(PAD_INDEX)):  # update [B,30]
            if a.sum().item() != 0:
                v = b.masked_select(a.unsqueeze(-1)).view(1, -1, self.hidden_size)
                n = v.size(1)
                gap = max_slot - n
                if gap > 0:
                    zeros = torch.zeros(1, 1 * gap, self.hidden_size, device=input_ids.device)
                    v = torch.cat([v, zeros], 1)
                mask = torch.cat([torch.ones(1,n,device=input_ids.device), torch.zeros(1,gap,device=input_ids.device)],dim=-1)
            else:
                v = torch.zeros(1, max_slot, self.hidden_size, device=input_ids.device)
                mask = torch.zeros(1,max_slot,device=input_ids.device)
            gathered_slot.append(v)
            masks.append(mask)
        entity_slot = torch.cat(gathered_slot)
        slot_mask = torch.cat(masks)
        slot_mask = slot_mask.unsqueeze(1)
        # attention start

        attn_e_slot = torch.bmm(state_output, entity_slot.permute(0, 2, 1))  # B,T,1
        # .masked_fill(mask, -1e9) 已增加mask
        slot_mask = slot_mask.expand(slot_mask.size()[0],30,slot_mask.size()[-1])

        attn_e_slot = attn_e_slot.masked_fill(slot_mask==0,-1e9)

        attn_slot = nn.functional.softmax(attn_e_slot, -1)  # B,T

        state_attn_slot = torch.bmm(attn_slot, entity_slot)  # B,1,D
        # attention end

        state_output = state_output + state_attn_slot
        # state classification
        state_scores = self.action_cls(self.dropout(state_output))  # B,J,4



        #slot filling

        if self.exclude_domain:
            domain_scores = torch.zeros(1, device=input_ids.device)  # dummy
        else:
            domain_scores = self.domain_cls(self.dropout(pooled_output))

        batch_size = state_scores.size(0)
        if op_ids is None:
            op_ids = state_scores.view(-1, self.n_op).max(-1)[-1].view(batch_size, -1)
        if max_update is None:
            max_update = op_ids.eq(self.update_id).sum(-1).max().item()
        # 最后预处理加max_slot
        gathered = []


        # b:30 * 768   a:30
        for b, a in zip(state_output, op_ids.eq(self.update_id)):  # update [B,30]

            # update的数量
            if a.sum().item() != 0:
                v = b.masked_select(a.unsqueeze(-1)).view(1, -1, self.hidden_size)
                n = v.size(1)
                gap = max_update - n
                if gap > 0:
                    zeros = torch.zeros(1, 1*gap, self.hidden_size, device=input_ids.device)


                    v = torch.cat([v, zeros], 1)
            else:
                v = torch.zeros(1, max_update, self.hidden_size, device=input_ids.device)
            gathered.append(v)

        decoder_inputs = torch.cat(gathered)
        # [B,num_C] [B,42,num_C] [B,30,num_C]  [B,2,768] [B,256,H] [B,1,H] [B,42,H]
        # decoder_inputs : 需要转移的槽位对应隐层
        return domain_scores, slot_scores,state_scores, decoder_inputs, sequence_output, pooled_output.unsqueeze(0)#,entity_slot


class Decoder(nn.Module):
    def __init__(self, config, bert_model_embedding_weights):
        super(Decoder, self).__init__()
        self.pad_idx = 0
        self.hidden_size = config.hidden_size
        self.vocab_size = config.vocab_size
        self.embed = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=self.pad_idx)
        self.embed.weight = bert_model_embedding_weights
        self.gru = nn.GRU(config.hidden_size, config.hidden_size, 1, batch_first=True)
        self.w_gen = nn.Linear(config.hidden_size*3, 1)
        # self.w_gen_slot = nn.Linear(config.hidden_size*3,1)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(config.dropout)

        for n, p in self.gru.named_parameters():
            if 'weight' in n:
                p.data.normal_(mean=0.0, std=config.initializer_range)

    def forward(self, x, decoder_input, encoder_output, hidden, max_len,teacher=None):
        #         [B,256] [B,num_op,768]   [B,256,H]   [B,1,H]      9   [B,42,H]
        # decoder_inputs : 需要转移的槽位对应隐层

        mask = x.eq(self.pad_idx)
        batch_size, n_update, _ = decoder_input.size()  # B,J',5 # long
        state_in = decoder_input
        all_point_outputs = torch.zeros(n_update, batch_size, max_len, self.vocab_size).to(x.device)
        result_dict = {}
        for j in range(n_update):
            w = state_in[:, j].unsqueeze(1)  # B,1,D
            slot_value = []
            for k in range(max_len):
                # w [B,1,768] 一个个解码
                w = self.dropout(w)
                _, hidden = self.gru(w, hidden)  # 1,B,D
                # B,T,D * B,D,1 => B,T
                attn_e = torch.bmm(encoder_output, hidden.permute(1, 2, 0))  # B,T,1

                attn_e = attn_e.squeeze(-1).masked_fill(mask, -1e9)

                attn_history = nn.functional.softmax(attn_e, -1)  # B,T

                # B,D * D,V => B,V
                attn_v = torch.matmul(hidden.squeeze(0), self.embed.weight.transpose(0, 1))  # B,V
                attn_vocab = nn.functional.softmax(attn_v, -1)

                # B,1,T * B,T,D => B,1,D
                context = torch.bmm(attn_history.unsqueeze(1), encoder_output)  # B,1,D

                # slot attention mj -----
                #attn_e_slot = torch.bmm(entity_slot, hidden.permute(1, 2, 0))  # B,T,1

                #attn_e_slot = attn_e_slot.squeeze(-1)# .masked_fill(mask, -1e9) 还要写mask

                #attn_slot = nn.functional.softmax(attn_e_slot, -1)  # B,T

                # B,D * D,V => B,V
                # attn_v_slot = torch.matmul(hidden.squeeze(0), self.embed.weight.transpose(0, 1))  # B,V
                # attn_vocab_slot = nn.functional.softmax(attn_v_slot, -1)

                # B,1,T * B,T,D => B,1,D
                #context_slot = torch.bmm(attn_slot.unsqueeze(1), entity_slot)  # B,1,D

                # slot attention mj -----

                p_gen = self.sigmoid(self.w_gen(torch.cat([w, hidden.transpose(0, 1),context], -1)))  # B,1 # 只用context-slot mj
                p_gen = p_gen.squeeze(-1)

                p_context_ptr = torch.zeros_like(attn_vocab).to(x.device)
                p_context_ptr.scatter_add_(1, x, attn_history)  # copy B,V

                #p_gen_slot = self.sigmoid(self.w_gen(torch.cat([w, hidden.transpose(0, 1), context_slot], -1)))
                #p_gen_slot = p_gen_slot.squeeze(-1)

                #p_slot_ptr = torch.zeros_like(attn_vocab_slot).to(x.device)
                #p_slot_ptr.scatter_add_(1, x, attn_slot)

                # p_final_slot = p_gen_slot * attn_vocab_slot + (1 - p_gen_slot) * p_slot_ptr

                p_final_context = p_gen * attn_vocab + (1 - p_gen) * p_context_ptr  # B,V

                p_final = p_final_context # + p_final_slot
                _, w_idx = p_final.max(-1)
                slot_value.append([ww.tolist() for ww in w_idx])
                if teacher is not None:
                    w = self.embed(teacher[:, j, k]).unsqueeze(1)
                else:
                    w = self.embed(w_idx).unsqueeze(1)  # B,1,D
                all_point_outputs[j, :, k, :] = p_final

        return all_point_outputs.transpose(0, 1)
