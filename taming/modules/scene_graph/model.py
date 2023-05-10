import torch


class Scene_graph_encoder(torch.nn.Module):
    def __init__(self,max_objects=11,n_embd=1024,max_obj_id=182):
        super(Scene_graph_encoder, self).__init__()
        self.max_objects=max_objects
        self.max_obj_id=max_obj_id
        self.token_start_position=n_embd
        pass

    def encode(self, x):
        device=x[0][0].device
        all_objs, all_triples=x
        sequence_list=list()
        location_matrix=torch.LongTensor([self.max_objects, 0, 1]).to(device=device)
        for i,objs in enumerate(all_objs):
            triples=all_triples[i]

            objs_padding=torch.LongTensor(self.max_objects-objs.shape[0]).fill_(999).to(device=device)
            #objs_padding=objs_padding
            if objs_padding.shape[0]==0:
                obj_token=objs
            else:
                obj_token=torch.cat((objs,objs_padding))

            relation_token=torch.LongTensor(45*2).fill_(999).to(device=device)
            relation_location_token=torch.einsum('ij,j->i', triples.float(), location_matrix.float())
            triple_len=relation_location_token.shape[0]

            relation_token[0:triple_len*2-1:2]=relation_location_token+200
            relation_token[1:triple_len*2:2]=triples[:,1]+400



            relation_token+=self.max_obj_id
            sequence=torch.cat((obj_token,relation_token))
            sequence+=self.token_start_position
            sequence_list.append(sequence)
            pass
        sequences=torch.stack(sequence_list,dim=0).to(device=device)
        #quant_c, _, [_, _, indices]
        #print('sequences.shape: ',sequences.shape)
        return None,None,[None,None,sequences]

        #return torch.Tensor((batch_size,sequence_len))
        #sequence_len=10+45*2=100
        pass

class Scene_graph_encoder_old(torch.nn.Module):
    def __init__(self,max_objects=11):
        super(Scene_graph_encoder_old, self).__init__()
        self.max_objects=max_objects
        self.obj_amounts=178
        self.token_start_position=1024
        pass

    def encode(self, x):
        device=x[0][0].device
        all_objs, all_triples=x
        sequence_list=list()
        location_matrix=torch.LongTensor([self.max_objects, 0, 1]).to(device=device)
        for i,objs in enumerate(all_objs):
            triples=all_triples[i]

            objs_padding=torch.LongTensor(self.max_objects-objs.shape[0]).fill_(999).to(device=device)
            #objs_padding=objs_padding
            if objs_padding.shape[0]==0:
                obj_token=objs
            else:
                obj_token=torch.cat((objs,objs_padding))

            relation_token=torch.LongTensor(45*2).fill_(999).to(device=device)
            #relation_token=relation_token.to(device)
            relation_location_token=torch.einsum('ij,j->i', triples.float(), location_matrix.float())
            triple_len=relation_location_token.shape[0]
            relation_token[0:triple_len*2-1:2]=relation_location_token
            relation_token[1:triple_len*2:2]=triples[:,1]
            relation_token+=self.obj_amounts
            sequence=torch.cat((obj_token,relation_token))
            sequence+=self.token_start_position
            sequence_list.append(sequence)
            pass
        sequences=torch.stack(sequence_list,dim=0).to(device=device)
        #quant_c, _, [_, _, indices]
        #print('sequences.shape: ',sequences.shape)
        return None,None,[None,None,sequences]

        #return torch.Tensor((batch_size,sequence_len))
        #sequence_len=10+45*2=100
        pass