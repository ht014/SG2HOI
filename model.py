from __future__ import print_function, division
import pickle
import torch
import numpy as np
import torch.nn as nn
from layers import ROIAlign
import torch.nn.functional as F
import torchvision.models as models
from torch.nn.utils import weight_norm
from structures.image_list import to_image_list
from utils import obj_edge_vectors,layer_init,get_box_info,get_box_pair_info
from utils import construct_pair_feature, load_all_vocabs, split_to_hum_obj, convert_to_roi_format,group_norm,make_fc


class FCNet(nn.Module):
	def __init__(self, in_size, out_size, activate=None, drop=0.0):
		super(FCNet, self).__init__()
		self.lin = weight_norm(nn.Linear(in_size, out_size), dim=None)
		self.drop_value = drop
		self.drop = nn.Dropout(drop)
		# in case of using upper character by mistake
		self.activate = activate.lower() if (activate is not None) else None
		if activate == 'relu':
			self.ac_fn = nn.ReLU()
		elif activate == 'sigmoid':
			self.ac_fn = nn.Sigmoid()
		elif activate == 'tanh':
			self.ac_fn = nn.Tanh()

	def forward(self, x):
		if self.drop_value > 0:
			x = self.drop(x)
		x = self.lin(x)
		if self.activate is not None:
			x = self.ac_fn(x)
		return x


class SelfAttention(nn.Module):
	def __init__(self, v_features, q_features, mid_features, glimpses, drop=0.0):
		super(SelfAttention, self).__init__()
		self.hidden_aug = 3
		self.glimpses = glimpses
		self.lin_v = FCNet(v_features, int(mid_features * self.hidden_aug), activate='relu',
						   drop=drop / 2.5)  # let self.lin take care of bias
		self.lin_q = FCNet(q_features, int(mid_features * self.hidden_aug), activate='relu', drop=drop / 2.5)

		self.h_weight = nn.Parameter(torch.Tensor(1, glimpses, 1, int(mid_features * self.hidden_aug)).normal_())
		self.h_bias = nn.Parameter(torch.Tensor(1, glimpses, 1, 1).normal_())

		self.drop = nn.Dropout(drop)

	def forward(self, v, q):

		v_num = v.size(1)
		q_num = q.size(1)

		v_ = self.lin_v(v).unsqueeze(1)  # batch, 1, v_num, dim
		q_ = self.lin_q(q).unsqueeze(1)  # batch, 1, q_num, dim
		v_ = self.drop(v_)

		h_ = v_ * self.h_weight  # broadcast:  batch x glimpses x v_num x dim
		logits = torch.matmul(h_, q_.transpose(2, 3))  # batch x glimpses x v_num x q_num
		logits = logits + self.h_bias

		atten = F.softmax(logits.view(-1, self.glimpses, v_num * q_num), 2)
		return atten.view(-1, self.glimpses, v_num, q_num)


class AttentionModule(nn.Module):
	def __init__(self, v_features, q_features, mid_features, glimpses, drop=0.0):
		super(AttentionModule, self).__init__()
		self.glimpses = glimpses
		layers = []
		for g in range(self.glimpses):
			layers.append(Correlation(v_features, q_features, mid_features, drop))
		self.glimpse_layers = nn.ModuleList(layers)

	def forward(self, v, q, atten):
		for g in range(self.glimpses):
			atten_h = self.glimpse_layers[g](v, q, atten[:, g, :, :])
			q = q + atten_h
		return q.sum(1)


class Correlation(nn.Module):
	def __init__(self, v_features, q_features, mid_features, drop=0.0):
		super(Correlation, self).__init__()
		self.lin_v = FCNet(v_features, mid_features, activate='relu', drop=drop)  # let self.lin take care of bias
		self.lin_q = FCNet(q_features, mid_features, activate='relu', drop=drop)
		self.lin_atten = FCNet(mid_features, mid_features, drop=drop)

	def forward(self, v, q, atten):

		v_ = self.lin_v(v).transpose(1, 2).unsqueeze(2)  # batch, dim, 1, num_obj
		q_ = self.lin_q(q).transpose(1, 2).unsqueeze(3)  # batch, dim, que_len, 1
		v_ = torch.matmul(v_, atten.unsqueeze(1))  # batch, dim, 1, que_len
		h_ = torch.matmul(v_, q_)  # batch, dim, 1, 1
		h_ = h_.squeeze(3).squeeze(2)  # batch, dim

		atten_h = self.lin_atten(h_.unsqueeze(1))

		return atten_h


class SGEncoder(nn.Module):
	def __init__(self, img_num_obj=81, img_num_rel=51):
		super(SGEncoder, self).__init__()
		self.embed_dim = 512
		self.hidden_dim = 512
		self.final_dim = 1024
		self.num_layer = 2
		self.margin = 1.0
		self.img_num_obj = img_num_obj
		self.img_num_rel = img_num_rel

		self.img_obj_embed = nn.Embedding(self.img_num_obj, self.embed_dim)
		self.img_rel_head_embed = nn.Embedding(self.img_num_obj, self.embed_dim)
		self.img_rel_tail_embed = nn.Embedding(self.img_num_obj, self.embed_dim)
		self.img_rel_pred_embed = nn.Embedding(self.img_num_rel, self.embed_dim)

		self.attention = weight_norm(SelfAttention(
			v_features=self.embed_dim * 2,
			q_features=self.embed_dim,
			mid_features=self.hidden_dim,
			glimpses=self.num_layer,
			drop=0.2, ), name='h_weight', dim=None)

		self.apply_attention = AttentionModule(
			v_features=self.embed_dim * 2,
			q_features=self.embed_dim,
			mid_features=self.hidden_dim,
			glimpses=self.num_layer,
			drop=0.2, )

		self.final_fc = nn.Sequential(*[nn.Linear(self.hidden_dim, self.hidden_dim),
										nn.ReLU(inplace=True),
										nn.Linear(self.hidden_dim, self.final_dim),
										nn.ReLU(inplace=True)
										])

	def encode(self, inp_dict):

		if len(inp_dict['relations']) == 0:
			inp_dict['relations'] = torch.zeros(1, 3).to(inp_dict['entities'].device).long()
		obj_encode = self.img_obj_embed(inp_dict['entities'])
		rel_tail_encode = self.img_rel_tail_embed(inp_dict['relations'][:, 1])
		rel_pred_encode = self.img_rel_pred_embed(inp_dict['relations'][:, 2])

		rel_encode = torch.cat(( rel_tail_encode, rel_pred_encode), dim=-1)
		atten = self.attention(rel_encode.unsqueeze(0), obj_encode.unsqueeze(0))
		sg_encode = self.apply_attention(rel_encode.unsqueeze(0), obj_encode.unsqueeze(0), atten)

		return self.final_fc(sg_encode).sum(0).view(1, -1)

	def forward(self, targets):
		encode_list = []
		for tg in targets:
			fg_img = tg.get_field('sg_data')
			fg_img['entities'] = fg_img['entities'].to(tg.bbox.device)
			fg_img['relations'] = fg_img['relations'].to(tg.bbox.device)
			fg_img_encode = self.encode(fg_img)
			encode_list.append(fg_img_encode)

		return torch.cat(encode_list)


class SG2HOI(nn.Module):
	def __init__(self,cfg):
		super(SG2HOI,self).__init__()
		self.cfg = cfg
		num_hois = 29
		model =models.resnet50(pretrained=True)
		input_size = 1024 * 7 * 7
		out_dim = 1024
		self.Conv_pretrain = nn.Sequential(*list(model.children())[0:7])#Resnets,resnext

		# For faster rcnn pre-trained model.
		# self.backbone_net = build_backbone(cfg)
		# self.union_box_feature_extractor = make_roi_relation_feature_extractor(out_dim)
		# self.box_feature_extractor = make_roi_box_feature_extractor(cfg, self.backbone_net.out_channels)

		self.hum_pool =  nn.Sequential(
			*[make_fc(input_size, out_dim*2), nn.ReLU(inplace=True),
			  make_fc(out_dim *2, out_dim), nn.ReLU(inplace=True),
			  ]
		)
		self.obj_pool = nn.Sequential(
			*[make_fc(input_size, out_dim*2), nn.ReLU(inplace=True),
			  make_fc(out_dim *2, out_dim), nn.ReLU(inplace=True),
			  ]
		)

		self.spt_emb = nn.Sequential(*[nn.Linear(32, out_dim // 2),
									   nn.ReLU(inplace=True),
									   nn.Linear(out_dim // 2, out_dim // 4),
									   nn.ReLU(inplace=True)
									   ])
		self.spt_emb2 = nn.Sequential(*[nn.Linear(32, out_dim ),
									   nn.ReLU(inplace=True),
									   nn.Linear(out_dim , out_dim ),
									   nn.ReLU(inplace=True)
									   ])
		self.conv_sp_map = nn.Sequential(
			nn.Conv2d(2, 64, kernel_size=(5, 5)), nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=(2, 2)),
			nn.Conv2d(64, 128, kernel_size=(5, 5)), nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=(2, 2)),
			nn.Conv2d(128, 256, kernel_size=(3, 3)), nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=(2, 2)),
			nn.Flatten()
		)
		self.sp_fc = nn.Sequential(
			*[make_fc(6400, out_dim * 2), nn.ReLU(inplace=True),
			  make_fc(out_dim * 2, out_dim), nn.ReLU(inplace=True),
			  ]
		)
		layer_init(self.spt_emb[0], xavier=True)
		layer_init(self.spt_emb[2], xavier=True)

		self.trans_per =  nn.Sequential(
			nn.Linear(out_dim * 4, out_dim),

		)
		self.trans_obj = nn.Sequential(
			nn.Linear(out_dim * 4, out_dim),

		)

		self.lin_visual_union = nn.Sequential(
			nn.Linear(out_dim*3 , out_dim),
			nn.ReLU(),
		)
		self.lin_graph_head = nn.Sequential(
			nn.Linear(out_dim * 2, out_dim),
			nn.ReLU(),
		)
		self.classifier_union_feats=nn.Sequential(
			nn.Linear(out_dim, num_hois),

		)
		self.classifier_hoi = nn.Sequential(
			nn.Linear(out_dim, num_hois),

		)
		self.classifier_spatial = nn.Sequential(
			nn.Linear(out_dim  , num_hois),

		)
		self.classifer_sg_message = nn.Sequential(
			nn.Linear(out_dim + 0, num_hois),

		)
		self.classifer_pair_box = nn.Sequential(
			nn.Linear(out_dim // 4 + 600, num_hois),
		)

		object_words = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
		vcoco_words_embed_vecs = obj_edge_vectors(['start'] + object_words, wv_dir='/mnt/hdd2/hetao/ne/neural-motifs/data',
										  wv_dim=300)

		self.vcoco_obj_embed = nn.Embedding(len(object_words) + 1, 300)

		with torch.no_grad():
			self.vcoco_obj_embed.weight.copy_(vcoco_words_embed_vecs, non_blocking=True)

		self.rel_classes = load_all_vocabs()
		obj_embed_vecs = obj_edge_vectors(['start'] + self.rel_classes, wv_dir='/mnt/hdd2/hetao/ne/neural-motifs/data', wv_dim=300)
		self.obj_embed = nn.Embedding(len(self.rel_classes) + 1, 300)
		with torch.no_grad():
			self.obj_embed.weight.copy_(obj_embed_vecs, non_blocking=True)

		self.roi_pool_layer =  ROIAlign(
                    (7,7), spatial_scale=0.0625, sampling_ratio=2
                )
		self.sg_encoder = SGEncoder()
		self.loss_hoi_classification = nn.BCELoss(reduction='none')
		self.mul_all_feats = False
		self.is_only_vis = False
		self.use_faster_rcnn_backbone = False
		self.rect_size = 28
		self.priors = pickle.load(open(self.cfg.prior,'rb'), encoding='latin1')

	def apply_prior(self,objs, feat):
		prior_mtx = torch.ones((feat.shape[0], 29)).to(feat.device)
		for index, prediction in enumerate(feat):
			prior_mtx[index] = torch.from_numpy(self.priors[int(objs[index])])
		return prior_mtx


	def construct_sematic_spatial_map(self,targets,p_embs,o_embs):
		mps = []
		offset = 0
		for tg in targets:
			rel_pair_idx = tg.get_field('pairs_info')
			person_embeddings = p_embs[offset:offset+rel_pair_idx.shape[0]]
			object_embeddings = o_embs[offset:offset+rel_pair_idx.shape[0]]
			head_proposal = tg[rel_pair_idx[:, 0]]
			tail_proposal = tg[rel_pair_idx[:, 1]]
			head_proposal = head_proposal.resize((self.rect_size, self.rect_size))
			tail_proposal = tail_proposal.resize((self.rect_size, self.rect_size))
			dumy = torch.zeros((1,2, self.rect_size, self.rect_size)).to(head_proposal.bbox.device)

			dumy[0,0, head_proposal.bbox[:,0].int().data.cpu().numpy():head_proposal.bbox[:,2].int().data.cpu().numpy(),
			head_proposal.bbox[:,1].int().data.cpu().numpy():head_proposal.bbox[:,3].int().data.cpu().numpy()] = person_embeddings
			dumy[0,1, tail_proposal.bbox[:,0].int():tail_proposal.bbox[:,2].int(),tail_proposal.bbox[:,1].int():tail_proposal.bbox[:,3].int()] = object_embeddings
			offset += rel_pair_idx.shape[0]
			mps.append(dumy)
		return torch.stack(dumy)

	def sg_message_passing(self,targets,roi_features_per,roi_features_obj,sg_adj_matrx):
		pairs_feats = []
		start_p = 0
		start_o = 0
		for batch_num, tg in enumerate(targets):
			human_feat_img = roi_features_per[start_p:start_p + tg.get_field('hoi_labels').shape[0]]
			img_objs = roi_features_obj[start_o:start_o + tg.get_field('hoi_labels').shape[1]]
			object_feat_img = img_objs[1:]

			none_object_feat = img_objs[0]
			if len(object_feat_img) == 0:
				human_refined_feats = human_feat_img
				objects_refined_feats = none_object_feat.view([1, roi_features_per.shape[-1]])
			else:
				adj_p2o = sg_adj_matrx[batch_num][0]
				adj_o2p = sg_adj_matrx[batch_num][1]

				tran_adj_p2o = adj_p2o @ self.obj_embed.weight
				tran_adj_o2p = adj_o2p @ self.obj_embed.weight

				human_refined_feats = human_feat_img + torch.einsum('ijk,jl->il', [tran_adj_p2o, object_feat_img])
				objects_refined_feats = object_feat_img + torch.einsum('ijk,jl->il', [tran_adj_o2p, human_feat_img])
				objects_refined_feats = torch.cat(
					(none_object_feat.view([1, roi_features_per.shape[-1]]), objects_refined_feats))

			pairs_idx = tg.get_field('pairs_info')
			for pair in pairs_idx:
				pair = pair.long()
				hum_feat = human_refined_feats[pair[0]]
				obj_feat = objects_refined_feats[pair[1] - tg.get_field('hoi_labels').shape[0]]
				pairs_feats.append(torch.cat((hum_feat, obj_feat)).view(1, -1))

			start_p += tg.get_field('hoi_labels').shape[0]
			start_o += tg.get_field('hoi_labels').shape[1]
		return torch.cat(pairs_feats)

	def calculate_loss(self,pred_vis_feats, pred_by_sg_mesage,pred_spatial,targets, prior_mask, pred_pair_box=None):

		score_confidence = torch.cat([t.get_field('per_mul_obj_scores') for t in targets]).view(-1, 1)
		if pred_pair_box is not None:
			predicted_HOI = F.sigmoid(pred_vis_feats) * F.sigmoid(pred_by_sg_mesage) * F.sigmoid(pred_spatial) \
							* F.sigmoid(pred_pair_box) * score_confidence #* prior_mask
		else:
			predicted_HOI = F.sigmoid(pred_vis_feats)  * F.sigmoid(pred_spatial) \
							  * score_confidence   * F.sigmoid(pred_by_sg_mesage)
		labels = torch.cat([t.get_field('HOI_labs') for t in targets])
		losses =  torch.sum(self.loss_hoi_classification(predicted_HOI, labels.float()))  / len(targets) / 29.

		return predicted_HOI,losses

	def forward(self,x, targets):

		if self.use_faster_rcnn_backbone:
			feat_map = self.backbone_net(x.tensors)
			rois = self.box_feature_extractor(feat_map,targets)

		else:
			feat_map = self.Conv_pretrain(x.tensors)
			rois_box = convert_to_roi_format(targets)
			rois = self.roi_pool_layer(feat_map, rois_box)

		union_box = torch.cat([tg.get_field('union_box') for tg in targets])
		spatial_attention = self.sp_fc(self.conv_sp_map(union_box))

		# scene graph layout embedding
		if 'sg_data' in targets[0].extra_fields:
			sg_embeddings = self.sg_encoder(targets)
		else:
			sg_embeddings = None

		out2_people, out2_objects = split_to_hum_obj(rois,targets)
		if not self.use_faster_rcnn_backbone:
			roi_features_per = self.hum_pool(out2_people.view(out2_people.size(0),-1))
			roi_features_obj = self.obj_pool(out2_objects.view(out2_objects.size(0),-1))

		else:
			roi_features_per,roi_features_obj = self.trans_per(out2_people), self.trans_obj(out2_objects)

		pairs, sg_adj_matrx, pair_box_feats = construct_pair_feature(roi_features_per, roi_features_obj, sg_embeddings, targets)

		pairs_graph = self.sg_message_passing(targets, roi_features_per, roi_features_obj,sg_adj_matrx)
		obj_classes = torch.cat([m.get_field('mask') for m in targets])
		losses = {}
		if self.mul_all_feats:
			final_feats = self.spt_emb2(pair_box_feats) * spatial_attention * self.lin_visual_union(pairs)   * self.lin_graph_head(pairs_graph)
			obj_classes = torch.cat([m.get_field('mask') for m in targets])
			prior_mask = self.apply_prior(obj_classes, final_feats, is_all_label=True)

			score_confidence = torch.cat([t.get_field('per_mul_obj_scores') for t in targets]).view(-1, 1)
			predicted_hoi = F.sigmoid(self.classifier_hoi(final_feats))*prior_mask*score_confidence
			labels = torch.cat([t.get_field('HOI_labs') for t in targets])

			loss= torch.sum(self.loss_hoi_classification(predicted_hoi, labels.float())) / len(targets) / 29.
			losses['binary loss'] = loss
		else:

			pred_spatial = self.classifier_spatial(spatial_attention)
			prior_mask = apply_prior(obj_classes, pred_spatial, is_all_label=True)

			trans_union = self.lin_visual_union(pairs)
			lin_t = trans_union * spatial_attention
			pred_vis_feats = self.classifier_union_feats(lin_t)


			trans_graph = self.lin_graph_head(pairs_graph)
			lin_graph_t = trans_graph #* spatial_attention
			pred_by_sg_mesage = self.classifer_sg_message(lin_graph_t)

			predicted_hoi,loss = self.calculate_loss(pred_vis_feats,pred_by_sg_mesage,pred_spatial, targets,prior_mask,pred_pair_box=None,)
			losses['binary loss']= loss

		return predicted_hoi,losses


