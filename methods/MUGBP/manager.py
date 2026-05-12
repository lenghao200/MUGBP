import torch
import torch.nn.functional as F
import logging
from torch import nn
from utils.functions import restore_model, save_model, EarlyStopping
from tqdm import trange, tqdm
from data.utils import get_dataloader
from utils.metrics import AverageMeter, Metrics
from transformers import AdamW, get_linear_schedule_with_warmup
from .model import MUGBP
from .loss import SupConLoss, InfoNCE, Multi_infoNCE, Multi_SupCon
import numpy as np
import os
import pandas as pd

from .GranularBall.cluster import gbcluster
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

__all__ = ['MUGBP_manager']


class MUGBP_manager:

    def __init__(self, args, data, labels_weight):

        self.logger = logging.getLogger(args.logger_name)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        args.device = self.device

        if not hasattr(args, 'purity_train'):
            args.purity_train = 0.95
            args.purity_get_ball = 0.95
            args.min_ball_train = 5  # 全局模式下，5 或 6 是最佳分裂阈值
            args.min_ball_get_ball = 2
            args.min_ball_select_ball = 1
            args.purity_select_ball = 0.95

        self.gb_cluster = gbcluster(args, data).to(self.device)

        self.model = MUGBP(args)
        self.model.to(self.device)
        self.optimizer, self.scheduler = self._set_optimizer(args, self.model)

        mm_dataloader = get_dataloader(args, data.mm_data)
        self.train_dataloader, self.eval_dataloader, self.test_dataloader = \
            mm_dataloader['train'], mm_dataloader['dev'], mm_dataloader['test']

        self.args = args
        self.labels_weight = labels_weight.to(self.device)

        # 恢复纯净版的分类 Loss
        self.criterion = nn.CrossEntropyLoss()

        print('manager_re_multi_view')
        print('loss', args.loss)
        print('align_method', args.aligned_method)
        if args.loss == 'InfoNCE':
            self.cons_criterion = Multi_infoNCE(temperature=args.temperature, reduction='mean',
                                                negative_mode='unpaired')
        if args.loss == 'SupCon':
            self.cons_criterion = Multi_SupCon(temperature=args.temperature)
        self.metrics = Metrics(args)

        if args.train:
            self.best_eval_score = 0
        else:
            self.model = restore_model(self.model, args.model_output_path, self.device)

    def _set_optimizer(self, args, model):
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay': args.weight_decay},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

        optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr, correct_bias=False)

        if args.learning_rate_method == 'Cosine annealing':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, eta_min=0.0,
                                                                   T_max=args.num_train_epochs // 5)
        else:
            num_train_optimization_steps = int(args.num_train_examples / args.train_batch_size) * args.num_train_epochs
            num_warmup_steps = int(
                args.num_train_examples * args.num_train_epochs * args.warmup_proportion / args.train_batch_size)
            scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps,
                                                        num_training_steps=num_train_optimization_steps)

        return optimizer, scheduler

    def _train(self, args):

        early_stopping = EarlyStopping(args)
        no_improve_epochs = 0
        self.best_eval_score = 0

        for epoch in trange(int(args.num_train_epochs), desc="Epoch"):

            # =====================================================================
            # [步骤 1] Epoch 级别全局快照：提取整个训练集的最新文本锚点特征
            # =====================================================================
            self.model.eval()
            global_feats = []
            global_labels = []

            with torch.no_grad():
                for batch in self.train_dataloader:
                    cons_text_feats = batch['cons_text_feats'].to(self.device)
                    condition_idx = batch['condition_idx'].to(self.device)
                    # 仅推理文本锚点部分以节省时间
                    cons_input_ids, cons_input_mask, cons_segment_ids = cons_text_feats[:, 0], cons_text_feats[:,
                                                                                               1], cons_text_feats[:, 2]
                    cons_outputs = self.model.anchor(
                        input_ids=cons_input_ids,
                        condition_idx=condition_idx,
                        token_type_ids=cons_segment_ids,
                        attention_mask=cons_input_mask
                    )
                    last_hidden_state = cons_outputs.last_hidden_state
                    cons_condition_tuple = tuple(
                        last_hidden_state[torch.arange(last_hidden_state.shape[0]), condition_idx.view(-1) + i,
                        :].unsqueeze(1) for i in range(args.label_len))
                    cons_condition = torch.cat(cons_condition_tuple, dim=1).mean(dim=1)

                    global_feats.append(cons_condition)
                    global_labels.append(batch['label_ids'])

            global_feats = torch.cat(global_feats, dim=0).to(self.device)
            global_labels = torch.cat(global_labels, dim=0).to(self.device)

            # 使用最精准的全局特征构建全局粒球！
            gb_centroids, gb_radii, gb_labels = self.gb_cluster(
                args, global_feats, global_labels, select=False
            )

            # 转化为 GPU 张量，供后续批次更新和计算使用
            if len(gb_centroids) > 0:
                gb_centroids = torch.tensor(np.array(gb_centroids), device=self.device, dtype=torch.float)
                gb_radii = torch.tensor(np.array(gb_radii), device=self.device, dtype=torch.float)
                gb_labels_np = np.array(gb_labels)
            else:
                gb_centroids = None

            # =====================================================================
            # [步骤 2] 正常 Batch 训练循环
            # =====================================================================
            self.model.train()
            loss_record = AverageMeter()
            cons_loss_record = AverageMeter()
            cls_loss_record = AverageMeter()

            for step, batch in enumerate(tqdm(self.train_dataloader, desc="Iteration")):

                text_feats = batch['text_feats'].to(self.device)
                cons_text_feats = batch['cons_text_feats'].to(self.device)
                condition_idx = batch['condition_idx'].to(self.device)
                video_feats = batch['video_feats'].to(self.device)
                audio_feats = batch['audio_feats'].to(self.device)
                label_ids = batch['label_ids'].to(self.device)

                with torch.set_grad_enabled(True):

                    # 1. 前向传播
                    logits, _, condition, cons_condition, text_condition, visual_condition, acoustic_condition = self.model(
                        text_feats, video_feats, audio_feats, cons_text_feats, condition_idx
                    )

                    # =====================================================================
                    # [步骤 3核心] 在线实时微调全局粒球 (Online EMA Update)
                    # 消除滞后性：让全局粒球在每个 Batch 吸纳最新的模型权重变化
                    # =====================================================================
                    momentum = getattr(args, 'momentum') # 动量系数：0.99 代表非常平滑的更新

                    gb_loss = torch.tensor(0.0).to(self.device)
                    if gb_centroids is not None:
                        for i in range(len(label_ids)):
                            current_label = label_ids[i]
                            current_anchor = cons_condition[i]  # 最新的文本锚点特征
                            current_vis_feat = visual_condition[i]
                            current_aud_feat = acoustic_condition[i]
                            current_fused_feat = condition[i]

                            target_indices = (gb_labels_np == current_label.item())
                            if target_indices.sum() == 0:
                                continue

                                # 找到该类别对应的所有全局球
                            target_centers = gb_centroids[target_indices]
                            target_radii = gb_radii[target_indices]

                            # --- EMA 动量同步更新（球心 + 半径） ---
                            with torch.no_grad():
                                # 1. 寻找当前最新特征离得最近的那个全局子球
                                anchor_dists = torch.norm(target_centers - current_anchor.detach(), p=2, dim=1)
                                min_anchor_idx = torch.argmin(anchor_dists)
                                # 获取在全局 gb_centroids 中的真实索引
                                global_ball_idx = np.where(target_indices)[0][min_anchor_idx.item()]

                                # 2. 【更新球心】：向最新特征拉近一点点
                                new_centroid = momentum * gb_centroids[global_ball_idx] + (
                                            1 - momentum) * current_anchor.detach()
                                gb_centroids[global_ball_idx] = new_centroid
                                target_centers[min_anchor_idx] = new_centroid  # 同步更新用于计算 loss 的临时变量

                                # 3. 【计算距离】：当前最新特征到“新球心”的欧氏距离
                                current_dist = torch.norm(current_anchor.detach() - new_centroid, p=2)

                                # 4. 【更新半径】：使用 EMA 更新平均距离（即新的半径）
                                new_radius = momentum * gb_radii[global_ball_idx] + (1 - momentum) * current_dist

                                # [重要保护机制]：防坍缩 (Anti-collapse)
                                # 随着网络训练，特征如果过度聚拢，半径可能会趋近于 0，导致惩罚变态严苛。
                                # 设置一个基于初始半径的底线（如初始半径的 50%），保证球不会坍缩成一个点。
                                original_radius = target_radii[min_anchor_idx]
                                min_radius_limit = original_radius * 0.5
                                new_radius = torch.max(new_radius, min_radius_limit)

                                gb_radii[global_ball_idx] = new_radius
                                target_radii[min_anchor_idx] = new_radius

                            # --- 计算基于最新全局球的 gb_loss ---
                            # 视频模态损失
                            dists_v = torch.norm(target_centers - current_vis_feat, p=2, dim=1)
                            min_dist_v, min_idx_v = torch.min(dists_v, dim=0)
                            loss_v = torch.relu(min_dist_v - target_radii[min_idx_v])

                            # 音频模态损失
                            dists_a = torch.norm(target_centers - current_aud_feat, p=2, dim=1)
                            min_dist_a, min_idx_a = torch.min(dists_a, dim=0)
                            loss_a = torch.relu(min_dist_a - target_radii[min_idx_a])

                            # 融合特征损失
                            dists_c = torch.norm(target_centers - current_fused_feat, p=2, dim=1)
                            min_dist_c, min_idx_c = torch.min(dists_c, dim=0)
                            loss_c = torch.relu(min_dist_c - target_radii[min_idx_c])

                            gb_loss += (loss_v + loss_a + loss_c)

                        gb_loss = gb_loss / (len(label_ids) + 1e-6)
                    # =====================================================================

                    # 4. 计算对比损失
                    cons_feature = torch.cat((condition.unsqueeze(1), cons_condition.unsqueeze(1)), dim=1)
                    text_feature = torch.cat((text_condition.unsqueeze(1), cons_condition.unsqueeze(1)), dim=1)
                    visual_feature = torch.cat((visual_condition.unsqueeze(1), cons_condition.unsqueeze(1)), dim=1)
                    acoustic_feature = torch.cat((acoustic_condition.unsqueeze(1), cons_condition.unsqueeze(1)), dim=1)

                    if args.loss == 'InfoNCE':
                        cons_loss = self.cons_criterion.compute_loss(text_anchor=cons_condition,
                                                                     text_view=text_condition,
                                                                     visual_view=visual_condition,
                                                                     acoustic_view=acoustic_condition,
                                                                     global_view=condition).to(self.device)
                    elif args.loss == 'SupCon':
                        cons_loss = self.cons_criterion.compute_loss(cons_feature, text_feature, visual_feature,
                                                                     acoustic_feature).to(self.device)

                    # 5. 分类损失
                    cls_loss = self.criterion(logits, label_ids).to(self.device)

                    # 6. 总 Loss
                    lambda_gb = getattr(args, 'lambda_gb', 0.05)
                    loss = cons_loss + cls_loss + (lambda_gb * gb_loss)

                    self.optimizer.zero_grad()
                    loss.backward()

                    loss_record.update(loss.item(), label_ids.size(0))
                    cons_loss_record.update(cons_loss.item(), label_ids.size(0))
                    #cls_loss_record.update(cls_loss.item(), label_ids.size(0))

                    if args.grad_clip != -1.0:
                        nn.utils.clip_grad_value_([param for param in self.model.parameters() if param.requires_grad],
                                                  args.grad_clip)

                    self.optimizer.step()
                    self.scheduler.step()

            # --- Eval 流程 ---
            outputs = self._get_outputs(args, self.eval_dataloader)
            eval_score = outputs[args.eval_monitor]

            gb_loss_val = gb_loss.item() if isinstance(gb_loss, torch.Tensor) else gb_loss
            eval_results = {
                'train_loss': round(loss_record.avg, 4),
                'cons_loss': round(cons_loss_record.avg, 4),
                'cls_loss': round(cls_loss_record.avg, 4),
                'gb_loss': round(gb_loss_val, 4),
                'eval_score': round(eval_score, 4),
                'best_eval_score': round(early_stopping.best_score, 4),
            }

            self.logger.info("***** Epoch: %s: Eval results *****", str(epoch + 1))
            for key in eval_results.keys():
                self.logger.info("  %s = %s", key, str(eval_results[key]))

            early_stopping(eval_score, self.model)

            if early_stopping.early_stop:
                self.logger.info(f'EarlyStopping at epoch {epoch + 1}')
                break

            if eval_score > self.best_eval_score:
                self.best_eval_score = eval_score
                no_improve_epochs = 0
                save_path = self.args.model_output_path
                torch.save(self.model.state_dict(), os.path.join(save_path, 'best_model.pth'))
                self.logger.info('The Best Model is Saved')
            else:
                no_improve_epochs += 1
                if args.learning_rate_method == 'decay' and no_improve_epochs >= 4:
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = param_group['lr'] / 2
                    self.logger.info(f'Learning rate decayed to {self.optimizer.param_groups[0]["lr"]}')
                    no_improve_epochs = 0

            if args.learning_rate_method == 'Cosine annealing':
                self.scheduler.step()

            current_lr = self.optimizer.param_groups[0]['lr']
            self.logger.info(f'Current learning rate: {current_lr:.6f}')

        if args.save_model:
            self.logger.info('Trained models are saved in %s', args.model_output_path)
            save_model(self.model, args.model_output_path)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _get_outputs(self, args, dataloader, show_results=False):

        self.model.eval()

        total_labels = torch.empty(0, dtype=torch.long).to(self.device)
        total_preds = torch.empty(0, dtype=torch.long).to(self.device)
        total_logits = torch.empty((0, args.num_labels)).to(self.device)
        total_features = torch.empty((0, args.feat_size)).to(self.device)

        for batch in tqdm(dataloader, desc="Iteration"):
            text_feats = batch['text_feats'].to(self.device)
            cons_text_feats = batch['cons_text_feats'].to(self.device)
            condition_idx = batch['condition_idx'].to(self.device)
            video_feats = batch['video_feats'].to(self.device)
            audio_feats = batch['audio_feats'].to(self.device)
            label_ids = batch['label_ids'].to(self.device)

            with torch.set_grad_enabled(False):
                logits, features, condition, cons_condition, text_condition, visual_condition, acoustic_condition \
                    = self.model(text_feats, video_feats, audio_feats, cons_text_feats, condition_idx)
                total_logits = torch.cat((total_logits, logits))
                total_labels = torch.cat((total_labels, label_ids))
                total_features = torch.cat((total_features, features))

        total_probs = F.softmax(total_logits.detach(), dim=1)
        total_maxprobs, total_preds = total_probs.max(dim=1)

        y_logit = total_logits.cpu().numpy()
        y_pred = total_preds.cpu().numpy()
        y_true = total_labels.cpu().numpy()
        y_prob = total_maxprobs.cpu().numpy()
        y_feat = total_features.cpu().numpy()

        outputs = self.metrics(y_true, y_pred, show_results=show_results)

        if args.save_pred and show_results:
            np.save('y_true_' + str(args.seed) + '.npy', y_true)
            np.save('y_pred_' + str(args.seed) + '.npy', y_pred)

        outputs.update(
            {
                'y_prob': y_prob,
                'y_logit': y_logit,
                'y_true': y_true,
                'y_pred': y_pred,
                'y_feat': y_feat
            }
        )

        return outputs

    def _test(self, args):
        save_path = self.args.model_output_path
        best_model_path = os.path.join(save_path, 'best_model.pth')

        print(f"Loading best model from: {best_model_path}")
        self.model.load_state_dict(torch.load(best_model_path))
        self.model.to(self.device)
        self.model.eval()

        test_results = {}
        ind_outputs = self._get_outputs(args, self.test_dataloader, show_results=True)

        if 'preds' in ind_outputs:
            preds = ind_outputs['preds']
        elif 'predictions' in ind_outputs:
            preds = ind_outputs['predictions']
        elif 'y_pred' in ind_outputs:
            preds = ind_outputs['y_pred']
        else:
            raise KeyError(f"无法找到预测值。Available keys: {list(ind_outputs.keys())}")

        if 'labels' in ind_outputs:
            true_labels = ind_outputs['labels']
        elif 'true_labels' in ind_outputs:
            true_labels = ind_outputs['true_labels']
        elif 'y_true' in ind_outputs:
            true_labels = ind_outputs['y_true']
        else:
            raise KeyError(f"无法找到真实标签。Available keys: {list(ind_outputs.keys())}")

        acc = accuracy_score(true_labels, preds)
        p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(true_labels, preds, average='weighted')
        _, r_macro, _, _ = precision_recall_fscore_support(true_labels, preds, average='macro')

        print("\n" + "=" * 50)
        print(f"Dataset: {args.dataset} | Best Eval Score: {self.best_eval_score:.4f}")
        print("-" * 50)
        print(f"{'Metric':<25} | {'Value (%)':<10}")
        print("-" * 50)
        print(f"{'ACC (Accuracy)':<25} | {acc * 100:.2f}")
        print(f"{'WF1 (Weighted F1)':<25} | {f1_weighted * 100:.2f}")
        print(f"{'WP (Weighted Precision)':<25} | {p_weighted * 100:.2f}")
        print(f"{'R (Recall)':<25} | {r_macro * 100:.2f}")
        print("=" * 50 + "\n")

        ind_outputs['best_eval_score'] = round(self.best_eval_score, 4)
        ind_outputs['ACC'] = acc
        ind_outputs['WF1'] = f1_weighted
        ind_outputs['WP'] = p_weighted
        ind_outputs['R'] = r_macro

        test_results.update(ind_outputs)

        return test_results
