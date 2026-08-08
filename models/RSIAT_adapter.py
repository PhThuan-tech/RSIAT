import copy
import logging
import numpy as np
import torch
from torch import nn
from torch.serialization import load
from tqdm import tqdm
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from utils.inc_net import SimpleVitNet
from torch.distributions.multivariate_normal import MultivariateNormal
from models.base import BaseLearner
from utils.toolkit import count_parameters, log_count_parameter, target2onehot, tensor2numpy
from utils.loss import AngularPenaltySMLoss
from utils.toolkit import AutoencoderSigmoid
from utils.research_losses import (
    adaptive_topk_separation_loss,
    moment_alignment_loss,
)
from models.projectors import build_projector
import math
num_workers = 8

class Learner(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        if 'adapter' not in args["convnet_type"]:
            raise NotImplementedError('Adapter requires Adapter backbone')
        self._network = SimpleVitNet(args, True)
        self.batch_size = args["batch_size"]
        self.init_lr = args["init_lr"]

        self.weight_decay = args["weight_decay"] if args["weight_decay"] is not None else 0.0005
        self.min_lr = args['min_lr'] if args['min_lr'] is not None else 1e-8
        self.args = args

        self._old_most_sentive = []
        self._update_grads = {}

        self.logit_norm = None
        self.tuned_epochs = None
        self.task_sizes = []
        self.rs_loss_func = RS_Loss(self.args["alpha"], self.args["rs_margin"])
        self.old_ae = None
        self.use_research_projector = (
            self.args.get("model_name", "adapter").lower() == "umt_adapter"
            or "projector_type" in self.args
        )
        self._loss_details = {}

    def _after_load_checkpoint(self, checkpoint):
        if self._cur_task >= 1:
            if self.use_research_projector:
                self.old_ae = build_projector(self.args, input_dim=self.feature_dim)
            else:
                self.old_ae = AutoencoderSigmoid(
                    input_dims=768,
                    code_dims=self.args["ae_code_dims"],
                )
            if "old_ae_state_dict" in checkpoint:
                self.old_ae.load_state_dict(checkpoint["old_ae_state_dict"])
            else:
                logging.warning(
                    "Checkpoint has no old_ae_state_dict. "
                    "Resume will use a freshly initialized autoencoder."
                )
            self.old_ae.to(self._device)

        self._network_module_ptr = self._network
        if hasattr(self._old_network, "module"):
            self.old_network_module_ptr = self._old_network.module
        else:
            self.old_network_module_ptr = self._old_network

    def after_task(self):
        self._known_classes = self._total_classes
        self._old_network = self._network.copy().freeze()
        if hasattr(self._old_network,"module"):
            self.old_network_module_ptr = self._old_network.module
        else:
            self.old_network_module_ptr = self._old_network


    def extract_features(self, trainloader, model, args):
        model = model.eval()
        embedding_list = []
        label_list = []
        with torch.no_grad():
            for i, batch in enumerate(trainloader):
                (_, data, label) = batch
                data = data.cuda()
                label = label.cuda()
                embedding = model.extract_vector(data)
                embedding_list.append(embedding.cpu())
                label_list.append(label.cpu())

        embedding_list = torch.cat(embedding_list, dim=0)
        label_list = torch.cat(label_list, dim=0)
        return embedding_list, label_list

    def incremental_train(self, data_manager):
        self._cur_task += 1
        
        if self._cur_task > 0 and self.use_research_projector:
            reset_projector = self.args.get("projector_reset_each_task", True)
            if self.old_ae is None or reset_projector:
                self.old_ae = build_projector(self.args, input_dim=self.feature_dim)
            self.old_ae.to(self._device)
        elif self._cur_task == 1:
            self.old_ae = AutoencoderSigmoid(input_dims=768, code_dims=self.args["ae_code_dims"])
            self.old_ae.to(self._device)
            
        task_size = data_manager.get_task_size(self._cur_task)
        self.task_sizes.append(task_size)
        self._total_classes = self._known_classes + task_size
        # self._network.update_fc(data_manager.get_task_size(self._cur_task)*4)
        self._network.update_fc(task_size)
        self._network_module_ptr = self._network
        logging.info("Learning on {}-{}".format(self._known_classes, self._total_classes))
    
        train_dataset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes), source="train",
                                                 mode="train")

        self.train_dataset = train_dataset
        print("The number of training dataset:", len(self.train_dataset))

        self.data_manager = data_manager
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=8)
        test_dataset = data_manager.get_dataset(np.arange(0, self._total_classes), source="test", mode="test")
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=8)

        if len(self._multiple_gpus) > 1:
            print('Multiple GPUs')
            self._network = nn.DataParallel(self._network, self._multiple_gpus)

      
        use_projector_transport = self.args.get("prototype_transport", False)
        if self._cur_task > 0 and not use_projector_transport:
            self._network.to(self._device)
            train_embeddings_old, _ = self.extract_features(self.train_loader, self._network, None)

        self._train(self.train_loader, self.test_loader)
        
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

      
        if self._cur_task > 0:
            if use_projector_transport:
                self._transport_old_statistics()
            else:
                train_embeddings_new, _ = self.extract_features(self.train_loader, self._network, None)
                old_class_mean = self._class_means[:self._known_classes]
                gap = self.displacement(train_embeddings_old, train_embeddings_new, old_class_mean, 4.0)
                if self.args['ssca'] is True:
                    old_class_mean +=gap
                    self._class_means[:self._known_classes] = old_class_mean

        self._network.fc.backup()
        self._compute_class_mean(data_manager, check_diff=False, oracle=False)
        if self._cur_task>0 and self.args['ca_epochs']>0 and self.args['ca'] is True:
            self._stage2_compact_classifier(task_size, self.args['ca_epochs'])
            if len(self._multiple_gpus) > 1:
                self._network = self._network.module

    def _train(self, train_loader, test_loader):
        self._network.to(self._device)
        if self._cur_task == 0:
            self.tuned_epochs = self.args["init_epochs"]
            param_groups = [
                {'params': self._network.convnet.blocks[-1].parameters(), 'lr': 0.01,
                 'weight_decay': self.args['weight_decay']},
                {'params': self._network.convnet.blocks[:-1].parameters(), 'lr': 0.01,
                 'weight_decay': self.args['weight_decay']},
                {'params': self._network.fc.parameters(), 'lr': 0.01, 'weight_decay': self.args['weight_decay']}
            ]

            if self.args['optimizer'] == 'sgd':
                optimizer = optim.SGD(param_groups, momentum=0.9, lr=self.init_lr, weight_decay=self.weight_decay)
            elif self.args['optimizer'] == 'adam':
                optimizer = optim.AdamW(param_groups, lr=self.init_lr, weight_decay=self.weight_decay)
                
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.tuned_epochs, eta_min=self.min_lr)
            log_count_parameter(param_groups)
            self._init_train(train_loader, test_loader, optimizer, scheduler, self.args['warmup_epoch'])
        else:
            self.tuned_epochs = self.args['inc_epochs']
            param_groups = []
            param_groups.append(
                {'params': self._network.convnet.parameters(), 'lr': self.init_lr, 'weight_decay': self.weight_decay})
            param_groups.append(
                {'params': self._network.fc.parameters(), 'lr': self.init_lr, 'weight_decay': self.weight_decay})
            param_groups.append(
                {
                    'params': self.old_ae.parameters(),
                    'lr': self.args.get('projector_lr', self.args['ae_init_lr']),
                    'weight_decay': self.args.get('projector_weight_decay', self.args['ae_weight_decay']),
                })
            
            if self.args['optimizer'] == 'sgd':
                optimizer = optim.SGD(param_groups, momentum=0.9)
            elif self.args['optimizer'] == 'adam':
                optimizer = optim.AdamW(param_groups, lr=self.init_lr, weight_decay=self.weight_decay)

            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.tuned_epochs, eta_min=self.min_lr)
            log_count_parameter(param_groups)
            self._init_train(train_loader, test_loader, optimizer, scheduler, self.args['warmup_epoch'])

    def _init_train(self, train_loader, test_loader, optimizer, scheduler, warmup_epoch):
        prog_bar = tqdm(range(self.tuned_epochs))
        eval_interval = self.args.get("eval_interval", 0)
        
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            losses = 0.0
            losses_c, losses_rt = 0.0, 0.0
            detail_sums = {}
            correct, total = 0, 0

            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                logits, loss_c, loss_rt = self._compute_rt_loss(inputs, targets, epoch, warmup_epoch)
                loss = loss_c + loss_rt
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()
                losses_c += loss_c.item()
                losses_rt += loss_rt.item()
                for name, value in self._loss_details.items():
                    detail_sums[name] = detail_sums.get(name, 0.0) + float(value)
                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)
            scheduler.step()

            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            if self._should_eval_epoch(epoch, self.tuned_epochs, eval_interval):
                test_acc = self._compute_accuracy(self._network, test_loader)
                test_acc_msg = "{:.2f}".format(test_acc)
            else:
                test_acc_msg = "-"

            info = "Task {}, Epoch {}/{} => Loss {:.3f}, Loss_c {:.3f}, Losses_rt {:.3f}, Train_accy {:.2f}, Test_accy {}".format(
                self._cur_task,
                epoch + 1,
                self.tuned_epochs,
                losses / len(train_loader),
                losses_c/len(train_loader),
                losses_rt/len(train_loader),
                train_acc,
                test_acc_msg,
            )
            if detail_sums:
                details = ", ".join(
                    "{} {:.4f}".format(name, value / len(train_loader))
                    for name, value in sorted(detail_sums.items())
                )
                info += ", " + details
            prog_bar.set_description(info)
        logging.info(info)

    def _inc_loss(self, features, features_old):
        if self.use_research_projector:
            return self._research_inc_loss(features, features_old)

        features_old = self.old_ae(features_old)
        loss_align = nn.MSELoss()(features, features_old)
        features_old_norm = F.normalize(features_old, p=2, dim=1)
        protos = torch.from_numpy(self._class_means).float().to(self._device,non_blocking=True)
        protos = self.old_ae(protos)
        protos = F.normalize(protos, p=2, dim=1)
        similarity = torch.matmul(protos, features_old_norm.t())
        loss_orth = similarity.sum() / (similarity.shape[0]*similarity.shape[1])
        return self.args["beta"] * loss_align + self.args["gamma"] * loss_orth

    def _research_inc_loss(self, features, features_old):
        projected_old = self.old_ae(features_old)
        loss_point = F.mse_loss(projected_old, features)
        loss_mean, loss_variance = moment_alignment_loss(
            features,
            projected_old,
            labels=getattr(self, "_current_targets", None),
            classwise=self.args.get("moment_classwise", True),
        )

        old_prototypes = torch.from_numpy(
            self._class_means[:self._known_classes]
        ).float().to(self._device, non_blocking=True)
        projected_prototypes = self.old_ae(old_prototypes)
        loss_separation, separation_diagnostics = adaptive_topk_separation_loss(
            features,
            projected_prototypes.detach(),
            topk=self.args.get("separation_topk", 10),
            threshold_min=self.args.get("separation_threshold_min", 0.1),
            threshold_max=self.args.get("separation_threshold_max", 0.5),
            reference_features=features_old,
            reference_prototypes=old_prototypes,
        )

        if hasattr(self.old_ae, "regularization_loss"):
            loss_complexity = self.old_ae.regularization_loss()
        else:
            loss_complexity = features.sum() * 0.0

        loss = (
            self.args.get("point_weight", self.args.get("beta", 1.0)) * loss_point
            + self.args.get("mean_weight", 0.1) * loss_mean
            + self.args.get("variance_weight", 0.01) * loss_variance
            + self.args.get("separation_weight", self.args.get("gamma", 1.0)) * loss_separation
            + self.args.get("complexity_weight", 0.0) * loss_complexity
        )
        self._loss_details = {
            "point": loss_point.detach().item(),
            "mean": loss_mean.detach().item(),
            "variance": loss_variance.detach().item(),
            "separation": loss_separation.detach().item(),
            "complexity": loss_complexity.detach().item(),
            "topk_sim": separation_diagnostics["topk_similarity"].item(),
            "active_sep": separation_diagnostics["active_fraction"].item(),
        }
        return loss

    @torch.no_grad()
    def _transport_old_statistics(self):
        """Move old class statistics into the current feature space."""
        if self._known_classes == 0:
            return

        was_training = self.old_ae.training
        self.old_ae.eval()
        mode = self.args.get("statistics_transport", "mean_only").lower()
        means = torch.from_numpy(
            self._class_means[:self._known_classes]
        ).float().to(self._device)

        if mode == "mean_only":
            transported_means = self.old_ae(means).cpu().numpy()
            self._class_means[:self._known_classes] = transported_means
        elif mode == "diagonal_mc":
            sample_count = int(self.args.get("statistics_transport_samples", 128))
            epsilon = float(self.args.get("statistics_epsilon", 1e-4))
            transported_means = []
            transported_covariances = self._class_covs[:self._known_classes].clone()
            for class_id in range(self._known_classes):
                class_mean = means[class_id]
                class_covariance = self._class_covs[class_id]
                class_std = torch.diagonal(class_covariance).clamp_min(epsilon).sqrt().to(self._device)
                samples = class_mean.unsqueeze(0) + torch.randn(
                    sample_count,
                    self.feature_dim,
                    device=self._device,
                ) * class_std.unsqueeze(0)
                projected_samples = self.old_ae(samples)
                projected_mean = projected_samples.mean(dim=0)
                projected_variance = projected_samples.var(dim=0, unbiased=False).clamp_min(epsilon)
                transported_means.append(projected_mean.cpu())
                transported_covariances[class_id] = torch.diag(projected_variance.cpu())

            self._class_means[:self._known_classes] = torch.stack(
                transported_means
            ).numpy()
            self._class_covs[:self._known_classes] = transported_covariances
        else:
            raise ValueError("Unknown statistics_transport: {}".format(mode))

        if was_training:
            self.old_ae.train()
        
    def _compute_rt_loss(self, inputs, targets, epoch=None, warmup_epoch=10):     
        loss_cos=AngularPenaltySMLoss(loss_type='cosface', eps=1e-7, s=self.args["scale"], m=self.args["margin"])
        features = self._network_module_ptr.extract_vector(inputs)
        logits = self._network_module_ptr.fc(features)["logits"]
        loss_c=loss_cos(logits[:, self._known_classes:], targets - self._known_classes)

        if self._cur_task == 0:
            self._loss_details = {}
            lambda_rs = self.args["lambda_rs"] * min(1.0, epoch / warmup_epoch)
            loss_base = lambda_rs * self.rs_loss_func(features, targets)
            return logits, loss_c, loss_base
        
        features_old = self.old_network_module_ptr.extract_vector(inputs)
        self._current_targets = targets
        loss_inc = self._inc_loss(features, features_old)
        return logits, loss_c, loss_inc
    
class RS_Loss(nn.Module):
    def __init__(self, lamda=0.5, margin=0.5):
        super(RS_Loss, self).__init__()
        self.lamda = lamda
        self.margin = margin

    def forward(self, features, labels):
        device = features.device
        features = F.normalize(features, p=2, dim=1)
        labels = labels[:, None]
        mask = torch.eq(labels, labels.t()).float().to(device)
        eye = torch.eye(mask.size(0), device=device)
        mask_pos = mask - eye
        mask_neg = 1.0 - mask
        dot_prod = torch.matmul(features, features.t())

        pos_loss = F.relu(1.0 - dot_prod) * mask_pos
        neg_loss = F.relu(dot_prod - self.margin) * mask_neg
        loss = pos_loss.sum() / (mask_pos.sum() + 1e-6) + \
               self.lamda * neg_loss.sum() / (mask_neg.sum() + 1e-6)

        return loss
