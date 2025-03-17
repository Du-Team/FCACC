import os
import pandas as pd
import torch
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

import datautils
from tools.augmentations import DataTransform
from models import FCACCEncoder
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
from models.Metrics import nmi, acc,rand_index_score
from sklearn.metrics import adjusted_rand_score as ari
from sklearn.metrics import fowlkes_mallows_score as fmi
from torch.optim.lr_scheduler import ReduceLROnPlateau

from tools.tool import generate_pos_neg_index

class FCACCModel:
    ''' The TSCC model '''

    def __init__(
            self,
            data_loader,
            dataset_size,
            timesteps_len,
            batch_size,
            pretraining_epoch,
            n_cluster,
            dataset_name,
            input_dims,
            MaxIter=100,
            m=1.5,
            T1=2,
            output_dims=32,
            hidden_dims=64,
            depth=10,
            device='cuda',
            lr=0.001,
            max_train_length=4000,
            temporal_unit=0,
            w_c = 0.2,
            hard_w = 0.2,
            log_dir = 0,
            ):

        ''' Initialize a TS2Vec model '''

        super().__init__()
        self.device = device
        self.lr = lr
        self.num_cluster = n_cluster
        self.batch_size = batch_size
        self.T1 = T1
        self.m = m
        self.pretraining_epoch = pretraining_epoch
        self.MaxIter1 = MaxIter
        self.data_loader = data_loader
        self.dataset_size = dataset_size
        self.timesteps_len = timesteps_len
        self.input_dims = input_dims
        self.dataset_name = dataset_name
        self.latent_size = output_dims
        self.max_train_length = max_train_length
        self.temporal_unit = temporal_unit

        self.draw = 0
        self.scaling_rate = 0.8
        self.w_c = w_c
        self.alpha = 0.95
        self.gamma = 0.5
        self.hard_w  = hard_w
        self.logdir = log_dir

        self.u_mean = torch.zeros([n_cluster, output_dims], device=torch.device('cuda'))

        self.encoder = FCACCEncoder(input_dims=input_dims, output_dims=output_dims, hidden_dims=hidden_dims, depth=depth).to(self.device)
        self.net = torch.optim.swa_utils.AveragedModel(self.encoder)
        self.net.update_parameters(self.encoder)





    def Pretraining(self):
        print('Pretraining...')
        self.encoder.train()
        for param in self.encoder.parameters():
            param.requires_grad = True
        optimizer = optim.AdamW(self.encoder.parameters(), lr=self.lr)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=12, verbose=True)
        loss_log = []
        acc_log = []
        nmi_log = []
        for T in range(0, self.pretraining_epoch):
            print('Pretraining Epoch: ', T + 1)
            total_loss = 0
            num_batches = 0
            for x, target, _ in self.data_loader:
                optimizer.zero_grad()
                x = x.to(self.device)
                out1, out2, out3 = self.crops_and_extract(x, tp_unit=self.temporal_unit, model=self.encoder, scaling_rate = self.scaling_rate)#cropping输出的是经过模型训练的特征
                loss = self.contrastive_loss(out1, out2, out3, temporal_unit=self.temporal_unit)
                loss.backward()
                optimizer.step()
                self.net.update_parameters(self.encoder)
                total_loss += loss.item()
                num_batches += 1

            average_loss = total_loss / num_batches
            loss_log.append(average_loss)
            scheduler.step(average_loss)

            ACC, NMI = self.Kmeans_model_evaluation(T)
            acc_log.append(ACC)
            nmi_log.append(NMI)

            print(f"Epoch #{T + 1}: loss={average_loss}")

        file = os.getcwd() + '\\pretraining.csv'
        data = pd.DataFrame.from_dict({'pretraining': loss_log, 'ACC': acc_log, 'NMI': nmi_log}, orient='index')
        data.to_csv(file, index=False)

        if T == self.pretraining_epoch-1:
            with open(self.dataset_name + '_Pretraining_phase','wb') as f:  # wb 模式表示以二进制写入模式打开文件。如果文件已存在，此模式会覆盖文件。如果文件不存在，则会创建新文件
                torch.save(self.encoder, f)

        if self.draw == 1:
            self.plotter(name=self.dataset_name + '_Pretraining_phase', save_fig=False)
        return self.encoder

    def Finetuning(self):
        self.encoder, self.u_mean = self.initialization()
        # self.encoder.cuda()
        self.encoder.train()
        for param in self.encoder.parameters():
            param.requires_grad = True
        optimizer = optim.AdamW(self.encoder.parameters(), lr=0.0001)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=12, verbose=True)

        loss_log = []
        acc_log = []
        nmi_log = []
        pseudo_labels = -torch.ones(self.dataset_size, dtype=torch.long)

        for T in range(0, self.MaxIter1):
            print('Finetuning Epoch: ', T + 1)
            total_loss = 0
            num_batches = 0
            if T % self.T1 == 1:
                self.u_mean = self.update_cluster_centers().to(self.device)
            for x, target, index in self.data_loader:
                real_batch_size = x.size(0)
                u = torch.zeros([self.num_cluster, real_batch_size, self.latent_size]).to(self.device)

                x = x.to(self.device)

                for kk in range(0, self.num_cluster):
                    y = self.encode_with_pooling(x).to(self.device)
                    u[kk, :, :] = y
                u = u.detach()


                p = self.cmp(u, self.u_mean)

                p = p.detach()
                pseudo_labels_cur, index_cur = self.generate_pseudo_labels(p, pseudo_labels[index].to(p.device), index)

                pseudo_labels[index_cur] = pseudo_labels_cur

                p = p.T
                p = torch.pow(p, self.m)

                loss_c = 0
                for i in range(0, self.num_cluster):
                    real_batch_size = x.size(0)

                    out1, out2, out3 = self.crops_and_extract(x, tp_unit=self.temporal_unit, model=self.encoder, scaling_rate = self.scaling_rate)
                    u1 = self.encode_with_pooling(x)
                    self.u_mean = self.u_mean.float()
                    loss_c += torch.matmul(p[i, :].unsqueeze(0), torch.sum(torch.pow(u1 - self.u_mean[i, :].unsqueeze(0).repeat(real_batch_size, 1), 2), dim=1))



                loss_r = self.contrastive_loss(out1, out2, out3, mask=True, pseudo_label=pseudo_labels_cur, temporal_unit=self.temporal_unit)

                loss = loss_r + self.w_c * loss_c
                if torch.isnan(loss):
                    print(loss)

                total_loss += loss.item()  # 累加loss
                num_batches += 1

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                self.net.update_parameters(self.encoder)
            average_loss = total_loss / num_batches
            scheduler.step(average_loss)

            ACC, NMI = self.model_evaluation(T)

            acc_log.append(ACC)
            nmi_log.append(NMI)
            loss_log.append(average_loss)

            if T == self.MaxIter1-1:
                with open(self.dataset_name + '_Finetuning_phase', 'wb') as f:
                    torch.save(self.encoder, f)
                with open(self.dataset_name + '_Centers', 'wb') as f:
                    torch.save(self.u_mean, f)
            print(f"Epoch #{T + 1}: loss={average_loss}")

        file = os.getcwd() + '\\finetuning.csv'
        data = pd.DataFrame.from_dict({'finetuning': loss_log, 'ACC': acc_log, 'NMI': nmi_log}, orient='index')  # orient='columns'
        data.to_csv(file, index=False)

        if self.draw == 1:
            self.plotter(name=self.dataset_name + '_Finetuning_phase', save_fig=False)

    def initialization(self):
        print("-----initialization mode--------")
        self.encoder = torch.load(self.dataset_name + '_Pretraining_phase')
        datas = np.zeros([self.dataset_size, self.latent_size])
        ii = 0
        for x, target, _ in self.data_loader:
            x = x.to(self.device)
            u = self.encode_with_pooling(x)

            real_batch_size = u.size(0)
            datas[ii * self.batch_size:(ii * self.batch_size) + real_batch_size, :] = u.data.cpu().numpy()
            ii = ii + 1

        kmeans = KMeans(n_clusters=self.num_cluster, init='k-means++', random_state=0).fit(datas)

        self.u_mean = kmeans.cluster_centers_
        self.u_mean = torch.from_numpy(self.u_mean).to(self.device)
        return self.encoder, self.u_mean

    def Kmeans_model_evaluation(self, T):
        self.encoder.eval()
        datas = np.zeros([self.dataset_size, self.latent_size])
        label_true = np.zeros(self.dataset_size)
        ii = 0
        for x, target, _ in self.data_loader:
            x = x.to(self.device)
            u = self.encode_with_pooling(x).to(self.device)

            real_batch_size = u.size(0)

            datas[ii * self.batch_size:(ii * self.batch_size) + real_batch_size, :] = u.data.cpu().numpy()
            label_true[ii * self.batch_size:(ii * self.batch_size) + real_batch_size] = target.numpy()
            ii = ii + 1


        kmeans = KMeans(n_clusters=self.num_cluster, random_state=0).fit(datas)

        label_pred = kmeans.labels_
        ACC = acc(label_true, label_pred, self.num_cluster)
        NMI = nmi(label_true, label_pred)
        print('ACC', ACC)
        print('NMI', NMI)
        if T == 0:
            np.save('./features/Start_Pretraining_R.npy', datas)
            np.save('./features/Start_Pretraining_y_true.npy', label_true)
        if T == self.pretraining_epoch-1:
            np.save('./features/End_Pretraining_R.npy', datas)
            np.save('./features/End_Pretraining_y_true.npy', label_true)
        return ACC, NMI

    def update_cluster_centers(self):
        self.encoder.eval()
        for param in self.encoder.parameters():
            param.requires_grad = False
        den = torch.zeros([self.num_cluster]).to(self.device)
        num = torch.zeros([self.num_cluster, self.latent_size]).to(self.device)

        for x, target, _ in self.data_loader:
            x = x.to(self.device)

            u = self.encode_with_pooling(x)

            p = self.cmp(u.unsqueeze(0).repeat(self.num_cluster, 1, 1), self.u_mean)
            p = torch.pow(p, self.m)
            for kk in range(0, self.num_cluster):
                den[kk] = den[kk] + torch.sum(p[:, kk])
                num[kk, :] = num[kk, :] + torch.matmul(p[:, kk].T, u)
        for kk in range(0, self.num_cluster):
            self.u_mean[kk, :] = torch.div(num[kk, :], den[kk])

        self.encoder.train()
        #解冻参数
        for param in self.encoder.parameters():
            param.requires_grad = True
        return self.u_mean

    def cmp(self, u, u_mean):
        real_batch_size = u.size(1)
        p = torch.zeros([real_batch_size, self.num_cluster]).to(self.device)
        for j in range(0, self.num_cluster):
            p[:, j] = torch.sum(torch.pow(u[j, :, :] - u_mean[j, :].unsqueeze(0).repeat(real_batch_size, 1), 2), dim=1)
        p = torch.pow(p, -1 / (self.m - 1))
        sum1 = torch.sum(p, dim=1)
        p = torch.div(p, sum1.unsqueeze(1).repeat(1, self.num_cluster))
        return p

    def generate_pseudo_labels(self, c, pseudo_label_cur, index):
        batch_size = c.shape[0]
        device = c.device
        alpha = self.alpha
        gamma = self.gamma
        cluster_num = self.num_cluster

        pseudo_label_nxt = -torch.ones(batch_size, dtype=torch.long).to(device)
        tmp = torch.arange(0, batch_size).to(device)

        prediction = c.argmax(dim=1)
        confidence = c.max(dim=1).values
        unconfident_pred_index = confidence < alpha
        pseudo_per_class = np.ceil(batch_size / cluster_num * gamma).astype(int)
        for i in range(cluster_num):
            class_idx = prediction == i
            if class_idx.sum() == 0:
                continue
            confidence_class = confidence[class_idx]
            num = min(confidence_class.shape[0], pseudo_per_class)
            confident_idx = torch.argsort(-confidence_class)
            for j in range(num):
                idx = tmp[class_idx][confident_idx[j]]
                pseudo_label_nxt[idx] = i

        todo_index = pseudo_label_cur == -1
        pseudo_label_cur[todo_index] = pseudo_label_nxt[todo_index]
        pseudo_label_nxt = pseudo_label_cur
        pseudo_label_nxt[unconfident_pred_index] = -1
        return pseudo_label_nxt.cpu(), index


    def model_evaluation(self, T):
        datas = np.zeros([self.dataset_size, self.latent_size])
        pred_labels = np.zeros(self.dataset_size)
        true_labels = np.zeros(self.dataset_size)
        ii = 0
        for x, target, _ in self.data_loader:
            x = x.to(self.device)
            u = self.encode_with_pooling(x)
            real_batch_size = u.size(0)
            datas[ii * self.batch_size:(ii * self.batch_size) + real_batch_size, :] = u.data.cpu().numpy()

            u = u.unsqueeze(0).repeat(self.num_cluster, 1, 1)
            p = self.cmp(u, self.u_mean)
            y = torch.argmax(p, dim=1)
            y = y.cpu()
            y = y.numpy()
            pred_labels[(ii) * self.batch_size:(ii * self.batch_size) + real_batch_size] = y
            true_labels[(ii) * self.batch_size:(ii * self.batch_size) + real_batch_size] = target.numpy()
            ii = ii + 1

        ACC = acc(true_labels, pred_labels, self.num_cluster)
        NMI = nmi(true_labels, pred_labels)
        print('ACC', ACC)
        print('NMI', NMI)
        if T == 0:
            np.save('./features/Start_Finetuning_R.npy', datas)
            np.save(f'./features/Start_Finetuning_y_pred.npy', pred_labels)
            np.save(f'./features/Start_Finetuning_y_true.npy', true_labels)
        if T == self.MaxIter1-1:
            np.save(f'./features/End_Finetuning_End_Finetuning_R.npy', datas)
            np.save(f'./features/End_Finetuning_y_pred.npy', pred_labels)
            np.save(f'./features/End_Finetuning_y_true.npy', true_labels)

        self.encoder.train()
        for param in self.encoder.parameters():
            param.requires_grad = True

        return ACC, NMI

    def encode_with_pooling(self, data):
        assert data.ndim == 3
        n_samples, ts_l, _ = data.shape

        org_training = self.net.training
        self.net.eval()

        if torch.is_tensor(data):
            dataset = TensorDataset(data)
        else:
            dataset = TensorDataset(torch.from_numpy(data).to(torch.float))
        loader = DataLoader(dataset, batch_size=self.batch_size)

        with torch.no_grad():
            output = []
            for batch in loader:
                x = batch[0]
                out = self.net(x.to(self.device, non_blocking=True))
                out = F.max_pool1d(out.transpose(1, 2), kernel_size=out.size(1), ).transpose(1, 2).cpu()

                out = out.squeeze(1)
                output.append(out)

            output = torch.cat(output, dim=0)

        self.net.train(org_training)

        if torch.is_tensor(data):
            return output.to(self.device)
        else:
            return output.numpy()


    def mask_instance_loss_with_mixup(self, z1, z2, pseudo_label=None):
        B, T = z1.size(0), z1.size(1)
        temp = 1.0
        if pseudo_label == None:
            pseudo_label = torch.full((B,), -1, dtype=torch.int64).to('cuda')

        if B == 1:
            return z1.new_tensor(0.)

        pseudo_label = pseudo_label.to(z1.device)

        hard_w = self.hard_w

        pos_indices, neg_indices = generate_pos_neg_index(pseudo_label)
        uni_z1 = hard_w * z1[pos_indices, :, :] + (1 - hard_w) * z1[neg_indices, :, :].view(z1.size())

        pos_indices, neg_indices = generate_pos_neg_index(pseudo_label)
        uni_z2 = hard_w * z2[pos_indices, :, :] + (1 - hard_w) * z2[neg_indices, :, :].view(z2.size())

        z = torch.cat([z1, z2, uni_z1, uni_z2], dim=0)
        z = z.transpose(0, 1)
        sim = torch.matmul(z[:, : 2 * B, :], z.transpose(1, 2))

        invalid_index = pseudo_label == -1

        mask = torch.eq(pseudo_label.view(-1, 1), pseudo_label.view(1, -1)).to(z1.device)
        mask[invalid_index, :] = False
        mask[:, invalid_index] = False

        # 屏蔽掉自身
        mask_eye = torch.eye(B).float().to(z1.device)
        mask &= ~(mask_eye.bool())

        mask = mask.float()

        mask = mask.repeat(2, 4)
        mask_eye = mask_eye.repeat(2, 4)

        logits_mask = torch.ones(2 * B, 4 * B).to(z1.device)

        rows = torch.arange(2 * B).view(-1, 1).to(z1.device)

        logits_mask = logits_mask.scatter(1, rows, 0)

        logits_mask *= 1 - mask
        mask_eye = mask_eye * logits_mask

        logits = sim
        logits_max = torch.max(logits, dim=-1, keepdim=True)[0]
        logits = logits - logits_max
        neg_exp_logits = torch.exp(logits) * logits_mask

        pos_exp_log = torch.exp(logits)

        neg_exp_log_sum = neg_exp_logits.sum(-1, keepdim=True)

        prob = pos_exp_log / (neg_exp_log_sum + 1e-10)

        prob = prob[:, 0:B, B:2 * B]

        mask = mask[:B, : B]
        self_mask = mask_eye[:B, B:2 * B]
        diffaug_cluster_mask = mask

        pos_mask = (self_mask + diffaug_cluster_mask)

        pos_prob_sum = (prob * pos_mask.unsqueeze(0)).sum(-1)

        log_prob = torch.log(pos_prob_sum + 1e-10)

        log_prob = log_prob.sum(dim=0) / T

        # loss
        instance_loss = -log_prob
        instance_loss = instance_loss.mean()

        return instance_loss

    def temporal_contrastive_loss_mixup(self,z1, z2, temp=1.0):
        B, T = z1.size(0), z1.size(1)
        alpha = 0.2
        beta = 0.2

        if T == 1:
            return z1.new_tensor(0.)

        uni_z1 = alpha * z1 + (1 - alpha) * z1[:, torch.randperm(z1.shape[1]), :].view(z1.size())
        uni_z2 = beta * z2 + (1 - beta) * z2[:, torch.randperm(z1.shape[1]), :].view(z2.size())

        z = torch.cat([z1, z2, uni_z1, uni_z2], dim=1)

        sim = torch.matmul(z[:, : 2 * T, :], z.transpose(1, 2)) / temp  # B x 2T x 2T
        logits = torch.tril(sim, diagonal=-1)[:, :, :-1]
        logits += torch.triu(sim, diagonal=1)[:, :, 1:]

        if T > 1500:
            z, sim = z.cpu(), sim.cpu()
            torch.cuda.empty_cache()

        logits = -F.log_softmax(logits, dim=-1)

        logits = logits[:, :2 * T, :(2 * T - 1)]

        t = torch.arange(T, device=z1.device)
        loss = (logits[:, t, T + t - 1].mean() + logits[:, T + t, t].mean()) / 2
        return loss


    def contrastive_loss(self, z1, z2, z3, mask = False, pseudo_label = None, alpha=0.5, temporal_unit=0):
        loss = torch.tensor(0., device=z1.device)
        d = 0

        while z1.size(1) > 1:
            if alpha != 0:
                if mask == False:
                    loss += alpha * self.mask_instance_loss_with_mixup(z1, z2)
                    loss += alpha * self.mask_instance_loss_with_mixup(z1, z3)
                if mask == True:
                    loss += alpha * self.mask_instance_loss_with_mixup(z1, z2, pseudo_label)
                    loss += alpha * self.mask_instance_loss_with_mixup(z1, z3, pseudo_label)

            if d >= temporal_unit:
                if 1 - alpha != 0:
                    loss += (1 - alpha) * self.temporal_contrastive_loss_mixup(z1, z2)
                    loss += (1 - alpha) * self.temporal_contrastive_loss_mixup(z1, z3)

            d += 1

            z1 = F.max_pool1d(z1.transpose(1, 2), kernel_size=2).transpose(1, 2)
            z2 = F.max_pool1d(z2.transpose(1, 2), kernel_size=2).transpose(1, 2)
            z3 = F.max_pool1d(z3.transpose(1, 2), kernel_size=2).transpose(1, 2)
        if z1.size(1) == 1:
            if alpha != 0:
                loss += alpha * self.mask_instance_loss_with_mixup(z1, z2, pseudo_label)
                loss += alpha * self.mask_instance_loss_with_mixup(z1, z3, pseudo_label)
            d += 1
        return loss / d


    def crops_and_extract(self, x, tp_unit,  model, scaling_rate):
        ts_l = x.size(1)
        crop_l = np.random.randint(low=2 ** (tp_unit + 1),
                                   high=ts_l + 1)

        crop_left = np.random.randint(ts_l - crop_l + 1)  # [0,s_l - crop_l + 1)
        crop_right = crop_left + crop_l

        crop_eleft = np.random.randint(crop_left + 1)
        crop_eright = np.random.randint(low=crop_right, high=ts_l + 1)

        crop_offset = np.random.randint(low=-crop_eleft, high=ts_l - crop_eright + 1,
                                        size=x.size(0))  # 返回batch_size个随机数

        indx1 = crop_offset + crop_eleft
        num_elem1 = crop_right - crop_eleft
        all_indx1 = indx1[:, None] + np.arange(num_elem1)


        indx2 = crop_offset + crop_left
        num_elem2 = crop_eright - crop_left
        all_indx2 = indx2[:, None] + np.arange(num_elem2)


        #out3选用重叠部分
        indx3 = crop_offset + crop_left
        num_elem3 = crop_l
        all_indx3 = indx3[:, None] + np.arange(num_elem3)

        out1 = model(x[torch.arange(all_indx1.shape[0])[:, None], all_indx1])
        out1 = out1[:, -crop_l:]


        out2 = x[torch.arange(all_indx2.shape[0])[:, None], all_indx2]
        out2 = model(out2)
        out2 = out2[:, :crop_l]


        out3 = x[torch.arange(all_indx3.shape[0])[:, None], all_indx3]
        # 应用 DataTransform 增强到 out3
        out3 = DataTransform(out3, scaling_rate)
        out3 = out3.to(self.device)

        out3 = model(out3.float())


        return out1, out2, out3


    # 定义评估真实数据的函数
    def eval_with_test_data(self, dataset_name, log_dir, data_loader, model, save=False):
        model.encoder.eval()
        data = np.zeros([self.dataset_size, self.timesteps_len, self.input_dims])
        reps = np.zeros([self.dataset_size, self.timesteps_len, self.latent_size])
        label_true = np.zeros(self.dataset_size)
        label_pred = np.zeros(self.dataset_size)

        ii = 0
        for x, target, _ in data_loader:
            x = x.to(self.device)
            u = model.encode_with_pooling(x)

            real_batch_size = u.size(0)

            features_without_pooling = model.encoder(x)
            reps[ii * self.batch_size: ii * self.batch_size + real_batch_size, :, :] = features_without_pooling.data.cpu().numpy()
            data[ii * self.batch_size: ii * self.batch_size + real_batch_size, :, :] = x.cpu().numpy()

            # Get predicted labels
            u = u.unsqueeze(0).repeat(self.num_cluster, 1, 1)
            p = model.cmp(u, model.u_mean)
            y = torch.argmax(p, dim=1)
            y = y.cpu().numpy()

            label_true[ii * self.batch_size: ii * self.batch_size + real_batch_size] = target.numpy()
            label_pred[ii * self.batch_size: ii * self.batch_size + real_batch_size] = y

            ii = ii + 1

        # Evaluate performance
        print("-------testdata_Evaluate---------")


        save_dir = f"./{self.logdir}/label/"
        os.makedirs(save_dir, exist_ok=True)


        df_true = pd.DataFrame(label_true, columns=['label_true'])
        df_true.to_csv(f'./{log_dir}/label/{dataset_name}_label_true.csv', index=False)


        df_pred = pd.DataFrame(label_pred, columns=['label_pred'])
        df_pred.to_csv(f'./{log_dir}/label/{dataset_name}_label_pred.csv', index=False)
        accuracy = acc(label_true, label_pred, self.num_cluster)
        nmi_score = nmi(label_true, label_pred)
        ari_score = ari(label_true, label_pred)
        fmi_score = fmi(label_true, label_pred)
        test_ri = rand_index_score(label_pred, label_true)

        model.encoder.train()
        return accuracy, nmi_score, ari_score, test_ri, fmi_score
