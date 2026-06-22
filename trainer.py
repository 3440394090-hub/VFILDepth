import numpy as np
import time
import logging
import cv2
import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from linear_warmup_cosine_annealing_warm_restarts_weight_decay import ChainedScheduler
import torch.distributed as dist
import os
import json
from utils import *
from kitti_utils import *
from layers import *
import datasets
from networks import *
import copy
import shutil
import warnings
import random
import networks
from get_monovit import get_monovit_pretrained
import torch.optim as optim

warnings.filterwarnings("ignore")


class Trainer:
    def __init__(self, options):

        self.opt = options
        self.log_path = os.path.join(self.opt.log_dir, self.opt.model_name)
        #
        # self.save_opts()

        # checking height and width are multiples of 32
        assert self.opt.height % 32 == 0, "'height' must be a multiple of 32"
        assert self.opt.width % 32 == 0, "'width' must be a multiple of 32"

        self.models = {}
        self.parameters_to_train = []
        self.parameters_to_train_pose = []

        self.device = self.opt.device

        # self.num_scales = len(self.opt.scales)
        self.num_input_frames = len(self.opt.frame_ids)
        self.num_pose_frames = 2 if self.opt.pose_model_input == "pairs" else self.num_input_frames

        assert self.opt.frame_ids[0] == 0, "frame_ids must start with 0"

        self.use_pose_net = not (self.opt.use_stereo and self.opt.frame_ids == [0])

        if self.opt.use_stereo:
            self.opt.frame_ids.append("s")
        # os.makedirs("debug_masks", exist_ok=True)

        # load pre-trained MonoViT (teacher network),the load_type present resolution
        self.monovit = get_monovit_pretrained(640)
        self.monovit.cuda(0)
        self.monovit.eval()

        # load VFI
        self.model_vfi_train = networks.IFRNet(scale="large")
        # self.model_vfi_test = IFRNet(scale="small")

        self.model_vfi_train.load_state_dict(
            torch.load("D:\Root\zjd_project_2\pretrain_models\VFI\IFRNet_L_KITTI.pth")["VFI"])
        # self.model_vfi_test.load_state_dict(torch.load("D:\Root\zjd_project_2\pretrain_models\VFI\IFRNet_S_KITTI.pth")["VFI"])

        self.model_vfi_train.to(self.device).eval()
        # self.model_vfi_test.to(self.device).eval()

        # load student model
        # encoder
        # self.models["encoder"] = networks.TinyViM_S(self.opt.pretrained_weights)
        self.models["encoder"] = networks.LiteMono(model=self.opt.model,
                                                   drop_path_rate=self.opt.drop_path,
                                                   width=self.opt.width, height=self.opt.height)

        self.models["encoder"].to(self.device)
        self.parameters_to_train += list(self.models["encoder"].parameters())

        # student decoder
        # self.models["depth"] = networks.DepthDecoder(
        #     self.models["encoder"].num_ch_enc, self.opt.scales, decoder_channel_scale=self.opt.decoder_channel_scale)
        # self.models["depth"] = networks.DepthDecoder(self.models["encoder"].num_ch_enc,
        #                                              self.opt.scales)
        self.models["depth"] = networks.FusionDecoder(self.models["encoder"].num_ch_enc,
                                                     self.opt.scales)

        self.models["depth"].to(self.device)
        self.parameters_to_train += list(self.models["depth"].parameters())

        """
        load pose network (resnet18 as encoder)
        """
        if self.use_pose_net:
            if self.opt.pose_model_type == "separate_resnet":
                self.models["pose_encoder"] = networks.ResnetEncoder(
                    self.opt.num_layers,
                    self.opt.weights_init == "pretrained",
                    num_input_images=self.num_pose_frames)

                self.models["pose_encoder"].to(self.device)
                self.parameters_to_train += list(self.models["pose_encoder"].parameters())

                self.models["pose"] = networks.PoseDecoder(
                    self.models["pose_encoder"].num_ch_enc,
                    num_input_features=1,
                    num_frames_to_predict_for=2)

            elif self.opt.pose_model_type == "shared":
                self.models["pose"] = networks.PoseDecoder(
                    self.models["encoder"].num_ch_enc, self.num_pose_frames)

            elif self.opt.pose_model_type == "posecnn":
                self.models["pose"] = networks.PoseCNN(
                    self.num_input_frames if self.opt.pose_model_input == "all" else 2)

            self.models["pose"].to(self.device)
            self.parameters_to_train += list(self.models["pose"].parameters())

        self.model_optimizer = optim.Adam(self.parameters_to_train, self.opt.learning_rate)
        self.model_lr_scheduler = optim.lr_scheduler.StepLR(
            self.model_optimizer, self.opt.scheduler_step_size, 0.1)


        if self.opt.mypretrain is not None:
            self.load_pretrain()

        self.ep_start = 0
        self.batch_start = 0
        self.step = 0
        #
        # assert self.opt.frame_ids[0] == 0, "frame_ids must start with 0"
        #
        # self.use_pose_net = not (self.opt.use_stereo and self.opt.frame_ids == [0])

        if self.opt.use_stereo:
            self.opt.frame_ids.append("s")

        # data
        datasets_dict = {"kitti": datasets.KITTIRAWDataset,
                         "kitti_odom": datasets.KITTIOdomDataset}
        self.dataset = datasets_dict[self.opt.dataset]

        fpath = os.path.join(os.path.dirname(__file__), "splits", self.opt.split, "{}_files.txt")

        train_filenames = readlines(fpath.format("train"))
        val_filenames = readlines(fpath.format("val"))

        img_ext = '.png' if self.opt.png else '.jpg'

        num_train_samples = len(train_filenames)
        self.num_steps_per_epoch = num_train_samples // self.opt.batch_size
        self.num_total_steps = self.num_steps_per_epoch * self.opt.num_epochs

        train_dataset = self.dataset(
            self.opt.data_path, train_filenames, self.opt.height, self.opt.width,
            self.opt.frame_ids, 4, is_train=True, img_ext=img_ext)
        self.train_loader = DataLoader(
            train_dataset, self.opt.batch_size, True,
            num_workers=self.opt.num_workers, pin_memory=True, drop_last=True, worker_init_fn=worker_seed_fn
        )
        val_dataset = self.dataset(
            self.opt.data_path, val_filenames, self.opt.height, self.opt.width,
            self.opt.frame_ids, 4, is_train=False, img_ext=img_ext)
        self.val_loader = DataLoader(
            val_dataset, self.opt.batch_size, True,
            num_workers=self.opt.num_workers, pin_memory=True, drop_last=True)
        self.val_iter = iter(self.val_loader)


        for k in self.models.keys():
            self.models[k].to(self.device)
            self.parameters_to_train += list(self.models[k].parameters())

        if self.opt.resume:
            checkpoint = self.load_ckpt()

        if not self.opt.no_ssim:
            self.ssim = SSIM()
            self.ssim.to(self.device)

        self.backproject_depth = BackprojectDepth(self.opt.batch_size, self.opt.height, self.opt.width)
        self.backproject_depth.to(self.device)

        self.project_3d = Project3D(self.opt.batch_size, self.opt.height, self.opt.width)
        self.project_3d.to(self.device)

        print("Using split:\n  ", self.opt.split)
        print("There are {:d} training items and {:d} validation items\n".format(
            len(train_dataset), len(val_dataset)))

        self.save_opts()

    def set_train(self):
        """Convert all models to training mode
        """
        for m in self.models.values():
            m.train()

    def set_eval(self):
        """Convert all models to testing/evaluation mode
        """
        for m in self.models.values():
            m.eval()

    def train(self):
        """Run the entire training pipeline
        """
        self.epoch = 0
        self.step = 0
        self.start_time = time.time()
        for self.epoch in range(self.opt.num_epochs):
            self.run_epoch()
            if (self.epoch + 1) % self.opt.save_frequency == 0:
                self.save_model()

    def load_pretrain(self):
        self.opt.mypretrain = os.path.expanduser(self.opt.mypretrain)
        path = self.opt.mypretrain
        model_dict = self.models["encoder"].state_dict()
        pretrained_dict = torch.load(path)['model']
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if (k in model_dict and not k.startswith('norm'))}
        model_dict.update(pretrained_dict)
        self.models["encoder"].load_state_dict(model_dict)
        print('mypretrain loaded.')

    def run_epoch(self):
        """Run a single epoch of training and validation
        """

        self.model_lr_scheduler.step()

        print("Training")
        self.set_train()
        start_data_time = time.time()
        for batch_idx, inputs in enumerate(self.train_loader):
            before_op_time = time.time()
            start_fp_time = time.time()
            outputs, losses = self.process_batch(inputs)

            start_bp_time = time.time()
            self.model_optimizer.zero_grad()
            losses["loss"].backward()

            if self.opt.clip_grad != -1:
                for params in self.model_optimizer.param_groups:
                    params = params['params']
                    nn.utils.clip_grad_norm_(params, max_norm=self.opt.clip_grad)

            self.model_optimizer.step()


            duration = time.time() - before_op_time

            # log less frequently after the first 2000 steps to save time & disk space
            early_phase = batch_idx % self.opt.log_frequency == 0 and self.step < 2000
            late_phase = self.step % 2000 == 0

            if early_phase or late_phase:
                self.log_time(batch_idx, duration, losses["loss"].cpu().data)

                self.val()

            self.step += 1

        self.batch_start = 0

    def log(self, mode, inputs, outputs, losses):
        """Write an event to the tensorboard events file
        """
        writer = self.writers[mode]
        for l, v in losses.items():
            writer.add_scalar("{}".format(l), v, self.step)

        for j in range(min(4, self.opt.batch_size)):  # write a maxmimum of four images
            for s in self.opt.scales:
                for frame_id in self.opt.frame_ids:
                    writer.add_image(
                        "color_{}_{}/{}".format(frame_id, s, j),
                        inputs[("color", frame_id, s)][j].data, self.step)
                    if s == 0 and frame_id != 0:
                        writer.add_image(
                            "color_pred_{}_{}/{}".format(frame_id, s, j),
                            outputs[("color", frame_id, s)][j].data, self.step)

                writer.add_image(
                    "disp_{}/{}".format(s, j),
                    normalize_image(outputs[("disp", s)][j]), self.step)

    def val(self):
        """Validate the model on a single minibatch
        """
        self.set_eval()
        try:
            # inputs = self.val_iter.next()
            inputs = next(self.val_iter)
        except StopIteration:
            self.val_iter = iter(self.val_loader)
            # inputs = self.val_iter.next()
            inputs = next(self.val_iter)

        with torch.no_grad():
            outputs, losses = self.process_batch(inputs)

            del inputs, outputs, losses

        self.set_train()

    def compute_depth_losses(self, inputs, outputs, losses):
        """Compute depth metrics, to allow monitoring during training

        This isn't particularly accurate as it averages over the entire batch,
        so is only used to give an indication of validation performance
        """
        depth_pred = outputs[("depth", 0, 0)]
        depth_pred = torch.clamp(F.interpolate(
            depth_pred, [375, 1242], mode="bilinear", align_corners=False), 1e-3, 80)
        depth_pred = depth_pred.detach()

        depth_gt = inputs["depth_gt"]
        mask = depth_gt > 0

        # garg/eigen crop
        crop_mask = torch.zeros_like(mask)
        crop_mask[:, :, 153:371, 44:1197] = 1
        mask = mask * crop_mask

        depth_gt = depth_gt[mask]
        depth_pred = depth_pred[mask]
        depth_pred *= torch.median(depth_gt) / torch.median(depth_pred)

        depth_pred = torch.clamp(depth_pred, min=1e-3, max=80)

        depth_errors = compute_depth_errors(depth_gt, depth_pred)

        for i, metric in enumerate(self.depth_metric_names):
            losses[metric] = np.array(depth_errors[i].cpu())

    def process_batch(self, inputs):
        """Pass a minibatch through the network and generate images and losses
        """
        for key, ipt in inputs.items():
            try:
                inputs[key] = ipt.to(self.device)
            except:
                pass

        embt = torch.tensor(0.5).view(1, 1, 1, 1).float().to(self.device)
        embt = embt.repeat(self.opt.batch_size, 1, 1, 1)
        img_n1 = inputs[("color", -1, 0)]
        img_p1 = inputs[("color", 1, 0)]
        img_0 = inputs[("color", 0, 0)]


        with torch.no_grad():
            img_nt, flow_nt_n1, flow_nt_0, merge_mask_nt = self.model_vfi_train(img_n1, img_0, embt)#-0.5
            img_pt, flow_pt_0, flow_pt_p1, merge_mask_pt = self.model_vfi_train(img_0, img_p1, embt)#0.5
            # img_new_0, _, _, _ = self.model_vfi_train(img_nt, img_pt, embt)

        K = inputs[("K", 0)]
        inv_K = inputs[("inv_K", 0)]

        losses = {}

        losses["loss_base"] = torch.tensor(0.0).to(self.device)
        losses["loss_dc"] = torch.tensor(0.0).to(self.device)

        pose_n1_0, pose_0_n1 = self.predict_poses(inputs[("color_aug", -1, 0)], inputs[("color_aug", 0, 0)])
        pose_0_p1, pose_p1_0 = self.predict_poses(inputs[("color_aug", 0, 0)], inputs[("color_aug", 1, 0)])
        pose_n1_nt, pose_nt_n1 = self.predict_poses(img_n1, img_nt)#-1 -0.5
        pose_nt_p1, pose_p1_nt = self.predict_poses(img_nt, img_p1)#-0.5 1
        pose_n1_pt, pose_pt_n1 = self.predict_poses(img_n1, img_pt)#-1 0.5
        pose_pt_p1, pose_p1_pt = self.predict_poses(img_pt, img_p1)#0.5 1

        ## predict single-frame depths

        # 教师网络权重不发生变化
        with torch.no_grad():
            teacher_outputs = self.monovit(inputs["color_aug", 0, 0])
            teacher_outputs_nt = self.monovit(img_nt)#-0.5
            teacher_outputs_pt = self.monovit(img_pt)#0.5
            teacher_outputs_f1 = self.monovit(img_p1)#1
            teacher_outputs_fn1 = self.monovit(img_n1)#-1
            # teacher_outputs_new = self.monovit(img_new_0)

        feats_0 = self.models["encoder"](inputs[("color_aug", 0, 0)])
        feats_nt = self.models["encoder"](img_nt)
        feats_pt = self.models["encoder"](img_pt)
        # feats_new_0 = self.models["encoder"](img_new_0)
        disp_0 = self.models["depth"](feats_0)
        disp_pt = self.models["depth"](feats_pt)#0.5
        disp_nt = self.models["depth"](feats_nt)#-0.5
        # disp_new_0 = self.models["depth"](feats_new_0)
        # _, depth_0 = disp_to_depth(disp_0[("disp", 0)], self.opt.min_depth, self.opt.max_depth)
        # _, depth_pt = disp_to_depth(disp_pt[("disp", 0)], self.opt.min_depth, self.opt.max_depth)#0.5
        # _, depth_nt = disp_to_depth(disp_nt[("disp", 0)], self.opt.min_depth, self.opt.max_depth)#-0.5

        ## calculate the self-supervised losses on single-frame depths
        img_n1_00 = self.generate_images_pred(disp_0, pose_0_n1, img_n1, K, inv_K)
        img_p1_00 = self.generate_images_pred(disp_0, pose_0_p1, img_p1, K, inv_K)
        loss_base, _ = self.compute_losses_base(disp_0, img_0, [img_n1_00, img_p1_00], [img_n1, img_p1], teacher_outputs = [teacher_outputs, teacher_outputs_f1], pose = [K, pose_0_p1])
        losses["loss_base"] += loss_base

        img_n1_pt = self.generate_images_pred(disp_pt, pose_pt_n1, img_n1, K, inv_K)
        img_p1_pt = self.generate_images_pred(disp_pt, pose_pt_p1, img_p1, K, inv_K)
        loss_base, _ = self.compute_losses_base(disp_pt, img_pt, [img_n1_pt, img_p1_pt], [img_n1, img_p1], teacher_outputs = [teacher_outputs_pt, teacher_outputs_fn1], pose = [K, pose_pt_n1])
        losses["loss_base"] += loss_base

        img_n1_nt = self.generate_images_pred(disp_nt, pose_nt_n1, img_n1, K, inv_K)
        img_p1_nt = self.generate_images_pred(disp_nt, pose_nt_p1, img_p1, K, inv_K)
        loss_base, _ = self.compute_losses_base(disp_nt, img_nt, [img_n1_nt, img_p1_nt], [img_n1, img_p1], teacher_outputs = [teacher_outputs_nt, teacher_outputs_f1], pose = [K, pose_nt_p1])
        losses["loss_base"] += loss_base

        # losses["loss_tc"] = self.compute_temporal_consistency_loss(teacher_outputs_new[("disp", 0)], disp_new_0[("disp", 0)])

        # losses["loss"] = losses["loss_base"] + losses["loss_tc"]
        losses["loss"] = losses["loss_base"]
        return None, losses


    def predict_poses(self, img_0, img1):
        """Predict poses between input frames for monocular sequences.
        """

        pose_inputs = [img_0, img1]
        pose_inputs = [self.models["pose_encoder"](torch.cat(pose_inputs, 1))]

        axisangle, translation = self.models["pose"](pose_inputs)

        pose = transformation_from_parameters(axisangle[:, 0], translation[:, 0], invert=False)
        pose_inv = transformation_from_parameters(axisangle[:, 0], translation[:, 0], invert=True)
        return pose, pose_inv

    def generate_images_pred(self, disp_tgt, pose_tgt_src, img_src, K, inv_K):
        """Generate the warped (reprojected) color images for a minibatch.
        Generated images are saved into the `outputs` dictionary.
        """
        disp = disp_tgt[("disp", 0)]
        _, depth = disp_to_depth(disp, self.opt.min_depth, self.opt.max_depth)

        cam_points = self.backproject_depth(depth, inv_K)
        pix_coords = self.project_3d(cam_points, K, pose_tgt_src)

        img_src_tgt = F.grid_sample(
            img_src,
            pix_coords,
            padding_mode="border", align_corners=True)

        return img_src_tgt

    def compute_temporal_consistency_loss(self, teacher, student):
        return torch.abs(teacher - student).mean()

    def compute_reprojection_loss(self, pred, target):
        """Computes reprojection loss between a batch of predicted and target images
        """
        abs_diff = torch.abs(target - pred)
        l1_loss = abs_diff.mean(1, True)

        if self.opt.no_ssim:
            reprojection_loss = l1_loss
        else:
            ssim_loss = self.ssim(pred, target).mean(1, True)
            reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss

        return reprojection_loss

    def compute_losses_base(self, disp_tgt, img_tgt, imgs_src_tgt, imgs_src, mask_rec=None, teacher_outputs=None, pose=None):
        """Compute the reprojection and smoothness losses for a minibatch
        """
        loss = 0
        reprojection_losses = []

        disp = disp_tgt[("disp", 0)]
        # teacher_disp = teacher_outputs[("disp", 0)]
        teacher = teacher_outputs[0]
        teacher_disp = teacher[("disp", 0)]

        if teacher_outputs:
            # depth = teacher_outputs[2]


            disp_before = F.interpolate(
                teacher_disp, [self.opt.height, self.opt.width], mode="bilinear", align_corners=False)
            _, depth_before = disp_to_depth(disp_before, self.opt.min_depth, self.opt.max_depth)
            # cross check view
            with torch.no_grad():
                B, C, H, W = depth_before.shape
                final_cross_mask = np.zeros([B, C, H, W])
                # features_src = self.models_before["encoder"](inputs["color_aug", 1, 0])  # frame = 1
                # outputs_src = self.models_before["depth"](features_src)
                outputs_src = teacher_outputs[1]
                disp_src = outputs_src[("disp", 0)]
                disp_src = F.interpolate(
                    disp_src, [self.opt.height, self.opt.width], mode="bilinear", align_corners=False)
                _, depth_src_batch = disp_to_depth(disp_src, self.opt.min_depth, self.opt.max_depth)

                for i in range(B):
                    depth_ref = depth_before[i, 0, :, :].detach().cpu().numpy()
                    depth_src = depth_src_batch[i, 0, :, :].detach().cpu().numpy()  # frame = 1
                    K = pose[0]
                    intrinsics_ref = K[i, :, :].detach().cpu().numpy()[:3, :3]
                    intrinsics_src = K[i, :, :].detach().cpu().numpy()[:3, :3]
                    T = pose[1]
                    T = T[i, :, :].detach().cpu().numpy()

                    cross_mask, _, _, _ = check_geometric_consistency(depth_ref, intrinsics_ref,
                                                                      depth_src, intrinsics_src,
                                                                      T)
                    # cross_mask = cross_mask >= 1    # 'threshold of num view'
                    final_cross_mask[i, 0, :, :] = cross_mask

                # outputs[("cross_mask", scale)] = final_cross_mask  # B,C,H,W
                final_cross_mask = torch.tensor(final_cross_mask).to(depth_before.device)
                print_cross_mask = final_cross_mask

        for i in range(len(imgs_src_tgt)):
            pred = imgs_src_tgt[i]
            reprojection_losses.append(self.compute_reprojection_loss(pred, img_tgt))

        reprojection_losses = torch.cat(reprojection_losses, 1)

        if not self.opt.disable_automasking:
            identity_reprojection_losses = []
            for i in range(len(imgs_src)):
                pred = imgs_src[i]
                identity_reprojection_losses.append(
                    self.compute_reprojection_loss(pred, img_tgt))

            identity_reprojection_losses = torch.cat(identity_reprojection_losses, 1)

            if self.opt.avg_reprojection:
                identity_reprojection_loss = identity_reprojection_losses.mean(1, keepdim=True)
            else:
                # save both images, and do min all at once below
                identity_reprojection_loss = identity_reprojection_losses

        if self.opt.avg_reprojection:
            reprojection_loss = reprojection_losses.mean(1, keepdim=True)
        else:
            reprojection_loss = reprojection_losses

        if not self.opt.disable_automasking:
            # add random numbers to break ties
            identity_reprojection_loss += torch.randn(
                identity_reprojection_loss.shape, device=self.device) * 0.00001

            combined = torch.cat((identity_reprojection_loss, reprojection_loss), dim=1)
        else:
            combined = reprojection_loss

        if combined.shape[1] == 1:
            to_optimise = combined
        else:
            to_optimise, idxs = torch.min(combined, dim=1)

        if mask_rec is not None:
            to_optimise *= mask_rec[:, 0]

        if not self.opt.disable_automasking:
            auto_mask = (idxs > identity_reprojection_loss.shape[1] - 1).float().unsqueeze(1)
        else:
            auto_mask = None

        loss += to_optimise.mean()
        mean_disp = disp.mean(2, True).mean(3, True)
        norm_disp = disp / (mean_disp + 1e-7)

        smooth_loss = get_smooth_loss(norm_disp, img_tgt)

        loss += self.opt.disparity_smoothness * smooth_loss
        # distillation_loss = self.compute_distillation_loss(teacher_disp, disp)
        diff = torch.log(torch.abs(disp - teacher_disp) + 1)
        distillation_loss = (diff * final_cross_mask).mean()
        # distillation_loss = torch.abs(depth - depth_before.detach()) * final_cross_mask
        loss += distillation_loss
        # if self.epoch >= 15 and self.step % 100 == 0:
        #     mask_np = print_cross_mask[0, 0].detach().cpu().numpy()
        #     mask_img = (mask_np * 255).astype(np.uint8)
        #     img_name1 = f"debug_masks/epoch{self.epoch}_step{self.step}_mask.png"
        #     img_tgt_tensor = img_tgt[0].detach().cpu()
        #     img_tgt_np = (img_tgt_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        #
        #     # PyTorch 默认是 RGB，OpenCV 需要 BGR，做一次颜色通道转换
        #     img_tgt_bgr = cv2.cvtColor(img_tgt_np, cv2.COLOR_RGB2BGR)
        #     img_name = f"debug_masks/epoch{self.epoch}_step{self.step}_img.png"
        #     cv2.imwrite(img_name, img_tgt_bgr)
        #
        #     # 5. 写入磁盘
        #     cv2.imwrite(img_name1, mask_img)
        #     print(f"[INFO] 成功保存跨视图几何一致性掩码至: {img_name}")

        return loss, auto_mask

    @staticmethod
    def compute_distillation_loss(good_disp, pred_disp):
        loss = torch.log(torch.abs(pred_disp - good_disp) + 1)
        loss = loss.mean()
        return loss


    def log_time(self, batch_idx, duration, loss):
        """Print a logging statement to the terminal
        """
        samples_per_sec = self.opt.batch_size / duration
        time_sofar = time.time() - self.start_time
        training_time_left = (
                                     self.num_total_steps / self.step - 1.0) * time_sofar if self.step > 0 else 0
        print_string = "epoch {:>3} | batch {:>6} | examples/s: {:5.1f}" + \
                       " | loss: {:.5f} | time elapsed: {} | time left: {}"
        print(print_string.format(self.epoch, batch_idx, samples_per_sec, loss,
                                  sec_to_hm_str(time_sofar), sec_to_hm_str(training_time_left)))

    def log_tensorboard(self, mode, losses):
        """Write an event to the tensorboard events file
        """
        writer = self.writers[mode]
        for l, v in losses.items():
            writer.add_scalar("{}".format(l), v, self.step)


    def save_opts(self):
        """Save options to disk so we know what we ran this experiment with
        """
        models_dir = os.path.join(self.log_path, "models")
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        to_save = self.opt.__dict__.copy()

        with open(os.path.join(models_dir, 'opt.json'), 'w') as f:
            json.dump(to_save, f, indent=2)


    def save_model(self):
        """Save model weights to disk
        """
        save_folder = os.path.join(self.log_path, "models", "weights_{}".format(self.epoch))
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        for model_name, model in self.models.items():
            save_path = os.path.join(save_folder, "{}.pth".format(model_name))
            to_save = model.state_dict()
            if model_name == 'encoder':
                # save the sizes - these are needed at prediction time
                to_save['height'] = self.opt.height
                to_save['width'] = self.opt.width
                # to_save['use_stereo'] = self.opt.use_stereo
            torch.save(to_save, save_path)

        save_path = os.path.join(save_folder, "{}.pth".format("adam"))
        torch.save(self.model_optimizer.state_dict(), save_path)

    def load_ckpt(self):
        """Load checkpoint to resume a training, used in training process.
        """
        logging.info(" ")
        load_path = os.path.join(self.log_path, "ckpt.pth")
        if not os.path.exists(load_path):
            logging.info("No checkpoint to resume, train from epoch 0.")
            return None

        logging.info("Resume checkpoint from {}".format(os.path.abspath(load_path)))
        checkpoint = torch.load(load_path, map_location='cpu')
        for model_name, model in self.models.items():
            model_dict = model.state_dict()
            pretrained_dict = checkpoint[model_name]
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)
        self.ep_start = checkpoint['epoch']
        self.batch_start = checkpoint['batch_idx']
        self.step = checkpoint['step_in_total']
        logging.info("Start at eopch {}, batch index {}".format(self.ep_start, self.batch_start))
        return checkpoint

    def load_pretrained_model(self):
        """Load pretrained model(s) from disk, used for initializing.
        """
        self.opt.pretrained_path = os.path.abspath(self.opt.pretrained_path)

        assert os.path.exists(self.opt.pretrained_path), \
            "Cannot find folder {}".format(self.opt.pretrained_path)
        logging.info("Loading pretrained model from folder: {}".format(self.opt.pretrained_path))

        checkpoint = torch.load(self.opt.pretrained_path, map_location='cpu')
        for model_name, model in self.models.items():
            model_dict = model.state_dict()
            pretrained_dict = checkpoint[model_name]
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)


def worker_seed_fn(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)