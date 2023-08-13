import torch.nn.functional as F


def model_loss_train(disp_ests, disp_gt, mask):
    weights = [1.0, 0.2]  # [cnn, bg]
    all_losses = []
    for disp_est, weight in zip(disp_ests, weights):
        all_losses.append(weight * F.smooth_l1_loss(disp_est[mask], disp_gt[mask], size_average=True))
    return sum(all_losses)


def model_loss_train_one(disp_ests, disp_gt, mask):
    weights = [1.0]
    all_losses = []
    for disp_est, weight in zip(disp_ests, weights):
        all_losses.append(weight * F.smooth_l1_loss(disp_est[mask], disp_gt[mask], size_average=True))
    return sum(all_losses)
