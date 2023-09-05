def double_pendulum_loss(pred, actual):
    # calculate loss as the mean squared error between predicted and actual final angles
    pred_theta1, pred_theta2 = pred[:, 0], pred[:, 1]
    actual_theta1, actual_theta2 = actual[:, 0], actual[:, 1]
    pred_omega1, pred_omega2 = pred[:, 2], pred[:, 3]
    actual_omega1, actual_omega2 = actual[:, 2], actual[:, 3]
    loss = torch.mean((pred_theta1 - actual_theta1) ** 2 + (pred_theta2 - actual_theta2) ** 2 + (pred_omega1 - actual_omega1) ** 2 + (pred_omega2 - actual_omega2) ** 2)
    return loss
