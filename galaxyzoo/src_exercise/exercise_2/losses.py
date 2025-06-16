import torch.nn as nn

def regression_loss(pred, target):
    mse = nn.MSELoss()
    # pred: [batch, 5], target: [batch, 7]
    loss = mse(pred, target[:, 2:])  # Only compare Class2/7 outputs
    # Constraint: Class2.1 + Class2.2 = Class1.2
    q2_sum = pred[:, 0] + pred[:, 1]
    q2_target = target[:, 1]
    loss += ((q2_sum - q2_target) ** 2).mean()
    # Constraint: Class7.1 + Class7.2 + Class7.3 = Class1.1
    q7_sum = pred[:, 2] + pred[:, 3] + pred[:, 4]
    q7_target = target[:, 0]
    loss += ((q7_sum - q7_target) ** 2).mean()
    return loss