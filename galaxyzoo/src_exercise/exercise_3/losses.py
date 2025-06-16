import torch.nn as nn

def weighted_regression_loss(pred, target, output_weights=None):
    mse = nn.MSELoss(reduction='none')
    base_loss = mse(pred, target)
    if output_weights is not None:
        base_loss = base_loss * output_weights
    base_loss = base_loss.mean()

    # Constraint: Class6.1 + Class6.2 = Class7.1
    class6_sum = pred[:, 2] + pred[:, 3]
    class7_1 = pred[:, 4]
    base_loss += ((class6_sum - class7_1) ** 2).mean()

    # Constraint: Class8.1 + ... + Class8.7 = Class6.1
    class8_sum = pred[:, 7:14].sum(dim=1)
    class6_1 = pred[:, 2]
    base_loss += ((class8_sum - class6_1) ** 2).mean()

    return base_loss