
# Focal Loss Implementation
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=0.5):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)

        if self.alpha is not None:
            alpha_factor = self.alpha[targets]  # index alpha per target class
        else:
            alpha_factor = 1.0

        focal_loss = alpha_factor * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()

