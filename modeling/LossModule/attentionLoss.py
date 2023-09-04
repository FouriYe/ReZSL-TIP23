import torch

def ADLoss(query, attr_group):
    """
    Attribute Decorelation Loss
    :param query:
    :return:
    """
    loss_sum = 0

    for key in attr_group:
        group = attr_group[key]
        proto_each_group = query[group]  # g1 * v
        channel_l2_norm = torch.norm(proto_each_group, p=2, dim=0)
        loss_sum += channel_l2_norm.mean()

    loss_sum = loss_sum.float() / len(attr_group)

    return loss_sum

def CPTLoss(atten_map, device):
    """

    :param atten_map: N, L, W, H
    :return:
    """

    N, L, W, H = atten_map.shape
    xp = torch.tensor(list(range(W))).long().unsqueeze(1).to(device)
    yp = torch.tensor(list(range(H))).long().unsqueeze(0).to(device)

    xp = xp.repeat(1, H)
    yp = yp.repeat(W, 1)

    atten_map_t = atten_map.view(N, L, -1)
    value, idx = atten_map_t.max(dim=-1)

    tx = idx // H
    ty = idx - H * tx

    xp = xp.unsqueeze(0).unsqueeze(0)
    yp = yp.unsqueeze(0).unsqueeze(0)
    tx = tx.unsqueeze(-1).unsqueeze(-1)
    ty = ty.unsqueeze(-1).unsqueeze(-1)

    pos = (xp - tx) ** 2 + (yp - ty) ** 2

    loss = atten_map * pos

    loss = loss.reshape(N, -1).mean(-1)
    loss = loss.mean()

    return loss