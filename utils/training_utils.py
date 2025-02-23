import numpy as np
import torch

from constants import forward_ratio


def random_training_split(B1, forward_ratio):
    indices = np.where(B1.flatten() == 1)[0]
    np.random.shuffle(indices)

    n = len(indices)

    f_indices = indices[:int(n * forward_ratio)]
    b_indices = indices[int(n * forward_ratio):]

    f_mask = np.zeros_like(B1.flatten())
    b_mask = np.zeros_like(B1.flatten())

    f_mask[f_indices] = 1
    b_mask[b_indices] = 1

    return f_mask.reshape(B1.shape), b_mask.reshape(B1.shape)


def create_batches(Z, forward, backward, batch_size):
    n = Z.shape[0]
    batches = []

    for i in range(0, n, batch_size):
        Z_batch = Z[i: i + batch_size, :]
        forward_batch = forward[i: i + batch_size, :]
        backward_batch = backward[i: i + batch_size, :]
        batches.append((Z_batch, forward_batch, backward_batch))

    return batches


def train_epoch(model, optimizer, Z_train, B_train_cpu, loss_fn, batch_size, device):

    model.train()
    epoch_loss = 0.0
    grad_norms = []

    B_forward, B_backward = random_training_split(B_train_cpu, forward_ratio=forward_ratio)

    B_forward_tensor = torch.tensor(B_forward, dtype=torch.int).to(device)
    B_backward_tensor = torch.tensor(B_backward, dtype=torch.int).to(device)

    batches = create_batches(Z_train, B_forward_tensor, B_backward_tensor, batch_size)

    for Z_batch, B_forward_batch, B_backward_batch in batches:
        optimizer.zero_grad()

        y_hat = model(Z_batch.unsqueeze(1) * B_forward_batch.unsqueeze(1)).squeeze(1)

        batch_loss = loss_fn(y_hat * B_backward_batch, # todo: B_forward_batch or B_backward_batch?
                             Z_batch * B_backward_batch)

        batch_loss.backward()
        optimizer.step()

        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        total_norm = total_norm ** 0.5
        grad_norms.append(total_norm)

        epoch_loss += batch_loss.item()

    epoch_grad_norm = sum(grad_norms) / len(grad_norms) if grad_norms else 0
    print(f"Average Gradient Norm for epoch: {epoch_grad_norm}")
    return epoch_loss / len(batches)
