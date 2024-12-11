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


def create_batches(X, f, b, batch_size):
    n = X.shape[0]
    batches = []

    for i in range(0, n, batch_size):
        X_batch = X[i: i + batch_size, :]
        f_batch = f[i: i + batch_size, :]
        b_batch = b[i: i + batch_size, :]
        batches.append((X_batch, f_batch, b_batch))

    return batches


def train_epoch(model, optimizer, X_train, B_train_cpu, loss_fn, batch_size, device):

    model.train()
    epoch_loss = 0.0

    f_mask, b_mask = random_training_split(B_train_cpu, forward_ratio=forward_ratio)

    f_mask_tensor = torch.tensor(f_mask, dtype=torch.int).to(device)
    b_mask_tensor = torch.tensor(b_mask, dtype=torch.int).to(device)

    batches = create_batches(X_train, f_mask_tensor, b_mask_tensor, batch_size)

    for X_batch, F_batch, B_batch in batches:
        optimizer.zero_grad()

        X_batch_shaped = X_batch.unsqueeze(1)

        y_hat = model(X_batch_shaped * F_batch.unsqueeze(1)).squeeze(1)
        batch_loss = loss_fn(y_hat * B_batch, X_batch * B_batch)

        batch_loss.backward()
        optimizer.step()

        epoch_loss += batch_loss.item()

    return epoch_loss / len(batches)
