import torch


def create_val_batches(Z_train, B_train, Z_val, B_val, batch_size):
    n = Z_train.shape[0]
    batches = []

    for i in range(0, n, batch_size):
        Z_train_batch = Z_train[i: i + batch_size, :]
        B_train_batch = B_train[i: i + batch_size, :]
        Z_val_batch = Z_val[i: i + batch_size, :]
        B_val_batch = B_val[i: i + batch_size, :]
        batches.append((Z_train_batch, B_train_batch, Z_val_batch, B_val_batch))

    return batches


def validate_model(model, Z_train, B_train, Z_val, B_val, loss_fn, batch_size, device):
    model.eval()
    test_loss = 0.0

    B_train_tensor = torch.tensor(B_train, dtype=torch.int).to(device)
    B_val_tensor = torch.tensor(B_val, dtype=torch.int).to(device)

    batches = create_val_batches(Z_train, B_train_tensor, Z_val, B_val_tensor, batch_size)

    with torch.no_grad():
        for Z_train_batch, B_train_batch, Z_val_batch, B_val_batch in batches:

            y_hat = model(Z_train_batch.unsqueeze(1)).squeeze(1)

            batch_loss = loss_fn(y_hat * B_val_batch,
                                 Z_val_batch * B_val_batch)

            test_loss += batch_loss.item()

    return test_loss / len(batches)
