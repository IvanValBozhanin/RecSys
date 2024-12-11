import torch


def create_val_batches(X, B, batch_size):
    n = X.shape[0]
    batches = []

    for i in range(0, n, batch_size):
        X_batch = X[i: i + batch_size, :]
        b_batch = B[i: i + batch_size, :]
        batches.append((X_batch, b_batch))

    return batches


def validate_model(model, X_val, B_val, loss_fn, batch_size, device):
    model.eval()
    test_loss = 0.0

    B_val_tensor = torch.tensor(B_val, dtype=torch.int).to(device)

    batches = create_val_batches(X_val, B_val_tensor, batch_size)

    with torch.no_grad():
        for X_batch, B_batch in batches:

            y_hat = model(X_batch.unsqueeze(1)).squeeze(1)

            batch_loss = loss_fn(y_hat * B_batch,
                                 X_batch * B_batch)
            test_loss += batch_loss.item()

    return test_loss / len(batches)
