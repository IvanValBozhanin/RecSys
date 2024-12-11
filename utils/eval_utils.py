import torch


def create_eval_batches(X_test, b, batch_size):
    n = X_test.shape[0]
    batches = []

    for i in range(0, n, batch_size):
        X_test_batch = X_test[i: i + batch_size, :]
        b_batch = b[i: i + batch_size, :]
        batches.append((X_test_batch, b_batch))

    return batches


def evaluate_model(model, X_eval, B_eval, loss_fn, batch_size, device):
    model.eval()
    test_loss = 0.0

    B_eval_tensor = torch.tensor(B_eval, dtype=torch.int).to(device)

    batches = create_eval_batches(X_eval, B_eval_tensor, batch_size)

    with torch.no_grad():
        for X_batch, B_batch in batches:
            X_batch_shaped = X_batch.unsqueeze(1)
            y_hat = model(X_batch_shaped).squeeze(1)

            batch_loss = loss_fn(y_hat * B_batch, X_batch * B_batch)
            test_loss += batch_loss.item()

    return test_loss / len(batches)
