#   OLD TRAINING LOOP
# for epoch in range(n_epochs):
#     gnn_model.train()
#     epoch_loss = 0.0
#
#     f_mask, b_mask = random_training_split(B1, 0.6)
#
#     f_mask_tensor = torch.tensor(f_mask, dtype=torch.int).to(device)
#     b_mask_tensor = torch.tensor(b_mask, dtype=torch.int).to(device)
#
#     batches = create_batches(X_train, f_mask_tensor, b_mask_tensor, batch_size)
#
#     for X_batch, F_batch, B_batch in batches:
#         optimizer.zero_grad()
#
#         X_batch_shaped = X_batch.unsqueeze(1)
#         # print(X_batch_shaped)
#         # print(X_batch_shaped.shape)
#
#         y_hat = gnn_model(X_batch_shaped * F_batch.unsqueeze(1)).squeeze(1)
#
#         batch_loss = loss_fn(y_hat * B_batch, X_batch * B_batch)
#
#         batch_loss.backward()
#         optimizer.step()
#         # print(f'Batch Loss: {batch_loss.item()}')
#         epoch_loss += batch_loss.item()
#
#     print(f'Epoch {epoch + 1}/{n_epochs}, Loss: {epoch_loss / len(batches)}')
#
# print("===============")




# TODO: To torch tensor, to device, to float32, unsqueeze(1) if needed
# TODO: pick 50% for the training forward pass, 40% for the training backward pass; 10% is already set for testing.
# TODO: Per each epoch, shuffle the data.
# TODO: Implement the forward pass,
#       backward pass,
#       and the loss function.
# TODO: Implement the training loop.
# TODO: Implement the evaluation on the test set.
# =============================================================================

# TODO: Denomralize z-score to the original values.
# TODO: Stochastic parameter search for the best hyperparameters.
# TODO: Check optimal number of epochs: Graphs & Elbow method.
# TODO: Train & Valid & Test split:
    # train and valid testing per each epoch to plot.



# TESTING LOOP
# gnn_model.eval()
# with torch.no_grad():
#
#     batches = create_testing_batches(X_train, B_eval, batch_size)
#     test_loss = 0.0
#     for X_batch, B_batch in batches:
#         X_batch_shaped = X_batch.unsqueeze(1)
#         y_hat = gnn_model(X_batch_shaped).squeeze(1)
#
#         batch_loss = loss_fn(y_hat * B_batch, X_batch * B_batch)
#         print(f'Batch Test Loss: {batch_loss.item()}')
#         test_loss += batch_loss.item()
#
#     print(f'Test Loss for the Masked Ratings: {test_loss}')



# from testing
# print(f"pre denorm: {all_predictions}")
    # print("=======")
    # print(f"first column mean and std: {np.mean(all_predictions[:, 0]), np.std(all_predictions[:, 0])}")
    # print(f"first column min and max: {np.min(all_predictions[:, 0]), np.max(all_predictions[:, 0])}")
    # print("=======")
    # print(f"Predictions: {denormalize_predictions}")
    # print(f"Actuals: {np.max(denormalize_actuals)}")

    # print all positions where predictions are different from actuals
    # pos = np.where(denormalize_predictions != denormalize_actuals)
    #
    # # print the cells where the predictions are different from the actuals
    # print(f"Predictions: {denormalize_predictions[pos]}")
    # print(f"Actuals: {denormalize_actuals[pos]}")



# from preprocessing
# print(f"Z1 0 0 {Z1[0,0]}")
# print(f"Z1 0 0 {X_train[0,0] * user_stds[0] + user_means[0]}")
# print(f"B eval: {B_eval.shape}")
# print(f"B eval: {B_eval.cpu().numpy()}")
# print(f"B eval non zero: {torch.nonzero(B_eval)}")

# print(f"X1: {X1}")

# print(f"Z1: {Z1}")

# print(f"Z1 denormalized: {denormalize_ratings(Z1, user_means, user_stds)}")

# print(f"X train: {X_train}")
# print(f"X train denormalized: {denormalize_ratings(X_train.cpu().numpy(), user_means, user_stds)}")
# print(f"x train multiplied by B eval: {X_train * B_eval}")
# print(f"x train multiplied by B eval non zero: {torch.max(X_train * B_eval)}")
# denorm_multiplied = denormalize_ratings((X_train * B_eval).cpu().numpy(), user_means, user_stds)

# pos_b_eval = np.where(B_eval.cpu().numpy() != 0.0)
# print(f"shape of pos b eval {len(pos_b_eval), len(pos_b_eval[0]), len(pos_b_eval[1])}")
# print(f"first pos b of X train: {X_train[0, 26]}")
# print(f"first pos b of B_eval: {B_eval.cpu().numpy()[0, 26]}")
# print(f"pos_b_eval: {pos_b_eval}")
# print(f"X train at pos: {(X_train[pos_b_eval])}")

# print(f"X train multiplied by B eval positions: {denorm_multiplied[pos_b_eval]}")
# print(f"x train multiplied by B eval denormalized: {denorm_multiplied[0, 0]}")
# print(f"x train multiplied by B eval denormalized non zero: {np.max(denorm_multiplied)}")
# print("==================\n"
#       "==================\n"
#       "==================")




#
# print(np.sum(B0), np.sum(B1), np.sum(B_train), np.sum(B_val) , np.sum(B_test))
#
# print(X_test)
# print(B_test)
# print(np.sum(~np.isnan(X0)), np.sum(~np.isnan(X1)), np.sum(~np.isnan(X_train)), np.sum(~np.isnan(X_val) ), np.sum(B_test * (~np.isnan(X_test))))

# print(X0)


# print(np.sum(B1 * B_val), np.sum(B1 * B_train))
# print(np.sum(B_train * B_val), np.sum(B_train * B_test), np.sum(B_val * B_test))
