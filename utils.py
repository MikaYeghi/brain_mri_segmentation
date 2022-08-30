import pdb

def make_train_step(model, loss_fn, optimizer):
    def train_step(images_batch, masks_batch):
        # pdb.set_trace()
        model.train()
        yhat = model(images_batch)
        loss = loss_fn(masks_batch, yhat)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        return loss.item()
    return train_step