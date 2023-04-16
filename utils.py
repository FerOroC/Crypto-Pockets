import numpy as np
import torch


def evaluate_model(model, loss, data_iter):
    model.eval()
    l_sum, n = 0.0, 0
    with torch.no_grad():
        for x, y in data_iter:
            y_pred = model(x).view(len(x), -1)
            l = loss(y_pred, y)
            l_sum += l.item() * y.shape[0]
            n += y.shape[0]
        return l_sum / n

def evaluate_mape(y, y_pred, threshold):
    y = y
    y_pred = y_pred
    mask = y > threshold
    if np.sum(mask)!=0:
        mape = np.abs(y[mask] - y_pred[mask])/y[mask]
        return mape
    else:
        return np.nan

def evaluate_metric(model, data_iter):
    model.eval()
    with torch.no_grad():
        mae, mape, mse = [], [], []
        for x, y in data_iter:
            y = y.cpu().numpy()
            y_pred = model(x).view(len(x), -1).cpu().numpy()
            d = np.abs(y - y_pred)
            mae += d.tolist()
            mape.append(evaluate_mape(y,y_pred,0))
            mse += (d**2).tolist()
        MAE = np.array(mae).mean()
        MAPE = np.array(mape).mean()
        RMSE = np.sqrt(np.array(mse).mean())
        return MAE, RMSE, MAPE