import matplotlib.pyplot as plt
import pandas as pd

def visualiz_MLPweights(model, dir):
    """
    visualize the paramters of this layer
    """
    weight_matrix = model.state_dict()["bert.encoder.layer.0.mlp.dense.weight"].detach().cpu()
    plt.imshow(weight_matrix, cmap='hot', interpolation='nearest')
    plt.colorbar()
    plt.savefig(dir / "weights.png")
    plt.clf()


def plot_test_train_loss(trainer_history, dir):
    """
    plot test train loss
    """
    trainer_history_train = [item for item in trainer_history if item.get("loss") is not None]
    trainer_history_eval = [item for item in trainer_history if item.get("eval_loss") is not None]
    trainer_history_train = pd.DataFrame(trainer_history_train[:-1]).set_index("step")
    trainer_history_eval = pd.DataFrame(trainer_history_eval[:-1]).set_index("step")

    trainer_history_train.loss.plot(label="Train Loss")
    trainer_history_eval.eval_loss.plot(label="Test Loss")
    plt.legend()
    plt.ylabel("loss")
    plt.savefig(dir / "loss.png")
    plt.clf()