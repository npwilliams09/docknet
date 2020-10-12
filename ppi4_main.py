from components.ppi4_data_loader import *
from components.utils import *
from pathlib import Path
from components.Docknet import docknet
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.metrics as metrics
import wandb
from wandb.keras import WandbCallback
import os

def getConfig():
    return {
        "filters": 128,
        "graph_blocks": 1,
        "graph_style": 'gcn',
        "drop_rate": 0.8,
        "combine_style": 'matmul',
        "squeeze_excite": False,
        "wave_blocks": 1,
        "epochs": 50
    }


def main():
    data_path = Path("/home/jupyter/output")
    train_file = 'trainList.txt'
    test_file = 'testList.txt'

    config = getConfig()
    g_mode = graph_mode(config["graph_style"])
    print(config)

    print("Loading training data...")
    train_ls, train_data = createSet(str(train_file), graph_mode=g_mode)
    print("Loading test data...")
    val_ls, test_data = createSet(str(test_file), graph_mode=g_mode)

    print("Building Model...")
    features, weight = get_parameters(train_data)

    model = docknet(config, features)

    wandb.init(project="ppi4-docknet-runs")
    wandb.config = config

    loss = get_crossentropy_loss(weight)
    lr_schedule = get_lr_schedule(len(train_ls))
    checkpoint = get_checkpoint_callback(os.path.join(wandb.run.dir, "mymodel.h5"))
    opt = Adam(learning_rate=lr_schedule)

    model.compile(loss=loss, optimizer=opt, metrics=[metrics.BinaryAccuracy(), metrics.AUC()], run_eagerly=True)

    model.summary()

    history = model.fit(
        seqGenerator(train_ls, train_data, aug=True),
        steps_per_epoch=len(train_ls),
        epochs=config["epochs"],
        batch_size=1,
        validation_data=seqGenerator(val_ls, test_data),
        validation_steps=len(val_ls),
        callbacks=[WandbCallback()],
    )

    hist_df = pd.DataFrame(history.history)
    hist_csv_file = data_path / 'history.csv'
    with open(hist_csv_file, mode='w') as f:
        hist_df.to_csv(f)

    print("Complete")


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
