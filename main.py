from components.data_loader import *
from components.utils import *
from pathlib import Path
from components.Docknet import docknet
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.metrics as metrics
import wandb
from wandb.keras import WandbCallback


def getConfig():
    return {
        "filters": 64,
        "graph_blocks": 2,
        "graph_style": 'gcn',
        "drop_rate": 0.2,
        "combine_style": 'matmul',
        "squeeze_excite": False,
        "wave_blocks": 1,
        "epochs": 50
    }


def main():
    data_path = Path("/home/jupyter/output")
    train_path = data_path / 'train'
    test_path = data_path / 'test'

    config = getConfig()
    g_mode = graph_mode(config["graph_style"])

    print("Loading training data...")
    train_ls, train_data = process_dataset(train_path, g_mode)
    print("Loading test data...")
    test_ls, test_data = process_dataset(test_path, g_mode)

    val_ls = test_ls[::2]  # halve for validation
    print("Building Model...")
    features, weight = get_parameters(train_data)

    model = docknet(config, features)

    wandb.init(project="docknet-30K")
    wandb.config = config

    loss = get_crossentropy_loss(weight)
    lr_schedule = get_lr_schedule(len(train_ls))
    checkpoint = get_checkpoint_callback(data_path / 'models')
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
        callbacks=[checkpoint, WandbCallback()],
        use_multiprocessing=True,
        max_queue_size=30,
        workers=16
    )

    hist_df = pd.DataFrame(history.history)
    hist_csv_file = data_path / 'history.csv'
    with open(hist_csv_file, mode='w') as f:
        hist_df.to_csv(f)

    print("Complete")


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
