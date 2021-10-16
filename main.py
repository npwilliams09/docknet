from components.data_loader import *
from components.utils import *
from pathlib import Path
from components.Docknet import docknet, dockcoder
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.metrics as metrics
import wandb
from wandb.keras import WandbCallback
import os
import tensorflow as tf
from tensorflow.keras.mixed_precision import experimental as mixed_precision
import json

def getConfig():
    return {
        "filters": 128,
        "graph_blocks": 1,
        "graph_style": 'gcn',
        "drop_rate": 0.2,
        "combine_style": 'matmul',
        "squeeze_excite": False,
        "wave_blocks": 1,
        "epochs": 14,
        "lr": 4e-3,
        "width": 256,
        "attn_heads" : 2,
        "dff_sz" : 128
    }

def main():
    data_path = Path("C:/Users/natha/dockdata/dataset")
    train_path = data_path / 'train'
    test_path = data_path / 'test'

    #cwd_path = Path("/scratch/gi80/docknet/")
    '''
    policy = mixed_precision.Policy('mixed_float16')
    mixed_precision.set_policy(policy)
    tf.config.optimizer.set_jit(True)
    
    os.environ["WANDB_MODE"] ="dryrun"
    os.environ["WANDB_API_KEY"] = "131c9435a433970b7ab833e257936757f2390599" #remove later
    wandb_path = data_path / "wandb"
    wandb_path.mkdir(exist_ok=True, parents=False)
    os.environ["WANDB_DIR"] = str(wandb_path)
    '''
    tf.get_logger().setLevel('ERROR')
    os.environ["TF_GPU_HOST_MEM_LIMIT_IN_MB"] = "100768" #100gb of mem

    config = getConfig()
    run = wandb.init()
    wandb.config = config
    #config = run.config
    g_mode = graph_mode(config["graph_style"])

    print("Loading training data...")
    #train_ls, train_data = process_dataset(train_path, g_mode)
    print("Loading test data...")
    test_ls, test_data = process_dataset(test_path, g_mode, width_limit=config["width"])
    #fore testing new features
    train_ls = test_ls
    train_data = test_data

    val_ls = test_ls[::2]  # halve for validation
    print("Building Model...")
    features, weight = get_parameters(train_data)

    model = dockcoder(config, features)

    wandb.init(project="docknet-30K")
    wandb.config = config

    loss = get_crossentropy_loss(weight)
    lr_schedule = get_lr_schedule(len(train_ls))
    checkpoint = get_checkpoint_callback(data_path / 'models')
    opt = Adam(learning_rate=config["lr"])
    opt = tf.keras.mixed_precision.experimental.LossScaleOptimizer(opt, "dynamic")

    model.compile(loss=loss, optimizer=opt, metrics=[metrics.AUC(dtype='float32')], run_eagerly=True)

    model.summary()
    
    mini_epoch = 2

    print(len(train_ls))

    history = model.fit(
        seqGenerator(train_ls, train_data, aug=True, pad_dim=config["width"]),
        steps_per_epoch=len(train_ls)/mini_epoch,
        epochs=config["epochs"]*mini_epoch,
        batch_size=1,
        validation_data=seqGenerator(val_ls, test_data, pad_dim=config["width"]),
        validation_steps=len(val_ls),
        #callbacks=[WandbCallback()],
        #use_multiprocessing=True,
        #max_queue_size=30,
        #workers=4
    )

    hist_df = pd.DataFrame(history.history)
    hist_csv_file = data_path / 'history.csv'
    with open(hist_csv_file, mode='w') as f:
        hist_df.to_csv(f)

    print("Complete")


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    '''
    sweep_config = {
        "name": "Test Sweep",
        "method": "bayesian",
        'metric': {
            'name': 'val_auc',
            'goal': 'maximize'
        },
        "parameters": {
            "filters": {
                "value": 32
            },
            "graph_blocks": {
                "value": 1
            },
            "drop_rate": {
                "values": [0.4, 0.6]
            },
            "combine_style": {
                "values": ['conc', 'matmul']
            },
            "squeeze_excite": {
                "value": False
            },
            "wave_blocks": {
                "value": 1
            },
            "epochs": {
                "value": 1
            },
            "graph_style": {
                "value": "gcn"
            }
        }
    }

    sweep_id = wandb.sweep(sweep_config, project="ppi4-docknet-test")
    wandb.agent(sweep_id, function=main)
    '''
    main()
