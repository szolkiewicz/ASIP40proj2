from trainer import Trainer
import argparse
import json

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=False, default="./34_1d_config/config_train.json")
    args = parser.parse_args()
    config = json.loads(open(args.config).read())
    trainer = Trainer(config)
    trainer.train()