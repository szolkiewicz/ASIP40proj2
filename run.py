import argparse
import json

from runner import Runner

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=False, default="./34_2d_config/config_run.json")
    args = parser.parse_args()
    config = json.loads(open(args.config).read())
    runner = Runner(config)
    runner.run()