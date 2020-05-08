##########
#
#
# Python script to predict ludwig models
# Author: Jaret Karnuta
# 
#
##########
import argparse
import os
import pandas as pd
from ludwig.api import LudwigModel

class LudwigPredictor:
    
    def __init__(self, ludwig_path):
        self.model = LudwigModel.load(ludwig_path)

    def predict(self, df):

        self.predictions = self.model.predict(df)
        return self.predictions

    def save(self, path):
        self.predictions.to_csv(path)

    def close(self):
        self.model.close()

def main(args):
    # update CUDA gpu assignment
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    lm = LudwigPredictor(args.ludwig_model)

    df = pd.read_csv(args.data_csv)
    predictions = lm.predict(df)

    c = pd.concat([df, predictions], axis = 1)
    c.to_csv('ludwig_resnet_raw.csv')

    lm.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Make predictions with Ludwig")
    parser.add_argument('--data-csv', type=str)
    parser.add_argument('--ludwig-model', type = str, required = True)
    parser.add_argument('--gpu', type = str, default = '3', help = "GPU to assign CUDA")
    args = parser.parse_args()
    main(args)