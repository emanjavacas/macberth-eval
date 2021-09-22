
from tensorflow import keras
from tensorflow.keras import layers

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=24)
    parser.add_argument('--margin', default=1.0, type=float)
    args = parser.parse_args()