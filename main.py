import argparse
from train_eval import train_eval
import torch

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./data', help='Directory for the dataset')
    parser.add_argument('--mode', type=str, default='normal', help='sparse or normal')
    parser.add_argument('--train_parts', type=int, default=100, help='Number of training parts')
    parser.add_argument('--val_parts', type=int, default=25, help='Number of validation parts')
    parser.add_argument('--test_parts', type=int, default=25, help='Number of test parts')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-3, help='Weight decay')
    parser.add_argument('--temperature', type=float, default=0.05, help='Temperature for softmax')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use for training')
    parser.add_argument('--hidden1', type=int, default=16, help='Size of the first hidden layer')
    parser.add_argument('--hidden2', type=int, default=16, help='Size of the second hidden layer')
    parser.add_argument('--k', type=int, default=3, help='Number of nearest neighbors')
    parser.add_argument('--input_dim', type=int, default=8, help='Input dimension')
    parser.add_argument('--output_dim', type=int, default=112, help='Output dimension')
    parser.add_argument('--edge_feature_dim', type=int, default=8, help='Edge feature dimension')

    args = parser.parse_args()
    data_dir = args.data_dir
    mode = args.mode
    train_parts = args.train_parts
    val_parts = args.val_parts
    test_parts = args.test_parts
    epochs = args.epochs
    lr = args.lr
    weight_decay = args.weight_decay
    temperature = args.temperature
    device = args.device
    hidden1 = args.hidden1
    hidden2 = args.hidden2
    k = args.k
    input_dim = args.input_dim
    output_dim = args.output_dim
    edge_feature_dim = args.edge_feature_dim

    train_eval(data_dir, mode, train_parts, val_parts, test_parts, epochs, lr, weight_decay, temperature, device, hidden1, hidden2, k, input_dim, output_dim, edge_feature_dim)
