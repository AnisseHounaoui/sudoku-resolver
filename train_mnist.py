import argparse

from dnn_framework import Network, FullyConnectedLayer, BatchNormalization, ReLU
from mnist import MnistTrainer


def main():
    parser = argparse.ArgumentParser(description='Train Backbone')
    parser.add_argument('--learning_rate', type=float, help='Choose the learning rate', required=True)
    parser.add_argument('--batch_size', type=int, help='Set the batch size for the training', required=True)
    parser.add_argument('--epoch_count', type=int, help='Choose the epoch count', required=True)
    parser.add_argument('--output_path', type=str, help='Choose the output path', required=True)

    parser.add_argument('--checkpoint_path', type=str, help='Choose the output path', default=None)

    args = parser.parse_args()

    network = create_network(args.checkpoint_path)
    trainer = MnistTrainer(network, args.learning_rate, args.epoch_count, args.batch_size, args.output_path)
    trainer.train()


def create_network(checkpoint_path):
    layers = [FullyConnectedLayer(784, 128), BatchNormalization(128), ReLU(),
              FullyConnectedLayer(128, 32), BatchNormalization(32), ReLU(),
              FullyConnectedLayer(32, 10)]
    network = Network(layers)
    if checkpoint_path is not None:
        network.load(checkpoint_path)

    return network


def grid_search():
    lr = [0.01, 0.005, 0.001]
    batch_size = [64, 256, 512]
    epochs = [5, 10, 15]

    best_acc = 0
    best_params = ''

    for rate in lr:
        for bs in batch_size:
            for e in epochs:
                print("Starting " + str(rate)+'_'+str(bs)+'_'+str(e))
                n = create_network(None)
                trainer = MnistTrainer(n, rate, e, bs, 'out/'+str(rate)+'_'+str(bs)+'_'+str(e)+'/')
                trainer.train()
                acc = trainer._test(trainer._network, trainer._test_dataset_loader)
                if acc > best_acc:
                    best_acc = acc
                    best_params = str(rate)+'_'+str(bs)+'_'+str(e)

    print(best_params)
    print(best_acc)


def validate_trained_models():
    lr = [0.01, 0.005, 0.001]
    batch_size = [64, 256, 512]
    epochs = [5, 10, 15]

    best_acc = 0
    best_params = ''

    for rate in lr:
        for bs in batch_size:
            for e in epochs:
                n = create_network('out/'+str(rate)+'_'+str(bs)+'_'+str(e)+'/checkpoint_epoch_'+str(e)+'.pkl')
                trainer = MnistTrainer(n, rate, e, bs, 'out/'+str(rate)+'_'+str(bs)+'_'+str(e)+'/')
                print('LR: '+ str(rate)+', Batchsize: '+str(bs)+', Nb epochs: '+str(e))
                acc = trainer._test(trainer._network, trainer._test_dataset_loader)

                if acc > best_acc:
                    best_acc = acc
                    best_params = str(rate)+'_'+str(bs)+'_'+str(e)
    print(best_params)
    print(best_acc)


if __name__ == '__main__':
    main()
    #grid_search()
    #validate_trained_models()
