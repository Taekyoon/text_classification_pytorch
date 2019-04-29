from fire import Fire

from train import build_model, prepare_dataset
from train import TrainManager

from utils import load_configs


def main(path: str):
    _configs = load_configs(path)

    _batch_size = _configs['hyperparams']['batch_size']
    _epochs = _configs['hyperparams']['epochs']
    _eval_steps = _configs['hyperparams']['evaluation_steps']

    _model_params = _configs['hyperparams']['model']

    print('Getting dataset...')
    _data_manager = prepare_dataset(_configs)
    _vocab_size = _data_manager.get_vocabulary_size()

    _model_params['vocab_size'] = _vocab_size

    print('Now build model...')
    model = build_model(_model_params)

    train_dataloader = _data_manager.get_train_data_loader(_batch_size)
    test_dataloader = _data_manager.get_test_data_loader(512)

    print(model)

    print('Train start!')
    train_manager = TrainManager(train_dataloader,
                                 test_dataloader,
                                 model,
                                 _epochs,
                                 _eval_steps)

    train_manager.train()


if __name__ == '__main__':
    Fire(main)
