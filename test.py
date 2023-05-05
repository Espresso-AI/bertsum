import argparse
import hydra
from src import *

parser = argparse.ArgumentParser()
parser.add_argument("--config-name", dest='config_name', default=None, type=str)
args = parser.parse_args()


@hydra.main(version_base=None, config_path='./config/', config_name=args.config_name)
def test(cfg: DictConfig):
    from transformers import AutoTokenizer
    from torch.utils.data import DataLoader

    model = BertSum_Ext(**cfg.model)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.base_checkpoint)

    # load train and validation datasets
    # The remainders will be published soon.
    if cfg.mode.dataset == 'cnn_dm':
        test_df = cnndm_test_df(cfg.dataset.path)
    else:
        raise ValueError("Sorry, 'cnn_dm' is now the only available dataset.")

    test_dataset = ExtSum_Dataset(test_df, tokenizer, cfg.max_seq_len)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    engine = ExtSum_Engine(model, test_df=test_df, **cfg.engine)
    cfg_trainer = Config_Trainer(cfg.trainer)()
    trainer = pl.Trainer(**cfg_trainer, logger=False)

    if 'test_checkpoint' in cfg:
        trainer.test(engine, test_loader, ckpt_path=cfg.test_checkpoint)
    else:
        raise RuntimeError('no checkpoint is given')



if __name__ == "__main__":
    test()
