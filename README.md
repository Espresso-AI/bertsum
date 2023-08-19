# BertSum

This is the implementation of [BertSum](https://arxiv.org/pdf/1908.08345.pdf) trained on CNN/DailyMail dataset. The current available model is BertSum-Ext for extractive summarization, and the abstractive model will be released in the near future.

Most implementations in the field of summarization still rely on [nlpyang/PreSumm](https://github.com/nlpyang/PreSumm). However, it is written nearly from the scratch. This repo contributes to the reproducibility of extractive summarization.

## Training
We used [bert-base-uncased](https://huggingface.co/bert-base-uncased) for the BERT checkpoints, and training was conducted on a single T4.

<img src="https://github.com/Espresso-AI/bertsum-korean/blob/main/misc/bertsum_training.png" width="850" height="350">

## Evaluation
The results of our model evaluated on CNN/DailyMail test set is as follows:
|rouge1|rouge2|rougeLsum|rougeL| 
|:---:|:---:|:---:|:---:|
|43.03|20.16|39.46|27.69|

The ROUGE scores were calculated using [rouge-score](https://pypi.org/project/rouge-score/) of [google-research](https://github.com/google-research/google-research/blob/master/rouge/rouge_scorer.py). This library provides two types of ROUGE-L , rougeL and rougeLsum:
* rougeL :  `\\n` ignored. It treats the summary as a single sentence.
* rougeLsum : `\\n` not ignored. It matches the sentence pair with the largest LCS from two summaries.

ROUGE-L provided by [pyrouge](https://pypi.org/project/pyrouge/), used in [nlpyang/PreSumm](https://github.com/nlpyang/PreSumm), is same with rougeLsum.

## Usage
To train the model, define the experiment in a YAML file and run the following command. 
```
python train.py -—config-name exp_0
```

To evaluate the model, enter the path of checkpoints file in `test_checkpoint` of the YAML file, and execute the following command.
```
python test.py —-config-name exp_0
```

## License
BSD 3-Clause License Copyright (c) 2022
