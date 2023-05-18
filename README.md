# nlp-biLSTM-attention

The code repository for the Nature Scientific Reports (2023) paper 
[Interpretable Sentiment Analysis of COVID-19 Tweets using Attention-based BiLSTM and Twitter-RoBERTa](https://arxiv.org/abs/2210.07182)



![Visualizations of some COVID-19 tweets](https://github.com/pdebench/PDEBench/blob/main/pdebench_examples.PNG)


Created and maintained by Md Abrar Jahin `<abrar.jahin.2652@gmail.com, md-jahin@oist.jp>`.

## Datasets

*Extended Datasets*

Each dataset contains a column of cleaned tweets which was obtained by preprocessing the raw tweets and comments, accompanied by sentiment label of negative (-1), neutral (0), and posititve (1). 

*External Datasets*

External Tweets and Comments were made on Narendra Modi and other leaders as well as people's opinion towards the next prime minister of India (in the context with general elections held in India - 2019). The external datasets were created with the help of the Tweepy and Reddit Apis. 

| Datasets  | Description |
| ------------- | ------------- |
| [UK Twitter COVID-19 Dataset]() | Content Cell  |
| [Global Twitter COVID-19 Dataset]() | Content Cell  |
| [USA Twitter COVID-19 Dataset]() | Content Cell  |
| [External Reddit Dataset]() | 36801 comments |
| [External Twitter Dataset]() | 162980 tweets |


## Pretrained Models

[1] [twitter-roberta-base-sentiment-latest](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest)

[2] [distilbert-base-uncased](https://huggingface.co/distilbert-base-uncased)

[3] [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)








