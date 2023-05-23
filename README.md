# nlp-roBERTa-biLSTM-attention

The code repository for the Nature Scientific Reports (2023) paper 
[Interpretable Sentiment Analysis of COVID-19 Tweets using Attention-based BiLSTM and Twitter-RoBERTa](https://arxiv.org/abs/2210.07182)


![Visualization of proposed model](https://github.com/Abrar2652/nlp-roBERTa-biLSTM-attention/blob/main/Twitter-RoBERTa%2BLSTM/model4_BiLSTM%2Battention/uk_twitter_data_all/collage.png)



Created and maintained by Md Abrar Jahin `<abrar.jahin.2652@gmail.com, md-jahin@oist.jp>`.

## Datasets

*Extended Datasets*

In order to address the research gaps identified by Qi and Shabrina (2023), we have expanded the existing COVID-19 Twitter dataset. Our datasets overcome the limitations highlighted in their paper, specifically the short timeline and geographical constraints of the tweets. Each dataset includes a column of cleaned tweets, which have undergone preprocessing of the raw tweets and comments. Additionally, the datasets are accompanied by sentiment labels categorizing the tweets as negative (-1), neutral (0), or positive (1).

*External Datasets*

To assess the robustness and generalizability of our proposed model, we employed external datasets for benchmarking purposes. These additional datasets were utilized to evaluate how well our model performs beyond the confines of the original dataset used for training and testing. By incorporating these external datasets, we aimed to obtain a more comprehensive understanding of our model's capabilities and its ability to handle diverse and unseen data. The inclusion of these benchmark datasets allowed us to gauge the model's performance under varying conditions and validate its effectiveness in real-world scenarios.

| Datasets  | Description |
| ------------- | ------------- |
| [UK Twitter COVID-19 Dataset](https://github.com/Abrar2652/nlp-roBERTa-biLSTM-attention/blob/main/UK_covid_twitter_data/sample_data_all.csv) | This dataset was developed by collecting COVID-19 tweets from only the major cities in the UK (Qi and Shabrina, 2023) |
| [Global Twitter COVID-19 Dataset](https://github.com/Abrar2652/nlp-roBERTa-biLSTM-attention/blob/main/Extended_datasets/Global_covid_twitter_data/Global.csv) | We extended the existing UK COVID-19 dataset by scraping additional 411885 tweets from 32 English-speaking countries |
| [USA Twitter COVID-19 Dataset](https://github.com/Abrar2652/nlp-roBERTa-biLSTM-attention/blob/main/Extended_datasets/Only_USA_covid_twitter_data/Only_USA.csv) | We extended the existing UK COVID-19 dataset by scraping additional 7500 tweets from only the USA |
| [External Reddit Dataset](https://www.kaggle.com/datasets/cosmos98/twitter-and-reddit-sentimental-analysis-dataset?select=Reddit_Data.csv) | 36801 comments |
| [External Twitter Dataset](https://www.kaggle.com/datasets/cosmos98/twitter-and-reddit-sentimental-analysis-dataset?select=Twitter_Data.csv) | 162980 tweets |
| [External Apple Twitter Dataset](https://www.kaggle.com/datasets/seriousran/appletwittersentimenttexts) | 1630 tweets |
| [External US Airline Twitter Dataset](https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment) | 14640 tweets |

## Classical Models

Qi and Shabrina (2023) benchmarked their UK COVID-19 Twitter dataset's 3000 observations using Random Forest, Multinomial NB, and SVM. We additionally benchmarked the same portion of the dataset using the existing tree-based gradient boosting models (LGBM, CatBoost, XGboost, GBM), RandomForest+KNN+MLP stacking, RandomForestBagging, and RandomForest+GBM voting. The evaluation of these traditional models was performed individually using [CountVectorizer](https://github.com/Abrar2652/nlp-roBERTa-biLSTM-attention/tree/main/BoW), [TF-IDF](https://github.com/Abrar2652/nlp-roBERTa-biLSTM-attention/tree/main/TF-IDF), and [word2vec](https://github.com/Abrar2652/nlp-roBERTa-biLSTM-attention/tree/main/word2vec) tokenizers as the tokenization methods.

We also showed how classical models and ensemble work on the pretrained transformer-based tokeizers: [BERT (classical and ensemble)](https://github.com/Abrar2652/nlp-roBERTa-biLSTM-attention/tree/main/BERT), [roBERTA (classical and ensemble)](https://github.com/Abrar2652/nlp-roBERTa-biLSTM-attention/tree/main/RoBERTa), [Sentence Transformer (classical and ensemble)](https://github.com/Abrar2652/nlp-roBERTa-biLSTM-attention/tree/main/SBERT)

### Pretrained Models

[1] [twitter-roberta-base-sentiment-latest](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest)

[2] [distilbert-base-uncased](https://huggingface.co/distilbert-base-uncased)

[3] [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)

## Deep-Learning Models

All the implemented DL model architectures with their associated codes and outputs can be found in [Twitter-RoBERTa+LSTM](https://github.com/Abrar2652/nlp-roBERTa-biLSTM-attention/tree/main/Twitter-RoBERTa%2BLSTM). Our proposed model [Attention-based biLSTM was trained](https://github.com/Abrar2652/nlp-roBERTa-biLSTM-attention/tree/main/Twitter-RoBERTa%2BLSTM/model4_BiLSTM%2Battention) on Twitter-RoBERTa tokenized inputs.

## XAI
You can find the relevant files in [XAI](https://github.com/Abrar2652/nlp-roBERTa-biLSTM-attention/tree/main/Twitter-RoBERTa%2BLSTM/model4_BiLSTM%2Battention/XAI)

### LIME

![LIME visualization](https://github.com/Abrar2652/nlp-roBERTa-biLSTM-attention/blob/main/Twitter-RoBERTa%2BLSTM/model4_BiLSTM%2Battention/XAI/Lime/lime2.png)

### SHAP
![SHAP visualization](https://github.com/Abrar2652/nlp-roBERTa-biLSTM-attention/blob/main/Twitter-RoBERTa%2BLSTM/model4_BiLSTM%2Battention/XAI/SHAP/shap_neu1.png)

## Requirements
The installation requirements for the Python packages are already included within the Notebooks, which are not discussed here.
### CPU Environment
The Jupyter Notebook can be executed on CPU using Google Colab or Kaggle, but it may take a significant amount of time to obtain the desired outputs.
### GPU Environment
Some of the notebooks were executed using Kaggle's GPU T4x2 and GPU P100. Kaggle provides a GPU quota of 30 hours per week, while Colab has a restricted usage limit.
### TPU Environment
Some of the notebooks were executed on Kaggle's TPU VM v3-8, which proved to be much faster than GPU. Kaggle provides a quota of 20 hours per week for TPU usage. However, the following additional code needs to be added before constructing the neural network model:

```
# detect and init the TPU
tpu = tf.distribute.cluster_resolver.TPUClusterResolver.connect()

# instantiate a distribution strategy
tpu_strategy = tf.distribute.experimental.TPUStrategy(tpu)

# instantiating the model in the strategy scope creates the model on the TPU
with tpu_strategy.scope():
    model = tf.keras.Sequential( … ) # define your model normally
    model.compile( … )

# train model normally
model.fit(training_dataset, epochs=EPOCHS, steps_per_epoch=…)
```

-------

## Directory Tour

Below is an illustration of the directory structure of PDEBench.

```
📁 nlp-roBERTa-biLSTM-attention
└── 📁 BERT
    📁 nlp-roBERTa-biLSTM-attention\BERT
    ├── 📄 all_models1.png
    ├── 📄 all_models2.png
    ├── 📄 all_models3.png
    ├── 📄 all_models4.png
    ├── 📄 lgb_knn_mlp.png
    ├── 📄 rf_knn_mlp.png
└── 📁 BoW
    📁 nlp-roBERTa-biLSTM-attention\BoW
    ├── 📄 all_models_1.png
    ├── 📄 all_models_2.png
    ├── 📄 all_models_3.png
    ├── 📄 all_models_4.png
    ├── 📄 dask_xgb.png
    ├── 📄 rf_bagging.png
    ├── 📄 rf_gb_voting.png
    ├── 📄 rf_knn_mlp.png
└── 📁 Data_scraping
    📁 nlp-roBERTa-biLSTM-attention\Data_scraping
    ├── 📄 Twint-data collection.ipynb
    ├── 📄 Twitter academic api.ipynb
└── 📁 Extended_datasets
    📁 nlp-roBERTa-biLSTM-attention\Extended_datasets
    ├── 📁 Global_covid_twitter_data
    │   📁 nlp-roBERTa-biLSTM-attention\Extended_datasets\Global_covid_twitter_data
    │   ├── 📄 Global.csv
    │   ├── 📄 Global_twitter_data_preprocessing.ipynb
    │   ├── 📄 best-model-global.ipynb
    │   ├── 📄 classification_report1.png
    │   ├── 📄 classification_report2.png
    │   ├── 📄 confusion_matrix.png
    │   ├── 📁 preprocessed_dataset
    │   │   📁 nlp-roBERTa-biLSTM-attention\Extended_datasets\Global_covid_twitter_data\preprocessed_dataset
    │   │   ├── 📄 sample_data_global_0.csv
    │   │   ├── 📄 sample_data_global_1.csv
    │   │   ├── 📄 sample_data_global_10.csv
    │   │   ├── 📄 sample_data_global_11.csv
    │   │   ├── 📄 sample_data_global_12.csv
    │   │   ├── 📄 sample_data_global_13.csv
    │   │   ├── 📄 sample_data_global_14.csv
    │   │   ├── 📄 sample_data_global_15.csv
    │   │   ├── 📄 sample_data_global_16.csv
    │   │   ├── 📄 sample_data_global_17.csv
    │   │   ├── 📄 sample_data_global_18.csv
    │   │   ├── 📄 sample_data_global_19.csv
    │   │   ├── 📄 sample_data_global_2.csv
    │   │   ├── 📄 sample_data_global_20.csv
    │   │   ├── 📄 sample_data_global_21.csv
    │   │   ├── 📄 sample_data_global_22.csv
    │   │   ├── 📄 sample_data_global_23.csv
    │   │   ├── 📄 sample_data_global_24.csv
    │   │   ├── 📄 sample_data_global_25.csv
    │   │   ├── 📄 sample_data_global_26.csv
    │   │   ├── 📄 sample_data_global_27.csv
    │   │   ├── 📄 sample_data_global_28.csv
    │   │   ├── 📄 sample_data_global_29.csv
    │   │   ├── 📄 sample_data_global_3.csv
    │   │   ├── 📄 sample_data_global_30.csv
    │   │   ├── 📄 sample_data_global_31.csv
    │   │   ├── 📄 sample_data_global_32.csv
    │   │   ├── 📄 sample_data_global_33.csv
    │   │   ├── 📄 sample_data_global_34.csv
    │   │   ├── 📄 sample_data_global_35.csv
    │   │   ├── 📄 sample_data_global_36.csv
    │   │   ├── 📄 sample_data_global_37.csv
    │   │   ├── 📄 sample_data_global_38.csv
    │   │   ├── 📄 sample_data_global_39.csv
    │   │   ├── 📄 sample_data_global_4.csv
    │   │   ├── 📄 sample_data_global_40.csv
    │   │   ├── 📄 sample_data_global_5.csv
    │   │   ├── 📄 sample_data_global_6.csv
    │   │   ├── 📄 sample_data_global_7.csv
    │   │   ├── 📄 sample_data_global_8.csv
    │   │   ├── 📄 sample_data_global_9.csv
    │   ├── 📄 tweets_distribution_global.png
    │   ├── 📄 word_cloud_global.png
    │   ├── 📄 word_freq.png
    ├── 📁 Only_USA_covid_twitter_data
    │   📁 nlp-roBERTa-biLSTM-attention\Extended_datasets\Only_USA_covid_twitter_data
    │   └── 📄 Only_USA.csv
    │   └── 📄 frequency.png
    │   └── 📁 model3_attention
    │       📁 nlp-roBERTa-biLSTM-attention\Extended_datasets\Only_USA_covid_twitter_data\model3_attention
    │       ├── 📄 accuracy.png
    │       ├── 📄 best-model-only-usa.ipynb
    │       ├── 📄 classification_report1.png
    │       ├── 📄 classification_reports.png
    │       ├── 📄 confusion_matrix.png
    │       ├── 📄 loss.png
    │       ├── 📄 model_architecture.png
    │   └── 📄 only_USA_twitter_data_preprocessing.ipynb
    │   └── 📄 sample_data_only_USA.csv
    │   └── 📄 uk_covid_twitter_sentiment.ipynb
    │   └── 📄 word_cloud.png
└── 📁 External_datasets
    📁 nlp-roBERTa-biLSTM-attention\External_datasets
    ├── 📁 Apple_twitter_sentiments
    │   📁 nlp-roBERTa-biLSTM-attention\External_datasets\Apple_twitter_sentiments
    │   ├── 📄 accuracy.png
    │   ├── 📄 best-model-apple-twitter.ipynb
    │   ├── 📄 classification_reports1.png
    │   ├── 📄 classification_reports2.png
    │   ├── 📄 confusion_matrix.png
    │   ├── 📄 loss.png
    ├── 📁 Reddit
    │   📁 nlp-roBERTa-biLSTM-attention\External_datasets\Reddit
    │   ├── 📄 Reddit_Data.csv
    │   ├── 📄 Screenshot 2023-05-08 025117.png
    │   ├── 📄 Screenshot 2023-05-08 025141.png
    │   ├── 📄 Screenshot 2023-05-08 025820.png
    │   ├── 📄 Screenshot 2023-05-08 025915.png
    │   ├── 📄 Screenshot 2023-05-08 025934.png
    │   ├── 📄 Screenshot 2023-05-08 025955.png
    │   ├── 📄 Screenshot 2023-05-08 030042.png
    │   ├── 📄 best-model-reddit.ipynb
    │   ├── 📄 classification_reports.png
    │   ├── 📄 cm.png
    ├── 📁 Twitter
    │   📁 nlp-roBERTa-biLSTM-attention\External_datasets\Twitter
    │   ├── 📄 Twitter_Data.csv
    │   ├── 📄 best-model-twitter-external.ipynb
    │   ├── 📄 classification_report1.png
    │   ├── 📄 classification_report2.png
    │   ├── 📄 confusion_matrix.png
    ├── 📁 US_airlines_twitter_sentiments
    │   📁 nlp-roBERTa-biLSTM-attention\External_datasets\US_airlines_twitter_sentiments
    │   ├── 📄 accuracy.png
    │   ├── 📄 best-model-us-airlines.ipynb
    │   ├── 📄 classification_report1.png
    │   ├── 📄 classification_report2.png
    │   ├── 📄 confusion_matrix.png
    │   ├── 📄 loss.png
    ├── 📄 token.txt
└── 📄 LICENSE
└── 📁 Previous_research
    📁 nlp-roBERTa-biLSTM-attention\Previous_research
    ├── 📄 1.png
    ├── 📄 2.png
    ├── 📄 Vaibhav 2022.pdf
    ├── 📄 Yuxing 2023.pdf
└── 📄 README.md
└── 📁 RoBERTa
    📁 nlp-roBERTa-biLSTM-attention\RoBERTa
    ├── 📄 cardiff_all_models_1.png
    ├── 📄 cardiff_all_models_2.png
    ├── 📄 cardiff_all_models_3.png
    ├── 📄 cardiff_all_models_4.png
    ├── 📄 lgb+knn+mlp.png
    ├── 📄 roberta_base_rf+knn+mlp.png
└── 📁 SBERT
    📁 nlp-roBERTa-biLSTM-attention\SBERT
    ├── 📄 all_models_1.png
    ├── 📄 all_models_2.png
    ├── 📄 all_models_3.png
    ├── 📄 all_models_4.png
    ├── 📄 all_models_5.png
    ├── 📄 lgb_knn_mlp.png
    ├── 📄 rf_knn_mlp.png
└── 📁 TF-IDF
    📁 nlp-roBERTa-biLSTM-attention\TF-IDF
    ├── 📄 all_models_1.png
    ├── 📄 all_models_2.png
    ├── 📄 all_models_3.png
    ├── 📄 all_models_4.png
    ├── 📄 rf_bagging.png
    ├── 📄 rf_knn_mlp.png
    ├── 📄 rf_stacking_voting.png
└── 📁 Target_lexicon_selection
    📁 nlp-roBERTa-biLSTM-attention\Target_lexicon_selection
    ├── 📄 target_lexicon_selection.ipynb
    ├── 📄 textblob1.png
    ├── 📄 textblob2.png
    ├── 📄 textblob3.png
    ├── 📄 textblob4.png
    ├── 📄 vader1.png
    ├── 📄 vader2.png
    ├── 📄 vader3.png
    ├── 📄 vader4.png
    ├── 📄 wordnet1.png
    ├── 📄 wordnet2.png
    ├── 📄 wordnet3.png
    ├── 📄 wordnet4.png
└── 📁 Twitter-RoBERTa+LSTM
    📁 nlp-roBERTa-biLSTM-attention\Twitter-RoBERTa+LSTM
    ├── 📁 BiLSTM+CNN
    │   📁 nlp-roBERTa-biLSTM-attention\Twitter-RoBERTa+LSTM\BiLSTM+CNN
    │   ├── 📄 accuracy.png
    │   ├── 📄 biLSTM+CNN.ipynb
    │   ├── 📄 classification_report.png
    │   ├── 📄 confusion_matrix.png
    │   ├── 📄 loss.png
    │   ├── 📄 model_architecture.png
    ├── 📁 model1_keras_1_dense_layers
    │   📁 nlp-roBERTa-biLSTM-attention\Twitter-RoBERTa+LSTM\model1_keras_1_dense_layers
    │   ├── 📄 Screenshot 2023-04-20 215305.png
    │   ├── 📄 accuracy1.png
    │   ├── 📄 classification_report.png
    │   ├── 📄 confusion_matrix.png
    │   ├── 📄 loss1.png
    │   ├── 📄 model_architecture.png
    │   ├── 📄 summary.png
    ├── 📁 model2_keras_3_dense_layers
    │   📁 nlp-roBERTa-biLSTM-attention\Twitter-RoBERTa+LSTM\model2_keras_3_dense_layers
    │   ├── 📄 accuracy1.png
    │   ├── 📄 classification_report.png
    │   ├── 📄 classification_report1.png
    │   ├── 📄 confusion_matrix.png
    │   ├── 📄 loss1.png
    │   ├── 📄 model_architecture.png
    │   ├── 📄 model_summary.png
    │   ├── 📄 train_val_loss.png
    ├── 📁 model3_BiLSTM
    │   📁 nlp-roBERTa-biLSTM-attention\Twitter-RoBERTa+LSTM\model3_BiLSTM
    │   ├── 📄 accuracy.png
    │   ├── 📄 classification_report1.png
    │   ├── 📄 classification_report2.png
    │   ├── 📄 confusion_matrix.png
    │   ├── 📄 loss.png
    │   ├── 📄 lr_vs_epoch.png
    │   ├── 📄 model_architecture.png
    │   ├── 📄 summary.png
    │   ├── 📄 target_val_counts.png
    │   ├── 📄 train_acc_vs_lr.png
    │   ├── 📄 train_loss_vs_lr.png
    │   ├── 📄 training_val.png
    │   ├── 📄 val_acc_vs_lr.png
    │   ├── 📄 val_loss_vs_lr.png
    ├── 📁 model4_BiLSTM+attention
    │   📁 nlp-roBERTa-biLSTM-attention\Twitter-RoBERTa+LSTM\model4_BiLSTM+attention
    │   └── 📁 XAI
    │       📁 nlp-roBERTa-biLSTM-attention\Twitter-RoBERTa+LSTM\model4_BiLSTM+attention\XAI
    │       ├── 📁 Lime
    │       │   📁 nlp-roBERTa-biLSTM-attention\Twitter-RoBERTa+LSTM\model4_BiLSTM+attention\XAI\Lime
    │       │   ├── 📄 lime1.png
    │       │   ├── 📄 lime2.png
    │       │   ├── 📄 lime3.png
    │       │   ├── 📄 lime4.png
    │       │   ├── 📄 lime5.png
    │       │   ├── 📄 lime6.png
    │       │   ├── 📄 lime7.png
    │       ├── 📁 SHAP
    │       │   📁 nlp-roBERTa-biLSTM-attention\Twitter-RoBERTa+LSTM\model4_BiLSTM+attention\XAI\SHAP
    │       │   └── 📄 shap_neg1.png
    │       │   └── 📄 shap_neg2.png
    │       │   └── 📄 shap_neg_bar_ascending.png
    │       │   └── 📄 shap_neg_bar_descending.png
    │       │   └── 📄 shap_neu1.png
    │       │   └── 📄 shap_neu2.png
    │       │   └── 📄 shap_neu_bar.png
    │       │   └── 📄 shap_neu_bar_ascending.png
    │       │   └── 📄 shap_neu_bar_descending.png
    │       │   └── 📄 shap_pos1.png
    │       │   └── 📄 shap_pos2.png
    │       │   └── 📄 shap_pos_bar_ascending.png
    │       │   └── 📄 shap_pos_bar_descending.png
    │   └── 📄 learning_rates.png
    │   └── 📄 model_architecture.png
    │   └── 📄 summary.png
    │   └── 📁 uk_twitter_data_3k
    │       📁 nlp-roBERTa-biLSTM-attention\Twitter-RoBERTa+LSTM\model4_BiLSTM+attention\uk_twitter_data_3k
    │       ├── 📄 accuracy.png
    │       ├── 📄 best-model_uk-tweet_3k.ipynb
    │       ├── 📄 classification_report.png
    │       ├── 📄 classification_report2.png
    │       ├── 📄 confusion_matrix.png
    │       ├── 📄 loss.png
    │       ├── 📄 train_val_loss.png
    │   └── 📁 uk_twitter_data_all
    │       📁 nlp-roBERTa-biLSTM-attention\Twitter-RoBERTa+LSTM\model4_BiLSTM+attention\uk_twitter_data_all
    │       └── 📄 accuracy.png
    │       └── 📄 best-model-uk-twitter-all.ipynb
    │       └── 📄 classification_report1.png
    │       └── 📄 classification_report2.png
    │       └── 📄 confusion_matrix.png
    │       └── 📄 loss.png
└── 📁 UK_covid_twitter_data
    📁 nlp-roBERTa-biLSTM-attention\UK_covid_twitter_data
    ├── 📄 all_cities.csv
    ├── 📄 sample_data_3000.csv
    ├── 📄 sample_data_all.csv
    ├── 📄 stacked bar graph.png
    ├── 📄 tweets distribution.png
    ├── 📄 uk_twitter_data_preprocessing.ipynb
└── 📄 list.md
└── 📄 uk-twitter-3k-classical-modelling.ipynb
└── 📁 word2vec
    📁 nlp-roBERTa-biLSTM-attention\word2vec
    └── 📄 all_models_1.png
    └── 📄 all_models_2.png
    └── 📄 all_models_3.png
    └── 📄 all_models_4.png
    └── 📄 rf_knn_mlp.png
    └── 📄 rf_stacking_voting.png


```


------

## Citations for datasets

#### Kaggle
```
 @misc{md abrar jahin_2023,
	title={Extended Covid Twitter Datasets},
	url={https://www.kaggle.com/ds/3205649},
	DOI={10.34740/KAGGLE/DS/3205649},
	publisher={Kaggle},
	author={Md Abrar Jahin},
	year={2023}
}
```
#### Mendeley
```
Jahin, Md Abrar (2023), “Extended Covid Twitter Datasets”, Mendeley Data, V1, doi: 10.17632/2ynwykrfgf.1
```


## License

MIT licensed, except where otherwise stated.
See `LICENSE.txt` file.





