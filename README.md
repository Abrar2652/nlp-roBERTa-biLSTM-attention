# nlp-biLSTM-attention

The code repository for the Nature Scientific Reports (2023) paper 
[Interpretable Sentiment Analysis of COVID-19 Tweets using Attention-based BiLSTM and Twitter-RoBERTa](https://arxiv.org/abs/2210.07182)



<!---
![Visualizations of some COVID-19 tweets](https://github.com/pdebench/PDEBench/blob/main/pdebench_examples.PNG)
--->


Created and maintained by Md Abrar Jahin `<abrar.jahin.2652@gmail.com, md-jahin@oist.jp>`.

## Datasets

*Extended Datasets*

Each dataset contains a column of cleaned tweets which was obtained by preprocessing the raw tweets and comments, accompanied by sentiment label of negative (-1), neutral (0), and posititve (1). 

*External Datasets*

External Tweets and Comments were made on Narendra Modi and other leaders as well as people's opinion towards the next prime minister of India (in the context with general elections held in India - 2019). The external datasets were created with the help of the Tweepy and Reddit Apis. 

| Datasets  | Description |
| ------------- | ------------- |
| [UK Twitter COVID-19 Dataset](https://github.com/Abrar2652/nlp-roBERTa-biLSTM-attention/blob/main/UK_covid_twitter_data/sample_data_all.csv) | This dataset was 
developed by collecting COVID-19 tweets from only the major cities in the UK (Qi and Shabrina, 2023) |
| [Global Twitter COVID-19 Dataset](https://github.com/Abrar2652/nlp-roBERTa-biLSTM-attention/blob/main/Extended_datasets/Global_covid_twitter_data/Global.csv) | We extended the existing UK COVID-19 dataset by scraping additional 411885 tweets from 32 English-speaking countries |
| [USA Twitter COVID-19 Dataset](https://github.com/Abrar2652/nlp-roBERTa-biLSTM-attention/blob/main/Extended_datasets/Only_USA_covid_twitter_data/Only_USA.csv) | We extended the existing UK COVID-19 dataset by scraping additional 7500 tweets from only the USA |
| [External Reddit Dataset](https://www.kaggle.com/datasets/cosmos98/twitter-and-reddit-sentimental-analysis-dataset?select=Reddit_Data.csv) | 36801 comments |
| [External Twitter Dataset](https://www.kaggle.com/datasets/cosmos98/twitter-and-reddit-sentimental-analysis-dataset?select=Twitter_Data.csv) | 162980 tweets |
| [External Apple Twitter Dataset](https://www.kaggle.com/datasets/seriousran/appletwittersentimenttexts) | 1630 comments |
| [External US Airline Twitter Dataset](https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment) | 14640 comments |


## Pretrained Models

[1] [twitter-roberta-base-sentiment-latest](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest)

[2] [distilbert-base-uncased](https://huggingface.co/distilbert-base-uncased)

[3] [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)

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
- 📂 __D:\\GitHub\\nlp\-roBERTa\-biLSTM\-attention__
   - 📂 __BERT__
     - 📄 [all\_models1.png](BERT/all_models1.png)
     - 📄 [all\_models2.png](BERT/all_models2.png)
     - 📄 [all\_models3.png](BERT/all_models3.png)
     - 📄 [all\_models4.png](BERT/all_models4.png)
     - 📄 [lgb\_knn\_mlp.png](BERT/lgb_knn_mlp.png)
     - 📄 [rf\_knn\_mlp.png](BERT/rf_knn_mlp.png)
   - 📂 __BoW__
     - 📄 [all\_models\_1.png](BoW/all_models_1.png)
     - 📄 [all\_models\_2.png](BoW/all_models_2.png)
     - 📄 [all\_models\_3.png](BoW/all_models_3.png)
     - 📄 [all\_models\_4.png](BoW/all_models_4.png)
     - 📄 [dask\_xgb.png](BoW/dask_xgb.png)
     - 📄 [rf\_bagging.png](BoW/rf_bagging.png)
     - 📄 [rf\_gb\_voting.png](BoW/rf_gb_voting.png)
     - 📄 [rf\_knn\_mlp.png](BoW/rf_knn_mlp.png)
   - 📂 __Data\_scraping__
     - 📄 [Twint\-data collection.ipynb](Data_scraping/Twint-data%20collection.ipynb)
     - 📄 [Twitter academic api.ipynb](Data_scraping/Twitter%20academic%20api.ipynb)
   - 📂 __Extended\_datasets__
     - 📂 __Global\_covid\_twitter\_data__
       - 📄 [Global.csv](Extended_datasets/Global_covid_twitter_data/Global.csv)
       - 📄 [Global\_twitter\_data\_preprocessing.ipynb](Extended_datasets/Global_covid_twitter_data/Global_twitter_data_preprocessing.ipynb)
       - 📄 [best\-model\-global.ipynb](Extended_datasets/Global_covid_twitter_data/best-model-global.ipynb)
       - 📄 [classification\_report1.png](Extended_datasets/Global_covid_twitter_data/classification_report1.png)
       - 📄 [classification\_report2.png](Extended_datasets/Global_covid_twitter_data/classification_report2.png)
       - 📄 [confusion\_matrix.png](Extended_datasets/Global_covid_twitter_data/confusion_matrix.png)
       - 📂 __preprocessed\_dataset__
         - 📄 [sample\_data\_global\_0.csv](Extended_datasets/Global_covid_twitter_data/preprocessed_dataset/sample_data_global_0.csv)
         - 📄 [sample\_data\_global\_1.csv](Extended_datasets/Global_covid_twitter_data/preprocessed_dataset/sample_data_global_1.csv)
         - 📄 [sample\_data\_global\_10.csv](Extended_datasets/Global_covid_twitter_data/preprocessed_dataset/sample_data_global_10.csv)
         - 📄 [sample\_data\_global\_11.csv](Extended_datasets/Global_covid_twitter_data/preprocessed_dataset/sample_data_global_11.csv)
         - 📄 [sample\_data\_global\_12.csv](Extended_datasets/Global_covid_twitter_data/preprocessed_dataset/sample_data_global_12.csv)
         - 📄 [sample\_data\_global\_13.csv](Extended_datasets/Global_covid_twitter_data/preprocessed_dataset/sample_data_global_13.csv)
         - 📄 [sample\_data\_global\_14.csv](Extended_datasets/Global_covid_twitter_data/preprocessed_dataset/sample_data_global_14.csv)
         - 📄 [sample\_data\_global\_15.csv](Extended_datasets/Global_covid_twitter_data/preprocessed_dataset/sample_data_global_15.csv)
         - 📄 [sample\_data\_global\_16.csv](Extended_datasets/Global_covid_twitter_data/preprocessed_dataset/sample_data_global_16.csv)
         - 📄 [sample\_data\_global\_17.csv](Extended_datasets/Global_covid_twitter_data/preprocessed_dataset/sample_data_global_17.csv)
         - 📄 [sample\_data\_global\_18.csv](Extended_datasets/Global_covid_twitter_data/preprocessed_dataset/sample_data_global_18.csv)
         - 📄 [sample\_data\_global\_19.csv](Extended_datasets/Global_covid_twitter_data/preprocessed_dataset/sample_data_global_19.csv)
         - 📄 [sample\_data\_global\_2.csv](Extended_datasets/Global_covid_twitter_data/preprocessed_dataset/sample_data_global_2.csv)
         - 📄 [sample\_data\_global\_20.csv](Extended_datasets/Global_covid_twitter_data/preprocessed_dataset/sample_data_global_20.csv)
         - 📄 [sample\_data\_global\_21.csv](Extended_datasets/Global_covid_twitter_data/preprocessed_dataset/sample_data_global_21.csv)
         - 📄 [sample\_data\_global\_22.csv](Extended_datasets/Global_covid_twitter_data/preprocessed_dataset/sample_data_global_22.csv)
         - 📄 [sample\_data\_global\_23.csv](Extended_datasets/Global_covid_twitter_data/preprocessed_dataset/sample_data_global_23.csv)
         - 📄 [sample\_data\_global\_24.csv](Extended_datasets/Global_covid_twitter_data/preprocessed_dataset/sample_data_global_24.csv)
         - 📄 [sample\_data\_global\_25.csv](Extended_datasets/Global_covid_twitter_data/preprocessed_dataset/sample_data_global_25.csv)
         - 📄 [sample\_data\_global\_26.csv](Extended_datasets/Global_covid_twitter_data/preprocessed_dataset/sample_data_global_26.csv)
         - 📄 [sample\_data\_global\_27.csv](Extended_datasets/Global_covid_twitter_data/preprocessed_dataset/sample_data_global_27.csv)
         - 📄 [sample\_data\_global\_28.csv](Extended_datasets/Global_covid_twitter_data/preprocessed_dataset/sample_data_global_28.csv)
         - 📄 [sample\_data\_global\_29.csv](Extended_datasets/Global_covid_twitter_data/preprocessed_dataset/sample_data_global_29.csv)
         - 📄 [sample\_data\_global\_3.csv](Extended_datasets/Global_covid_twitter_data/preprocessed_dataset/sample_data_global_3.csv)
         - 📄 [sample\_data\_global\_30.csv](Extended_datasets/Global_covid_twitter_data/preprocessed_dataset/sample_data_global_30.csv)
         - 📄 [sample\_data\_global\_31.csv](Extended_datasets/Global_covid_twitter_data/preprocessed_dataset/sample_data_global_31.csv)
         - 📄 [sample\_data\_global\_32.csv](Extended_datasets/Global_covid_twitter_data/preprocessed_dataset/sample_data_global_32.csv)
         - 📄 [sample\_data\_global\_33.csv](Extended_datasets/Global_covid_twitter_data/preprocessed_dataset/sample_data_global_33.csv)
         - 📄 [sample\_data\_global\_34.csv](Extended_datasets/Global_covid_twitter_data/preprocessed_dataset/sample_data_global_34.csv)
         - 📄 [sample\_data\_global\_35.csv](Extended_datasets/Global_covid_twitter_data/preprocessed_dataset/sample_data_global_35.csv)
         - 📄 [sample\_data\_global\_36.csv](Extended_datasets/Global_covid_twitter_data/preprocessed_dataset/sample_data_global_36.csv)
         - 📄 [sample\_data\_global\_37.csv](Extended_datasets/Global_covid_twitter_data/preprocessed_dataset/sample_data_global_37.csv)
         - 📄 [sample\_data\_global\_38.csv](Extended_datasets/Global_covid_twitter_data/preprocessed_dataset/sample_data_global_38.csv)
         - 📄 [sample\_data\_global\_39.csv](Extended_datasets/Global_covid_twitter_data/preprocessed_dataset/sample_data_global_39.csv)
         - 📄 [sample\_data\_global\_4.csv](Extended_datasets/Global_covid_twitter_data/preprocessed_dataset/sample_data_global_4.csv)
         - 📄 [sample\_data\_global\_40.csv](Extended_datasets/Global_covid_twitter_data/preprocessed_dataset/sample_data_global_40.csv)
         - 📄 [sample\_data\_global\_5.csv](Extended_datasets/Global_covid_twitter_data/preprocessed_dataset/sample_data_global_5.csv)
         - 📄 [sample\_data\_global\_6.csv](Extended_datasets/Global_covid_twitter_data/preprocessed_dataset/sample_data_global_6.csv)
         - 📄 [sample\_data\_global\_7.csv](Extended_datasets/Global_covid_twitter_data/preprocessed_dataset/sample_data_global_7.csv)
         - 📄 [sample\_data\_global\_8.csv](Extended_datasets/Global_covid_twitter_data/preprocessed_dataset/sample_data_global_8.csv)
         - 📄 [sample\_data\_global\_9.csv](Extended_datasets/Global_covid_twitter_data/preprocessed_dataset/sample_data_global_9.csv)
       - 📄 [tweets\_distribution\_global.png](Extended_datasets/Global_covid_twitter_data/tweets_distribution_global.png)
       - 📄 [word\_cloud\_global.png](Extended_datasets/Global_covid_twitter_data/word_cloud_global.png)
       - 📄 [word\_freq.png](Extended_datasets/Global_covid_twitter_data/word_freq.png)
     - 📂 __Only\_USA\_covid\_twitter\_data__
       - 📄 [Only\_USA.csv](Extended_datasets/Only_USA_covid_twitter_data/Only_USA.csv)
       - 📄 [frequency.png](Extended_datasets/Only_USA_covid_twitter_data/frequency.png)
       - 📂 __model3\_attention__
         - 📄 [accuracy.png](Extended_datasets/Only_USA_covid_twitter_data/model3_attention/accuracy.png)
         - 📄 [best\-model\-only\-usa.ipynb](Extended_datasets/Only_USA_covid_twitter_data/model3_attention/best-model-only-usa.ipynb)
         - 📄 [classification\_report1.png](Extended_datasets/Only_USA_covid_twitter_data/model3_attention/classification_report1.png)
         - 📄 [classification\_reports.png](Extended_datasets/Only_USA_covid_twitter_data/model3_attention/classification_reports.png)
         - 📄 [confusion\_matrix.png](Extended_datasets/Only_USA_covid_twitter_data/model3_attention/confusion_matrix.png)
         - 📄 [loss.png](Extended_datasets/Only_USA_covid_twitter_data/model3_attention/loss.png)
         - 📄 [model\_architecture.png](Extended_datasets/Only_USA_covid_twitter_data/model3_attention/model_architecture.png)
       - 📄 [only\_USA\_twitter\_data\_preprocessing.ipynb](Extended_datasets/Only_USA_covid_twitter_data/only_USA_twitter_data_preprocessing.ipynb)
       - 📄 [sample\_data\_only\_USA.csv](Extended_datasets/Only_USA_covid_twitter_data/sample_data_only_USA.csv)
       - 📄 [uk\_covid\_twitter\_sentiment.ipynb](Extended_datasets/Only_USA_covid_twitter_data/uk_covid_twitter_sentiment.ipynb)
       - 📄 [word\_cloud.png](Extended_datasets/Only_USA_covid_twitter_data/word_cloud.png)
   - 📂 __External\_datasets__
     - 📂 __Reddit__
       - 📄 [Reddit\_Data.csv](External_datasets/Reddit/Reddit_Data.csv)
       - 📄 [Screenshot 2023\-05\-08 025117.png](External_datasets/Reddit/Screenshot%202023-05-08%20025117.png)
       - 📄 [Screenshot 2023\-05\-08 025141.png](External_datasets/Reddit/Screenshot%202023-05-08%20025141.png)
       - 📄 [Screenshot 2023\-05\-08 025820.png](External_datasets/Reddit/Screenshot%202023-05-08%20025820.png)
       - 📄 [Screenshot 2023\-05\-08 025915.png](External_datasets/Reddit/Screenshot%202023-05-08%20025915.png)
       - 📄 [Screenshot 2023\-05\-08 025934.png](External_datasets/Reddit/Screenshot%202023-05-08%20025934.png)
       - 📄 [Screenshot 2023\-05\-08 025955.png](External_datasets/Reddit/Screenshot%202023-05-08%20025955.png)
       - 📄 [Screenshot 2023\-05\-08 030042.png](External_datasets/Reddit/Screenshot%202023-05-08%20030042.png)
       - 📄 [best\-model\-reddit.ipynb](External_datasets/Reddit/best-model-reddit.ipynb)
       - 📄 [classification\_reports.png](External_datasets/Reddit/classification_reports.png)
       - 📄 [cm.png](External_datasets/Reddit/cm.png)
     - 📂 __Twitter__
       - 📄 [Twitter\_Data.csv](External_datasets/Twitter/Twitter_Data.csv)
       - 📄 [best\-model\-twitter\-external.ipynb](External_datasets/Twitter/best-model-twitter-external.ipynb)
       - 📄 [cm.png](External_datasets/Twitter/cm.png)
       - 📄 [cr.png](External_datasets/Twitter/cr.png)
       - 📄 [cr2.png](External_datasets/Twitter/cr2.png)
     - 📄 [token.txt](External_datasets/token.txt)
   - 📄 [LICENSE](LICENSE)
   - 📂 __Previous\_research__
     - 📄 [1.png](Previous_research/1.png)
     - 📄 [2.png](Previous_research/2.png)
     - 📄 [Vaibhav 2022.pdf](Previous_research/Vaibhav%202022.pdf)
     - 📄 [Yuxing 2023.pdf](Previous_research/Yuxing%202023.pdf)
   - 📄 [README.md](README.md)
   - 📂 __RoBERTa__
     - 📄 [cardiff\_all\_models\_1.png](RoBERTa/cardiff_all_models_1.png)
     - 📄 [cardiff\_all\_models\_2.png](RoBERTa/cardiff_all_models_2.png)
     - 📄 [cardiff\_all\_models\_3.png](RoBERTa/cardiff_all_models_3.png)
     - 📄 [cardiff\_all\_models\_4.png](RoBERTa/cardiff_all_models_4.png)
     - 📄 [lgb+knn+mlp.png](RoBERTa/lgb%2Bknn%2Bmlp.png)
     - 📄 [roberta\_base\_rf+knn+mlp.png](RoBERTa/roberta_base_rf%2Bknn%2Bmlp.png)
   - 📂 __SBERT__
     - 📄 [all\_models\_1.png](SBERT/all_models_1.png)
     - 📄 [all\_models\_2.png](SBERT/all_models_2.png)
     - 📄 [all\_models\_3.png](SBERT/all_models_3.png)
     - 📄 [all\_models\_4.png](SBERT/all_models_4.png)
     - 📄 [all\_models\_5.png](SBERT/all_models_5.png)
     - 📄 [lgb\_knn\_mlp.png](SBERT/lgb_knn_mlp.png)
     - 📄 [rf\_knn\_mlp.png](SBERT/rf_knn_mlp.png)
   - 📂 __TF\-IDF__
     - 📄 [all\_models\_1.png](TF-IDF/all_models_1.png)
     - 📄 [all\_models\_2.png](TF-IDF/all_models_2.png)
     - 📄 [all\_models\_3.png](TF-IDF/all_models_3.png)
     - 📄 [all\_models\_4.png](TF-IDF/all_models_4.png)
     - 📄 [rf\_bagging.png](TF-IDF/rf_bagging.png)
     - 📄 [rf\_knn\_mlp.png](TF-IDF/rf_knn_mlp.png)
     - 📄 [rf\_stacking\_voting.png](TF-IDF/rf_stacking_voting.png)
   - 📂 __Twitter\-RoBERTa+LSTM__
     - 📂 __BiLSTM+CNN__
       - 📄 [accuracy.png](Twitter-RoBERTa+LSTM/BiLSTM+CNN/accuracy.png)
       - 📄 [biLSTM+CNN.ipynb](Twitter-RoBERTa+LSTM/BiLSTM+CNN/biLSTM%2BCNN.ipynb)
       - 📄 [classification\_report.png](Twitter-RoBERTa+LSTM/BiLSTM+CNN/classification_report.png)
       - 📄 [confusion\_matrix.png](Twitter-RoBERTa+LSTM/BiLSTM+CNN/confusion_matrix.png)
       - 📄 [loss.png](Twitter-RoBERTa+LSTM/BiLSTM+CNN/loss.png)
       - 📄 [model\_architecture.png](Twitter-RoBERTa+LSTM/BiLSTM+CNN/model_architecture.png)
     - 📂 __model1\_keras\_1\_dense\_layers__
       - 📄 [Screenshot 2023\-04\-20 215305.png](Twitter-RoBERTa+LSTM/model1_keras_1_dense_layers/Screenshot%202023-04-20%20215305.png)
       - 📄 [accuracy1.png](Twitter-RoBERTa+LSTM/model1_keras_1_dense_layers/accuracy1.png)
       - 📄 [classification\_report.png](Twitter-RoBERTa+LSTM/model1_keras_1_dense_layers/classification_report.png)
       - 📄 [confusion\_matrix.png](Twitter-RoBERTa+LSTM/model1_keras_1_dense_layers/confusion_matrix.png)
       - 📄 [loss1.png](Twitter-RoBERTa+LSTM/model1_keras_1_dense_layers/loss1.png)
       - 📄 [model\_architecture.png](Twitter-RoBERTa+LSTM/model1_keras_1_dense_layers/model_architecture.png)
       - 📄 [summary.png](Twitter-RoBERTa+LSTM/model1_keras_1_dense_layers/summary.png)
     - 📂 __model2\_keras\_3\_dense\_layers__
       - 📄 [accuracy1.png](Twitter-RoBERTa+LSTM/model2_keras_3_dense_layers/accuracy1.png)
       - 📄 [classification\_report.png](Twitter-RoBERTa+LSTM/model2_keras_3_dense_layers/classification_report.png)
       - 📄 [classification\_report1.png](Twitter-RoBERTa+LSTM/model2_keras_3_dense_layers/classification_report1.png)
       - 📄 [confusion\_matrix.png](Twitter-RoBERTa+LSTM/model2_keras_3_dense_layers/confusion_matrix.png)
       - 📄 [loss1.png](Twitter-RoBERTa+LSTM/model2_keras_3_dense_layers/loss1.png)
       - 📄 [model\_architecture.png](Twitter-RoBERTa+LSTM/model2_keras_3_dense_layers/model_architecture.png)
       - 📄 [model\_summary.png](Twitter-RoBERTa+LSTM/model2_keras_3_dense_layers/model_summary.png)
       - 📄 [train\_val\_loss.png](Twitter-RoBERTa+LSTM/model2_keras_3_dense_layers/train_val_loss.png)
     - 📂 __model3\_BiLSTM__
       - 📄 [accuracy.png](Twitter-RoBERTa+LSTM/model3_BiLSTM/accuracy.png)
       - 📄 [classification\_report1.png](Twitter-RoBERTa+LSTM/model3_BiLSTM/classification_report1.png)
       - 📄 [classification\_report2.png](Twitter-RoBERTa+LSTM/model3_BiLSTM/classification_report2.png)
       - 📄 [confusion\_matrix.png](Twitter-RoBERTa+LSTM/model3_BiLSTM/confusion_matrix.png)
       - 📄 [loss.png](Twitter-RoBERTa+LSTM/model3_BiLSTM/loss.png)
       - 📄 [lr\_vs\_epoch.png](Twitter-RoBERTa+LSTM/model3_BiLSTM/lr_vs_epoch.png)
       - 📄 [model\_architecture.png](Twitter-RoBERTa+LSTM/model3_BiLSTM/model_architecture.png)
       - 📄 [summary.png](Twitter-RoBERTa+LSTM/model3_BiLSTM/summary.png)
       - 📄 [target\_val\_counts.png](Twitter-RoBERTa+LSTM/model3_BiLSTM/target_val_counts.png)
       - 📄 [train\_acc\_vs\_lr.png](Twitter-RoBERTa+LSTM/model3_BiLSTM/train_acc_vs_lr.png)
       - 📄 [train\_loss\_vs\_lr.png](Twitter-RoBERTa+LSTM/model3_BiLSTM/train_loss_vs_lr.png)
       - 📄 [training\_val.png](Twitter-RoBERTa+LSTM/model3_BiLSTM/training_val.png)
       - 📄 [val\_acc\_vs\_lr.png](Twitter-RoBERTa+LSTM/model3_BiLSTM/val_acc_vs_lr.png)
       - 📄 [val\_loss\_vs\_lr.png](Twitter-RoBERTa+LSTM/model3_BiLSTM/val_loss_vs_lr.png)
     - 📂 __model4\_BiLSTM+attention__
       - 📄 [learning\_rates.png](Twitter-RoBERTa+LSTM/model4_BiLSTM+attention/learning_rates.png)
       - 📄 [lime1.png](Twitter-RoBERTa+LSTM/model4_BiLSTM+attention/lime1.png)
       - 📄 [lime2.png](Twitter-RoBERTa+LSTM/model4_BiLSTM+attention/lime2.png)
       - 📄 [lime3.png](Twitter-RoBERTa+LSTM/model4_BiLSTM+attention/lime3.png)
       - 📄 [lime4.png](Twitter-RoBERTa+LSTM/model4_BiLSTM+attention/lime4.png)
       - 📄 [lime5.png](Twitter-RoBERTa+LSTM/model4_BiLSTM+attention/lime5.png)
       - 📄 [lime6.png](Twitter-RoBERTa+LSTM/model4_BiLSTM+attention/lime6.png)
       - 📄 [lime7.png](Twitter-RoBERTa+LSTM/model4_BiLSTM+attention/lime7.png)
       - 📄 [model\_architecture.png](Twitter-RoBERTa+LSTM/model4_BiLSTM+attention/model_architecture.png)
       - 📄 [shap\_neg1.png](Twitter-RoBERTa+LSTM/model4_BiLSTM+attention/shap_neg1.png)
       - 📄 [shap\_neg2.png](Twitter-RoBERTa+LSTM/model4_BiLSTM+attention/shap_neg2.png)
       - 📄 [shap\_neg\_bar\_ascending.png](Twitter-RoBERTa+LSTM/model4_BiLSTM+attention/shap_neg_bar_ascending.png)
       - 📄 [shap\_neg\_bar\_descending.png](Twitter-RoBERTa+LSTM/model4_BiLSTM+attention/shap_neg_bar_descending.png)
       - 📄 [shap\_neu1.png](Twitter-RoBERTa+LSTM/model4_BiLSTM+attention/shap_neu1.png)
       - 📄 [shap\_neu2.png](Twitter-RoBERTa+LSTM/model4_BiLSTM+attention/shap_neu2.png)
       - 📄 [shap\_neu\_bar.png](Twitter-RoBERTa+LSTM/model4_BiLSTM+attention/shap_neu_bar.png)
       - 📄 [shap\_neu\_bar\_ascending.png](Twitter-RoBERTa+LSTM/model4_BiLSTM+attention/shap_neu_bar_ascending.png)
       - 📄 [shap\_neu\_bar\_descending.png](Twitter-RoBERTa+LSTM/model4_BiLSTM+attention/shap_neu_bar_descending.png)
       - 📄 [shap\_pos1.png](Twitter-RoBERTa+LSTM/model4_BiLSTM+attention/shap_pos1.png)
       - 📄 [shap\_pos2.png](Twitter-RoBERTa+LSTM/model4_BiLSTM+attention/shap_pos2.png)
       - 📄 [shap\_pos\_bar\_ascending.png](Twitter-RoBERTa+LSTM/model4_BiLSTM+attention/shap_pos_bar_ascending.png)
       - 📄 [shap\_pos\_bar\_descending.png](Twitter-RoBERTa+LSTM/model4_BiLSTM+attention/shap_pos_bar_descending.png)
       - 📄 [summary.png](Twitter-RoBERTa+LSTM/model4_BiLSTM+attention/summary.png)
       - 📂 __uk\_twitter\_data\_3k__
         - 📄 [accuracy.png](Twitter-RoBERTa+LSTM/model4_BiLSTM+attention/uk_twitter_data_3k/accuracy.png)
         - 📄 [best\-model\_uk\-tweet\_3k.ipynb](Twitter-RoBERTa+LSTM/model4_BiLSTM+attention/uk_twitter_data_3k/best-model_uk-tweet_3k.ipynb)
         - 📄 [classification\_report.png](Twitter-RoBERTa+LSTM/model4_BiLSTM+attention/uk_twitter_data_3k/classification_report.png)
         - 📄 [confusion\_matrix.png](Twitter-RoBERTa+LSTM/model4_BiLSTM+attention/uk_twitter_data_3k/confusion_matrix.png)
         - 📄 [cr.png](Twitter-RoBERTa+LSTM/model4_BiLSTM+attention/uk_twitter_data_3k/cr.png)
         - 📄 [loss.png](Twitter-RoBERTa+LSTM/model4_BiLSTM+attention/uk_twitter_data_3k/loss.png)
         - 📄 [train\_val\_loss.png](Twitter-RoBERTa+LSTM/model4_BiLSTM+attention/uk_twitter_data_3k/train_val_loss.png)
       - 📂 __uk\_twitter\_data\_all__
         - 📄 [accuracy.png](Twitter-RoBERTa+LSTM/model4_BiLSTM+attention/uk_twitter_data_all/accuracy.png)
         - 📄 [best\-model\-uk\-twitter\-all.ipynb](Twitter-RoBERTa+LSTM/model4_BiLSTM+attention/uk_twitter_data_all/best-model-uk-twitter-all.ipynb)
         - 📄 [classification\_report1.png](Twitter-RoBERTa+LSTM/model4_BiLSTM+attention/uk_twitter_data_all/classification_report1.png)
         - 📄 [classification\_report2.png](Twitter-RoBERTa+LSTM/model4_BiLSTM+attention/uk_twitter_data_all/classification_report2.png)
         - 📄 [confusion\_matrix.png](Twitter-RoBERTa+LSTM/model4_BiLSTM+attention/uk_twitter_data_all/confusion_matrix.png)
         - 📄 [loss.png](Twitter-RoBERTa+LSTM/model4_BiLSTM+attention/uk_twitter_data_all/loss.png)
   - 📂 __UK\_covid\_twitter\_data__
     - 📄 [all\_cities.csv](UK_covid_twitter_data/all_cities.csv)
     - 📄 [sample\_data\_3000.csv](UK_covid_twitter_data/sample_data_3000.csv)
     - 📄 [sample\_data\_all.csv](UK_covid_twitter_data/sample_data_all.csv)
     - 📄 [stacked bar graph.png](UK_covid_twitter_data/stacked%20bar%20graph.png)
     - 📄 [tweets distribution.png](UK_covid_twitter_data/tweets%20distribution.png)
     - 📄 [uk\_twitter\_data\_preprocessing.ipynb](UK_covid_twitter_data/uk_twitter_data_preprocessing.ipynb)
   - 📄 [list.md](list.md)
   - 📄 [uk\-twitter\-3k\-classical\-modelling.ipynb](uk-twitter-3k-classical-modelling.ipynb)
   - 📂 __word2vec__
     - 📄 [all\_models\_1.png](word2vec/all_models_1.png)
     - 📄 [all\_models\_2.png](word2vec/all_models_2.png)
     - 📄 [all\_models\_3.png](word2vec/all_models_3.png)
     - 📄 [all\_models\_4.png](word2vec/all_models_4.png)
     - 📄 [rf\_knn\_mlp.png](word2vec/rf_knn_mlp.png)
     - 📄 [rf\_stacking\_voting.png](word2vec/rf_stacking_voting.png)


```


------


## License

MIT licensed, except where otherwise stated.
See `LICENSE.txt` file.





