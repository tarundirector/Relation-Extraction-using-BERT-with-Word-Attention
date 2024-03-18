
# 1. Introduction

![relation-extraction-1](https://github.com/Chandra0505/Data-Science-Resources/assets/85684655/6c7a823b-d86d-427d-8f47-a841e61676ac)

Figure: Example of Relation Extraction (Source: https://explosion.ai/blog/relation-extraction)

Relation extraction involves identifying the most relevant word that explains the relationship between entities within a given phrase. In the context of the SemEval-2010 Task 8 dataset, the focus is on semantic relations between pairs of nominals. For example, consider the sentence: "The chef prepared pasta with tomato sauce." The word "prepared" refers to the relationship between "chef" and "pasta." It indicates that the chef is preparing pasta. This relationship fits in with the "AGENT-INSTRUMENT" relationship, in which the agent (chef) uses an instrument (tomato sauce) to complete the task.

In natural language processing (NLP), BERT (Bidirectional Encoder Representations from Transformers) has emerged as a pivotal tool, specifically in its variant known as BERT-base-uncased plays a crucial role. BERT represents a significant advancement in language understanding, leveraging the Transformer architecture to enable bidirectional processing of textual data. The “base” version of BERT refers to a moderate-sized model, making it computationally feasible for various NLP tasks. The “uncased” indicates that during training, text is converted to lowercase, facilitating case-insensitive operations. BERT-base-uncased comprises 12 standard transformers, totalling approximately 110 million parameters. In our research, we utilize the pre-trained BERT model proposed by Devlin et al. (2018) to address our specific task.

BiLSTM, on the other hand, represents a distinct variant of recurrent neural networks (RNNs). It is designed to tackle the challenge of capturing contextual information from both past and future states within sequential data. This network architecture incorporates bidirectional processing, allowing inputs to be processed in forward and backward direction. By leveraging the memory cells of LSTM units, BiLSTM networks are highly effective at identifying complex patterns and contextual details found in textual data.



# 2. Related Work

Several approaches have been explored for relation classification, ranging from kernel methods and traditional machine learning algorithms to recent deep learning models. (Zeng et al., 2014) this paper introduces a convolutional neural network (CNN) approach for relation classification, demonstrating its effectiveness on benchmark datasets. (Ma et al., 2016) this paper introduces an approach based on Long Short-Term Memory (LSTM) networks for relation classification, focusing on shortest dependency paths. (Nguyen et al, 2016) extract relations for SemEval-2010 Task 8, this work integrates distributed semantics with feature engineering methods.

A graph convolutional network (GCN) model for relation extraction operates directly on dependency graphs to capture relational information (Marcheggiani et al., 2017). An end-to-end relation extraction framework based on LSTM networks is capable of handling both sequential and tree-structured inputs (Yao et al., 2016). 
Notably, the introduction of pre-trained models like BERT has set new benchmarks across various NLP tasks such as (Yu et al., 2019) proposed a relation extraction model that leverages BERT embeddings and knowledge graph context to enhance relation extraction performance. However, BERT's application to relation classification often does not incorporate detailed entity-specific information (Wu et al., 2019). 

On the other hand, attention mechanisms within neural networks have been effectively used to focus on relevant parts of the text, as seen in (Shen et al., 2016) work, which employs an attention-based CNN for semantic relation extraction. 
Our methodology aims to bridge these approaches, combining the contextual depth of BERT with the focused relevance provided by attention mechanisms.

# 3. Methodology

## 3.1	Dataset
The SemEval-2010 Task 8 dataset focuses on multi-way classification of semantic relations between nominal pairs. The dataset is used to evaluate the two relation classification algorithms BERT-uncased-base and BiLSTM. The dataset includes nine semantic relation types and one artificial relation type named ‘Other’. The nine relation types are Cause-Effect, Component-Whole, Content-Container, Entity-Destination, Entity-Origin, Instrument-Agency, Member-Collection, Message-Topic, and Product-Producer. The dataset contains 10717 annotated sentences, which is divided into 8000 training and 2717 test samples. Each annotated sentence has two nominals e1 and e2 that correspond to a specific relation type. The relations are direction, which means that the component whole (e1, e2) is different from component whole (e2, e1).
 
## 3.2	BERT-Uncased-base Model

### 3.2.1	Model Architecture

![Beige and Pink Modern Business Process Flowchart Diagram](https://github.com/Chandra0505/Data-Science-Resources/assets/85684655/5688030b-f40f-4139-825d-0030f605080b)

Figure 1: Model Architecture for Att-BERT

The provided model architecture in Figure 1 extends BERT for relation extraction tasks. It begins with a BERT backbone, utilizing a pre-trained BERT model to encode input text sequences into contextualized representations. Following this, the model computes representations for each entity (e1 and e2) using attention mechanisms over the input embeddings, focusing on the tokens corresponding to the entity mentions. These entity representations are then concatenated with the pooled output of BERT, which represents the entire input sequence. Subsequently, the concatenated representations undergo fully connected layers with tanh activation to extract features. Finally, another fully connected layer is employed for predicting the label of the relation between the entities. This architecture enables the model to effectively capture contextual information from the input text and make predictions regarding the relationships between entities in each sentence.

### 3.2.2	Pre-processing and Embedding Layer

In the pre-processing stage, input sentences are tokenized and entities are identified. For a sentence with two target entities (e1 and e2), special tokens ‘$’ and ‘#’ are inserted around the entities. Additionally, ‘[CLS]’ is added to the beginning of each sentence. This format adjustment helps BERT focus on entity locations for better extraction of contextual features.
The BERT tokenizer ensures that input is in a compatible format for subsequent stages. For each token, comprehensive representations are generated by concatenating two types of embeddings: Word Embeddings, which utilize pre-trained GloVe embeddings to capture semantic context, and Position Embeddings, which encode relative positions of tokens to entities using trainable embeddings.
The enhanced tokens are then fed into a fine-tuned pre-trained BERT model, The output consists of token-level embeddings that capture global context and entity positions.
 
### 3.2.3	Attention-Based Feature Extraction and Relation Classification
To distill relevant information from the BERT-encoded sentence for relation classification the attention layer computes token weights based on their significance in determining the relation between entities. This process involves calculating attention scores for each token using the attention network. Subsequently, this feature vector passes through fully connected layers, culminating in a softmax layer that produces probabilities for each relation class. The model undergoes end-to-end training, optimizing the relation classification task using a cross-entropy loss function. This training process adjusts weights in the fully connected and attention layers and  fine-tunes the underlying BERT model for a specific purpose. The final sentence representation is obtained by a weighted sum of token embeddings, determined by the attention scores.

# 4. Results

| Parameter           |   |
|---------------------|-------|
| Batch Size          | 32    |
| Number of Epochs    | 50    |
| No. of TRAIN Examples | 8000 |
| No. of TEST Examples  | 2717 |

| Model         | F1     | Execution Time |
|---------------|--------|----------------|
| BERT (with Att)| 84.36 | 2hrs 50 mins   |


BERT(with Attention) is favored for its contextual understanding which strikes a balance between complexity and effectiveness.
BERT's bidirectional contextual representation and large text corpus pretraining enable it to capture semantic details with high accuracy. However, its large model size poses memory challenges, and processing long sequences still requires a lot of computational power. 

An essential indicator that strikes a balance between precision (correct positive predictions) and recall (capturing actual positive instances) is the F1 score. These are combined by the F1 score using their harmonic mean. A score of 1 indicates perfect performance, while 0 signifies poor classification ability. It is preferred when there is an unequal distribution of classes or when both recall and precision are equally important.
F1=(precision+recall)2×(precision×recall)​


# 5. Conclusion & Future Work

In our investigation, a comprehensive analysis of the capabilities of BERT (with Attention) for the task of semantic relation classification using the SemEval-2010 Task 8 dataset was presented. An innovative word attention approach aimed at enhancing the extraction of contextually relevant information surrounding target entities within sentences was introduced. The results underscore the effectiveness of leveraging pre-trained models like BERT, which provides a deep understanding of language nuance and context, combined with the nuanced attention mechanisms that prioritize critical elements within text for relation classification.

Future work will extend the model from sentence-level to document-level datasets like DocRED, necessitating methodological adaptations. With increased computational resources, enhancing performance by scaling the number of training epochs is possible. Expanding tests to diverse and other language datasets to assess the model's generalizability can be undertaken. End-to-end systems that simultaneously perform named entity recognition (NER) along with relation extraction can be developed to streamline the pipeline for extracting structured knowledge from text, improving overall efficiency and accuracy. Employing Neural Architecture Search(NAS) to automatically discover optimal model architectures which could help identify the most effective combinations of embeddings, attention mechanisms, and neural network layers for various relation extraction tasks.
# 6. Installation


To install PyTorch, please refer to the instructions provided on the official website.

For TorchText, you can install it using the following command:

```
pip install torchtext
```

For spaCy with English language support, use the following command:

```
python -m spacy download en
```

Finally, install the transformers library with the following command:

```
pip install transformers
```

These libraries are essential for running the code related to natural language processing tasks, including BERT and BiLSTM models for relation extraction. If you encounter any issues during installation or have any questions, feel free to reach out for assistance.

### Data Format

The TextMiningCW directory contains several subdirectories, each serving a specific purpose in the context of the project. Here's an overview of the structure and contents of each directory:
```
TextMiningCW
│
├── data_RE
│   ├── relation2id.tsv
│   ├── vector_50d.txt
│   ├── relation2id.txt
│   ├── label.txt
│   ├── dev.tsv
│   ├── test.tsv
│   ├── train.tsv
│   ├── test.json
│   ├── train.json
│   ├── valid.json
│   ├── cached_train_semeval_bert-base-uncased_384
│   ├── cached_test_semeval_bert-base-uncased_384
│   ├── test1.json
│   ├── valid1.json
│   ├── train1.json
│   └── tut2-model_ner.pt
│
├── model
│   ├── vocab.txt
│   ├── pytorch_model.bin
│   ├── config.json
│   ├── model.safetensors
│   └── training_args.bin
│
├── eval
│   └── proposed_answers.txt
│
└── embeddings
    └── glove.6B.300d.txt
```
Note: Some of these files have been generated by the code during testing and evaluation.

# 7. Reproducing Model Results

Both of the model implementations provided in this project were executed on Google Colab using the T4 GPU, utilizing Torch CUDA for accelerated computation. The execution time for each model is documented in the accompanying paper, along with the corresponding results.

The results of the model executions can be observed directly in the provided Jupyter Notebook files after training and testing the models. However, if you wish to reproduce the results or run the code yourself, follow these steps:

1. Create the necessary directories in the specified format on your Google Drive. Ensure that the directories `data_RE`, `model`, `eval`, and `embeddings` are created within the `TextMiningCW` directory.

2. Download the SemEval 2010 dataset from [here](https://huggingface.co/datasets/sem_eval_2010_task_8/tree/main) (optional, already provided in folders).

3. Obtain the pre-trained BERT model from [here](https://huggingface.co/google-bert/bert-base-uncased)(optional, already provided in folders).

4. Modify any path directories in the code that are dependent on the location where you load the data. This step is crucial for ensuring that the code can access the required files and directories correctly.

5. Run the provided Jupyter Notebook files on Google Colab, ensuring that the paths are correctly configured and the necessary dependencies are installed.

By following these steps, you can replicate the experiments conducted in the paper or explore further variations of the models as needed.


# References

- Relation Classification via Convolutional Deep Neural Network by Daojian Zeng, Kang Liu, Siwei Lai, Guangyou Zhou, and Jun Zha, (COLING 2014).

- Classifying Relations via Long Short-Term Memory Networks along Shortest Dependency Paths by Xuezhe Ma and Eduard Hovy, (NAACL-HLT 2016).
 
- Combining Distributed Semantics and Feature Engineering for Relation Extraction from Texts by Dat Quoc Nguyen, Richard Billingsley, Lan Du, and Mark Johnson, (IJCNN 2016).

- Graph Convolutional Networks for Relation Extraction from Text by Diego Marcheggiani and Ivan Titov, (EACL 2017).
 
- End-to-End Relation Extraction using LSTMs on Sequences and Tree Structures by Yao Zhu, Igor Titov, and Eric Cambria, (ACL 2016).

- BERT for Relation Extraction via Knowledge Graph Context by Yu Su, Honglei Liu, Semih Yavuz, Izzeddin Gur, and Huan Sun, (ACL 2019).

- Enriching Pre-trained Language Model with Entity Information for Relation Classification by Shanchan Wu and Yifan He, (arXiv:1905.08284 2019).

- BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding by Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova, (arXiv:1810.04805 2019).

- Long short-term memory by Sepp Hochreiter and Jurgen Schmidhuber, (Neural Computation, 9(8):1735–1780 1997).

- Generating sequences with recurrent neural networks by Alex Graves, (arXiv preprint arXiv:1308.0850 2013).

- Improving neural networks by preventing co adaptation of feature detectors by Geoffrey E Hinton, Nitish Srivastava, Alex Krizhevsky, Ilya Sutskever, and Ruslan R Salakhutdinov, (arXiv preprint arXiv:1207.0580 2012).

- Neural machine translation by jointly learning to align and translate by Dzmitry Bahdanau, Kyunghyun Cho, and Yoshua Bengio, (arXiv preprint arXiv:1409.0473 2014).