# NLP-Based Conversational Chatbot for AI Learning

An NLP-based conversational chatbot designed to answer questions about Natural Language Processing and related AI concepts. The system uses a pretrained BERT model to represent user queries as embeddings, compares them with stored intent patterns using cosine similarity, and returns a predefined response corresponding to the most relevant intent.

The project demonstrates semantic text representation, similarity-based intent classification, and response selection using Python and deep learning libraries.

## Features

### Intent Classification

The chatbot identifies the likely intent of a user's query by comparing its semantic representation with stored example patterns.

Supported intents include:

- Greetings and farewells
- General help and thanks
- Natural Language Processing (NLP)
- Tokenization
- Sentiment Analysis
- Word Embeddings
- Transformer Models
- Named Entity Recognition (NER)
- Machine Translation
- Deep Learning
- Chatbots

### Semantic Similarity Matching

The system uses the pretrained `bert-base-uncased` model from Hugging Face Transformers to generate contextual representations of text.

Cosine similarity measures the similarity between the user's query representation and stored pattern representations. The chatbot selects the most relevant intent based on the similarity scores.

### Response Generation

After predicting an intent, the system selects an appropriate predefined response from its response dataset.

If the similarity score falls below the configured threshold, the query is assigned to an unknown intent.

### Data Visualization

The project includes visualizations to explore text representations and similarity results, including:

- t-SNE embedding visualization
- Intent similarity heatmap
- Similarity score visualization

## Architecture

The chatbot follows an embedding-based intent classification and response selection architecture.

```text
                 User Query
                      |
                      v
             Text Preprocessing
                      |
                      v
             BERT Tokenization
                      |
                      v
          Pretrained BERT Encoder
             bert-base-uncased
                      |
                      v
            Query Representation
                      |
                      v
       Cosine Similarity Calculation
                      |
                      v
       Best-Matching Intent Selection
                      |
                      v
          Similarity Threshold Check
                      |
             +--------+--------+
             |                 |
             v                 v
      Relevant Intent     Unknown Intent
             |                 |
             v                 v
      Predefined Reply    Fallback Reply
             |                 |
             +--------+--------+
                      |
                      v
                Chatbot Output
```

### Model Specifications

| Component | Description |
|---|---|
| Base Model | `bert-base-uncased` |
| Model Type | Pretrained Transformer encoder |
| Embedding Dimension | 768 |
| Similarity Metric | Cosine similarity |
| Intent Prediction | Similarity-based matching |
| Response Mechanism | Predefined intent responses |
| Unknown Intent Handling | Configurable similarity threshold |
| Deep Learning Framework | PyTorch |

**Note:** BERT is used to create text representations. The chatbot does not use a generative language model to generate new responses.

## Dataset and Intent Categories

The chatbot uses predefined intent categories, example patterns, and associated responses.

The patterns provide reference examples against which incoming queries are compared. The predicted intent determines which response category is selected.

The implementation contains 13 defined intent categories, excluding the unknown-intent fallback:

1. Greeting
2. Goodbye
3. Thanks
4. What is NLP
5. Tokenization
6. Sentiment Analysis
7. Word Embeddings
8. Transformers
9. Named Entity Recognition
10. Machine Translation
11. Deep Learning
12. Chatbots
13. Help

The dataset is intended for an educational chatbot focused on introductory NLP and AI concepts.

## Installation and Setup

### Prerequisites

- Python 3.8 or later
- Google Colab or a local Python environment
- Internet access to download pretrained model files when they are not already cached

### Clone the Repository

```bash
git clone https://github.com/Gargipal12/nlp-conversational-chatbot.git
cd nlp-conversational-chatbot
```

### Install Dependencies

Install the libraries required by the notebook:

```bash
pip install numpy torch nltk transformers scikit-learn matplotlib
```

If additional dependencies are imported in the notebook, install those as well.

## Usage

### Run in Google Colab

1. Open the project notebook in Google Colab.
2. Run the setup and import cells.
3. Load the intent patterns and responses.
4. Initialize the pretrained BERT tokenizer and model.
5. Generate the stored pattern representations.
6. Run the chatbot interface cell.
7. Enter a question related to the supported NLP topics.
8. Review the predicted intent, similarity score, and chatbot response.

Use `exit`, `quit`, or the other configured exit commands to end the interactive session.

## Working Principle

The chatbot processes a user query through the following stages:

**1. Input Processing**

The system accepts a text query from the user and prepares it for model processing.

**2. Tokenization**

The BERT tokenizer converts the text into token IDs and other model inputs required by the pretrained Transformer.

**3. Text Representation**

The pretrained BERT model processes the tokenized input to produce a numerical representation of the text.

**4. Similarity Calculation**

Cosine similarity compares the query representation with stored pattern representations to estimate which pattern is most similar.

**5. Intent Classification**

The system identifies the intent associated with the best-matching pattern.

**6. Threshold-Based Decision**

The predicted match is checked against the configured similarity threshold. Low-scoring queries can be classified as unknown.

**7. Response Selection**

The chatbot returns a predefined response associated with the selected intent.

## Model Evaluation

The notebook includes a small set of test queries to examine intent prediction and response selection.

In the recorded evaluation, the chatbot achieved:

| Metric | Recorded Result |
|---|---:|
| Test Queries | 10 |
| Correct Predictions | 9 |
| Incorrect Predictions | 1 |
| Accuracy | 90% |
| Average Similarity Score | 0.9951 |

These results represent the notebook's recorded evaluation on a small test set. They should not be interpreted as evidence of general performance across all possible user queries.

Cosine similarity scores are matching scores, not calibrated probabilities. A high score does not guarantee that the predicted intent is correct.

## Project Structure

```text
nlp-conversational-chatbot/
|
├── NLP-Based Conversational Chatbot for AI Learning.ipynb
└── README.md
```

The notebook contains the implementation, model initialization, intent matching, response selection, evaluation, and visualization code.

The exact notebook filename in the repository may differ.

## Technologies Used

- **Python:** Core programming language
- **PyTorch:** Deep learning framework used to load and run BERT
- **Hugging Face Transformers:** BERT tokenizer and pretrained model
- **NumPy:** Numerical operations
- **NLTK:** Natural language processing utilities
- **Scikit-learn:** Similarity calculations and related utilities
- **Matplotlib:** Data visualization
- **Google Colab:** Notebook-based development and execution

## Applications

- Educational conversational systems
- Introductory NLP concept demonstrations
- Intent classification experiments
- Semantic similarity analysis
- Learning applications for Transformer-based text processing

## Limitations

- The chatbot supports a limited set of predefined intents.
- Responses are selected from stored responses rather than generated dynamically.
- Queries outside the supported topics may be incorrectly classified.
- Similarity-based classification depends on the quality and coverage of the example patterns.
- The recorded evaluation uses only a small test set and does not establish performance on unseen real-world data.

## Future Improvements

- Expand the dataset with more diverse patterns for each intent.
- Improve intent classification and unknown-query detection.
- Add more NLP and AI topics.
- Evaluate performance using a larger, separate test dataset.
- Introduce a web-based interface.
- Explore retrieval-based response selection and more advanced conversational techniques.

## References

- Devlin, J., Chang, M.-W., Lee, K., and Toutanova, K. *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.* NAACL-HLT, 2019.
- Hugging Face Transformers documentation.
- PyTorch documentation.
- Scikit-learn documentation.

## Author

**Gargi Pal**

GitHub: https://github.com/Gargipal12
