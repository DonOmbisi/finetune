**IMDB Sentiment Analysis with DistilBERT**

**Overview**

This Jupyter Notebook demonstrates how to perform sentiment analysis on the IMDB movie review dataset using a pre-trained DistilBERT model. The project includes data loading, preprocessing, model initialization, evaluation, and fine-tuning.

**Key Components**

**1\. Data Loading and Preprocessing**

* Loads the IMDB dataset from a CSV file containing 50,000 reviews  
    
* Cleans text by removing HTML tags and extra whitespace  
    
* Converts sentiment labels ('positive'/'negative') to numeric values (1/0)  
    
* Creates a balanced subset of 10,000 samples for faster training  
    
* Splits data into training (8,000 samples) and test (2,000 samples) sets

**2\. Model Setup**

* Uses the distilbert-base-uncased pre-trained model  
    
* Initializes tokenizer and sequence classification model  
    
* Tokenizes text data with a maximum length of 256 tokens  
    
* Creates PyTorch datasets and data loaders with batch size of 16

**3\. Baseline Evaluation**

* Evaluates model performance before fine-tuning  
    
* Baseline metrics:

  *     Accuracy: 0.5040


  *     F1 Score: 0.6702


  *     Average Loss: 0.6946

**4\. Model Training**

* Fine-tunes the model for 2 epochs using AdamW optimizer  
    
* Tracks training loss and evaluates on test set after each epoch  
    
* Final performance after fine-tuning:

  *        **Test Accuracy: 0.8865**


  *        **Test F1: 0.8930**


  *        **Test Loss: 0.2803**

**Technical Details**  
**Dependencies**

* Python 3.10  
    
* Key libraries:  
    
  * PyTorch


  * Transformers


  * Pandas


  * Scikit-learn


  * Tqdm

**Model Architecture**

* Pre-trained DistilBERT model  
    
* Sequence classification head with 2 output classes  
    
* 66,955,010 total parameters

**Training Parameters**

* Learning rate: 2e-5  
    
* Batch size: 16  
    
* Epochs: 2  
    
* Weight decay: 0.01  
    
* Max sequence length: 256 tokens

**Usage**

1. Install required packages:  
   **Bash**

   **pip install transformers datasets torch scikit-learn matplotlib seaborn tqdm**  
     
2. Ensure the IMDB dataset is available as 'imdb.csv' with columns 'review' and 'sentiment'  
3. Run the notebook sequentially to:  
     
* Load and preprocess data


* Initialize the model


* Evaluate baseline performance


* Fine-tune the model


* View training progress and final metrics

**Results**  
The fine-tuned model achieves:

* 88.65% accuracy on the test set  
  * 89.30% F1 score  
  * 0.2803 average loss

This demonstrates significant improvement over the baseline model, showing the effectiveness of fine-tuning pre-trained language models for sentiment analysis tasks.

