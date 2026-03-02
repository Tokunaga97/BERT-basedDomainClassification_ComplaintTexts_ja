<h1 align="center">
BERT-based Domain Classification for Japanese Complaint Texts
</h1>

<p align="center">
Japanese Complaint Domain Classification using BERT<br>
Pretrained Model Available on Hugging Face Hub
</p>

<hr>

<h2>Project Overview</h2>

<p>
This project implements a BERT-based domain classification model
for Japanese complaint texts.
</p>

<p>
The trained model is publicly available on Hugging Face Hub,
allowing direct inference without retraining.
</p>

<hr>

<h2>Technical Highlights</h2>

<ul>
<li>Transformer-based classification (BERT)</li>
<li>Custom Dataset pipeline</li>
<li>Train / Validation / Test split (9:0.5:0.5)</li>
<li>Evaluation using accuracy metric</li>
<li>Public deployment of trained model on Hugging Face Hub</li>
</ul>

<hr>

<h2>Project Structure</h2>

<pre>
BERT-basedDomainClassification_ComplaintTexts_ja
├ requirements.txt
├ src
│  ├ train.py
│  ├ dataset.py
│  └ bert_test.py
└ corpus (hosted on Hugging Face Hub)
</pre>

<hr>

<h2>Dataset</h2>

<p>
Corpus hosted at:
</p>

<p>
<a href="https://huggingface.co/datasets/SHSK0118/BERT-basedDomainClassification_ComplaintTexts_ja">
Dataset Repository
</a>
</p>

<hr>

<h2>Trained Model</h2>

<p>
The trained model is available at:
</p>

<p>
<a href="https://huggingface.co/SHSK0118/BERT-basedDomainClassification_ComplaintTexts_ja">
Model Repository on Hugging Face Hub
</a>
</p>

<p>
Model files include:
</p>

<ul>
<li>config.json</li>
<li>model.safetensors</li>
<li>tokenizer.json</li>
<li>spiece.model</li>
<li>special_tokens_map.json</li>
<li>tokenizer_config.json</li>
</ul>

<hr>

<h2>How to Use the Trained Model</h2>

<pre>
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained(
    "SHSK0118/BERT-basedDomainClassification_ComplaintTexts_ja"
)

model = AutoModelForSequenceClassification.from_pretrained(
    "SHSK0118/BERT-basedDomainClassification_ComplaintTexts_ja"
)
</pre>

<hr>

<h2>Experimental Result</h2>

<p>
Test Accuracy: <strong>73.0%</strong>
</p>

<hr>

<h2>Analysis</h2>

<p>
Performance is influenced by domain mismatch:
</p>

<ul>
<li>Training corpus: formal written text (Wikimedia)</li>
<li>Complaint corpus: informal conversational style</li>
</ul>

<p>
Future improvement may involve domain adaptation or additional fine-tuning
on conversational datasets.
</p>

<hr>

<h2>Purpose</h2>

<p>
This project demonstrates:
</p>

<ul>
<li>Implementation of BERT-based classification</li>
<li>End-to-end ML pipeline construction</li>
<li>Model deployment via Hugging Face Hub</li>
</ul>

<hr>

<h2>Author</h2>

<p>
Shota Tokunaga<br>
Independent researcher | NLP / Machine Learning
</p>
