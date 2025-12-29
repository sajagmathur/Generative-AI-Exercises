What is Natural Language Processing (NLP)
Natural language processing (NLP) is the discipline of building machines that can manipulate human language — or data that resembles human language — in the way that it is written, spoken, and organized. 

Use of NLP: 
 

Here are 11 tasks that can be solved by NLP:

Sentiment analysis is the process of classifying the emotional intent of text. Generally, the input to a sentiment classification model is a piece of text, and the output is the probability that the sentiment expressed is positive, negative, or neutral. Typically, this probability is based on either hand-generated features, word n-grams, TF-IDF features, or using deep learning models to capture sequential long- and short-term dependencies. Sentiment analysis is used to classify customer reviews on various online platforms as well as for niche applications like identifying signs of mental illness in online comments.
NLP sentiment analysis illustration
Toxicity classification is a branch of sentiment analysis where the aim is not just to classify hostile intent but also to classify particular categories such as threats, insults, obscenities, and hatred towards certain identities. The input to such a model is text, and the output is generally the probability of each class of toxicity. Toxicity classification models can be used to moderate and improve online conversations by silencing offensive comments, detecting hate speech, or scanning documents for defamation. 
Machine translation automates translation between different languages. The input to such a model is text in a specified source language, and the output is the text in a specified target language. Google Translate is perhaps the most famous mainstream application. Such models are used to improve communication between people on social-media platforms such as Facebook or Skype. Effective approaches to machine translation can distinguish between words with similar meanings. Some systems also perform language identification; that is, classifying text as being in one language or another. 
Named entity recognition aims to extract entities in a piece of text into predefined categories such as personal names, organizations, locations, and quantities. The input to such a model is generally text, and the output is the various named entities along with their start and end positions. Named entity recognition is useful in applications such as summarizing news articles and combating disinformation. For example, here is what a named entity recognition model could provide: 
named entity recognition NLP
Spam detection is a prevalent binary classification problem in NLP, where the purpose is to classify emails as either spam or not. Spam detectors take as input an email text along with various other subtexts like title and sender’s name. They aim to output the probability that the mail is spam. Email providers like Gmail use such models to provide a better user experience by detecting unsolicited and unwanted emails and moving them to a designated spam folder. 
Grammatical error correction models encode grammatical rules to correct the grammar within text. This is viewed mainly as a sequence-to-sequence task, where a model is trained on an ungrammatical sentence as input and a correct sentence as output. Online grammar checkers like Grammarly and word-processing systems like Microsoft Word use such systems to provide a better writing experience to their customers. Schools also use them to grade student essays. 
Topic modeling is an unsupervised text mining task that takes a corpus of documents and discovers abstract topics within that corpus. The input to a topic model is a collection of documents, and the output is a list of topics that defines words for each topic as well as assignment proportions of each topic in a document. Latent Dirichlet Allocation (LDA), one of the most popular topic modeling techniques, tries to view a document as a collection of topics and a topic as a collection of words. Topic modeling is being used commercially to help lawyers find evidence in legal documents. 
Text generation, more formally known as natural language generation (NLG), produces text that’s similar to human-written text. Such models can be fine-tuned to produce text in different genres and formats — including tweets, blogs, and even computer code. Text generation has been performed using Markov processes, LSTMs, BERT, GPT-2, LaMDA, and other approaches. It’s particularly useful for autocomplete and chatbots.
Autocomplete predicts what word comes next, and autocomplete systems of varying complexity are used in chat applications like WhatsApp. Google uses autocomplete to predict search queries. One of the most famous models for autocomplete is GPT-2, which has been used to write articles, song lyrics, and much more. 
Chatbots automate one side of a conversation while a human conversant generally supplies the other side. They can be divided into the following two categories:
Database query: We have a database of questions and answers, and we would like a user to query it using natural language. 
Conversation generation: These chatbots can simulate dialogue with a human partner. Some are capable of engaging in wide-ranging conversations. A high-profile example is Google’s LaMDA, which provided such human-like answers to questions that one of its developers was convinced that it had feelings.
Information retrieval finds the documents that are most relevant to a query. This is a problem every search and recommendation system faces. The goal is not to answer a particular query but to retrieve, from a collection of documents that may be numbered in the millions, a set that is most relevant to the query. Document retrieval systems mainly execute two processes: indexing and matching. In most modern systems, indexing is done by a vector space model through Two-Tower Networks, while matching is done using similarity or distance scores. Google recently integrated its search function with a multimodal information retrieval model that works with text, image, and video data.
 
information retrieval illustration
Summarization is the task of shortening text to highlight the most relevant information. Researchers at Salesforce developed a summarizer that also evaluates factual consistency to ensure that its output is accurate. Summarization is divided into two method classes:
Extractive summarization focuses on extracting the most important sentences from a long text and combining these to form a summary. Typically, extractive summarization scores each sentence in an input text and then selects several sentences to form the summary.
Abstractive summarization produces a summary by paraphrasing. This is similar to writing the abstract that includes words and sentences that are not present in the original text. Abstractive summarization is usually modeled as a sequence-to-sequence task, where the input is a long-form text and the output is a summary.
Question answering deals with answering questions posed by humans in a natural language. One of the most notable examples of question answering was Watson, which in 2011 played the television game-show Jeopardy against human champions and won by substantial margins. Generally, question-answering tasks come in two flavors:
Multiple choice: The multiple-choice question problem is composed of a question and a set of possible answers. The learning task is to pick the correct answer. 
Open domain: In open-domain question answering, the model provides answers to questions in natural language without any options provided, often by querying a large number of texts.


How Does Natural Language Processing (NLP) Work?

Step 1. Data Preprocessing:

Stemming and lemmatization: 

Stemming is an informal process of converting words to their base forms using heuristic rules. For example, “university,” “universities,” and “university’s” might all be mapped to the base univers. emmatization is a more formal way to find roots by analyzing a word’s morphology using vocabulary from a dictionary. Stemming and lemmatization are provided by libraries like spaCy and NLTK. 


Sentence segmentation breaks a large piece of text into linguistically meaningful sentence units. This is obvious in languages like English, where the end of a sentence is marked by a period, but it is still not trivial. 

Stop word removal aims to remove the most commonly occurring words that don’t add much information to the text. For example, “the,” “a,” “an,” and so on.

Tokenization splits text into individual words and word fragments. The result generally consists of a word index and tokenized text in which words may be represented as numerical tokens for use in various deep learning methods. 



Feature extraction: 

Most conventional machine-learning techniques work on the features – generally numbers that describe a document in relation to the corpus that contains it – 

Bag-of-Words: Bag-of-Words counts the number of times each word or n-gram (combination of n words) appears in a document. For example, below, the Bag-of-Words model creates a numerical representation of the dataset based on how many of each word in the word_index occur in the document. 

TF-IDF: In Bag-of-Words, we count the occurrence of each word or n-gram in a document. In contrast, with TF-IDF, we weight each word by its importance. To evaluate a word’s significance, we consider two things:
Term Frequency: How important is the word in the document?
TF(word in a document)= Number of occurrences of that word in document / Number of words in document


Inverse Document Frequency: How important is the term in the whole corpus?
IDF(word in a corpus)=log(number of documents in the corpus / number of documents that include the word)

A word is important if it occurs many times in a document. But that creates a problem. Words like “a” and “the” appear often. And as such, their TF score will always be high. We resolve this issue by using Inverse Document Frequency, which is high if the word is rare and low if the word is common across the corpus. The TF-IDF score of a term is the product of TF and IDF. 

Word2Vec, introduced in 2013, uses a vanilla neural network to learn high-dimensional word embeddings from raw text. It comes in two variations: Skip-Gram, in which we try to predict surrounding words given a target word, and Continuous Bag-of-Words (CBOW), which tries to predict the target word from surrounding words. After discarding the final layer after training, these models take a word as input and output a word embedding that can be used as an input to many NLP tasks. 

GLoVE is similar to Word2Vec as it also learns word embeddings, but it does so by using matrix factorization techniques rather than neural learning. The GLoVE model builds a matrix based on the global word-to-word co-occurrence counts. 


Step 2: Modeling


After data is preprocessed, it is fed into an NLP architecture that models the data to accomplish a variety of tasks.

Language Models: In very basic terms, the objective of a language model is to predict the next word when given a stream of input words. Probabilistic models that use Markov assumption are one example:
P(Wn)=P(Wn|Wn−1)


Top NLP Techniques:
1. Traditional NLP Techniques:
i. Logistic Regression: Logistic regression is a supervised classification algorithm that aims to predict the probability that an event will occur based on some input. In NLP, logistic regression models can be applied to solve problems such as sentiment analysis, spam detection, and toxicity classification.

ii. Naive Bayes is a supervised classification algorithm that finds the conditional probability distribution P(label | text) using the following Bayes formula:
P(label | text) = P(label) x P(text|label) / P(text) 

and predicts based on which joint distribution has the highest probability. The naive assumption in the Naive Bayes model is that the individual words are independent. Thus: 

P(text|label) = P(word_1|label)*P(word_2|label)*…P(word_n|label)

In NLP, such statistical methods can be applied to solve problems such as spam detection or finding bugs in software code. 


Decision trees are a class of supervised classification models that split the dataset based on different features to maximize information gain in those splits

Deep Learning Techniques:
Convolutional Neural Networks: See a document as an image. However, instead of pixels, the input is sentences or documents represented as a matrix of words.


Recurrant Neural Networks: CNNs don’t learn the sequential structure of the data, where every word is dependent on the previous word or a word in the previous sentence. RNNs remember previous information using hidden states and connect it to the current task. 

Auto Encoder, Encoder Decoder etc. 

Transformers: The transformer, a model architecture first described in the 2017 paper “Attention Is All You Need” (Vaswani, Shazeer, Parmar, et al.), forgoes recurrence and instead relies entirely on a self-attention mechanism to draw global dependencies between input and output. Since this mechanism processes all words at once (instead of one at a time) that decreases training speed and inference cost compared to RNNs, especially since it is parallelizable. The transformer architecture has revolutionized NLP in recent years, leading to models including BLOOM, Jurassic-X, and Turing-NLG. It has also been successfully applied to a variety of different vision tasks, including making 3D images.



What to use:
Pytorch, TensorFlow, Python - Spacy, Natural Language Toolkit


Course 2: Complete guide towards neural networks
https://towardsdatascience.com/neural-networks-intuitively-and-exhaustively-explained-0153f85c1007/


Neural networks take direct inspiration from the human brain, which is made up of billions of incredibly complex cells called neurons.

A neural network is made up of elements called "perceptrons", which are directly inspired by neurons.

Since these perceptrons always have a linear relationship, the neural network usually has linearity issues. To counter this, you introduce an activation function that introduces non linearity into the model.

In AI there are many popular activation functions, but the industry has largely converged on three popular ones: ReLU, Sigmoid, and Softmax, which are used in a variety of different applications. ReLu is the most popular one.

Model training:
Weights are adjusted so that our model starts giving responses closer to what is desired.

---> Folder 2:Neural_Network: Implementing Neural Network from Scratch