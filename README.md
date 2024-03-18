This repo contains a list of sources, weblinks, blogs and Youtube channels from where LLMs can and should be learned.

* **History of NLP**

![image](https://github.com/ParthaPRay/LLM-Learning-Sources/assets/1689639/9cb88e08-1501-49fd-af2a-a23ef1e6909f)

 It has been quite a journey to arrive at a ChatGPT model! It took some time before we thought about modeling language as a probabilistic generative process. NLP studies the interactions between computers and human language, and it is as old as computers themselves.

Warren Weaver was the first to suggest an algorithmic approach to machine translation (MT) in 1949, and this led to the Georgetown experiment, the first computer application to MT, in 1955. In 1957, Chomsky established the first grammar theory. ELIZA (1964) and SHRDLU (1968) can be considered to be the first natural-language understanding computer programs.

The 60s and early 70s marked the era of grammar theories. During the 70s, the concept of conceptual ontologies became quite fashionable. Conceptual ontologies are similar to knowledge graphs, where concepts are linked to each other by how they are associated. The famous ones are MARGIE (1975), TaleSpin (1976), QUALM (1977), SAM (1978), PAM (1978), Politics (1979) and Plot Units (1981).

The 80s showed a great period of success for symbolic methods. In 1983, Charniak proposed Passing Markers, a mechanism for resolving ambiguities in language comprehension by indicating the relationship between adjacent words. In 1986, Riesbeck and Martin proposed Uniform Parsing, a new approach to natural language processing that combines parsing and inferencing in a uniform framework for language learning. In 1987, Hirst proposed a new approach to resolving ambiguity: Semantic Interpretation.

The 90s saw the advent of statistical models. It was the beginning of thinking about language as a probabilistic process. In 1989, Balh proposed a tree-based method to predict the next word in a sentence. IBM presented a series of models for statistical machine translation. In 1990 Chitrao and Grishman demonstrated the potential of statistical parsing techniques for processing messages and Brill et al introduced a method for automatically inducing a part-of-speech tagger by training on a large corpus of text. In 1991, Brown proposed a method for aligning sentences in parallel corpora for machine translation applications.

In 2003, Bengio proposed the first neural language model, a simple feed-forward model. In 2008, Collobert and Weston applied multi-task learning with ConvNet. In 2011, Hinton built a generative text model with Recurrent Neural Networks. In 2013, Mikolov introduced Word2Vec. In 2014, Sutskever suggested a model for sequence-to-sequence learning. In 2017, Vaswani gave us the Transformer architecture that led to a revolution in model performance. In 2018, Devlin presented BERT, which popularized Transformers. And in 2022, we finally got to experience ChatGPT, which completely changed the way the public perceived AI!

* NLP metrics: a small subset

   ![image](https://github.com/ParthaPRay/LLM-Learning-Sources/assets/1689639/7761677e-e11d-4eea-ba79-fcf4f64ad830)



# Large Language Model

* How Large Language Models Work, 
https://www.youtube.com/watch?v=5sLYAQS9sWQ&ab_channel=IBMTechnology

* **Andrej Karpathy**

    * 1hr Talk Intro to Large Language Models Lecture by **Andrej Karpathy**, https://www.youtube.com/watch?v=zjkBMFhNj_g&ab_channel=AndrejKarpathy
    
      Slide PDF: https://drive.google.com/file/d/1pxx_ZI7O-Nwl7ZLNk5hI3WzAsTLwvNU7/view
    
      Slide PPT Keynote: https://drive.google.com/file/d/1FPUpFMiCkMRKPFjhi9MAhby68MHVqe8u/view


              Makemore implementation from Andrej Karpathy
      
              https://github.com/karpathy/makemore
   
    
    * **Neural Networks: Zero to Hero** Lecture by **Andrej Karpathy**

      A course on neural networks that starts all the way at the basics. The course is a series of YouTube videos where we code and train neural networks together. The Jupyter notebooks we build in the videos are then captured here inside the lectures directory [https://github.com/karpathy/nn-zero-to-hero/blob/master/lectures]. Every lecture also has a set of exercises included in the video description. (This may grow into something more respectable).
      
      https://github.com/karpathy/nn-zero-to-hero/tree/master
      
    * Let's build GPT: from scratch, in code, spelled out.,
          
        ![image](https://github.com/ParthaPRay/LLM-Learning-Sources/assets/1689639/d4894e83-ddb2-46fa-bcab-7dba61aeaac6)
      
       https://www.youtube.com/watch?v=kCc8FmEb1nY&ab_channel=AndrejKarpathy

       https://colab.research.google.com/drive/1JMLa53HDuA-i7ZBmqV7ZnA3c_fvtXnx-?usp=sharing
    

     * Let's build the GPT Tokenizer, 
        https://www.youtube.com/watch?v=zduSFxRajkE&ab_channel=AndrejKarpathy

       https://colab.research.google.com/drive/1y0KnCFZvGVf_odSfcNAws6kcDD7HsI0L?usp=sharing

       https://github.com/karpathy/minbpe
       
        
* Create a Large Language Model from Scratch with Python – Tutorial, https://www.youtube.com/watch?v=UU1WVnMk4E8&t=24s&ab_channel=freeCodeCamp.org

* [Build a Large Language Model (From Scratch)] (https://www.manning.com/books/build-a-large-language-model-from-scratch) by Sebastian Raschka, MEAP publications 2025.

    ![image](https://github.com/ParthaPRay/LLM-Learning-Sources/assets/1689639/86352650-cc89-4590-afc5-6ac1e5bfb481)


   ![image](https://github.com/ParthaPRay/LLM-Learning-Sources/assets/1689639/d2e8cc02-9ca4-4c14-ad7e-ccc236083084)

   Codes: https://github.com/rasbt/LLMs-from-scratch
  
* How to Build an LLM from Scratch | An Overview,
 https://www.youtube.com/watch?v=ZLbVdvOoTKM&pp=ygUdQ3JlYXRlIGEgTGFyZ2UgTGFuZ3VhZ2UgTW9kZWw%3D
  
* Train your own language model with nanoGPT | Let’s build a songwriter, https://www.youtube.com/watch?v=XS8eRtlcCGU&ab_channel=SophiaYang

* A Hackers' Guide to Language Models, https://www.youtube.com/watch?v=jkrNMKz9pWU&ab_channel=JeremyHoward

* Create your own Local Chatgpt for FREE, Full Guide: PDF, Image, & Audiochat (Langchain, Streamlit), https://www.youtube.com/watch?v=CUjO8b6_ZuM&t=452s&ab_channel=LeonExplainsAI
  
* LLM Evaluation Essentials: Statistical Analysis of Hallucination LLM Evaluations, https://www.youtube.com/watch?v=IH45ltIMC3k&ab_channel=ArizeAI

  https://docs.arize.com/phoenix/llm-evals/running-pre-tested-evals/hallucinations

* Fine Tuning and Evaluating LLMs with Anyscale and Arize, https://www.youtube.com/watch?v=b-MfkFz-A2E&ab_channel=ArizeAI

* Advanced LLM Evaluation: Synthetic Data Generation, https://www.youtube.com/watch?v=AYehm7q6Oks&ab_channel=ArizeAI

* Constructing an Evaluation Approach for Generative AI Models with Hugging Face's Rajiv Shah, https://www.youtube.com/watch?v=PtXOQDHPddE&ab_channel=ArizeAI

* LLM Evaluation Essentials: Benchmarking and Analyzing Retrieval Approaches, https://www.youtube.com/watch?v=ExO3U0M3y_0&ab_channel=ArizeAI

* Building And Troubleshooting An Advanced LLM Query Engine, https://www.youtube.com/watch?v=_zDDErOaUqc&ab_channel=ArizeAI

* Model Monitoring for LLMs, 
https://www.youtube.com/watch?v=zR1X5R_1TUw&ab_channel=SethJuarez

* Let's pretrain a 3B LLM from scratch: on 16+ H100 GPUs, no detail skipped.
  https://youtu.be/aPzbR1s1O_8?si=2VEoUt9FFRUftctv

* A simple generative ML model with just KNN, https://www.youtube.com/watch?v=aFuHPiJu0QA


* The N Implementation Details of RLHF with PPO,
  https://huggingface.co/blog/the_n_implementation_details_of_rlhf_with_ppo

* Optimizing your LLM in production

  https://huggingface.co/blog/optimize-llm

* LLM Tutorial, https://www.youtube.com/watch?v=JvLiEdTKKgk&list=PLpqh-PUKX-i4TT-vZXhFwI8Jdqr7J742n&pp=iAQB

* Serve a custom LLM for over 100 customers

  https://youtu.be/1TU9ZrZhqw0?si=LwtZJ0V2K6xQvSBA

* State of GPT | BRK216HFS, https://www.youtube.com/watch?v=bZQun8Y4L2A&ab_channel=MicrosoftDeveloper

* Building Systems with the ChatGPT API, https://www.deeplearning.ai/short-courses/building-systems-with-chatgpt/  

* Flash Attention 2.0 with Tri Dao (author)! | Discord server talks, https://www.youtube.com/watch?v=IoMSGuiwV3g&ab_channel=AleksaGordi%C4%87-TheAIEpiphany


* Outcome-based reward model (ORM)

  Meet Stepwise ORMs (SORMs)

 https://arxiv.org/abs/2402.10963
  ![image](https://github.com/ParthaPRay/LLM-Learning-Sources/assets/1689639/1ef40ceb-e53f-48d4-a4a4-29a45cb328c1)


* Datasets for Large Language Models: A Comprehensive Survey

  https://arxiv.org/abs/2402.18041v1

 
  LLM datasets from five perspectives:
   - (1) Pre-training Corpora;
   - (2) Instruction Fine-tuning Datasets;
   - (3) Preference Datasets;
   - (4) Evaluation Datasets;
   - (5) Traditional Natural Language Processing (NLP) Datasets.


  ![image](https://github.com/ParthaPRay/LLM-Learning-Sources/assets/1689639/7641da34-51ec-44b4-abd0-36d1ee54870d) 
  A timeline of some representative LLM datasets. Orange represents pre-training corpora, yellow represents instruction fine-tuning datasets, green represents preference datasets, and pink represents evaluation datasets



* GPT-Fast - blazingly fast inference with PyTorch (w/ Horace He)
  
  https://www.youtube.com/watch?v=18YupYsH5vY&ab_channel=AleksaGordi%C4%87-TheAIEpiphany

  https://pytorch.org/blog/accelerating-generative-ai-2/
  
  https://github.com/pytorch-labs/gpt-fast


 * Genie: Generative Interactive Environments

    ![image](https://github.com/ParthaPRay/LLM-Learning-Sources/assets/1689639/a06534bb-c44c-4bfd-995e-66ac9cd0b82f)
    A whole new world: Genie is capable of converting a variety of different prompts into interactive, playable environments that can be easily created, stepped into, and explored. This is made possible via a latent action interface, learned fully unsupervised from Internet videos. On the right we see a few generated steps for taking two latent actions.
   

   https://sites.google.com/view/genie-2024/

   https://arxiv.org/abs/2402.15391

   https://www.youtube.com/watch?v=lhg7DOCGqtU&ab_channel=code_your_own_AI
   

* 3 ways to train LLMs

  ![image](https://github.com/ParthaPRay/LLM-Learning-Sources/assets/1689639/59fa5381-a0ad-4508-9637-07ca37c1937d)

  Transformers can be used for many learning tasks, and the only difference comes from the way we prepare the data, the modeling head we choose, and the loss function we use to optimize the model.

With Causal Language Modeling, the model learns the language statistics by focusing on predicting the next word in a sequence. This is the more common way to perform Language modeling nowadays, and it has been the approach taken in GPT-1, GPT2, and GPT-3. Causality is ensured by applying a mask to the attention matrices computed within the attention layers. To avoid paying attention to words later in the sequence, we just set the attention to 0 for those words. To train this model, we just need to shift the inputs by removing the first word to create the labels.

For text classification, we want to associate the input text data with some category. For example, in the context of sentiment analysis, we may want to categorize the input sentence into the following three categories: [POSITIVE], [NEGATIVE] and [NEUTRAL]. In the context of text classification, we only need one prediction vector, and the typical strategy is usually to choose one of the hidden states and project it into the prediction space. This works because, although there are as many hidden states as there are input tokens, after passing through multiple transformer blocks, they all represent an entangled representation of the entire sentence. To train that model, we only need to compare the prediction vectors to the categorical labels by using a loss function such as cross-entropy.

The token classification learning task is often used for applications such as Named Entity Recognition (NER). We want to categorize each of the tokens in the input sentence. For example, we may want to associate each of the words with their grammatical categories: [NOUN], [VERB], and [ADJECTIVE]. For each of the inputs in the sequence, we need a prediction vector of the size of the number of categories we want to predict. At training time, we compare that prediction matrix for all the tokens to their categories in the labels with a cross-entropy loss function and update the model weights.

  
* How LLMs generate text?
  
  ![image](https://github.com/ParthaPRay/LLM-Learning-Sources/assets/1689639/e5e8ac44-97b8-45b4-acca-57584058b749)

  Generating text is by no means a trivial task! LLMs are optimized to predict the probability of the next token, but how do we generate text with that?

The naive approach is to use the probability vector generated by the model, choose the word with the highest probability, and autoregress. This is the greedy approach, but this tends to generate repetitive sentences that degenerate when they are too long. Another approach is to use the probabilities generated by the model and perform a sampling of the words based on those probabilities. Typically, we use a temperature parameter to adjust the level of randomness of this process. This allows to generate less repetitive and more creative sentences.

But those 2 techniques have a problem. When we generate a sentence, we want to maximize the probability of the whole output sequence and not just the next token:

P(Output sequence | Prompt) 

Fortunately, we can express this probability as a product of the probabilities to predict the next token:

P(token 1, .., token N | Prompt) = P(token 1| Prompt) x ... P(token N |Prompt, token 1, ..., token N - 1)

But solving this problem exactly is an NP-hard problem. So, instead, we can approximate the problem by choosing k candidate tokens at each iteration, testing them, and keeping the k sequences that maximize the probability of the whole sequence. In the end, we just choose the sequence with the highest probability. This is called the Beam search generation and can be mixed with the greedy and the multinomial approach. 

Another approach is the contrastive search, where we take into account additional metrics like fluency or diversity. At each iteration, we choose candidate tokens, penalize the probabilities with a similarity metric of the tokens that were previously generated, and choose the tokens that maximize the new score. 


* Self-attention vs cros-attention

  ![image](https://github.com/ParthaPRay/LLM-Learning-Sources/assets/1689639/dc8623dc-0638-4353-ba69-d4f99600c805)
  What is the difference between Self-Attention and Cross-Attention? They are actually very similar! The self-attention computes the interactions between the different elements of an input sequence (for example, the different words in a sentence), and the cross-attention computes the interactions between the elements of 2 different input sequences (for example, how words in one sentence influence words of the next another sentence).

Both of those attentions can be computed by the same process. We have 3 matrices, Wk, Wq, and Wv, and they project the input vectors into Keys, Queries, and Values vectors. The self-attentions are computed by using the same input vectors, whereas the cross-attentions are computed by using vectors coming from 2 different sources. Those input vectors in the case of self-attention can be internal hidden states within a Transformer, for example, and they can be the encoder output and the internal hidden states of a decoder in the case of an encoder-decoder Transformer for the cross-attentions. For the cross-attentions, the encoder output gets projected as Keys and Values, whereas the decoder hidden states get projected as Queries.

Then, the softmax transformation of the matrix multiplication between Keys and Queries creates the attentions, self, or cross depending on the input vectors. The output of the attention layer is just the matrix multiplication between the attention matrix and the Values vectors. 

* How to handle short sentences in LLMs?

  ![image](https://github.com/ParthaPRay/LLM-Learning-Sources/assets/1689639/cb32709e-1db4-418e-befb-45da6ee33ad0)
  
It is much easier to train Language Models now than it used to be! The amount of text processing needed to obtain usable models was pretty intense. I remember spending many hours testing all the tricks like stemming or lemmatization in Spacy or NLTK!

Now, LLMs can take text pretty much as such. We just need to tokenize it! Tokenizing means we break down the text into sub-word units, but it also means that we need to add special tokens like the beginning or end of sentence tokens ([BOS], [EOS]). One particular token is the Padding token [PAD]. 

When we train LLMs, we apply the batched backpropagation algorithm. To parallelize the computations, we need the input sentences to all have the same length so we can treat the whole batch as one tensor. The [PAD] token allows to pad shorter sentences in the batch. 

Those [PAD] tokens are semantically meaningless, and they should not contribute to the computed attentions within the transformer architecture. The trick is to add a padding mask to the attention computations by setting the elements related to the [PAD] tokens within the attention matrix to zero. This way, they don't contribute to the overall prediction process and text generation. We just need to make sure not to use the hidden states related to those [PAD] tokens for anything other than getting a tensor of the right size!


* How to create tokens from words in LLMs?

  ![image](https://github.com/ParthaPRay/LLM-Learning-Sources/assets/1689639/3495dd90-7ce0-4d97-882d-10709db2fcdb)

  Why do we keep talking about "tokens" in LLMs instead of words? It happens to be much more efficient to break the words into sub-words (tokens) for model performance!

The typical strategy used in most modern LLMs (GPT-1, GPT-2, GPT-3, ChatGPT, Llama 2, etc.) is the Byte Pair Encoding (BPE) strategy. The idea is to use as tokens sub-word units that appear often in the training data. The algorithm works as follows:

- We start with a character-level tokenization
- we count the pair frequencies
- We merge the most frequent pair
- We repeat the process until the dictionary is as big as we want it to be

The size of the dictionary becomes a hyperparameter that we can adjust based on our training data. For example, GPT-1 has a dictionary size of ~40K merges, GPT-2, GPT-3, ChatGPT have a dictionary size of ~50K, and Llama 2 only 32K.


* How masked language moldeling works?

   ![image](https://github.com/ParthaPRay/LLM-Learning-Sources/assets/1689639/79e832c0-43be-4586-a33d-38c2c7767061)

  What is Language Modeling? That is the modeling task of learning the distribution of words in text data. One typical approach is Masked Language Modeling. We mask some tokens of the input data, and we want to predict what were those masked tokens. This has been the original way to train transformers since BERT.

We want to train the model to learn what are the probabilities of the words in the sequence. The prediction matrix for each sample in a batch has a dimension [Sequence size, Vocabulary size]. For each position in the token sequence, we have a probability for each token in the vocabulary. Of course, what interests us the most are the positions where the words are masked in the input data.

To get the prediction matrix with this dimension, we need to be careful about the prediction head we are using. For each input in the sequence, we get a hidden state coming out of the LLM. For each sample within a batch, the resulting tensor coming out of the LLM has a dimension [Sequence size, Hidden state size]. Therefore, the Language modeling head is a simple linear layer with the number of input features to be [Hidden state size] and the number of output features to be [Vocabulary size]. Think about the linear layer as a projection matrix of size [Hidden state size, Vocabulary size] that will resize the hidden state to the vocabulary size.

To train the model, we simply need to compare the predictions for the words that are masked and all the other words are ignored. Typically, we use the cross-entropy loss function for the LLM to learn to predict the masked words.

To generate a sequence at inference time, there might be multiple strategies. The simplest one is to choose the word with the highest predicted probability and to auto-regress. Let’s say we have the first word being “Machine“ as input. Using this as input, we choose the second word in the sequence with the highest probability. Let’s say it is “learning“; now the sequence becomes “Machine learning“. Using those two words as input, we choose the word with the highest probability for the 3rd word in the sequence. We iterate this process until we meet an ending condition, such as the maximum number of tokens or an <END SEQUENCE> token.

* The RNN Encoder-Decoder Architecture

  https://lnkd.in/gnGFsdJe
  
 ![image](https://github.com/ParthaPRay/LLM-Learning-Sources/assets/1689639/65ba0423-f978-43cf-bf7e-6a62ca5f0221)


* Attention mechanisms before transformers
  
  ![image](https://github.com/ParthaPRay/LLM-Learning-Sources/assets/1689639/ce5357f6-ca08-42a3-bbf8-910fdfe64bfd)

  The Attention Mechanism didn't start with Transformers! It was designed to alleviate typical weaknesses related to RNN. The idea was to be able to predict the next word in a sentence by taking into account the signal of all the words in the input sentence.

It was proposed in 2014 by Bahdanau and later improved by Luong in 2015, and it solved some concerns that were seen in the RNN encoder-decoder architecture. Recurring networks generate two types of output vectors: the output vectors at the last layer for each of the input words and the hidden states at the last time step for each layer in the recurring network. Because we may want to generate an output sequence that has a different size than the input sequence, it was considered a better idea to use the hidden states from the encoder encoding the input sequence that would be independent of the input sequence size. This tensor would be used as input to the decoder that was used to decode the output sequence. The hidden states are a tensor representation of the input sequence, but they lose the information related to the different words and their order. The Attention mechanism was just a way to use the output vectors instead that were dependent on the input sequence size and provide more refined information about the input sequence.

* Attention is all you need

  ![image](https://github.com/ParthaPRay/LLM-Learning-Sources/assets/1689639/e8006c86-140e-407d-bf4b-f52d8e41f426)
  
  Transformers are taking every domain of ML by storm! I think it is becoming more and more important to understand the basics, so pay attention because Attention is there to stay!

At the center of Transformers is the self-attention mechanism, and once you get the intuition, it is not too difficult to understand. Let me try to break it down:

As inputs to a transformer, we have a series of contiguous inputs, for example, words (or tokens) in a sentence. When it comes to contiguous inputs, it is not too difficult to see why time series, images, or sound data could fit the bill as well.

Each has its vector representation in an embedding matrix. As part of the attention mechanism, we have 3 matrices Wq, Wk, and Wv, that project each of the input embedding vectors into 3 different vectors: the Query, the Key, and the Value. This jargon comes from retrieval systems, but I don't find them particularly intuitive!

For each word, we take its related Key vector and compute the dot products to the Query vectors of all the other words. This gives us a sense of how similar the Queries and the Keys are, and that is the basis behind the concept of "attention": how much attention should a word pay to another word in the input sequence for the specific learning task? A Softmax transform normalizes and further accentuates the high similarities of the resulting vector. This resulting matrix is called the self-attentions!

This results in one vector for each word. For each of the resulting vectors, we now compute the dot products to the Value vectors of all the other words. We now have computed hidden states or context vectors!

Repeat this process multiple times with multiple attention layers, and this gives you a multi-head attention layer. This helps diversify the learning of the possible relationships between the words. The resulting hidden states are combined into final hidden states by using a linear layer.

The original Transformer block is just an attention layer followed by a set of feed-forward layers with a couple of residual units and layer normalizations. A "Transformer" model is usually multiple Transformer blocks, one after the other. Most language models follow this basic architecture. I hope this explanation helps people trying to get into the field!


* How to augment LLMs with Agents and Tools

  ![image](https://github.com/ParthaPRay/LLM-Learning-Sources/assets/1689639/4ba94d10-1248-4c8b-9e96-080cfd40b0a5)

 ere is how to augment LLMs with tools!

We build a prompt with the following items:

- a list of the possible and description of what they are and how to use them
- the template of the Reasoning-Act (ReAct) prompt technique
- the scratch book showing the results of the previous steps
- the output indicator to guide the LLM in formatting its output correctly

The ReAct technique forces the LLM to think about the next step to solve the question and choose a tool and a tool input to get more information based on that thought. We then extract the tool name and input with Regex and programmatically call the tool with the input and get the response. For example, one tool could be the Python package of the Wikipedia search engine.

We use the tool response to help further the LLM investigation to find the right answer. An agent is a wrapper around an LLM that is augmented with a bunch of tools. The agent iterates until the answer is found:

agent -> prompt with past steps -> LLM -> next steps -> tool -> reponse -> agent -> ...

* Diffusion Models
  
  ![image](https://github.com/ParthaPRay/LLM-Learning-Sources/assets/1689639/07236430-131f-495d-907e-6bd75b78bcdd)

  What is a Diffusion model in Machine Learning? Conceptually, it is very simple! You add some noise to an image, and you learn to remove it. Train a machine learning model that takes as input a noisy image and as output a denoised image, and you have a denoising model.

The typical way to do it is to assume a normal distribution of the noise and to parametrize the distribution mean and standard deviation matrix. Effectively, we can actually reduce the problem to just learning the mean matrix. The process can be divided into the forward process, where white noise (Gaussian distributed) is progressively added to a clean image, and the reverse process, where a learner progressively learns to denoise the noisy image until it is back to being clean: https://lnkd.in/gJ7gRJij.

Why is that called a diffusion model? What does that have to do with the diffusive process of particles in a fluid with a gradient of concentration (https://lnkd.in/gn_FR_Ua)? This is due to the way mathematicians have abused the jargon of the physical process to formalize a mathematical concept. It happens that physical phenomena like Fick diffusion (https://lnkd.in/gKRreTpn), heat diffusion (https://lnkd.in/gB5tWpp6), and Brownian motion (https://lnkd.in/gpKRbkak) are all well described by the diffusion equation: https://lnkd.in/gB5tWpp6, the first time derivative of a state function is equal to the second space derivative of that state function. That diffusion equation has an equivalent stochastic formulation known as the Langevin equation: https://lnkd.in/g9Fjwtxx. At the core of the Langevin equation is a mathematical object called the Wiener process: https://lnkd.in/gmf54HPX. Interestingly enough, this process is also called a Brownian motion (not to be confused with the physical process). It can be thought of as a Random Walk with infinitely small steps: https://lnkd.in/gh6ef5RB. The key feature of the Wiener process is that a time increment of that object is Normal distributed. That is why the concept of "diffusion" is intertwined with the white noise generation process, and that is why those ML models are called diffusion models.

Those diffusion models are generative models as data is generated using a Gaussian prior, and they are the core of the text-to-image generative models such as Stable Diffusion, DALL-E 2, and MidJourney.


* How to summarize texts with LLMs
  
 ![1692207869281](https://github.com/ParthaPRay/LLM-Learning-Sources/assets/1689639/92c5ff75-e4fa-499f-a039-65dcb5424fca)

  With LangChain, it is not difficult to summarize text of any length. To summarize text with a LLM, there are a few strategies.

If the whole text fits in the context window, then you can simply feed the raw data and get the result. LangChain refers to that strategy as the “stuff“ chain type. Often, the number of tokens contained in the text is larger than the LLM's maximum capacity. A typical strategy is to break down the data into multiple chunks, summarize each chunk, and summarize the concatenated summaries in a final "combine" step. LangChain refers to this strategy as “map-reduce“.

Another strategy is to begin the summary with the first chunk and refine it little by little with each of the following chunks. LangChain refers to this as “refine“. For example here is the prompt template used by LangChain for the Refine step:

"""
Your job is to produce a final summary 
We have provided an existing summary up to a certain point: {existing_answer} 
We have the opportunity to refine the existing summary (only if needed) with some more context below.

------------ 
{text}
------------

Given the new context, refine the original summary 
If the context isn't useful, return the original summary.
"""

----


* How to 16x Llama 2's context window size?

![image](https://github.com/ParthaPRay/LLM-Learning-Sources/assets/1689639/518eafc4-4f60-46f2-a3e5-14a8af7b4743)

Did you know that LLama 2 is probably the best choice if you need a large context window? At first glance, LLama 2 has a context window size of 4096, which seems smaller than ChatGPT's 16K, GPT-4's 32K, and Claude 2's 100K, but the magic in the open source!

4096 tokens, that is about 3000 words. Not bad but it limits the possible applications. The typical Transformer architecture is composed of Embeddings to encode the text input, multiple transformer blocks, and a prediction head specific to the learning task the LLM is used for. To encode the text, we use a text embedding matrix T that has the size of the token vocabulary and a positional embedding P that encodes the position of the token in the input sequence. That position embedding size defines the context size. That embedding can be learned or it can be a simple sin function of the position index. Typically they are added together T + P such that the same word is encoded differently at positions i and j.

The great thing about LLama 2 is that it uses Rotary Positional Embeddings (RoPE) as opposed to the typical sin function encoding. Each Attention layer is modified using that embedding and it ensures the computed attention between input tokens to be only dependent on the distance between those tokens. If token T1 is at position i and a token T2 at position j, the attention A(T1, T2) = f(j - i) is a function of j - i. The attention is not dependent on the specific token's locations but on their relative positions. 

The technique they use at Meta to extend the context window is to interpolate at non-integer positions. Basically, if the original window size is L, you can extend it to L' (with L' > L) by rescaling the integer positions

i' = i * L / L'

As an example, if you wanted to have a text input of 16,384 tokens (so 4x the window size of LLama 2) into LLama 2, you would just need to divide every integer position by 4: i' = i / 4. To be clear, if you look at the implementation of LLama 2 available on GitHub (line 50 in model.py today https://lnkd.in/gGvUye6K), you would just need to replace the following line of code 
 
t = torch.arange(end, device=freqs.device) 
by
t = torch.arange(end, device=freqs.device) / 4

How simple is that? Because the model was not trained for that position embedding, you would need to fine-tune a bit the model to adapt it to that new context window and position embedding. When we think that LLama 2 will most likely be used to be fine-tuned on private data, that is the icing on the cake to be able to dynamically adapt the context window to our needs as we fine-tune it.

You can look at the method here: https://lnkd.in/gPUzdBPi. They were able to extend LLama's context window by 16 times while keeping the performance at the same level!




* V-JEPA: Video Joint Embedding Predictive Architecture
  
  ![image](https://github.com/ParthaPRay/LLM-Learning-Sources/assets/1689639/fead596e-d9fd-4029-9fd5-add93f3e8f82)


  ![image](https://github.com/ParthaPRay/LLM-Learning-Sources/assets/1689639/b794d4c2-aa35-4da6-8fbc-6e028f197d5c)

  https://www.youtube.com/watch?v=4X_26j5Z43Y&ab_channel=AIPapersAcademy
  

  https://ai.meta.com/blog/v-jepa-yann-lecun-ai-model-video-joint-embedding-predictive-architecture/
  
  https://github.com/facebookresearch/jepa




* Gorilla
  
Gorilla is a LLM that can provide appropriate API calls. It is trained on three massive machine learning hub datasets: Torch Hub, TensorFlow Hub and HuggingFace. We are rapidly adding new domains, including Kubernetes, GCP, AWS, OpenAPI, and more. Zero-shot Gorilla outperforms GPT-4, Chat-GPT and Claude. 

  
  <img width="1184" alt="image" src="https://github.com/ParthaPRay/LLM-Learning-Sources/assets/1689639/080b8eef-4e6c-44d3-8a43-f966d0f183dd">


Gorilla is extremely reliable, and significantly reduces hallucination errors.Gorilla enables LLMs to use tools by invoking APIs. Given a natural language query, Gorilla comes up with the semantically- and syntactically- correct API to invoke. With Gorilla, we are the first to demonstrate how to use LLMs to invoke 1,600+ (and growing) API calls accurately while reducing hallucination. We also release APIBench, the largest collection of APIs, curated and easy to be trained on! Join us, as we try to expand the largest API store and teach LLMs how to write them! Hop on our Discord, or open a PR, or email us if you would like to have your API incorporated as well.

https://gorilla.cs.berkeley.edu/


https://github.com/ShishirPatil/gorilla

https://colab.research.google.com/drive/1DEBPsccVLF_aUnmD0FwPeHFrtdC0QIUP?usp=sharing






* **Mixture of Experts (MoEs)**


   
        * What is a Mixture-of-Experts (MoE)?

           ![image](https://github.com/ParthaPRay/LLM-Learning-Sources/assets/1689639/536aecab-1e37-46d2-b2c8-82711b7f03cd)
          
        * towards understanding mixture of experts in deep learning

           https://arxiv.org/abs/2208.02813

        * Mixture-of-Experts Meets Instruction Tuning: A Winning Combination for Large Language Models

          https://arxiv.org/abs/2305.14705

        * Mixture of Experts Explained
      
          https://huggingface.co/blog/moe

        * Welcome Mixtral - a SOTA Mixture of Experts on Hugging Face

          https://huggingface.co/blog/mixtral

        * SegMoE: Segmind Diffusion Mixture of Experts (MoEs) Model,  https://www.youtube.com/watch?v=gIz7Td6WfEo
      
        * Mixtral Fine tuning and Inference, https://www.youtube.com/watch?v=EXFbZfp8xCI&ab_channel=TrelisResearch
       
        * Understanding Mixture of Experts, https://www.youtube.com/watch?v=0U_65fLoTq0&ab_channel=TrelisResearch
      
        * How To Install Uncensored Mixtral Locally For FREE! (EASY), https://www.youtube.com/watch?v=DC2te4CZXeM&ab_channel=WorldofAI
      
        * Fully Uncensored MIXTRAL Is Here 🚨 Use With EXTREME Caution, https://www.youtube.com/watch?v=q2KpPUOsBCs&ab_channel=MatthewBerman
      
        * Depliy your AI Streamlit App, https://youtu.be/74c3KaAXPvk?si=mHuW18-fvW1sJswn
          
        * makeMoE: Implement a Sparse Mixture of Experts Language Model from Scratch
      
          https://huggingface.co/blog/AviSoori1x/makemoe-from-scratch
      
          https://github.com/AviSoori1x/makeMoE/tree/main
      



* How to play a Chess Game ChatGPT and Llama 2

  ![1690391653971](https://github.com/ParthaPRay/LLM-Learning-Sources/assets/1689639/68421e27-c225-4508-947d-d3aff345530c)

It is not tomorrow that LLama 2 is going to replace ChatGPT, and it is not tomorrow that those LLMs are going to take over the world! In my opinion, LLama 2 only makes sense if you need to fine tune your model with your own data. The biggest LLama 2 model is 70B of parameters. With 4 bytes per parameter, that's a 240 GB model, so count ~400GB of GPU hardware to have one standing model for inference. Using AWS GPU pricing, that's $4 / hr on the low end. With ChatGPT on the other hand, the cost is $0.0015/1K tokens. If you count 4 tokens per word, to get to $4/hr, you need to send 700K words / hr to the API. That's about 10 books with 300 pages each. If your model consumes less input than that, don't bother with LLama2. 

A fine-tuned model is another story. For both models, you need to swallow the training cost, but LLama inference's cost remains the same where the inference on a fine-tuned GPT-3 is 0.12 / 1K (~100 times the cost of the non-fine-tuned model) as OpenAI charges very differently for hosting custom models.

In terms of performance evaluation, what about a little chess tournament? I used the [replicate API to use LLama] (https://replicate.com/meta/llama-2-70b-chat) and the OpenAI API for ChatGPT and GPT-4. The AiEdge used the [Python Chess package for the game structure] (https://python-chess.readthedocs.io/en/latest/). The AiEdge feed the current state of the board, the history of the past moves and the current available legal moves within the prompt to guide the LLMs. After multiple rounds, ChatGPT destroyed LLama, it was a tie between GPT-4 and LLama and a tie between GPT-4 and ChatGPT (for some reason!). GPT-4 was not the greatest at chess but it was great at making a big hole in my bank account due to its cost! LLama seemed to play like a bored gold fish, moving the same pieces back and forth, not being really clear on what it was supposed to do. 

The AiEdge tried to use the non-official Bard API (https://lnkd.in/gJUGA4fV) but that model is about as good as a 3 year old toddler listening to commands within the prompts. Whatever way I would engineer my prompts, Bard could not follow the basic instructions to get my code to work and would ramble like a drunk Woody Allen so The AiEdge gave up. Painful experience!

The AiEdge would have loved to get Claude 2 to participate but Anthropic keeps "forgetting" to provide API access to their customers. The AiEdge used a chess engine (https://lnkd.in/dG8TvhBQ) to compare and it crushed any of the LLMs in a few moves every time. It seems that LLMs are unable to form coherent strategies to solve these kinds of problems. LLMs are not ready to replace us anytime soon!


* Merge Large Language Models with mergekit
  https://huggingface.co/blog/mlabonne/merge-models


  https://colab.research.google.com/drive/1_JS7JKJAQozD48-LhYdegcuuZ2ddgXfr?usp=sharing

* Merge LLMs with Mergekit: create your own medical mixture of experts
  
  https://youtu.be/eKDz-K3UvbY?si=limrl7Raf86bdqS7






# Transformers


* Want to understand the Transformer architecture?

   - the encoder
   - the decoder
   - the position embedding
   - the encoder block
   - the self-attention layer
   - the layer-normalization
   - the position-wise feed-forward network 
   - the decoder block
   - the cross-attention layer
   - the predicting head
  
   ![image](https://github.com/ParthaPRay/LLM-Learning-Sources/assets/1689639/4e058dd1-f753-4728-a5b5-a16547b9cbb9)


  ![image](https://github.com/ParthaPRay/LLM-Learning-Sources/assets/1689639/345e49b2-d3fc-4493-ae97-724910c242de)


  ![image](https://github.com/ParthaPRay/LLM-Learning-Sources/assets/1689639/f0a2092b-6b21-4a01-aa3a-6cbd54a8946f)


  ![image](https://github.com/ParthaPRay/LLM-Learning-Sources/assets/1689639/1d85bd3e-29c3-4b24-840f-6e235246e6be)


  ![image](https://github.com/ParthaPRay/LLM-Learning-Sources/assets/1689639/78e4d46c-50cf-4e40-a2bf-98ba17b8d53b)

   ![image](https://github.com/ParthaPRay/LLM-Learning-Sources/assets/1689639/64d0f0f3-b434-4532-b421-73c2c19bd48d)


  ![image](https://github.com/ParthaPRay/LLM-Learning-Sources/assets/1689639/6856f4e1-ce64-4eb9-bae0-1b4ac7198e81)

  ![image](https://github.com/ParthaPRay/LLM-Learning-Sources/assets/1689639/713b6435-7152-4944-8a78-ac2427a25505)


  ![image](https://github.com/ParthaPRay/LLM-Learning-Sources/assets/1689639/fb067d72-b164-49e0-952a-780dd238ffac)



* How to feed data to a Transformer

  ![image](https://github.com/ParthaPRay/LLM-Learning-Sources/assets/1689639/5a363911-4186-4534-86e1-b7a187495c8a)

  If you think about Transformers, chances are you are thinking about NLP applications, but how can we use Transformers for data types other than text? Actually, you can use Transformers on any data that you are able to express as a sequence of vectors, which is what Transformers feed on! Typically, any sequence or time series of data points should be able to fit the bill.

Let's consider image data, for example. An image is not per se a sequence of data, but the local correlation of the pixels sure resembles the concept. For the Vision Transformer (ViT: https://lnkd.in/gPC_iFaV), the guys at Google simply created patches of an image that were flattened through linear transformations into a vector format. By feeding images to Transformers through this process, they realized that typical CNNs were performing better on a small amount of data, but Transformers were getting better than CNNs if the scale of the data was very high.

Time series are obviously good candidates for Transformers. For example, for the Temporal Fusion Transformer (https://lnkd.in/gfMTHYBc), they transform the time series into the right-sized vector through LSTM layers, as they say, to capture the short-term correlations of the data where the multihead attention layers take care of capturing the long term correlations. They beat all the time series benchmarks with this model, but I wonder how scalable it is with those LSTM layers! You can use it in PyTorch: https://lnkd.in/gzisFCUF

Sequencing proteins seems to be an obvious application of Transformers, considering the language analogy of amino acid sequences. Here, you just need to have an amino acid embedding to capture the semantic representation of protein unit tokens. Here is a Nature article on generating new proteins with Transformers: https://lnkd.in/gzeiuZ8w, and here is its BioaRXiv version: https://lnkd.in/gQgHg-sm.

Reinforcement Learning expressed at a Markov chain sequence of states, actions, and rewards is another good one. For the Decision Transformer (https://lnkd.in/giJCnXJX), they encoded each state, action, and reward as a vector and concatenated them into 1 final vector. For example, in the case of video games, a state can simply be the image on the screen at a time t, and you extract the latent features with a CNN. An action can be encoded with embedding, and a scalar reward can be seen as a vector with 1 dimension. Apparently, they beat all the benchmarks as well! You can find the code here: https://lnkd.in/gwFdrZHX. 

Looking forward to seeing what Transformers are going to achieve in the coming years!
  


* What are Transformers and GPTs?, https://www.youtube.com/watch?v=ucityipiNtA&ab_channel=RicardoCalix

* High overview of the original Transformer architecture for Large Language Models, https://www.youtube.com/watch?v=zxVhAYkSYcY&ab_channel=RicardoCalix

*  Coding a Transformer from scratch on Pytorch with full explanation training and Inference, https://youtu.be/ISNdQcPhsts?si=EA3BSRVo1Tr4Z4NC
  
    * GPTs, BERTs, Full Transformers, in PyTorch (Part 1), https://www.youtube.com/watch?v=s6gys0iozLk&ab_channel=RicardoCalix
    * GPTs, BERTs, Full Transformers, in PyTorch (Part 2), https://www.youtube.com/watch?v=a1qomZy_yfo&ab_channel=RicardoCalix
    * GPU Scholar cloud, GPTs, BERTs, Full Transformers, in PyTorch (Part 3), https://www.youtube.com/watch?v=klQnQMoy9zI&ab_channel=RicardoCalix
    * Embeddings, GPTs, BERTs, Full Transformers, in PyTorch (Part 4), https://www.youtube.com/watch?v=yNZCcF6a7a4&ab_channel=RicardoCalix
    * The simple linear algebra for Attention, GPTs, BERTs, and Full Transformers in PyTorch (part 5), https://www.youtube.com/watch?v=zgH69JoAB_k&ab_channel=RicardoCalix

* Implementing a simple GPT in PyTorch, https://www.youtube.com/watch?v=RsQxg913eXY&ab_channel=RicardoCalix
* Implementing a simple GPT in PyTorch (Take Two), https://www.youtube.com/watch?v=zyDzpVu9lyA&ab_channel=RicardoCalix


* Starting with GPTs (A Hello World Example), https://www.youtube.com/watch?v=oPcJg3QrKf4&ab_channel=RicardoCalix
  
* Intro to Reinforcement Learning through Human Feedbacks (RLHF), https://www.youtube.com/watch?v=A8YqZKGRTAM&ab_channel=RicardoCalix


* What is an instruct model? - Instruction and Chat Fine-Tuning,

  As you browse the ever growing global catalogue of generative AI models, you will see some of the Large Language Models (LLMs) being listed with the suffix 'instruct' or 'chat'. What does this mean?
  
  TL:DR; The 'instruct' version of the model has been fine-tuned to be able to follow prompted instructions. These models 'expect' to be asked to do something. Models with the 'chat' suffix have been fine-tuned to work in chatbots. These models 'expect' to be involved in a conversation with different actors. In contrast non-instruct tuned models will simply generate an output that follows on from the prompt. If you are making a chatbot, implementing RAG or using agents, use instruct or chat models. If in doubt us an instruct model.

https://community.aws/content/2ZVa61RxToXUFzcuY8Hbut6L150/what-is

* **Stanford CS25 - Transformers United** **Course**

  https://www.youtube.com/playlist?list=PLoROMvodv4rNiJRchCzutFw5ItR_Z27CM


 ![image](https://github.com/ParthaPRay/LLM-Learning-Sources/assets/1689639/4bc1a0d7-fcc7-4c01-8d44-b9b7f381785c)

 When we think about Transformers, we tend to think about LLMs, but it revolutionized the world of Computer Vision as well! The Vision Transformer has slowly been replacing typical convolutional networks when it comes to image analysis tasks.

Nothing in the Transformer architecture is intrinsically bound to NLP applications! As long as you can format your data into a sequence of vectors, you can feed it to a Transformer. It might feel odd to think about an image as a sequence of vectors, though.

The idea is to build visual tokens by breaking down the image into patches of pixels and flattening them down into vectors through a linear transformation. With a convolutional layer, we can transform an image into a sequence of vectors in one shot. As soon as we have vectors, we can pass them into a Transformer, as you would any textual tokens. 










  

# Ollama

  * Ollama, https://github.com/ollama/ollama
    
  * Installing Ollama to Customize My Own LLM,    https://www.youtube.com/watch?v=xa8pTD16SnM&ab_channel=Decoder
  
  * Use Your Self-Hosted LLM Anywhere with Ollama Web UI,   https://www.youtube.com/watch?v=syR0fT0rkgY&ab_channel=Decoder

  * Importing Open Source Models to Ollama, https://www.youtube.com/watch?v=fnvZJU5Fj3Q&ab_channel=Decoder
 
  * Ollama has a Python library!, https://www.youtube.com/watch?v=JwYwPiOh72w&ab_channel=LearnDatawithMark
    
  * Building a local ChatGPT with Chainlit, Mixtral, and Ollama, https://www.youtube.com/watch?v=MiJQ_zlnBeo&ab_channel=LearnDatawithMark
    
  * Langroid: Chat to a CSV file using Mixtral (via Ollama), https://www.youtube.com/watch?v=XFTFEKYLxyU
    
  * Few Shot Prompting with Llama2 and Ollama, https://www.youtube.com/watch?v=ocfzGBnhhDE
 
  * Hugging Face GGUF Models locally with Ollama, https://www.youtube.com/watch?v=7BH4C6-HP14&ab_channel=LearnDatawithMark
 
  * Autogen: Ollama integration 🤯 Step by Step Tutorial. Mind-blowing!, https://www.youtube.com/watch?v=UQw04VW60U0&ab_channel=MervinPraison
    
  * Writing Better Code with Ollama, https://www.youtube.com/watch?v=NNBWmIve3fQ&ab_channel=MattWilliams
 
  * Ollama meets LangChain, https://www.youtube.com/watch?v=k_1pOF1mj8k&ab_channel=SamWitteveen
    
  * Running Mixtral on your machine with Ollama, https://www.youtube.com/watch?v=rfr4p0srlqs&ab_channel=LearnDatawithMark
    
  * Running Mistral AI on your machine with Ollama, https://www.youtube.com/watch?v=NFgEgqua-fg&ab_channel=LearnDatawithMark
    
  * Ollama Python Library Released! How to implement Ollama RAG? https://www.youtube.com/watch?v=4HfSfFvLn9Q&ab_channel=MervinPraison

  * Ollama Web UI 🤯 How to run LLMs 100% LOCAL in EASY web interface? CRAZY!!🚀 (Step-by-Step Tutorial), https://www.youtube.com/watch?v=84vGNkW1A8s&ab_channel=MervinPraison
 
  * How TO Install Ollama Web UI | ChatGPT LIKE Interface, https://www.youtube.com/watch?v=bt4AR7sK9tk&ab_channel=DataScienceBasics
 
  * Ollama: The Easiest Way to Run Uncensored Llama 2 on a Mac, https://www.youtube.com/watch?v=tIRx-Sm3xDQ&ab_channel=IanWootten
 
  * Using Ollama To Build a FULLY LOCAL "ChatGPT Clone", https://www.youtube.com/watch?v=rIRkxZSn-A8&ab_channel=MatthewBerman

  * Hugging Face GGUF Models locally with Ollama, https://www.youtube.com/watch?v=7BH4C6-HP14&t=8s&ab_channel=LearnDatawithMark
 
  * Using the Chat Endpoint in the Ollama API, https://www.youtube.com/watch?v=QUJHEvCqhdw&ab_channel=MattWilliams
 
  * Adding Custom Models to Ollama, https://www.youtube.com/watch?v=0ou51l-MLCo&t=211s&ab_channel=MattWilliams
 
  * Finally Ollama has an OpenAI compatible API, https://www.youtube.com/watch?v=38jlvmBdBrU&ab_channel=MattWilliams

  * Hosting Ollama Starts With Environment Variables, https://www.youtube.com/watch?v=H_cqBjDVinw&ab_channel=MattWilliams

  * Understanding How Ollama Stores Models, https://www.youtube.com/watch?v=6bF1uCHTFyk&ab_channel=MattWilliams
    
  * Run any AI model remotely for free on google colab, https://www.youtube.com/watch?v=Qa1h7ygwQq8&ab_channel=TechwithMarco

    https://github.com/marcogreiveldinger/videos/tree/main/ollama-ai/run-on-colab

  * Run Mixtral 8x7B MoE in Google Colab, https://www.youtube.com/watch?v=Zo3CTapKJ4I&ab_channel=PromptEngineering

    https://github.com/dvmazur/mixtral-offloading?tab=readme-ov-file

    https://huggingface.co/lavawolfiee/Mixtral-8x7B-Instruct-v0.1-offloading-demo

  * Run Mixtral 8x7B Hands On Google Colab for FREE | End to End GenAI Hands-on Project
 
    https://www.youtube.com/watch?v=vzUJ-yjA8Bw&ab_channel=AnalyticsVidhya

    https://drive.google.com/drive/folders/1Bo4sJu9vEnjzV_h4FmBNb6dSZ8BxZxpa

    https://drive.google.com/drive/folders/1AuReI63WzKRSdzRIlCxl6WuBkNMryPv9

  * Unleash the power of Local LLM's with Ollama x AnythingLLM, https://www.youtube.com/watch?v=IJYC6zf86lU&ab_channel=TimCarambat

    Any LLM, unlimited documents, and fully private. All on your desktop. https://useanything.com/download

  * Ollama: How To Create Custom Models From HuggingFace ( GGUF ), https://www.youtube.com/watch?v=TFwYvHZV6j0&t=72s&ab_channel=DataScienceBasics

  * How to run Ollama on Docker, https://www.youtube.com/watch?v=ZoxJcPkjirs&t=127s&ab_channel=MattWilliams
  * Easy Access to GPUs for Ollama, https://www.youtube.com/watch?v=QRot1WtivqI&ab_channel=MattWilliams

    Fine-tune, train, or deploy. Use your own notebook, or one of ours. SSH too. CUDA, Python, Jupyter Lab, all set up.

    https://brev.dev/

    Tailscale is a zero config VPN for building secure networks. Install on any device in minutes. Remote access from any network or physical location.

     https://tailscale.com/

 *  Using Ollama as a local LLM for chat apps

  https://www.youtube.com/watch?v=zEN_oKrttK0&ab_channel=PamelaFox

* How to Access Ollama Model with Public IP Remotely

 https://www.youtube.com/watch?v=QSfvLWaJc2s&t=20s&ab_channel=FahdMirza


* Let's use Ollama's Embeddings to Build an App

  ![image](https://github.com/ParthaPRay/LLM-Learning-Sources/assets/1689639/48225956-67b8-431b-9f03-bc6f88e2ff1e)

 https://www.youtube.com/watch?v=6QAIbThWomc&ab_channel=MattWilliams

 https://github.com/technovangelist/videoprojects


* JSON agents with Ollama & LangChain

  Learn to implement an open-source Mixtral agent that interacts with a graph database Neo4j through a semantic layer

   ![image](https://github.com/ParthaPRay/LLM-Learning-Sources/assets/1689639/100767aa-5125-4a47-b2f4-cd2c3177675d)

  https://blog.langchain.dev/json-based-agents-with-ollama-and-langchain/






# Fine Tuning




    
  * Fine-tuning Tiny LLM on Your Data | Sentiment Analysis with TinyLlama and LoRA on a Single GPU, https://www.youtube.com/watch?v=_KPEoCSKHcU&ab_channel=VenelinValkov
    
  * Make LLM Fine Tuning 5x Faster with Unsloth, https://www.youtube.com/watch?v=sIFokbuATX4&ab_channel=AIAnytime
  
  * Fine-Tuning Your Own Llama 2 Model, https://www.youtube.com/watch?v=Pb_RGAl75VE&ab_channel=DataCamp
  
  * Fine Tune a Multimodal LLM "IDEFICS 9B" for Visual Question Answering, https://www.youtube.com/watch?v=usoTCfyQxjU&ab_channel=AIAnytime
  
  * Anyone can Fine Tune LLMs using LLaMA Factory: End-to-End Tutorial, https://www.youtube.com/watch?v=iMD7ba1hHgw&t=15s&ab_channel=AIAnytime
  
  * Fine Tune Phi-2 Model on Your Dataset, https://www.youtube.com/watch?v=eLy74j0KCrY&ab_channel=AIAnytime
  
  * LLM Fine Tuning Crash Course: 1 Hour End-to-End Guide, https://www.youtube.com/watch?v=mrKuDK9dGlg
  
  * Fine-tuning LLMs with PEFT and LoRA, https://www.youtube.com/watch?v=Us5ZFp16PaU&ab_channel=SamWitteveen
  
  *  🤗 PEFT welcomes new merging methods
        
    https://huggingface.co/blog/peft_merging

  * Prompt Tuning With PEFT

      ![image](https://github.com/ParthaPRay/LLM-Learning-Sources/assets/1689639/a7e31223-0703-4710-b2e9-e815ed4a3129)


    https://huggingface.co/learn/cookbook/prompt_tuning_peft

    https://colab.research.google.com/github/huggingface/cookbook/blob/main/notebooks/en/prompt_tuning_peft.ipynb
 

  * Train a Small Language Model for Disease Symptoms | Step-by-Step Tutorial, https://www.youtube.com/watch?v=1ILVm4IeNY8&ab_channel=AIAnytime
 
  * Fine tuning Whisper for Speech Transcription, https://www.youtube.com/watch?v=anplUNnkM68&ab_channel=TrelisResearch
 
  * Efficient Fine-Tuning for Llama-v2-7b on a Single GPU, https://www.youtube.com/watch?v=g68qlo9Izf0&t=17s&ab_channel=DeepLearningAI


   
  * Direct Preference Optimization (DPO), https://www.youtube.com/watch?v=E5kzAbD8D0w&ab_channel=TrelisResearch

 
  * Fine Tune LLaMA 2 In FIVE MINUTES! - "Perform 10x Better For My Use Case", https://www.youtube.com/watch?v=74NSDMvYZ9Y&ab_channel=MatthewBerman
 
  * How to Fine-Tune Mistral 7B on Your Own Data, https://www.youtube.com/watch?v=kmkcNVvEz-k&ab_channel=brev
    
  * Fine-Tune Your Own Tiny-Llama on Custom Dataset, https://www.youtube.com/watch?v=OVqe6GTrDFM&ab_channel=PromptEngineering
 
  * Fine-tune Mixtral 8x7B (MoE) on Custom Data - Step by Step Guide, https://www.youtube.com/watch?v=RzSDdosu_y8&ab_channel=PromptEngineering
    
  * Mistral: Easiest Way to Fine-Tune on Custom Data, https://www.youtube.com/watch?v=lCZRwrRvrWg&ab_channel=PromptEngineering
    
  * Self-Play Fine-Tuning (SPIN), https://www.youtube.com/watch?v=khPq69GgPAo&ab_channel=FahdMirza

    The official implementation of Self-Play Fine-Tuning (SPIN), https://github.com/uclaml/SPIN
    
    https://uclaml.github.io/SPIN/

    
  * Building Production-Ready RAG Applications: Jerry Liu, https://www.youtube.com/watch?v=TRjq7t2Ms5I&t=10s&ab_channel=AIEngineer

  * Custom Fine-tuning 30x Faster on T4 GPUs with UnSloth AI, https://www.youtube.com/watch?v=R4CUKAHShyE&ab_channel=PromptEngineering

    https://unsloth.ai/introducing

  *  To Fine Tune or not Fine Tune? That is the question, https://www.youtube.com/watch?v=XPU8PH0_d6g&ab_channel=SethJuarez

  * Fine-tune TinyLlama 1.1B locally on own custom dataset, https://youtu.be/VoDHpnCN6PA?si=Aq7soXO6k83mJJVs
    
  * Llama Factory: How to Fine-Tune LLMs easily?, https://youtu.be/G5ENOwfPHFE?si=2BZ6Zh5x55TDr2dl
    
  * How to create custom datasets to train Llama-2? https://youtu.be/z2QE12p3kMM?si=j52ptrx0GMnj9OSy
  
  * LocalGPT: Convert your chats with Docs to Fine-Tuing datasets, https://youtu.be/2_o6epQToVY?si=CZMdu1u2IU0wXUz8
    
  * D2SLM (Doc to Dataset to Fine-Tune Small Language Model), https://www.youtube.com/watch?v=khIDeJwBf4k&ab_channel=AIMakerspace
    
  * LLAMA-2 🦙: EASIET WAY To FINE-TUNE ON YOUR DATA 🙌, https://www.youtube.com/watch?v=LslC2nKEEGU&t=2s&ab_channel=PromptEngineering
 
  * The EASIEST way to finetune LLAMA-v2 on local machine!, https://www.youtube.com/watch?v=3fsn19OI_C8&ab_channel=AbhishekThakur
    
  * Stable Diffusion XL (SDXL) DreamBooth: Easy, Fast & Free | Beginner Friendly, https://www.youtube.com/watch?v=3fsn19OI_C8&ab_channel=AbhishekThakur
    
  * Fine-tuning Notebook on how to fine-tune MPT-7B on a free Google Colab instance to turn the model into a Chatbot. MPT7b sharded version + LoRA adapter  

    https://colab.research.google.com/drive/1HCpQkLL7UXW8xJUJJ29X7QAeNJKO0frZ?usp=sharing

    Dataset: https://huggingface.co/datasets/timdettmers/openassistant-guanaco

    
  * Fine tuning with LlamaIndex

    https://docs.llamaindex.ai/en/stable/optimizing/fine-tuning/fine-tuning.html
    
  * Fine tuning Google Colab notebook - This notebook shows how to fine-tune a 4bit model on a downstream task using the Hugging Face ecosystem. We show that it is possible to fine tune _GPT-neo-X 20B_ on a Google Colab instance!

     https://colab.research.google.com/drive/1VoYNfYDKcKRQRor98Zbf2-9VQTtGJ24k?usp=sharing
     
  * Fine Tune pre-trained GPT and BERT models with the Huggingface library, https://www.youtube.com/watch?v=g1dAsgibRcw&ab_channel=RicardoCalix
 
  * Fine-Tuning HF examples on GPU Scholar, scratch disk space, https://www.youtube.com/watch?v=_S01y-JY8k4&ab_channel=RicardoCalix

  * Fine-tune Multi-modal Vision and Language Models, https://www.youtube.com/watch?v=eIziN2QUt8U&ab_channel=TrelisResearch

 * Fine-Tuning Gemma Models in Hugging Face

   https://huggingface.co/blog/gemma-peft

 
  * The AiEdge+: How to fine-tune Large Language Models with Intermediary models

    https://newsletter.theaiedge.io/p/the-aiedge-how-to-fine-tune-large


* ReFT: Reasoning with Reinforced Fine-Tuning
  
  Aligning LLMs: ReFT

  https://www.youtube.com/watch?v=K_8a056X4ys&ab_channel=AIMakerspace
  



* Dickens: an LLM that writes Great Expectations

https://colab.research.google.com/drive/1MdZvYtm3xrkPrxzD71SZ6H9GTkG46VRF?usp=sharing

* Question Answering on FAQs of GST (Goods and Services Tax) in India

https://medium.com/analytics-vidhya/how-to-fine-tune-llms-without-coding-41cf8d4b5d23

https://colab.research.google.com/drive/1RQc035W1_7CTEViYrsnRwYvOtObvXo-B?usp=sharing

* Intent Classification with LLMs: Fine-Tuning on Support Call Transcripts using Ludwig

https://colab.research.google.com/drive/17fmNaq-2KwqJLHt4ZZ0X6FbmMlssq_vR?usp=sharing


* Democratize and Automate the Feature Engineering of Tabular Data using fine-tuned LLMs


https://colab.research.google.com/drive/1NLmQqbiXc-dU9C0ulNsUuubB3vbhaJbi?usp=sharing

* Assessing Health Data with ML and Becoming More Aware
  
https://colab.research.google.com/drive/16Ofyeg2wse1UFEMwROCN5qqWHKgWZNIR?usp=sharing

  * NODES 2023 - Fine-Tuning an Open-Source LLM for Text-to-Cypher Translation
    https://www.youtube.com/watch?v=TB6URe5f3MA&ab_channel=Neo4j
    
  * Fine-tuning a Code LLM on Custom Code on a single GPU

    https://github.com/huggingface/cookbook/tree/main/notebooks/en

  * Fine-tuning Zephyr-7B to znalyze customer support call logs

   https://youtu.be/cwT5JAqtTM4?si=x5NZgXKzgNx6xlt-

    https://pbase.ai/ZephyrWebinarSlides

   https://pbase.ai/ZephyrCustomerSupport


* Building an LLM fine-tuning dataset,
  
https://youtu.be/pCX_3p40Efc?si=UKvB7DSVb366Zzbe

https://github.com/Sentdex/LLM-Finetuning


  * Fine tuning LLMs for Memorization

    https://www.youtube.com/watch?v=_GkHZQYFOGM&ab_channel=TrelisResearch

    https://docs.google.com/presentation/d/1Un-H9d3ghlR23VddD3aR8aSWHHg9vjIwvYC45o0Vn7g/edit?usp=sharing

    https://huggingface.co/datasets/Trelis/touch-rugby-rules-memorisation
    
  *  Fine-tuning a large language model on Kaggle Notebooks (or even on your own computer) for solving real-world tasks
  
    https://huggingface.co/blog/lmassaron/fine-tuning-llms-on-kaggle-notebooks

         Code references:

         Fine-tune Llama-2 for Sentiment Analysis: https://www.kaggle.com/code/lucamassaron/fine-tune-llama-2-for-sentiment-analysis
         Fine-tune Mistral v0.2 for Sentiment Analysis: https://www.kaggle.com/code/lucamassaron/fine-tune-mistral-v0-2-for-sentiment-analysis
         Fine-tune Phi 2 for Sentiment Analysis: https://www.kaggle.com/code/lucamassaron/fine-tune-phi-2-for-sentiment-analysis
         LSTM Baseline for Sentiment Analysis): https://www.kaggle.com/code/lucamassaron/lstm-baseline-for-sentiment-analysis



  * Phinetuning 2.0

    Meet Phi-2, Microsoft’s newly released small model, remarkably powerful yet compact. This tutorial will guide you through fine-tuning Phi-2, demonstrating how to build a unique dataset and fine-tune the model using QLoRA.

    https://huggingface.co/blog/g-ronimo/phinetuning

 
  * Fine-tuning Language Models for Structured Responses with QLoRa, https://www.youtube.com/watch?v=OQdp-OeG1as&ab_channel=TrelisResearch


  * Fine-tuning Llama 2 on Your Own Dataset | Train an LLM for Your Use Case with QLoRA on a Single GPU, https://www.youtube.com/watch?v=MDA3LUKNl1E&ab_channel=VenelinValkov

    https://github.com/curiousily/Get-Things-Done-with-Prompt-Engineering-and-LangChain

    
  * Easily Train a Specialized LLM: PEFT, LoRA, QLoRA, LLaMA-Adapter, and More, https://cameronrwolfe.substack.com/p/easily-train-a-specialized-llm-peft#:~:text=LoRA%3A%20Low%2DRank%20Adaptation%20of%20Large%20Language%20Models%20%5B1%5D&text=LoRA%20leaves%20the%20pretrained%20layers,of%20the%20model%3B%20see%20below.
    
  * Efficient Fine-Tuning for Llama 2 on Custom Dataset with QLoRA on a Single GPU in Google Colab, https://www.youtube.com/watch?v=YyZqcNo4hdo&pp=ygUQZmluZSB0dW5pbmcgTExNXA%3D%3D

 * QLoRA—How to Fine-tune an LLM on a Single GPU (w/ Python Code), https://www.youtube.com/watch?v=XpoKB3usmKc&ab_channel=ShawTalebi

    https://colab.research.google.com/drive/1AErkPgDderPW0dgE230OOjEysd0QV1sR?usp=sharing


  * Preference Tuning LLMs with **Direct Preference Optimization** Methods

    https://huggingface.co/blog/pref-tuning

  * Fine-tune Llama 2 with DPO
    
    https://huggingface.co/blog/dpo-trl


  * Practical Fine-Tuning of LLMs

    https://www.youtube.com/watch?v=Jp-6hyf_CoE&ab_channel=AIMakerspace

    https://www.canva.com/design/DAF-v_5WxcU/s2SCPuVA7ikGR0VSJOG6Rw/view?utm_content=DAF-v_5WxcU&utm_campaign=designshare&utm_medium=link&utm_source=editor
    
    https://colab.research.google.com/drive/1Jw9jthx_S62MPwKH9lqb6xPRwec4OiI6?usp=sharing
    

  * How to Train a Multi Modal Large Language Model with Images?

    https://huggingface.co/HuggingFaceM4/idefics-9b

    https://www.youtube.com/watch?v=ojjIYAbWP6U&ab_channel=MervinPraison
   
  * Fine-tuning Llama 2 70B using PyTorch FSDP
    
    https://huggingface.co/blog/ram-efficient-pytorch-fsdp


 * How to fine tune a model locally on mistralai/Mistral-7B-Instruct-v0.2 using HuggingFaceTB/cosmopedia-20k or Elriggs/openwebtext-100k dataset

   https://youtu.be/9GjLAyn12MU?si=NYd1BmNv4vfVtde4

   https://huggingface.co/cloudyu/mistral_pretrain_demo


 * Fine-Tune Whisper For Multilingual ASR with 🤗 Transformers

 https://huggingface.co/blog/fine-tune-whisper

 https://colab.research.google.com/github/sanchit-gandhi/notebooks/blob/main/fine_tune_whisper.ipynb

 

 * Fine-Tune W2V2-Bert for low-resource ASR with 🤗 Transformers

   https://huggingface.co/blog/fine-tune-w2v2-bert

 * Google Gemma Finetuning: how to teach a large language model?,
    https://youtu.be/RevZAM9taFk?si=QuNJAVrLdqs7SUgE

 * Steps to Master Fine Tuning LLMs To Ultimate AI Proficiency : A Definitive Guide

   https://www.youtube.com/watch?v=GK860luUyEk&ab_channel=KamalrajMM

* Fine tuing optimization DoRA, NEFT, LoRA+, Unsloth
  
https://youtu.be/ae2lbmtTY5A?si=0NXaw8tOXqh800x2

  supervised fine tuning 
  https://huggingface.co/docs/trl/main/en/sft_trainer


* Building with Instruction-Tuned LLMs: A Step-by-Step Guide

  https://www.youtube.com/watch?v=eTieetk2dSw&ab_channel=DeepLearningAI

    ![image](https://github.com/ParthaPRay/LLM-Learning-Sources/assets/1689639/144ea4d5-62cd-4bc1-8a72-c660f354f149)





* Fine Tune Large Language Model (LLM) on a Custom Dataset with QLoRA

  https://dassum.medium.com/fine-tune-large-language-model-llm-on-a-custom-dataset-with-qlora-fb60abdeba07

* Unsloth: How to Train LLM 5x Faster and with Less Memory Usage?

   ![image](https://github.com/ParthaPRay/LLM-Learning-Sources/assets/1689639/6f6e4936-84d7-46df-a573-073bf3f54494)

     https://www.youtube.com/watch?v=Gpyukc6c0w8&t=16s&ab_channel=MervinPraison




* ​**Direct Preference Optimization (DPO)**

    - [Direct Preference Optimization (DPO)](https://huggingface.co/papers/2305.18290)
    - [Identity Preference Optimisation (IPO)] (https://huggingface.co/papers/2310.12036)
    - [Kahneman-Tversky Optimisation (KTO)] (https://github.com/ContextualAI/HALOs)

  https://huggingface.co/blog/pref-tuning

* **Reinforcement Learning with AI Feedback - RLAIF** Github Link

  Reinforcement Learning from AI Feedback (RLAIF) is a concept that describes a type of machine learning approach where an AI agent learns by receiving feedback or guidance from another AI system. This concept is closely related to the field of Reinforcement Learning (RL), which is a type of machine learning where an agent learns to make a sequence of decisions in an environment to maximize a cumulative reward.

  https://github.com/mengdi-li/awesome-RLAIF
  

* **​Reasoning with Reinforced Fine-Tuning (ReFT)**

  https://github.com/lqtrung1998/mwp_ReFT
  

* Illustrating **Reinforcement Learning from Human Feedback (RLHF)**

  Reinforcement learning from Human Feedback (also referenced as RL from human preferences) is a challenging concept because it involves a multiple-model training process and different stages of deployment. In this blog post, we’ll break down the training process into three core steps:

     - Pretraining a language model (LM),

         ![image](https://github.com/ParthaPRay/LLM-Learning-Sources/assets/1689639/f2d564e2-13d3-4f53-aedc-3234963cb871)



     - gathering data and training a reward model, and

       ![image](https://github.com/ParthaPRay/LLM-Learning-Sources/assets/1689639/28125d05-ea71-4210-82fe-20b116298ba4)

       
     - fine-tuning the LM with reinforcement learning.

       ![image](https://github.com/ParthaPRay/LLM-Learning-Sources/assets/1689639/3442441b-52ee-43a1-b0a5-af8150af15c0)
  Technical detail note: The above diagram makes it look like both models generate different responses for the same prompt, but what really happens is that the RL policy generates text, and that text is fed into the initial model to produce its relative probabilities for the KL penalty. This initial model is untouched by gradient updates during training.
  

  https://huggingface.co/blog/rlhf


 _Open-source tools for RLHF_
 
    The first code released to perform RLHF on LMs was from OpenAI in TensorFlow in 2019.
    
    Today, there are already a few active repositories for RLHF in PyTorch that grew out of this. The primary repositories are Transformers Reinforcement Learning (TRL), TRLX which originated as a fork of TRL, and Reinforcement Learning for Language models (RL4LMs).
    
    TRL is designed to fine-tune pretrained LMs in the Hugging Face ecosystem with PPO. TRLX is an expanded fork of TRL built by CarperAI to handle larger models for online and offline training. At the moment, TRLX has an API capable of production-ready RLHF with PPO and Implicit Language Q-Learning ILQL at the scales required for LLM deployment (e.g. 33 billion parameters). Future versions of TRLX will allow for language models up to 200B parameters. As such, interfacing with TRLX is optimized for machine learning engineers with experience at this scale.
    
    RL4LMs offers building blocks for fine-tuning and evaluating LLMs with a wide variety of RL algorithms (PPO, NLPO, A2C and TRPO), reward functions and metrics. Moreover, the library is easily customizable, which allows training of any encoder-decoder or encoder transformer-based LM on any arbitrary user-specified reward function. Notably, it is well-tested and benchmarked on a broad range of tasks in recent work amounting up to 2000 experiments highlighting several practical insights on data budget comparison (expert demonstrations vs. reward modeling), handling reward hacking and training instabilities, etc. RL4LMs current plans include distributed training of larger models and new RL algorithms.
    
    Both TRLX and RL4LMs are under heavy further development, so expect more features beyond these soon.

    There is a [large dataset](https://huggingface.co/datasets/Anthropic/hh-rlhf) created by Anthropic available on the Hub.


  
*  How to fine tune LLMs?

  ![image](https://github.com/ParthaPRay/LLM-Learning-Sources/assets/1689639/a4b04d0d-05ec-410e-b367-1c3b3869423a)

  Fine-tuning an LLM may not be as trivial as we may think! Depending on your data, it may lead to the model forgetting what it learned in the pretraining phase! You want to fine-tune it but you also may want to retain its coding or chatting abilities. Because you most likely don't have the right benchmark data to validate it on different learning tasks, it might be difficult to understand the abilities it lost in the process! 

Why would we want to fine-tune an LLM in the first place? There 2 main reasons! First, we may want to augment the model's data bank with private data, and second, we may want the model to specialize in specific learning tasks. A full fine-tuning takes time and money and generates a very large resulting model file. The typical way to go about it is to use Low-Rank Adaptaters (LoRA) to minimize the fine-tuning cost. 

The idea is to replace within the model some of the large matrices with smaller ones for the gradient computation. Let's call W0 the weights of the pre-trained model for a specific layer matrix. After a gradient update ΔW, the weights will be

W = W0 + ΔW

and, if x is the input to that layer, the output of that layer will be

W . x = W0 . x + ΔW . x

If we use the LLama2 with 70B parameters, we need to update all the parameters for each backward pass: computationally very expensive! Instead, with LoRA, we insert next to each layer matrix of the pre-trained model, 2 matrices A and B such that the update is approximated by a lower rank decomposition:
ΔW ~ B . A 

The trick is that if ΔW has dimensions (R, C), we can create B with dimensions (R, r) and A with dimensions (r, C) such that r << R, C. For example if R = 10K, C = 20K and r = 4, then 

ΔW has R x C = 10K x 20K = 200M elements
B has R x r = 10K x 4 = 40K elements
and A has r x C= 20K x 4 = 80K elements

Therefore A and B combined have 120K elements which is 1666 times less elements than ΔW. When we fine-tune, we only update the weights of those newly inserted matrices. The gradient matrices are much smaller and therefore require much less GPU memory space. Because the pre-trained weights are frozen, we don't need to compute the gradients for a vast majority of the parameters. 

To gain even more space, we may want to quantize the float parameters into integers while applying LoRA (QLoRA). Now, the number of fine-tuned weights is just a fraction of the original model size and we can more easily store those weights for each of the learning tasks we needed fine-tuning for. When we need to deploy an inference server, we can use the original pre-trained model and combine it with the fine-tuned LoRA adapters for the specific learning task needed on that server. 

That is worth a read: https://lnkd.in/d8sXWD_X


*  How to fine-tune LLMs for text encoding

   ![image](https://github.com/ParthaPRay/LLM-Learning-Sources/assets/1689639/98c37c7e-c276-4dcc-995e-9bc8675b179e)

   Being able to encode text of any size into an embedding is one of the superpowers of LLMs! Do you remember when Word2Vec was the best we could do?!

Transformers are great candidates to project the text representation of a sentence into its latent space. The latent space is represented by vector representations of the text representation. This vector representation encodes the text into a shorter format. This text encoding can be used as input for other models or as an index for vector databases. A simple way to extract a text encoding is to pick one of the hidden states. Each of them captures a vector representation of the whole input sentence. Different pre-training tasks (language modeling, sentence classification, etc.) may lead to different vector representations that can be more or less useful depending on how they are used.

It is possible that the size of the hidden states is not adapted to the applications we may want to use the text encoding for, in which case, we want to resize the text encoding by using a linear layer to project the vectors onto the desired dimension. To train that projection layer, we need to plug a specific modeling head and fine-tune the model on the related learning task.

In the context of RAG, we want the text encoding a question to be similar to its answer. The text encodings described above will capture semantic similarity, but a question is not always semantically similar to its answer. We can enforce similarity in the vector representations of questions and their respective answers by using contrastive learning. The idea is to train the model such that the dot product (or the cosine similarity) computed on the questions and their related answers is ~1:

Vector(question) x Vector(answer) ~ 1

To do that, we need to construct a data set where pairs of related (Question, answer) are labeled 1 (similar) and 0 otherwise (dissimilar). We can train the model using contrastive learning where the weights are updated, such that the vector representations of the related (Question, answer) are similar.


*  Fine-tuning large language models (LLMs) in 2024

   Life Cycle of LLM
    ![image](https://github.com/ParthaPRay/LLM-Learning-Sources/assets/1689639/ed9f1bcc-f61b-4d93-b46f-e2c299cf13a1)

   Fine Tuning
  
   ![image](https://github.com/ParthaPRay/LLM-Learning-Sources/assets/1689639/dd0d55d8-31ee-4e9f-9412-20c82a206b18)

    Supervised fine-tuning (SFT)
    ![image](https://github.com/ParthaPRay/LLM-Learning-Sources/assets/1689639/be10b8d8-4a45-4bf9-b816-1a0a4a499214)

      ![image](https://github.com/ParthaPRay/LLM-Learning-Sources/assets/1689639/05dc88b9-226d-4724-931e-b2464efd7349)

    **Fine-tuning methods**

       - Instruction fine-tuning: It's about training the machine learning model using examples that demonstrate how the model should respond to the query. The dataset you use for fine-tuning large language models has to serve the purpose of your instruction. 

             ![image](https://github.com/ParthaPRay/LLM-Learning-Sources/assets/1689639/4cd9d6f7-9808-4463-a912-32a122f11a64)
               
       -  Full fine-tuning: Instruction fine-tuning, where all of the model's weights are updated, is known as full fine-tuning
       -  Parameter-efficient fine-tuning:  PEFT methods only update a small set of parameters

   **Other types of fine-tuning**

      - Transfer learning:  Transfer learning is about taking the model that had learned on general-purpose, massive datasets and training it on distinct, task-specific data. This dataset may include labeled examples related to that domain. Transfer learning is used when there is not enough data or a lack of time to train data; the main advantage of it is that it offers a higher learning rate and accuracy after training. You can take existing LLMs that are pre-trained on vast amounts of data, like GPT ¾ and BERT, and customize them for your own use case.
      - Task-specific fine-tuning: Task-specific fine-tuning is a method where the pre-trained model is fine-tuned on a specific task or domain using a dataset designed for that domain. This method requires more data and time than transfer learning but can result in higher performance on the specific task.
      - Multi-task learning: Multi-task fine-tuning is an extension of single-task fine-tuning, where the training dataset consists of example inputs and outputs for multiple tasks.
      - Sequential fine-tuning: Sequential fine-tuning is about sequentially adapting a pre-trained model on several related tasks. After the initial transfer to a general domain, the LLM might be fine-tuned on a more specific subset.
      


 * **Benefits of Fine Tuning**

    ![image](https://github.com/ParthaPRay/LLM-Learning-Sources/assets/1689639/46df27cd-0902-443d-b4eb-aea9831ed2dc)

   https://www.superannotate.com/blog/llm-fine-tuning?source=post_page-----fb60abdeba07--------------------------------

 
 

 * **RAG Vs Fine-Tuning: How to Optimize LLM Performance**
      
        https://www.e2enetworks.com/blog/rag-vs-fine-tuning-how-to-optimize-llm-performance#:~:text=Trade%2Doffs%3A%20Fine%2Dtuning%20may%20provide%20more%20control%20over,reliability%20of%20the%20knowledge%20base.

 * **Full-model Fine-tuning vs. LoRA vs. RAG**

   https://www.blog.dailydoseofds.com/p/full-model-fine-tuning-vs-lora-vs

    ![1b402882-3dc1-4d5b-ba11-7f4f6d40d888_914x1116-ezgif com-webp-to-gif-converter](https://github.com/ParthaPRay/LLM-Learning-Sources/assets/1689639/803e206e-7362-495f-833b-812e39b951d8)

   


 * **Trade-Offs**
  
      The decision to employ fine-tuning or RAG depends on the specific goals of a task and the nature of the knowledge required. Here are some considerations and trade-offs:
      
      Fine-tuning Considerations: Fine-tuning is suitable for tasks where specific, task-oriented improvements are needed. It is effective for refining a model's performance in a particular domain. However, fine-tuning may exhibit instability and might not be the optimal choice for addressing broad knowledge deficits.
      RAG Considerations: RAG excels in knowledge-intensive tasks where external information is valuable which is provided by feeding data to the knowledge base. It can address both knowledge deficits and factual errors by incorporating diverse knowledge from external sources. RAG's effectiveness relies on the quality and coverage of the knowledge base.
      Trade-offs: Fine-tuning may provide more control over specific task-related improvements, but it might struggle with broader knowledge adaptation. RAG, while powerful in leveraging external knowledge, depends on the availability and reliability of the knowledge base.

* H2O LLM DataStudio: Streamlining Data Curation and Data Preparation for LLMs related tasks
  https://h2o.ai/blog/2023/streamlining-data-preparation-for-fine-tuning-of-large-language-models/
* H2O LLM DataStudio Part II: Convert Documents to QA Pairs for fine tuning of LLMs
  https://h2o.ai/blog/2023/h2o-llm-datastudio-part-ii-convert-documents-to-qa-pairs-for-fine-tuning-of-llms/










# RAG


**RAG = Dense vector Retrieval (R) + In-Contsxt learning (AG)**


  * 3 Ways to build multimodal RAG pipeline

    ![image](https://github.com/ParthaPRay/LLM-Learning-Sources/assets/1689639/392ce5b1-b086-4416-a463-00215782aba6)

    Text is not the only data type we use in RAG pipelines! We are still in the infancy of Generative AI, and text is now the primary information that we feed to LLMs, but that is going to change quickly! There is a lot more information contained in the different documents we use on a daily basis beyond just text data. 

For example, GPT-4, Bard, and LlaVA are multimodal LLMs that can ingest images as well as text. The images are passed through a Vision Transformer, resulting in visual tokens. The visual tokens are then passed through a projection layer that specializes in aligning visual tokens with text tokens. The visual and text tokens are then provided to the LLM, which cannot make the difference between the different data modes. 

In the context of RAG, the LLM plays a role at indexing time, where it generates a vector representation of the data to index it in a vector database. It is also used at retrieval time, where it uses the retrieved documents to provide an answer to a user question. A multimodal LLM can generate embedding representations of images and text and answer questions using those same data types. If we want to answer questions that involve information in different data modes, using a multimodal LLM at indexing and retrieval time is the best option.

If you want to build your RAG pipeline using API providers like OpenAI, there are currently no available options for multimodal LLMs. However, OpenAI is likely to release its API to ingest images with GPT-4 pretty soon, so it will be available for question-answering using multimodal prompts. Even if it is available for text generation, it might not be available for embedding generation. Remains creating embedding for images then? This can be achieved by prompting a multimodal LLM to describe in text the images we need to index. We can then index the images using the text descriptions and their vector representations.

The complexity of generating a text description of an image is not the same as answering questions using a large context of different data types. With a small multimodal LLM, we might get satisfactory results in describing images but subpar results in answering questions. For example, it is pretty simple to build an image description pipeline with [LlaVA models](https://huggingface.co/mys/ggml_llava-v1.5-7b/tree/main)  and Llama.cpp as [LLM backbone](https://github.com/ggerganov/llama.cpp). Those descriptions can be used for indexing as well as for answering questions that may involve those images. The LLM answering questions would use the text description of images instead of the images themselves. Today that might be the simplest option to build a multimodal RAG pipeline. It might not be as performant, but the technology is going to improve very fast! 


* How to optimize your RAG pipelines

  ![image](https://github.com/ParthaPRay/LLM-Learning-Sources/assets/1689639/af81ba01-a922-4cb5-bbc2-e212a689fb33)

  In RAG, the data you retrieve doesn't have to be the data you used to index it! Typically, when we talk about RAG, we assume that the data is stored in its vector representation in a vector database. When we query the database, we then retrieve the most similar data to the query vector. But it doesn't have to be the case!

In a typical RAG (Retrieval Augmented Generation), we have a document, we convert the document into its vector representation, and when a query vector is similar to the vector, we retrieve the document. However, the vector that is used to index the document doesn't have to be its direct vector representation.

For example, the document could be quite large and could contain multiple conflicting information about different concepts. The query vector usually comes from a question about a single concept, so it is unlikely that the vector representation of the question will be similar to the large document. Instead, we could break down the large document into smaller chunks, convert those into their vector representations, and index the large document multiple times using the child documents' vectors. The small child documents have more chance to contain a unique concept, so they are great for indexing the data for similarity search, but they don't contain a lot of context to answer the question, so it is better to retrieve the larger document.

We can also index the document by the questions that the document answers. As part of the indexing pipeline, we can have an LLM prompted with the task of generating the questions that the document could answer. We then get the embeddings of the questions and index the document by those embeddings. When we have a question, the resulting query vector will be much more similar to the questions about the document than the document itself. However, the data retrieved should be the document so that the LLM has all the context necessary to answer the question.

We could also index the document by its summary. Again, as part of the indexing pipeline, we could have an LLM tasked to summarize the incoming documents. The resulting text will be more concise and "semantically purer", so it could be a better option for a similarity search. This is a great option when your document contains tables (like .csv). Tables contain numbers, and it might be difficult to get a question whose vector representation could be similar to the table's. However, if, as part of the indexing pipeline, we have an LLM tasked to provide a text description of the table data, we can then index the table data using its text description. This will make it much easier on the similarity search! The retrieved data will be the original table data as it contains more information to answer the question. 



 * Problems with RAG

   ![image](https://github.com/ParthaPRay/LLM-Learning-Sources/assets/1689639/013bfe81-88a2-4bea-81c5-3b949aee0002)

   Augmenting LLMs with databases is great, but there are major flaws in that approach! We see a lot of debates around fine-tuning versus Retriever Augmented Generation (RAG) with LLMs these days. Augmenting LLMs with small additional data is better served by RAG, but it is important to understand the shortcomings of that approach! 

The idea with RAG is to encode the data you want to expose to your LLM into embeddings and index that data into a vector database. When a user asks a question, it is converted to an embedding, and we can use it to search for similar embeddings in the database. Once we found similar embeddings, we construct a prompt with the related data to provide context for an LLM to answer the question. Similarity here is usually measured using the cosine similarity metric.

The first problem is that a question is usually not semantically similar to its answers. At least, it is possible for the search to retrieve documents containing the same words as the question or that are used in the same context without providing relevant information to answer the question. Because the search retrieves the most similar documents to the question, depending on the data, too many irrelevant documents may show higher cosine similarity than the documents actually containing the answer.

To be fair, high cosine similarity does not exactly translate to semantic similarity with Transformers. High cosine similarity can also capture the high co-occurrence of 2 different terms within the same sub-text of the training data, which often happens for a specific question and its related answer.

Another problem may be related to the way the data has been indexed. If the data have been broken down into big chunks of text, then it is likely to contain multiple different and unrelated information within each chunk. If you perform a similarity search on that data, the pertinent information may be diluted, and the search may return irrelevant documents instead. It is important to break down the data so that each chunk contains no more than a few paragraphs to ensure more "uniqueness" in the concepts developed in each text.

With the RAG approach, it is very important to limit the type of questions we ask the LLM. If we ask questions that require aggregating data all over the database, the answers are most likely going to be wrong, but the LLM won't be able to know that. If the right information is local to one or a few documents, a similarity search may find it. However, if the information requires scanning all the documents to find the answer, a similarity search won't find it. Imagine each document is dated, and we ask, "What is the earliest document?". In that case, we can only know the answer if we scan the entire database, and a similarity search won't be helpful. 


 * Vector Database vs Graph Database for RAG

   ![image](https://github.com/ParthaPRay/LLM-Learning-Sources/assets/1689639/e1092baf-9f4e-4473-9646-ce755e93a7e3)

    Graph Databases should be the better choice for Retrieval Augmented Generation (RAG)! We have seen the debate RAG vs fine-tuning, but what about Vector databases vs Graph databases?

In both cases, we maintain a database of information that an LLM can query to answer a specific question. In the case of vector databases, we partition the data into chunks, encode the chunks into vector representations using an LLM, and index the data by their vector representations. Once we have a question, we retrieve the nearest neighbors to the vector representation of the question. The advantage is the fuzzy matching of the question to chunks of data. We don't need to query a specific word or concept; we simply retrieve semantically similar vectors. The problem is that the retrieved data may contain a lot of irrelevant information, which might confuse the LLM.

In the context of graphs, we extract the relationships between the different entities in the text, and we construct a knowledge base of the information contained within the text. An LLM is good at extracting that kind of triplet information:

[ENTITY A] -> [RELATIONSHIP] -> [ENTITY B]
 
For example: 
- A [cow] IS an [animal]
- A [cow] EATS [plants]
- An [animal] IS a [living thing]
- A [plant] IS a [living thing]

Once the information is parsed, we can store it in a graph database. The information stored is the knowledge base, not the original text. For information retrieval, the LLM needs to come up with an Entity query related to the question to retrieve the related entities and relationships. The retrieved information is much more concise and to the point than in the case of vector databases. This context should provide much more useful information for the LLM to answer the question. The problem is that the query matching needs to be exact, and if the entities captured in the database are slightly semantically or lexically different, the query will not return the right information.

I wonder if there is a possibility to merge the advantages of vector and graph databases. We could parse the entities and relationships, but we index them by their vector representations in a graph database. This way, the information retrieval could be performed using approximate nearest neighbor search instead of exact matching. Does that exist already?
     
  * What is Retrieval-Augmented Generation (RAG)?, https://www.youtube.com/watch?v=T-D1OfcDW1M&t=265s&ab_channel=IBMTechnology

  * Community Paper Reading: RAG vs Fine-tuning, https://www.youtube.com/watch?v=EbEPHOABgSY&ab_channel=ArizeAI

    
  * User-Selected metadata in RAG Applications with Qdrant, https://www.youtube.com/watch?v=qcn7YAJfCeE&ab_channel=LearnDatawithMark
  
  * Ollama Python Library Released! How to implement Ollama RAG? https://www.youtube.com/watch?v=4HfSfFvLn9Q&ab_channel=MervinPraison
  
  * Building a Multimodal RAG App for Medical Applications, https://www.youtube.com/watch?v=fbbFrCfaF0w&ab_channel=AIAnytime
  
  * Track and Monitor RAG Pipelines using Weights & Biases (wandb), https://www.youtube.com/watch?v=8-exaASey6o&ab_channel=AIAnytime
 
  * Unlocking RAG Potential with LLMWare's CPU-Friendly Smaller Models, https://www.youtube.com/watch?v=qXEUqhqjHdg&ab_channel=AIAnytime
 
  * RAG Implementation using Zephyr 7B Beta LLM: Is this the best 7B LLM? https://www.youtube.com/watch?v=btuN-rrPhsM&ab_channel=AIAnytime
 
  * Better RAG with Merger Retriever (LOTR) and Re-ranking Retriever (Long Context Reorder), https://www.youtube.com/watch?v=uYZftCq2efg&ab_channel=AIAnytime
 

 
  * Pinecone + LlamaIndex on Retrieval Augmented Generation (RAG) Systems, https://www.youtube.com/watch?v=FgLf5HjxI8w&ab_channel=ArizeAI
    
  * Optimizing RAG With LLMS: Exploring Chunking Techniques and Reranking for Enhanced Results, https://youtube.com/watch?v=QpRTdZDR4tE&ab_channel=ArizeAI

  * Check Hallucination of LLMs and RAGs using Open Source Evaluation Model by Vectara, https://www.youtube.com/watch?v=O-VYDADgc68&ab_channel=AIAnytime
 
  * Learn to Evaluate LLMs and RAG Approaches, https://www.youtube.com/watch?v=97ftVtITKfo&ab_channel=AIAnytime
 
  * Evaluating Biases in LLMs using WEAT and Demographic Diversity Analysis, https://www.youtube.com/watch?v=eTenkUPsjko&ab_channel=AIAnytime
 
  * RAG with LlamaIndex - Qdrant and Azure OpenAI in 9 minutes, https://www.youtube.com/watch?v=h4F09fWhyhg&ab_channel=AmbarishGangulyAcademy

    https://github.com/ambarishg/llama-index


  * LLM Search & Retrieval Systems with Arize and LlamaIndex: Powering LLMs on Your Proprietary Data, https://www.youtube.com/watch?v=hbQYDpJayFw&ab_channel=ArizeAI
 
  * Building A RAG System With OpenAI Latest Embeddings, https://www.youtube.com/watch?v=OvvgaR1S4Xc&ab_channel=RichmondAlake

  * Transform RAG and Search with Azure AI Document Intelligence, https://www.youtube.com/watch?v=SOBdR-xxE04&ab_channel=SethJuarez

  * Best retrieval strategies for Generative AI applications: Semantic Search Benchmarking, https://www.youtube.com/watch?v=BvnOln6YZ_8&ab_channel=SethJuarez
    
  * Building corrective RAG from scratch with open source, local LLMs, https://youtu.be/E2shqsYwxck?si=LEeA5KXOQ6idzDd2

  * RAG from scratch, https://youtube.com/playlist?list=PLfaIDFEXuae2LXbO1_PKyVJiQ23ZztA0x&si=BtJ6KCTMfqBzIGya
 
    
  * CrewAI RAG: How I Created AI Assistants to Run My News Agency, https://www.youtube.com/watch?v=77xSbC-9yn4&ab_channel=MervinPraison

    
  * RAG Evaluation Using Synthetic data and LLM-As-A-Judge, https://github.com/huggingface/cookbook/tree/main/notebooks/en
    
  * Bert Score for Contextual Similarity for RAG Evaluation, https://youtube.com/watch?v=7AVjk2k8Mbs&ab_channel=AIAnytime
 
  * Testing Framework Giskard for LLM and RAG Evaluation (Bias, Hallucination, and More), https://www.youtube.com/watch?v=KeY6qPAvyq0&ab_channel=AIAnytime

  * RAG Evaluation

    https://huggingface.co/learn/cookbook/rag_evaluation

    https://colab.research.google.com/github/huggingface/cookbook/blob/main/notebooks/en/rag_evaluation.ipynb
    
    
  * Getting started with RAG in DSPy!,
  https://youtu.be/CEuUG4Umfxs?si=Dz_S5uOXSlo3yiIN

  * RAG Time! Evaluate RAG with LLM Evals and Benchmarking

     ![image](https://github.com/ParthaPRay/LLM-Learning-Sources/assets/1689639/1d20253d-7f6f-41d2-86cd-b68bc9e46233)
    
    https://www.youtube.com/watch?v=LrMguHcbpO8&ab_channel=ArizeAI

  * Gemma with transformers: how to teach structured English quotes to LLM
    https://youtu.be/qeJgBkPLCxo?si=YzFFkJop1ptC_YBM

  * Unlock AI Agents, Function Calls and Multi-Step RAG with LLMWare
    https://www.youtube.com/watch?v=cQfdaTcmBpY&ab_channel=llmware


 * Chat with documents with Chainlit, Langchain, Ollama & Mistral, https://youtu.be/2IL0Sd3neWc?si=eXSH7WZa_bczTfTv
    
 * How I created AI Research Assistantand it costs 0$ to run, Ollama + qdrant + Gptforall + langchain, https://youtu.be/f1ihg20fQiU?si=VjaYv9yr9g-Ujvdk


 * Question Answer Generator App using Mistral LLM, Langchain, and FastAPI, https://www.youtube.com/watch?v=Hcqmhhx30Pg&ab_channel=AIAnytime

 * RAG with LlamaParse, Qdrant and groq

 https://youtu.be/w7Ap6gZFXl0?si=liBk9uDsOm9DbSi4
 
 * Better Retrieval Augmented Generation (RAG) with LangChain Parent-Child Retriever, https://www.youtube.com/watch?v=wSi0fxkH6e0

 * Advanced RAG on HuggingFace documentation using langchain, https://huggingface.co/learn/cookbook/advanced_rag

      https://github.com/huggingface/cookbook/tree/main/notebooks/en

      ![image](https://github.com/ParthaPRay/LLM-Learning-Sources/assets/1689639/a3b7c78b-c349-481c-a7c7-94191c00ea19)


* LangChain RAG featuring Shopify's Madhav Thaker, https://www.youtube.com/watch?v=IUEny5cbys8&ab_channel=ArizeAI

    https://shopify.engineering/topics/data-science-engineering

* RAG-VectorDB-Embedings-LlamaIndex-Langchain,   https://github.com/lucifertrj/Awesome-RAG

* Q&A with RAG, https://python.langchain.com/docs/use_cases/question_answering/

   ![image](https://github.com/ParthaPRay/LLM-Learning-Sources/assets/1689639/b1f416e5-9ba8-43e4-89da-cb6baa256ddb)


   Table of contents:
  
    - [Quickstart](https://python.langchain.com/docs/use_cases/question_answering/quickstart): We recommend starting here. Many of the following guides assume you fully understand the architecture shown in the Quickstart.
    - [Returning sources](https://python.langchain.com/docs/use_cases/question_answering/sources): How to return the source documents used in a particular generation.
    - [Streaming](https://python.langchain.com/docs/use_cases/question_answering/streaming): How to stream final answers as well as intermediate steps.
    - [Adding chat history](https://python.langchain.com/docs/use_cases/question_answering/chat_history): How to add chat history to a Q&A app.
    - [Per-user retrieval](https://python.langchain.com/docs/use_cases/question_answering/per_user): How to do retrieval when each user has their own private data.
    - [Using agents](https://python.langchain.com/docs/use_cases/question_answering/conversational_retrieval_agents): How to use agents for Q&A.
    - [Using local models](https://python.langchain.com/docs/use_cases/question_answering/local_retrieval_qa): How to use local models for Q&A.


* Google Gemma Fully LOCAL RAG ChatBot using Ollama|LangChain|Chainlit|Chat with Docs #ai #ollama #llm, https://www.youtube.com/watch?v=8uo-GCIKim8&ab_channel=DataInsightEdge

  https://github.com/InsightEdge01/RAGGemmaModel/tree/main

* Beyond RAG:How to Build an App with LOCAL LLMs to Generate Custom Datasets to Fine-tune Your LLMs, https://www.youtube.com/watch?v=vBC6Ym0cb0Y&ab_channel=DataInsightEdge

* How to use MongoDB as vector store for RAG -Atlas vector search index,

https://youtu.be/IPbv5Fs3mis?si=5_frUdnXNLoVJEpM


* Multi Needle in a Haystack, https://youtu.be/UlmyyYQGhzc?ref=blog.langchain.dev

  ![image](https://github.com/ParthaPRay/LLM-Learning-Sources/assets/1689639/ce782f56-fae6-40d1-b369-6e2dde3da301)


  https://github.com/gkamradt/LLMTest_NeedleInAHaystack/tree/main?ref=blog.langchain.dev
  
  https://blog.langchain.dev/multi-needle-in-a-haystack/
  

* LangGraph for Code Generation, https://www.youtube.com/watch?v=MvNdgmM7uyc&ref=blog.langchain.dev

   
   ![image](https://github.com/ParthaPRay/LLM-Learning-Sources/assets/1689639/931f2caa-e72d-40eb-bc6f-4f67fafbbb92)


    ![image](https://github.com/ParthaPRay/LLM-Learning-Sources/assets/1689639/201563fe-9952-4bdc-b7b7-9f5c2558fce3)

     Flow for AlphaCodium

    **The recent AlphaCodium work showed that code generation can be improved by using a flow paradigm rather than a naive prompt:answer paradigm: answers can be iteratively constructed by (1) testing answers and (2) reflecting on the results of these tests in order to improve the solution.**



  https://blog.langchain.dev/code-execution-with-langgraph/


*  How to use Langchain with multimodal AI to analyze images in financial reports using langchain and GPT-4
  
  https://youtu.be/Rcqy92Ik6Uo?si=PPeKxtD5GHArV9iN

  https://docs.google.com/presentation/d/1EJqIvYGbF5IGHX7orXaUSKVN3PVbQh7kOP7m5BEoyKQ/edit?usp=sharing

  https://github.com/langchain-ai/langchain/blob/master/cookbook/Multi_modal_RAG.ipynb



* How to analyze tables in large financial reports using GPT-4 with LkamaIndex

  https://youtu.be/xT6JpDELKPg?si=nULiN7_jpQXExfhH


https://docs.google.com/presentation/d/1ug9jHtMFsGjNV7zp85hPUNjiiCGKz53wQb9mZh0B_ZI/edit?usp=sharing

https://colab.research.google.com/drive/1DldMhszgSI4KKI2UziNHHM4w8Cb5OxEL#scrollTo=Ht4oSN2PvzUJ



* A sample app for the Retrieval-Augmented Generation pattern running in Azure, using Azure AI Search for retrieval and Azure OpenAI large language models to power ChatGPT-style and Q&A experiences.

https://www.youtube.com/live/uVqrZhNdUAI?si=58gCEN7BW613l43a

https://github.com/Azure-Samples/azure-search-openai-demo


* Going Meta - ep 22: RAG with knowledge graph, neo4j

https://www.youtube.com/live/9DxwgIKVSHY?si=nXqLEDVbcWwfmzqf

https://github.com/jbarrasa/goingmeta


* Bhuilding RAG with knowledge graphs
workshop with LlamaIndex, 
  
https://youtu.be/VEvFPRlCcvI?si=rz_TMnuNrQuncusa


* RAGArch: Building a No-Code RAG Pipeline Configuration & One-Click RAG Code Generation Tool Powered by LlamaIndex

  https://www.llamaindex.ai/blog/ragarch-building-a-no-code-rag-pipeline-configuration-one-click-rag-code-generation-tool-powered-b6e8eeb70089

  https://github.com/AI-ANK/RAGArch

  https://huggingface.co/spaces/AI-ANK/RAGArch

* MultiModal RAG for Advanced Video Processing with LlamaIndex & LanceDB

  https://www.llamaindex.ai/blog/multimodal-rag-for-advanced-video-processing-with-llamaindex-lancedb-33be4804822e

  https://github.com/run-llama/llama_index/blob/main/docs/examples/multi_modal/multi_modal_video_RAG.ipynb


* Introducing LlamaCloud and LlamaParse for production-grade context-augmentation to LLM and RAG applications

  https://github.com/run-llama/llama_parse

  https://github.com/run-llama/llama_parse/blob/main/examples/demo_basic.ipynb
  
  https://github.com/run-llama/llama_parse/blob/main/examples/demo_advanced.ipynb


* Chunking Strategies in RAG: Optimising Data for Advanced AI Responses

  https://www.youtube.com/watch?v=pIGRwMjhMaQ&ab_channel=MervinPraison

  https://mer.vin/2024/03/chunking-strategy/


* Building a RAG system with Google Gemma, Huggingface and MongoDB

  
  https://youtu.be/BNUpRW-Dk90?si=84DKcxms8RHWmSda
  

* Building A RAG System with Gemma, MongoDB and Open Source Models

  https://huggingface.co/learn/cookbook/rag_with_hugging_face_gemma_mongodb

  https://colab.research.google.com/github/huggingface/cookbook/blob/main/notebooks/en/rag_with_hugging_face_gemma_mongodb.ipynb



* **SubDocument RAG**: If You Are NOT Using This, You're OUTDATED Already! (step-by-step LlamaIndex)

  ![image](https://github.com/ParthaPRay/LLM-Learning-Sources/assets/1689639/001a2ee8-8636-4ec6-b13b-eb239e3d3c39)

   ![image](https://github.com/ParthaPRay/LLM-Learning-Sources/assets/1689639/88a53e1e-e0ea-4ef6-80ea-78e01f04d28f)

  ![image](https://github.com/ParthaPRay/LLM-Learning-Sources/assets/1689639/cac7b5a8-b3c1-46e4-bc73-f8347b7374a0)

  ![image](https://github.com/ParthaPRay/LLM-Learning-Sources/assets/1689639/ff167387-c330-424d-8bf2-deef46a0c80f)


https://github.com/run-llama/llama_index/blob/main/llama-index-packs/llama-index-packs-subdoc-summary/examples/subdoc-summary.ipynb

 https://www.youtube.com/watch?v=m6P1Rp91AzM&t=63s&ab_channel=TwoSetAI

 https://mlnotes.substack.com/p/advanced-rag-technique-subdoc-summary?r=164sm1&utm_campaign=post&utm_medium=web&triedRedirect=true

 

* **Command-R**
  
  C4AI Command-R is a research release of a 35 billion parameter highly performant generative model. Command-R is a large language model with open weights optimized for a variety of use cases including reasoning, summarization, and question answering. Command-R has the capability for multilingual generation evaluated in 10 languages and highly performant RAG capabilities.
  
  https://huggingface.co/CohereForAI/c4ai-command-r-v01

  https://www.youtube.com/watch?v=YQFLdE3osws&ab_channel=FahdMirza


* Transforming business process automation with retrieval-augmented generation and LLMs

    - Retrieval-augmented generation in practice
    - RAG in supply chain
    - RAG in retail
    - RAG in finance and insurance
    - Case study: RFP processing with RAG
    - Assembling RAG flows: From basic building blocks to valuable use cases
    - Architecture of retrieval-augmented generation
    - Orchestrating RAG processes
    - Constructing the RAG pipeline: Essential building blocks
    - Conclusion: The benefits of retrieval-augmented generation and large language models

  https://www.griddynamics.com/blog/retrieval-augmented-generation-llm






* Implementing semantic cache to improve a RAG system with FAISS.

 ![image](https://github.com/ParthaPRay/LLM-Learning-Sources/assets/1689639/91d52d20-a75c-432c-b939-897e4ecc796e)



  In this notebook, they explore a typical RAG solution where we will utilize an open-source model and the vector database Chroma DB. However, we will integrate a semantic cache system that will store various user queries and decide whether to generate the prompt enriched with information from the vector database or the cache.

  A semantic caching system aims to identify similar or identical user requests. When a matching request is found, the system retrieves the corresponding information from the cache, reducing the need to fetch it from the original source.

  As the comparison takes into account the semantic meaning of the requests, they don’t have to be identical for the system to recognize them as the same question. They can be formulated differently or contain inaccuracies, be they typographical or in the sentence structure, and we can identify that the user is actually requesting the same information.


   https://huggingface.co/learn/cookbook/semantic_cache_chroma_vector_database


   https://colab.research.google.com/github/huggingface/cookbook/blob/main/notebooks/en/semantic_cache_chroma_vector_database.ipynb





 * **GraphRAG**

   https://www.microsoft.com/en-us/research/blog/graphrag-unlocking-llm-discovery-on-narrative-private-data/

  https://www.microsoft.com/en-us/research/publication/can-generalist-foundation-models-outcompete-special-purpose-tuning-case-study-in-medicine/

  Violent Incident Information from News Articles (VIINA)  https://github.com/zhukovyuri/VIINA

  Base repositories https://github.com/microsoft/graspologic

  Comparison, https://arxiv.org/pdf/2303.08896.pdf




* AI RAG Chat App Evaluation,

  https://www.youtube.com/watch?v=mM8pZAI2C5w&ab_channel=PamelaFox

  https://github.com/Azure-Samples/ai-rag-chat-evaluator

  developed by using https://github.com/Azure-Samples/azure-search-openai-demo/


* AI RAG Chat App: CI/CD Deployment, 

  https://www.youtube.com/watch?v=GMy3v6UXkYs&ab_channel=PamelaFox  
  
   https://github.com/Azure-Samples/azure-search-openai-demo/



* **Building A RAG Ebook “Librarian” Using LlamaIndex**
  https://huggingface.co/learn/cookbook/rag_llamaindex_librarian

  https://colab.research.google.com/github/huggingface/cookbook/blob/main/notebooks/en/rag_llamaindex_librarian.ipynb



* Build applications with LLMs: LangChain

  ![image](https://github.com/ParthaPRay/LLM-Learning-Sources/assets/1689639/e2cbe213-eb19-41f8-8f3c-079b4c23d594)


* Universal Document Loader with langchain-airbyte, https://www.youtube.com/watch?v=zQU_1sCLSMU&ab_channel=LangChain

* Build with LangChain, https://youtube.com/playlist?list=PLfaIDFEXuae06tclDATrMYY0idsTdLg9v&si=0ypsn2axHsDSMs6b
    
* LangGraph python, https://youtube.com/playlist?list=PLfaIDFEXuae16n2TWUkKq5PgJ0w6Pkwtg&si=haMafIbDjtLZ9hFU

  
* Hands on with LangGraph Agent Workflows: Build a LangChain Coding Agent with Custom Tools
 
    https://www.youtube.com/watch?v=oMRJ--GJCKQ&ab_channel=DeployingAI


* RAG from Scratch

  https://www.youtube.com/watch?v=wd7TZ4w1mSw&list=PLfaIDFEXuae2LXbO1_PKyVJiQ23ZztA0x&pp=iAQB

 * LangGraph (Python)

    https://www.youtube.com/watch?v=5h-JBkySK34&list=PLfaIDFEXuae16n2TWUkKq5PgJ0w6Pkwtg&pp=iAQB

 * AutoPrompt Builder

    https://www.youtube.com/watch?v=mmBo8nlu2j0&list=PLfaIDFEXuae06tclDATrMYY0idsTdLg9v&pp=iAQB

 * LangSmith Platform Overview

     https://www.youtube.com/watch?v=3wAON0Lqviw&list=PLfaIDFEXuae2WCZ63usrRoriORSGmDQsg&pp=iAQB
   
 * Langchain Expression Languare

   https://www.youtube.com/watch?v=9M8x485j_lU&list=PLfaIDFEXuae1Ed60mXaLZRXC_jv0IwxPl&pp=iAQB
   
 * Deep Dive: How to Build a Smart Chatbot in 10 mins with LangChain

   https://newsletter.theaiedge.io/p/deep-dive-building-a-smart-chatbot


 * Building long context RAG with RAPTOR from scratch
   
   https://youtu.be/jbGchdTL7d0?si=8AgkTzEqy9VKN_LX

* Super Easy Way To Parse PDF | LlamaParse From LlamaIndex | LlamaCloud

  https://www.youtube.com/watch?v=wRMnHbiz5ck&ab_channel=DataScienceBasics

  https://www.llamaindex.ai/blog/introducing-llamacloud-and-llamaparse-af8cedf9006b

  https://github.com/run-llama/llama_parse

    Getting Started: https://github.com/run-llama/llama_parse/blob/main/examples/demo_basic.ipynb
  
    Advanced: https://github.com/run-llama/llama_parse/blob/main/examples/demo_advanced.ipynb
  
    Advacned RAG Example: https://github.com/run-llama/llama_parse/blob/main/examples/demo_advanced.ipynb

    LlamParse Example: https://github.com/run-llama/llama_parse/tree/main/examples
  
    RAW API usage: https://github.com/run-llama/llama_parse/blob/main/examples/demo_api.ipynb

  
    - LlamaCloud: https://cloud.llamaindex.ai/
    - Ollama: https://ollama.ai/
    - GitHub repo for code: https://github.com/sudarshan-koirala/llamaparser-example



  * Superior RAGs for complex PDFs with LlamaParse

    https://www.youtube.com/live/7qsxz2rURG4?si=GbuRI1hfqrwpA6XU


  * RAG over your **code**: a project by Akshay on creating a local code assistant using **LlamaIndex**, MistralAI, and Streamlit to index and query GitHub repositories, offering a foundational guide for advanced code QA
 
      ![image](https://github.com/ParthaPRay/LLM-Learning-Sources/assets/1689639/08cea68a-9670-46b8-8f08-3aa3396cce7f)


     https://www.youtube.com/watch?v=3V-rpBofej8&ab_channel=AkshayPachaar
    
    https://lightning.ai/lightning-ai/studios/chat-with-your-code-using-rag?__s=u4pvflfacap82vd4gibe&utm_source=drip&utm_medium=email&utm_campaign=LlamaIndex+news+2024-03-12


 * Build a real-time RAG chatbot using Google Drive and Sharepoint

   Keep your chatbot’s knowledge base up-to-date with Pathway and LlamaIndex

   In this post, they explore how to build a real-time RAG app with up-to-date information from your files stored in Google Drive or Sharepoint. This means that your chatbot will always have access to the most recent version of your knowledge base—no manual pipeline reruns needed. By the end of this tutorial, you’ll use Pathway and LlamaIndex to build a RAG chatbot that instantly updates.


   https://blog.streamlit.io/build-a-real-time-rag-chatbot-google-drive-sharepoint/?__s=u4pvflfacap82vd4gibe&utm_source=drip&utm_medium=email&utm_campaign=LlamaIndex+news+2024-03-12

   https://www.youtube.com/watch?v=JLVsFIXtvKE&ab_channel=Streamlit
      
   

 * Build an AI Browser Copilot

   **LaVague** is designed to automate menial tasks on behalf of its users. Many of these tasks are repetitive, time-consuming, and require little to no cognitive effort. By automating these tasks, LaVague aims to free up time for more meaningful endeavors, allowing users to focus on what truly matters to them.

By providing an engine turning natural language queries into Selenium code, LaVague is designed to make it easy for users or other AIs to automate easily express web workflows and execute them on a browser.

One of the key usages we see is to automate tasks that are personal to users and require them to be logged in, for instance automating the process of paying bills, filling out forms or pulling data from specific websites.

LaVague is built on open-source projects and leverages open-sources models, either locally or remote, to ensure the transparency of the agent and ensures that it is aligned with users' interests.

 Large Action Model framework to automate browser interaction

   A project by Daniel Huynh that demonstrates how to create a browser agent using RAG, local embeddings, and Mixtral to execute browser tasks from a Colab notebook, showcased with a video on navigating HuggingFace datasets

    ![image](https://github.com/ParthaPRay/LLM-Learning-Sources/assets/1689639/a176c50a-7a1c-47fb-8b84-73f6c6cdda01)
     LaVague interacting with Hugging Face's website.


Features:

  - Natural Language Processing: Understands instructions in natural language to perform browser interactions.
  - Selenium Integration: Seamlessly integrates with Selenium for automating web browsers.
  - Open-Source: Built on open-source projects such as transformers and llama-index, and leverages open-source models, either locally or remote, to ensure the transparency of the agent and ensures that it is aligned with users' interests.
  - Local models for privacy and control: Supports local models like Gemma-7b so that users can fully control their AI assistant and have privacy guarantees.
  - Advanced AI techniques: Uses a local embedding (bge-small-en-v1.5) first to perform RAG to extract the most relevant HTML pieces to feed the LLM answering the query, as directly dropping the full HTML code would not fit in context. Then leverages Few-shot learning and Chain of Thought to elicit the most relevant Selenium code to perform the action without having to finetune the LLM (Nous-Hermes-2-Mixtral-8x7B-DPO) for code generation.

   https://github.com/lavague-ai/LaVague

   https://colab.research.google.com/github/dhuynh95/LaVague/blob/main/LaVague.ipynb

   
* LlamaIndex and Anthropic Cookbooks for RAG

  LlamaIndex is a data framework for LLM-based applications that benefit from context augmentation.

Here they provide cookbooks for building LLM applications using Anthropic and LlamaIndex.

    - [Basic_RAG_With_LlamaIndex.ipynb](https://github.com/anthropics/anthropic-cookbook/blob/main/third_party/LlamaIndex/Basic_RAG_With_LlamaIndex.ipynb) - Notebook to help you build RAG pipelines with LlamaIndex.
    - [Router_Query_Engine.ipynb](https://github.com/anthropics/anthropic-cookbook/blob/main/third_party/LlamaIndex/Multi_Document_Agents.ipynb) - Notebook to help you use RouterQueryEngine to route user queries to different indices.
    - [SubQuestion_Query_Engine](https://github.com/anthropics/anthropic-cookbook/blob/main/third_party/LlamaIndex/Multi_Modal.ipynb) - Notebook to help you to use SubQuestionQueryEngine to answer complex user queries spanning multiple documents.
    - [ReAct_Agent.ipynb](https://github.com/anthropics/anthropic-cookbook/blob/main/third_party/LlamaIndex/ReAct_Agent.ipynb) - Notebook to help you to use ReActAgent for using Tools and QueryEngine Tools.
    - [Multi_Document_Agents.ipynb](https://github.com/anthropics/anthropic-cookbook/blob/main/third_party/LlamaIndex/Router_Query_Engine.ipynb) - Notebook to help you build an efficient RAG pipeline for a large number of documents.
    - [Multi_Modal.ipynb](https://github.com/anthropics/anthropic-cookbook/blob/main/third_party/LlamaIndex/SubQuestion_Query_Engine.ipynb) - Notebook to help you build Multi-Modal applications using LlamaIndex.

  https://github.com/anthropics/anthropic-cookbook/tree/main/third_party/LlamaIndex
   

* **CodeHierarchyAgentPack** from LlamaIndex

  The CodeHierarchyAgentPack is useful to split long code files into more reasonable chunks, while creating an agent on top to navigate the code. What this will do is create a "Hierarchy" of sorts, where sections of the code are made more reasonable by replacing the scope body with short comments telling the LLM to search for a referenced node if it wants to read that context body.

Nodes in this hierarchy will be split based on scope, like function, class, or method scope, and will have links to their children and parents so the LLM can traverse the tree.

  https://llamahub.ai/l/llama-packs/llama-index-packs-code-hierarchy?from=llama-packs

  https://github.com/run-llama/llama_index/tree/main/llama-index-packs/llama-index-packs-code-hierarchy


* **VideoDB Retriever** from LlamaIndex: RAG: Instantly Search and Stream Video Results 📺
  
  RAG: Instantly Search and Stream Video Results

  VideoDB is a serverless database designed to streamline the storage, search, editing, and streaming of video content. VideoDB offers random access to sequential video data by building indexes and developing interfaces for querying and browsing video content. Learn more at [docs.videodb.io](https://docs.videodb.io/).

  Constructing a RAG pipeline for text is relatively straightforward, thanks to the tools developed for parsing, indexing, and retrieving text data. However, adapting RAG models for video content presents a greater challenge. Videos combine visual, auditory, and textual elements, requiring more processing power and sophisticated video pipelines.

While Large Language Models (LLMs) excel with text, they fall short in helping you consume or create video clips. VideoDB provides a sophisticated database abstraction for your MP4 files, enabling the use of LLMs on your video data. With VideoDB, you can not only analyze but also instantly watch video streams of your search results.

In this notebook, we introduce VideoDBRetriever, a tool specifically designed to simplify the creation of RAG pipelines for video content, without any hassle of dealing with complex video infrastructure.


* StreamRAG: GPT-Powered Video Retrieval & Streaming 🚀

  Video Search Agent for ChatGPT


    ![image](https://github.com/ParthaPRay/LLM-Learning-Sources/assets/1689639/c0823868-a004-47e3-b46d-7dfcd68a6992)

  What does it do? 🤔
  
   It enables developers to:
   
   - 📚 Upload multiple videos to create a library or collection.
   - 🔍 Search across these videos and get real-time video responses or compilations.
   - 🛒 Publish your searchable collection on the ChatGPT store.
   - 📝 Receive summarized text answers (RAG).
   - 🌟 Gain key insights from specific videos (e.g. "Top points from episode 31").

  https://github.com/video-db/StreamRAG

  https://colab.research.google.com/github/video-db/videodb-cookbook/blob/main/quickstart/quickstart.ipynb



* Semi-structured RAG - Langchain using Mistral 7B, Qdrant, Fastembed on pdf text using Tabular Data,
  https://colab.research.google.com/drive/1rLWrDwePwgtZAOUTL7RNpsS7tTQ3oWWQ?usp=sharing
  
https://youtu.be/2Id2KTrES2s?si=44IA8s3qHQYEUTkR



*  Improved Retrieval Augmented Generation with ALL-SORT (Assisted Large Language Sorting)

   ![image](https://github.com/ParthaPRay/LLM-Learning-Sources/assets/1689639/857fa6b7-fa02-46dc-a225-461036fa45de)

   https://docs.google.com/presentation/d/1poQa3t5fuBfAdfXvACicFKKNUPNsk0lfsNgc10TiIUE/edit#slide=id.gcb9a0b074_1_0

   Smaug 34B Model: https://huggingface.co/abacusai/Smaug-34B-v0.1
   
   E5 Embedding Model: https://huggingface.co/intfloat/e5-large-v2


   <img src="https://github.com/ParthaPRay/LLM-Learning-Sources/assets/1689639/1ffaccd1-4db7-41cb-9555-c6cc39773acf" width="200" height="150">
   
   Strucured Text Generation: https://github.com/outlines-dev/outlines, https://outlines-dev.github.io/outlines/
   
   
   https://www.youtube.com/watch?v=biJmRQF8bmY&ab_channel=TrelisResearch


   

* Building STORM from scratch with LangGraph, https://www.youtube.com/watch?v=1uUORSZwTz4&ab_channel=LangChain


    ![image](https://github.com/ParthaPRay/LLM-Learning-Sources/assets/1689639/91a7b5e1-d529-40f2-b6a5-e3518a61036b)

     https://github.com/langchain-ai/langgraph/blob/main/examples/storm/storm.ipynb



* Create Medical Chatbot with Google Gemma 7B LLM LlamaIndex Colab Demo Qdrant FastEmbed Gradio

   https://www.youtube.com/watch?v=23BU5Csi_3w&ab_channel=RitheshSreenivasan

   

    https://colab.research.google.com/drive/1XBohRbAQchvxXVMi1Nap7JuRihjX-N9e?usp=sharing


* **Retrieval Augmented Fine Tuning (RAFT)**

  🦍 RAFT: Adapting Language Model to Domain Specific RAG

    ![image](https://github.com/ParthaPRay/LLM-Learning-Sources/assets/1689639/f41b11c5-9013-4955-86bc-77fb38df66ed)

      How to preapre a LLM for an Exam? Closed-Book vs. Open-Book vs. RAFT

     ![image](https://github.com/ParthaPRay/LLM-Learning-Sources/assets/1689639/9e05b387-4559-46f8-85d7-d62a8959bbd1)

    Train and Test Configuration for RAFT




   https://gorilla.cs.berkeley.edu/blogs/9_raft.html

    https://aka.ms/raft-blog

   RAFT is a recipie to adapting LLMs to domain-specific RAG. You can learn more in our release-blogs here and here. RAFT takes an input document from the user and creates a dataset using the document, consisting of synthetically generated { question, answer, documents } triplets. The dataset can then be used to fine-tune models for improved question-answering and retrieval.

  The input data from the user can be either a general text document (pdf, json, or txt) for general QA or an API documentation in the API Zoo JSONL format for API calling.


     https://github.com/ShishirPatil/gorilla/tree/main/raft
  




# Dataset 


 *  **Augmentoolkit**
   
    Convert Compute And Books Into Instruct-Tuning Datasets.

    Turn any raw text into a high-quality dataset using local models. Make data gathering a painless step of the model creation process. Augmentoolkit is the easy-to-use, 
    customizable, open-source, and cost-effective data generation solution. No OpenAI needed.

     ![image](https://github.com/ParthaPRay/LLM-Learning-Sources/assets/1689639/297cb4b6-9a40-4b74-9724-cd531eab0602)


    https://github.com/e-p-armstrong/augmentoolkit

   


 * Convert Any Text to LLM Dataset Locally - Demo with Example


   https://www.youtube.com/watch?v=ZiyCe_dRksM&ab_channel=FahdMirza

   **NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO**
   
   togetherai: The fastest cloud platform for building and running generative AI. 
  
   https://api.together.xyz/


* Install Genstruct 7B Locally - Best Model to Create Datasets of Any Domain

  Genstruct 7B is an instruction-generation model, designed to create valid instructions given a raw text corpus. This enables the creation of new, partially synthetic instruction finetuning datasets from any raw-text corpus.
  
  **https://huggingface.co/NousResearch/Genstruct-7B**
  

 | Feature             | ChatGPT        | Few-shot prompting | RAG  | Ada-Instruct | Genstruct |
|---------------------|----------------|--------------------|------|--------------|-----------|
| Open models         | ❌             | ☑️                 | ☑️   | ✅           | ✅        |
| Grounded generation | ❌             | ❌                 | ✅   | ❌           | ✅        |
| Complex questions   | ❌             | ❌                 | ❌   | ☑️           | ✅        |
| Complex responses   | ✅             | ☑️                 | ❌   | ☑️           | ✅        |


* Ada-Instruct: Adapting Instruction Generators for Complex Reasoning

  https://arxiv.org/abs/2310.04484



* H2O LLM DataStudio Part II: Convert Documents to QA Pairs for fine tuning of LLMs

  
 https://h2o.ai/blog/2023/h2o-llm-datastudio-part-ii-convert-documents-to-qa-pairs-for-fine-tuning-of-llms/

* H2O LLM DataStudio: Streamlining Data Curation and Data Preparation for LLMs related tasks
  
 https://h2o.ai/blog/2023/streamlining-data-preparation-for-fine-tuning-of-large-language-models/



* Fine-Tuned Q&A - create Q&A (Some Part Deprecated)

 https://cookbook.openai.com/examples/fine-tuned_qa/olympics-2-create-qa

 

* **Detecting Issues in a Text Dataset with Cleanlab**

  In this 5-minute quickstart tutorial, they use Cleanlab to detect various issues in an intent classification dataset composed of (text) customer service requests at an online bank. We consider a subset of the Banking77-OOS Dataset containing 1,000 customer service requests which are classified into 10 categories based on their intent (you can run this same code on any text classification dataset). Cleanlab automatically identifies bad examples in our dataset, including mislabeled data, out-of-scope examples (outliers), or otherwise ambiguous examples. Consider filtering or correcting such bad examples before you dive deep into modeling your data!

  https://huggingface.co/learn/cookbook/issues_in_text_dataset

  Dataset:  [Banking77-OOS Dataset] (https://arxiv.org/abs/2106.04564)
  
  https://colab.research.google.com/github/huggingface/cookbook/blob/main/notebooks/en/issues_in_text_dataset.ipynb

      CleanLab: https://github.com/cleanlab/cleanlab

      https://cleanlab.ai/
      The standard data-centric AI package for data quality and machine learning with messy, real-world data and labels.



 * From screenshots to HTML code: Introducing the WebSight dataset

https://huggingface.co/blog/websight

  Dataset: https://huggingface.co/datasets/HuggingFaceM4/WebSight

  ![image](https://github.com/ParthaPRay/LLM-Learning-Sources/assets/1689639/fd3dab18-1696-4bbd-8a30-077114f0e3f9)

Examples of web pages included in WebSight.

  ![image](https://github.com/ParthaPRay/LLM-Learning-Sources/assets/1689639/0c730368-edb0-45b2-abde-4aee39611281)

Comparison of an original web page (input) on the left, and the rendering of the code generated by our model, Sightseer, (output) on the right.

https://colab.research.google.com/drive/1LdamGKR2oacrDk-kYwz_Wfc1-RBUdzcO?usp=sharing







# Vector Database and Embeddings 




 ![image](https://github.com/ParthaPRay/LLM-Learning-Sources/assets/1689639/5722e5f6-e3d6-473a-86c0-25ce3cc6cc1d)

We have recently seen a surge in vector databases in this era of generative AI. The idea behind vector databases is to index the data with vectors that relate to that data. Hierarchical Navigable Small World (HNSW) is one of the most efficient ways to build indexes for vector databases. The idea is to build a similarity graph and traverse that graph to find the nodes that are the closest to a query vector. 

Navigable Small World (NSW) is a process to build efficient graphs for search. We build a graph by adding vectors one after the other and connecting each new node to the most similar neighbors.

When building the graph, we need to decide on a metric for similarity such that the search is optimized for the specific metric used to query items. Initially, when adding nodes, the density is low, and the edges will tend to capture nodes that are far apart in similarity. Little by little, the density increases, and the edges start to be shorter and shorter. As a consequence, the graph is composed of long edges that allow us to traverse long distances in the graph and short edges that capture closer neighbors. Because of it, we can quickly traverse the graph from one side to the other and look for nodes at a specific location in the vector space.

When we want to find the nearest neighbor to a query vector, we initiate the search by starting at one node (i.e., node A in that case). Among its neighbors (D, G, C), we look for the closest node to the query (D). We iterate over that process until there are no closer neighbors to the query. Once we cannot move anymore, we found a close neighbor to the query. The search is approximate, and the found node may not be the closest as the algorithm may be stuck in a local minima. 

The problem with NSW, is we spend a lot of iterations traversing the graph to arrive at the right node. The idea for Hierarchical Navigable Small World is to build multiple graph layers where each layer is less dense compared to the next. Each layer represents the same vector space, but not all vectors are added to the graph. Basically, we include a node in the graph at layer L with a probability P(L). We include all the nodes in the final layer (if we have N layers, we have P(N) = 1), and the probability gets smaller as we get toward the first layers. We have a higher chance of including a node in the following layer, and we have P(L) < P(L + 1).

The first layer allows us to traverse longer distances at each iteration, whereas in the last layer, each iteration will tend to capture shorter distances. When we search for a node, we start first in layer 1 and go to the next layer if the NSW algorithm finds the closest neighbor in that layer. This allows us to find the approximate nearest neighbor in fewer iterations on average.




 ![image](https://github.com/ParthaPRay/LLM-Learning-Sources/assets/1689639/5454d672-52e0-4e27-8cd4-9665c6f09702)


          Vector databases are often used for recommender engines, where we learn vector representations of users and items we want to recommend. This allows to quickly find similar items by using an approximate nearest neighbor search. As long as we can learn a vector representation of a piece of data, we can index it in a vector database. With the recent advent of LLMs, it became easier to compute vector representations of text documents, capturing the semantic meaning of that text, and vector databases make it easier to find semantically similar text documents.

When looking for the nearest neighbors, it is often not important to be perfectly accurate. Product Quantization (PQ) is a way to quantize the vector space to represent vectors with less precision. The idea is to cluster vectors and index the cluster centroids instead of the vectors themselves. When looking for the nearest neighbors to a query vector, we just need to pull the vectors from the closest clusters. It is a faster search, and indexing the vectors takes much less memory space.

We first need to partition each vector into smaller vectors and run a K-means algorithm on each partition. Instead of indexing the vectors, we index the centroid of the clusters they belong to. If we use 2 clusters per partition and have 6 vectors, that's 3X data compression. Obviously, compression would be much higher with more vectors. Each vector now maps to a set of clusters and their related centroids.

If we want to find the nearest neighbors from a query vector, we measure the squared Euclidean distance for each cluster in each partition and return the vectors with the lowest summed squared Euclidean distances.

Instead of having to iterate through each vector, we just need to iterate through the clusters' centroids. There is a balance between search latency and accuracy. The more clusters we use, the better the hash will be and the more accurate the returned nearest neighbors, but it will increase the search latency as we will need to iterate through more clusters.

This is still a brute force approach as the algorithm scales with the number of clusters, but it can be used in combination with other algorithms to have blasting fast retrieval.





 ![image](https://github.com/ParthaPRay/LLM-Learning-Sources/assets/1689639/e327ce0e-82d1-4b85-8aa0-69d9286f57a6)

   There are tons of vector database providers: Pinecone, Deep Lake, Milvus, Qdrant, Weaviate, ... They all tend to provide similar capabilities with efficient similarity search, optimized storage formats for AI applications, unstructured data accessibility, and cloud-native infrastructure. Most of the game is about how to index billions of vectors for fast retrieval. One such indexing algorithm is Locality-sensitive hashing (LSH).

LSH aims to group vectors together based on similarity. For example, we could partition the vector space into multiple buckets, and we could call “nearest neighbors” whatever vectors belong to the same bucket. In practice, it is done a bit differently. An efficient way to partition the space is to project the vectors onto a space of a specific dimensionality and “binarize“ each component. The projection is done using a random matrix M of dimension (C, R) where C is the dimension of the original vector V and R is the dimension of the space we want to project the vectors into

V' = V. M

For example, if C = 2 and R = 3, we would project from a plane to a 3D space. We can now partition the space with regions above and below the hyperplanes passing by the origin. If we have, for example, a vector A = [0.5, -1.5, 0.3], we look at each of the components and assign a 1 if it is positive and 0 otherwise. The vector A would be hashed to [1, 0, 1] under that process. Every vector assigned the same hash will be close in the vector space and can be labelled “nearest neighbors”. The time complexity to hash a vector V is O(R x C + R) = O(R x C), and retrieving the vectors with the same hash can be done in constant time.

The hash of a vector under the LSH hashing process is a binary vector. To measure how different 2 binary vectors are, we use the Hamming Distance. The Hamming distance counts the number of times 2 strings have different characters. When we have strings of binary numbers, the Hamming distance can be computed using the XOR operation, and the number of resulting 1s can be counted.


* Embeddings: the superpower of deep learning
  
 ![image](https://github.com/ParthaPRay/LLM-Learning-Sources/assets/1689639/3ccc604f-8879-4db8-916a-f13b62d1ac77)

  Deep Learning finds its strength in its ability to model efficiently with different types of data at once. It is trivial to build models from multimodal datasets nowadays. It is not a new concept, though, nor was it impossible to do it prior to the advent of DL, but the level of complexity of feature processing and modeling was much higher with much lower performance levels!

One key aspect of this success is the concept of Embedding: a lower dimensionality representation of the data. This makes it possible to perform efficient computations while minimizing the effect of the curse of dimensionality and providing more robust representations when it comes to overfitting. In practice, this is just a vector living in a "latent" or "semantic" space.

The first great success of embedding for word encoding was Word2Vec back in 2013 and later GloVe in 2014. Since AlexNet back in 2012, many Convolutional network architectures (VGG16 (2014), ResNet (2015), Inception (2014), …) have been used as feature extractors for images. As of 2018, starting with BERT, Transformer architectures have been used quite a bit to extract semantic representations from sentences.

One domain where embeddings changed everything is recommender engines. It all started with Latent Matrix Factorization methods made popular during the Netflix competition in 2009. The idea is to have a vector representation for each user and product and use that as base features. In fact, any sparse feature could be encoded within an embedding vector, and modern rec engines typically use hundreds of embedding matrices for different categorical variables.

Dimensionality reduction is by all accounts not a new concept in Unsupervised Learning! PCA, for example, dates back to 1901; the concept of Autoencoder was introduced in 1986, and the variational Autoencoders (VAE) were introduced in 2013. For example, VAE is a key component of Stable Diffusion. The typical difficulty with Machine Learning is the ability to have labeled data. Self-supervised learning techniques like Word2Vec, Autoencoders, and generative language models allow us to build powerful latent representations of the data at a low cost. Meta came out with Data2Vec 2.0 to learn latent representations of any data modality using self-supervised learning.

Besides learning latent representations, a lot of work is being done to learn aligned representations between different modalities. For example, CLIP is a recent contrastive learning method to learn semantically aligned representations between text and image data.


* How LLMs answer questions with databases

  ![image](https://github.com/ParthaPRay/LLM-Learning-Sources/assets/1689639/f9b111eb-626b-4db0-be7f-7a1b86e93e09)
  How does an LLM ask a question to a database? The typical process is to use another LLM to encode the question into a vector representation and use this vector to query a vector database. By finding "similar" vectors in that database, we assume that the related documents should contain the answer to the original question. By feeding those documents into a prompt, we hope the LLM will have enough context to answer that question.

This process is called Retrieval Augmented Generation (RAG), and it suffers a simple problem: there is no reason for a question to be semantically similar to its answer. RAG can lead to many irrelevant documents being fed to the LLM without being provided the right context for an answer.

One solution to that is to use the Hypothetical Document Embeddings (HyDE) technique. The idea is to use the LLM to generate a hypothetical answer, embed that answer, and use this embedding to query the vector database. The hypothetical answer will be wrong, but it has more chance to be semantically similar to the right answer. 


* How to build Google image search engine

  ![image](https://github.com/ParthaPRay/LLM-Learning-Sources/assets/1689639/99066218-f2bd-439f-8faf-e072b05f8656)

  We can frame this problem as a ranking problem. We need a model that takes as input two images and returns a similarity score. Using that model, we can then rank the images based on that similarity score. A typical modeling approach is to utilize models that can learn a vectorial representation (embedding) of the images and compute a similarity metric on those vectors. We need a model that can extract the image features to learn a vector representation of images, and we need a model that can extract the text features to learn a vector representation of text inputs. We need to co-train the image and text models so the vector representations are semantically aligned.

  To ensure fast retrieval, we need a way to store the existing images and quickly search for similar images. Since we are encoding the images into their vector representations, it seems logical to index the images into a vector database. The indexing pipeline converts the original images into their vector representations and indexes them into a vector database.

When a user inputs a text or image query, we need to return a list of images. The embedding generation service generates an embedding encoding of the input query. The embedding query is sent to the vector database that returns the nearest neighbors of the query. The reranking service is mainly used to rerank the nearest neighbors using a better model than the embedding generation model. It could be used to personalize the ranking to the specific user by using user-specific data. The resulting list is a list of image IDs, and it is then sent to the image store to retrieve the actual images to return to the user.


* **LanceDB**, a free, open-source, serverless vectorDB that requires no setup.   It integrates into python data ecosystem so you can simply start using these in your existing data pipelines in pandas, arrow, pydantic etc. LanceDB has native Typescript SDK using which you can run vector search in serverless functions!
  
   ![image](https://github.com/ParthaPRay/LLM-Learning-Sources/assets/1689639/fb5edda5-8ac4-4e28-9c59-3f220604f444)
  
  https://github.com/lancedb/vectordb-recipes/tree/main


* Building Multi-Modal Search with Vector Databases
  
 https://www.youtube.com/watch?v=3WUobZryyok&t=6s&ab_channel=DeepLearningAI

https://docs.google.com/presentation/d/1sS-bxJ-k9EuESH2VhpwnybY3QyV_9FdxHLmZLCSpuSM/edit?usp=sharing


* How to select embedding model?

  https://www.rungalileo.io/blog/mastering-rag-how-to-select-an-embedding-model

  https://huggingface.co/spaces/mteb/leaderboard
  


* **Fastmbed**, FastEmbed is a lightweight, fast, Python library built for embedding generation. We support popular text models. Please open a Github issue if you want us to add a new model.

   https://www.youtube.com/watch?v=1mMLVQE11Io&ab_channel=LearnDatawithMark

  https://github.com/qdrant/fastembed

  https://qdrant.github.io/fastembed/

  https://simonwillison.net/2023/Oct/23/embeddings/
  
  
 * Embedding multimodal data for similarity search using 🤗 transformers, 🤗 datasets and FAISS

   https://github.com/huggingface/cookbook/tree/main/notebooks/en

   https://colab.research.google.com/github/huggingface/cookbook/blob/main/notebooks/en/faiss_with_hf_datasets_and_clip.ipynb


 * Introduction to Matryoshka Embedding Models

   https://huggingface.co/blog/matryoshka


* Ollama 0.1.26 Makes Embedding 100x Better**
  
  https://www.youtube.com/watch?v=Ml179HQoy9o&ab_channel=MattWilliams
  
  nomic-embed-text works very faster than llama2 as of now. 

  https://huggingface.co/nomic-ai/nomic-embed-text-v1


* From HuggingFace dataset to Qdrant vector database in 12 minutes flat

  https://www.gptechblog.com/from-huggingface-dataset-to-qdrant-vector-database-in-12-minutes-flat/


* Transformers and Quadrant: Revolutionizing Data Integration for NLP Tasks

  https://huggingface.co/blog/Andyrasika/qdrant-transformers

* Ollama Embedding: How to Feed Data to AI for Better Response?

   Model
  
   ![image](https://github.com/ParthaPRay/LLM-Learning-Sources/assets/1689639/e5581945-1d68-492f-b17b-34854d8bb927)

   Web
  
   ![image](https://github.com/ParthaPRay/LLM-Learning-Sources/assets/1689639/c40a56fe-588d-43fe-a2a1-23a1d7f9f88a)



   https://www.youtube.com/watch?v=jENqvjpkwmw&t=17s&ab_channel=MervinPraison










* Nomic's new embedding model : nomic-embed-text, 
  https://youtu.be/LpcaeQZDVB8?si=VrJzmRSrwJRxHwzv

* Crazy fast RAG, Ollama, Nomic embedding model, groq
  
  https://youtu.be/TMaQt8rN5bE?si=4KnO2DFdVYiWjkg6


 * Mixedbread mxbai-embed-large-v1 embedding model

   This is a base sentence embedding model. It was trained using [AnglE](https://arxiv.org/abs/2309.12871)
   loss on our high-quality large scale data. It achieves SOTA performance on BERT-large scale. Find out more in our [blog post](
   https://mixedbread.ai/blog/mxbai-embed-large-v1)

 https://huggingface.co/mixedbread-ai/mxbai-embed-large-v1
   
https://youtu.be/CXDOkHFboAU?si=m8OoaiPa0JHMDs1e


| Model                       | Avg (56 datasets) | Classification (12 datasets) | Clustering (11 datasets) | PairClassification (3 datasets) | Reranking (4 datasets) | Retrieval (15 datasets) | STS (10 datasets) | Summarization (1 dataset) |
|-----------------------------|-------------------|-------------------------------|--------------------------|----------------------------------|------------------------|-------------------------|-------------------|---------------------------|
| mxbai-embed-large-v1        | 64.68             | 75.64                         | 46.71                    | 87.2                             | 60.11                  | 54.39                   | 85.00             | 32.71                     |
| bge-large-en-v1.5           | 64.23             | 75.97                         | 46.08                    | 87.12                            | 60.03                  | 54.29                   | 83.11             | 31.61                     |
| mxbai-embed-2d-large-v1     | 63.25             | 74.14                         | 46.07                    | 85.89                            | 58.94                  | 51.42                   | 84.90             | 31.55                     |
| nomic-embed-text-v1         | 62.39             | 74.12                         | 43.91                    | 85.15                            | 55.69                  | 52.81                   | 82.06             | 30.08                     |
| jina-embeddings-v2-base-en  | 60.38             | 73.45                         | 41.73                    | 85.38                            | 56.98                  | 47.87                   | 80.70             | 31.60                     |
| Proprietary Models          |                   |                               |                          |                                  |                        |                         |                   |                           |
| OpenAI text-embedding-3-large | 64.58           | 75.45                         | 49.01                    | 85.72                            | 59.16                  | 55.44                   | 81.73             | 29.92                     |
| Cohere embed-english-v3.0   | 64.47             | 76.49                         | 47.43                    | 85.84                            | 58.01                  | 55.00                   | 82.62             | 30.18                     |
| OpenAI text-embedding-ada-002 | 60.99           | 70.93                         | 45.90                    | 84.89                            | 56.32                  | 49.25                   | 80.97             | 30.80                     |




* RAG Databases with Johannes Jolkkonen: When to Choose a Graph Database vs Alternatives

  https://www.youtube.com/watch?v=1Iuuvk6yJME&ab_channel=Neo4j

* Pdf reader using genai-stack using Langchain + Docker + Neo4j + Ollama

  https://github.com/docker/genai-stack/blob/main/pdf_bot.py


* NODES 2023 - Using LLMs to Convert Unstructured Data to Knowledge Graphs

 https://www.youtube.com/watch?v=qLdkRReMPvM&ab_channel=Neo4j







* **Index Guide**


  * Guidelines to choose an FAISS index


 Selecting the appropriate FAISS index is crucial for optimizing performance and depends on the specific requirements of your project, such as dataset size, query frequency, and latency constraints. Here's a guide to selecting different indexes based on these criteria:

    - For Small Datasets:
   
      * FlatL2 or FlatIP: Ideal for smaller datasets due to their simplicity and moderate memory consumption. They perform exhaustive searches across all vectors and provide precise results.
      * LSH (Locality-Sensitive Hashing): Suitable for small to medium datasets and recommended for vectors up to 128 dimensions. LSH is faster than exhaustive search but may trade off a bit of accuracy for speed.
   
    - For Medium to Large Datasets:
     
     * HNSW (Hierarchical Navigable Small World): Extremely fast for both indexing and querying and supports higher-dimensional data. However, it requires more memory, making it suitable for medium-sized datasets.
     * IVF (Inverted File Indexing): Ideal for large datasets. It segments the search space into a predefined number of clusters and only searches within the most relevant clusters. IVF indexes balance between memory usage and search speed, making them efficient for large-scale applications.
   
    - For Very Large Datasets:
   
      * Advanced versions of IVF, such as IVFADC (Inverted File with Asymmetric Distance Computation) or IVFPQ (Product Quantization), can be used. These indexes further compress the dataset and reduce the search space, optimizing both memory usage and search speed at the scale of millions of vectors.
        
   When integrating a semantic cache with a FAISS-based RAG system, it's essential to:
   
     - Choose the right index type based on your dataset size and query characteristics.
     - Consider the trade-offs between accuracy and speed, as some indexes may offer faster retrieval at the expense of precision.
     - Test and evaluate different indexes to find the best configuration for your specific use case.
       
 https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index

  https://github.com/facebookresearch/faiss



* LlamaIndex Indexing Guide

   -VectorStoreIndex
   - Summary Index
   - Tree Index
   - Keyword Table Index
   - Knowledge Graph Index
   - Knowledge Graph Query Engine
   - Knowledge Graph RAG Query Engine
   - REBEL + Knowledge Graph Index
   - REBEL + Wikipedia Filtering
   - SQL Index
   - SQL Query Engine with LlamaIndex + DuckDB
   - Document Summary Index
   - The ObjectIndex Class

    https://docs.llamaindex.ai/en/stable/module_guides/indexing/index_guide.html




* **FlagEmbedding**

  FlagEmbedding focuses on **retrieval-augmented LLMs**, consisting of the following projects currently:

   - Long-Context LLM: Activation Beacon
   - Fine-tuning of LM : LM-Cocktail
   - Dense Retrieval: BGE-M3, LLM Embedder, BGE Embedding
   - Reranker Model: BGE Reranker
   - Benchmark: C-MTEB


  https://github.com/FlagOpen/FlagEmbedding
  
  https://huggingface.co/BAAI/bge-base-en-v1.5

  


* CPU Optimized Embeddings with 🤗 Optimum Intel and fastRAG

 SFR-Embedding by Salesforce Research

 
Should dense vectors always be used for information retrieval? The two dominant approaches have trade-offs:

    * Sparse retrieval matches n-grams, phrases, or metadata to search large collections efficiently and at scale. However, it may miss relevant documents due to lexical gaps between the query and the document.
    
    *  Semantic retrieval encodes text into dense vectors, capturing context and meaning better than bag-of-words. It can retrieve semantically related documents despite lexical mismatches. However, it's computationally intensive, has higher latency, and requires sophisticated encoding models compared to lexical matching like BM25.


    Optimum Intel is an open-source library that accelerates end-to-end pipelines built with Hugging Face libraries on Intel Hardware. Optimum Intel includes several techniques to accelerate models such as low-bit quantization, model weight pruning, distillation, and an accelerated runtime.

The runtime and optimizations included in Optimum Intel take advantage of Intel® Advanced Vector Extensions 512 (Intel® AVX-512), Vector Neural Network Instructions (VNNI) and Intel® Advanced Matrix Extensions (Intel® AMX) on Intel CPUs to accelerate models. Specifically, it has built-in BFloat16 (bf16) and int8 GEMM accelerators in every core to accelerate deep learning training and inference workloads. AMX accelerated inference is introduced in PyTorch 2.0 and Intel Extension for PyTorch (IPEX) in addition to other optimizations for various common operators.

Optimizing pre-trained models can be done easily with Optimum Intel; many simple examples can be found here.
 
  https://huggingface.co/blog/intel-fast-embedding

.





# Essentials on LoRA, Quantization and Sharding Variants

**LoRA**

   * What is LoRA?
      
       Edward Hu, https://edwardjhu.com/
      
       https://lightning.ai/lightning-ai/studios/code-lora-from-scratch
      
   * LoRA training scripts of the world, unite!
      
       https://huggingface.co/blog/sdxl_lora_advanced_script
     
   * Parameter-Efficient LLM Finetuning With Low-Rank Adaptation (LoRA)
      
      https://lightning.ai/pages/community/tutorial/lora-llm/
      
   * Finetuning LLMs with LoRA and QLoRA: Insights from Hundreds of Experiments
      
        https://lightning.ai/pages/community/lora-insights/


   
      
   * Practical Tips for Finetuning LLMs Using LoRA (Low-Rank Adaptation)
      
         https://magazine.sebastianraschka.com/p/practical-tips-for-finetuning-llms
      
         Hackable implementation of state-of-the-art open-source LLMs based on nanoGPT. Supports flash attention, 4-bit and 8-bit quantization, LoRA and LLaMA-Adapter fine-tuning, pre-training. Apache 2.0-licensed.
        
         https://github.com/Lightning-AI/lit-gpt
        
      
   * Optimzie LLM utilization with LoRA

     ![image](https://github.com/ParthaPRay/LLM-Learning-Sources/assets/1689639/7c7661c2-8e0a-481e-9729-804f82ad219a)

      How can we optimize machine utilization for multiple fine-tuned LLMs? Let’s consider OpenAI as an example and its API to fine-tune models. 
In the case of OpenAI, “fine-tuning“ means that the model is specialized by using some proprietary data, and it is then deployed on GPU hardware for API access. Naively, we could think that for each new customer wanting to fine-tune their model, we would need to deploy a new model on a new GPU cluster. However, it is unlikely that OpenAI proceed this way!

GPU hardware is really expensive, and they would need to allocate a GPU cluster for each new customer. OpenAI pricing model is based on model usage, meaning customers only pay when they use the model, but for OpenAI, the cost of serving the model never stops! It is very likely that there have been thousands of customers who just wanted to test OpenAI’s fine-tuning capabilities, and the resulting fine-tuned models were never actually used. Would OpenAI just handle the serving cost for each of those models?

One strategy to fine-tune LLMs is to use adapters that can be “plugged“ into the base model. The idea is to avoid updating the weights of the base model and have the adapters capture the information about the fine-tuning tasks. We can plug in and out different adapters that specialize the model on different tasks. The most common and efficient adapter type is the Low-Rank Adapter (LoRA). The idea is to replace some of the large matrices within the model with smaller ones for the gradient computation.

Because of the small size of those adapters and their simple additive logic, it is easy to add multiple adapters at once for different fine-tuning tasks. Those adapters can be trained separately and plugged together at serving time. We just need a logic to route the inputs to their respective task.

This is extremely beneficial when we have a low request volume for some of the tasks. In the case of OpenAI, with multiple LoRA adapters, it becomes easy for them to deploy multiple fine-tuned models on the same GPU cluster. After the LoRA weights have been trained during a fine-tuning process, we just store those in a model registry. The cost of storing those weights instead of a full fine-tuned model is going to be much lower! At serving time, we can plug multiple adapters into the same base model and route the customer’s request to its own adapter.

OpenAI can easily measure the adapter utilization and the customers’ request volume for the different fine-tuned models. If the volume is low, it can be deployed along with other low-utilization adapters on the same base model, and if it is high, the adapter can be allocated its own base model such that the users don’t wait too long for their requests to be completed.
     
   *  Making LLMs even more accessible with bitsandbytes, 4-bit quantization and QLoRA
      
        https://huggingface.co/blog/4bit-transformers-bitsandbytes


   * Improving LoRA: Implementing Weight-Decomposed Low-Rank Adaptation **(DoRA)** from Scratch
      
        https://magazine.sebastianraschka.com/p/lora-and-dora-from-scratch
      
        https://github.com/rasbt/dora-from-scratch
      
            
   * Rank-Stabilized LoRA: Unlocking the Potential of LoRA Fine-Tuning
      
        https://huggingface.co/blog/damjan-k/rslora
         
   * Finetuning LLMs with LoRA and QLoRA: Insights from Hundreds of Experiments
      
        https://lightning.ai/pages/community/lora-insights/
        
      
   * A Gentle Introduction to 8-bit Matrix Multiplication for transformers at scale using Hugging Face Transformers, Accelerate and bitsandbytes
      
        https://huggingface.co/blog/hf-bitsandbytes-integration
      
   * SDXL in 4 steps with Latent Consistency LoRAs
      
        https://huggingface.co/blog/lcm_lora




**Quantization**

         The Two Types of LLM Quantization: PTQ and QAT
         
         While there are several quantization techniques, the most notable of which we detail later in this guide, generally speaking, LLM quantization falls into two categories:
         
         Post-Training Quantization (PTQ): this refers to techniques that quantize an LLM after it has already been trained. PTQ is easier to implement than QAT, as it requires less training data and is faster. However, it can also result in reduced model accuracy from lost precision in the value of the weights. 
         
         Quantization-Aware Training (QAT): this refers to methods of fine-tuning on data with quantization in mind. In contrast to PTQ techniques, QAT integrates the weight conversion process, i.e., calibration, range estimation, clipping, rounding, etc., during the training stage. This often results in superior model performance, but is more computationally demanding. 



   ![image](https://github.com/ParthaPRay/LLM-Learning-Sources/assets/1689639/dcde7a2d-d48e-47c9-8b16-e20082e02410)
  
 Not too long ago, the largest Machine Learning models most people would deal with merely reached a few GB in memory size. Now, every new generative model coming out is between 100B and 1T parameters! To get a sense of the scale, one float parameter that's 32 bits or 4 bytes, so those new models scale between 400 GB to 4 TB in memory, each running on expensive hardware. Because of the massive scale increase, there has been quite a bit of research to reduce the model size while keeping performance up. There are 5 main techniques to compress the model size.

- Model pruning is about removing unimportant weights from the network. The game is to understand what "important" means in that context. A typical approach is to measure the impact on the loss function of each weight. This can be done easily by looking at the gradient and second-order derivative of the loss. Another way to do it is to use L1 or L2 regularization and get rid of the low-magnitude weights. Removing whole neurons, layers or filters is called "structured pruning" and is more efficient when it comes to inference speed.

- Model quantization is about decreasing parameter precision, typically by moving from float (32 bits) to integer (8 bits). That's 4X model compression. Quantizing parameters tends to cause the model to deviate from its convergence point, so it is typical to fine-tune it with additional training data to keep model performance high. We call this "Quantization-aware training". When we avoid this last step, it is called "Post training quantization", and additional heuristic modifications to the weights can be performed to help performance.

- Low-rank decomposition comes from the fact that neural network weight matrices can be approximated by products of low-dimension matrices. A N x N matrix can be approximately decomposed into a product of 2 N x 1 matrices. That's an O(N^2) -> O(N) space complexity gain! 

- Knowledge distillation is about transferring knowledge from one model to another, typically from a large model to a smaller one. When the student model learns to produce similar output responses, that is response-based distillation. When the student model learns to reproduce similar intermediate layers, it is called feature-based distillation. When the student model learns to reproduce the interaction between layers, it is called relation-based distillation. 

- Lightweight model design is about using knowledge from empirical results to design more efficient architectures. That is probably one of the most used methods in LLM research.
   

* Quantization

  https://huggingface.co/docs/optimum/concept_guides/quantization

* A Guide to Quantization in LLMs

 https://symbl.ai/developers/blog/a-guide-to-quantization-in-llms/

* Quantization in LLMs: Why Does It Matter?

  https://blog.dataiku.com/quantization-in-llms-why-does-it-matter
  
* What are Quantized LLMs?
  
  https://www.tensorops.ai/post/what-are-quantized-llms#:~:text=LLM%20Quantization%20is%20enabled%20thanks,allowing%20it%20to%20be%20run

* The LLM Revolution: Boosting Computing Capacity with Quantization Methods

  https://blog.gopenai.com/the-llm-revolution-boosting-computing-capacity-with-quantization-methods-b8666cdb4b6a
    

* Which Quantization Method is Right for You? (GPTQ vs. GGUF vs. AWQ)
  https://www.maartengrootendorst.com/blog/quantization/

* Quantization and LLMs - Condensing Models to Manageable Sizes
  https://www.exxactcorp.com/blog/deep-learning/what-is-quantization-and-llms
  
* Best LLM quantization (accuracy and speed)
  
  https://scifilogic.com/best-llm-quantization-accuracy-and-speed/
  
* Serving Quantized LLMs on NVIDIA H100 Tensor Core GPUs

  https://www.databricks.com/blog/serving-quantized-llms-nvidia-h100-tensor-core-gpus

* New Tutorial on LLM Quantization w/ QLoRA, GPTQ and Llamacpp, LLama 2

  https://www.youtube.com/watch?v=YEVyupJxt1Q
  
*  How to make your LLMs lighter with GPTQ quantization

  https://bdtechtalks.com/2023/11/08/llm-quantization-gptq/
  
* Model Quantization with 🤗 Hugging Face Transformers and Bitsandbytes Integration

  https://medium.com/@rakeshrajpurohit/model-quantization-with-hugging-face-transformers-and-bitsandbytes-integration-b4c9983e8996


* How to Quantize an LLM with GGUF or AWQ

  https://www.youtube.com/watch?v=XM8pllpBVA0
  
* Effective Post-Training Quantization for Large Language Models

  https://medium.com/intel-analytics-software/effective-post-training-quantization-for-large-language-models-with-enhanced-smoothquant-approach-93e9d104fb98


* Overview of natively supported quantization schemes in 🤗 Transformers

   https://huggingface.co/blog/overview-quantization-transformers


* How to quantization an LLM with GGUF or AWQ
  
  https://youtu.be/XM8pllpBVA0?si=v_jLj78pCnOXIv2i

  https://tinyurl.com/2s58xnam
  
* Making LLMs lighter with AutoGPTQ and transformers
  
  GPTQ blogpost – gives an overview on what is the GPTQ quantization method and how to use it.

  https://huggingface.co/blog/gptq-integration
  
  https://colab.research.google.com/drive/1_TIrmuKOFhuRRiTWN94iLKUFu6ZX4ceb?usp=sharing

  
* bistandbytes 4-bit quantization blogpost - This blogpost introduces 4-bit quantization and QLoRa, an efficient finetuning approach

  This blogpost introduces 4-bit quantization and QLoRa, an efficient finetuning approach.

  https://huggingface.co/blog/4bit-transformers-bitsandbytes


* A Gentle Introduction to 8-bit Matrix Multiplication for transformers at scale using Hugging Face Transformers, Accelerate and bitsandbytes

  bistandbytes 8-bit quantization blogpost - This blogpost explains how 8-bit quantization works with bitsandbytes.

  https://huggingface.co/blog/hf-bitsandbytes-integration

  Basic usage Google Colab notebook for bitsandbytes - This notebook shows how to use 4-bit models in inference with all their variants, and how to run GPT-neo-X (a 20B parameter model) on a free Google Colab instance.

  https://colab.research.google.com/drive/1ge2F1QSK8Q7h0hn3YKuBCOAS0bK8E0wf?usp=sharing

* Comparing the Performance of LLMs: A Deep Dive into Roberta, Llama 2, and Mistral for Disaster Tweets Analysis with Lora
    https://huggingface.co/blog/Lora-for-sequence-classification-with-Roberta-Llama-Mistral


* Introduction to Quantization cooked in 🤗 with 💗🧑‍🍳

  Merve's blogpost on quantization - This blogpost provides a gentle introduction to quantization and the quantization methods supported natively in transformers.

  https://huggingface.co/blog/merve/quantization


* Democratizing LLMs: 4-bit Quantization for Optimal LLM Inference

  https://towardsdatascience.com/democratizing-llms-4-bit-quantization-for-optimal-llm-inference-be30cf4e0e34



 * Quantize any LLM with GGUF and Llama.cpp, https://www.youtube.com/watch?v=wxQgGK5K0rE&ab_channel=AIAnytime

  
 * **Half Quadratic Quantization (HQQ)**
   
  HQQ is a fast and accurate model quantizer that skips the need for calibration data. It's super simple to implement (just a few lines of code for the optimizer). It can crunch through quantizing the Llama2-70B model in only 4 minutes! 


  Supported Models
  
 LLMs

   Llama (Hugging Face + VLLM) 🦙
   Mistral (Hugging Face)
   Mixtral-8x7B (Hugging Face)
   Phi + Phi_opt (Hugging Face)
   
 Vision
  ViT-CLIP (timm) 🖼️

https://huggingface.co/posts/macadeliccc/282259361762056

   AutoHQQ: https://colab.research.google.com/drive/1cG_5R_u9q53Uond7F0JEdliwvoeeaXVN?usp=sharing

https://huggingface.co/macadeliccc/Nous-Hermes-2-Mixtral-8x7B-DPO-HQQ
    https://mobiusml.github.io/hqq_blog/

  
  https://github.com/mobiusml/hqq



**Sharding**


      How to shard LLMs locally, https://youtu.be/F0pkj2trRcI?si=zAsZmmbhsp1wqlBe












# LLM Apps

  * Visual Question Answering with IDEFICS 9B Multimodal LLM, https://www.youtube.com/watch?v=hyP1ekLKtiI&ab_channel=AIAnytime
  
  * Outfit Anyone: A Diffusion Project for Virtual Try On, https://www.youtube.com/watch?v=V21GfgSFuGk&ab_channel=AIAnytime
    
  * Oncology RAG App - Powered by Meditron 7B Medical LLM, https://www.youtube.com/watch?v=kvbjB-q5Dss&ab_channel=AIAnytime
 
  * Investment Banker RAG Chatbot using Intel's Neural Chat LLM, https://www.youtube.com/watch?v=d9wCHH3iknM&ab_channel=AIAnytime
 
  * Deploy RAG App built using Create Llama on Vercel: Free and Easy Method, https://www.youtube.com/watch?v=euYa4iesOm8&ab_channel=AIAnytime
    
  * Create a LlamaIndex App with Create Llama: No Code tool for RAG, https://www.youtube.com/watch?v=JkGU3d8IM1c&ab_channel=AIAnytime

  * AutoLLM: Ship RAG based LLM Apps and API in Seconds, https://www.youtube.com/watch?v=iTGbwD-sSxM&ab_channel=AIAnytime

  * Query Your CSV using LIDA: Automatic Generation of Visualizations with LLMs, https://www.youtube.com/watch?v=U9K1Cu45nMQ&ab_channel=AIAnytime
 
  * Chat with Data App: RAG using Mistral 7B, Haystack, and Chainlit, https://www.youtube.com/watch?v=01_2-Dy57ys&ab_channel=AIAnytime

  * Building LLM Applications with Langchain, https://www.youtube.com/watch?v=HmZzbhL8Tf8&list=PLfaIDFEXuae2Zb0phFLWAxgrJT7f416xq&pp=iAQB

  * RAG Implementation using Mistral 7B, Haystack, Weaviate, and FastAPI, https://www.youtube.com/watch?v=C5mqILmVUEo&ab_channel=AIAnytime

  * Let's Build an AI News Anchor Generator App using Generative AI, https://www.youtube.com/watch?v=cddahTnCo10&ab_channel=AIAnytime

  * Screenshot to Code Generation: 10x Faster Frontend/UI Development, https://www.youtube.com/watch?v=52Xq6AaRnT4&ab_channel=AIAnytime

  * ComfyUI GUI for Image and Video Generation: Google Colab Setup, https://www.youtube.com/watch?v=PYEnK_iQeZU&ab_channel=AIAnytime

  * Build Generative AI Agents using Dialogflow CX and Vertex AI on GCP, https://www.youtube.com/watch?v=cDY8lm6vg7w&ab_channel=AIAnytime

  * Build a Containerized Transcription API using Whisper Model and FastAPI, https://www.youtube.com/watch?v=NU406wZz1eU&ab_channel=AIAnytime

  * Build Your RAG-based ChatGPT Web App with Azure: LawGPT Use Case Tutorial, https://www.youtube.com/watch?v=wmfAJWwyaQA&ab_channel=AIAnytime
 
  * Creating a Veterinary Chatbot using Llama 2: Harnessing Gen AI for Pet Care, https://www.youtube.com/watch?v=Iyzvka711pc&ab_channel=AIAnytime

  * Build Your API for Llama 2 on AWS: Lambda Function and API Gateway, https://www.youtube.com/watch?v=Nlo7WclRBXc&t=512s&pp=ygUGb2xsYW1h

  * Deploy Llama 2 for your Entire Organisation, https://www.youtube.com/watch?v=Ror2xOOA-VE&ab_channel=TrelisResearch

  * Install and Run Mistral 7B on AWS, https://www.youtube.com/watch?v=aSh66tG1B5o&pp=ygUNb2xsYW1hIG9uIEFXUw%3D%3D

  * Deploy Llama 2 on AWS SageMaker using DLC (Deep Learning Containers), https://www.youtube.com/watch?v=rQq1m2aJ_fk&ab_channel=AIAnytime
 
  * Enterprise Chat App using Azure Cognitive Search and Azure OpenAI: End-to-End Tutorial, https://www.youtube.com/watch?v=hkSnPhhjm1Y&ab_channel=AIAnytime

  * Containerizing LLM-Powered Apps: Part 1 of the Chatbot Deployment, https://www.youtube.com/watch?v=7CeAJ0EbzDA&ab_channel=AIAnytime

  * Deploy LLM Powered Apps on Azure App Service: Part 2 of the Chatbot Deployment, https://www.youtube.com/watch?v=vYIlhgVHAls&ab_channel=AIAnytime

  * Serve a Custom LLM for Over 100 Customers, https://www.youtube.com/watch?v=1TU9ZrZhqw0&ab_channel=TrelisResearch

  * Long Context Summarization, https://www.youtube.com/watch?v=I83TH4x9keo&ab_channel=TrelisResearch

  * Function Calling Datasets, Training and Inference, https://www.youtube.com/watch?v=hHn_cV5WUDI&ab_channel=TrelisResearch
 
  * How to Build an OpenAI LLM on a Private Network with AWS, https://www.youtube.com/watch?v=6LGGQERxrQo&ab_channel=SingleStore
 
  * Amazon Bedrock: Generative AI on AWS without the Headaches, https://www.youtube.com/watch?v=Yj_7FuFgPyI

  * FULLY LOCAL Mistral AI PDF Processing Hands-on Tutorial, https://www.youtube.com/watch?v=wZDVgy_14PE&pp=ygUNb2xsYW1hIG9uIEFXUw%3D%3D

  * PrivateGPT 2.0 - FULLY LOCAL Chat With Docs (PDF, TXT, HTML, PPTX, DOCX, and more), https://www.youtube.com/watch?v=XFiof0V3nhA&ab_channel=MatthewBerman
 
  * AutoLLM: Create RAG Based LLM Web Apps in SECONDS!, https://www.youtube.com/watch?v=kPaiZe_qD34&ab_channel=WorldofAI
 
  * Use OpenChat and LM Studio with LLMWare, https://www.youtube.com/watch?v=h2FDjUyvsKE&ab_channel=llmware

  * Compare Embedding Models for Side by Side Queries Using Postgres with LLMWare, https://www.youtube.com/watch?v=Bncvggy6m5Q&ab_channel=llmware
    
  * AutoGen Studio with 100% Local LLMs (LM Studio), https://www.youtube.com/watch?v=ob45YmYD2KI&ab_channel=PromptEngineering
  
  * AutoGen Studio UI 2.0: Easiest Way to Create Custom Agents, https://www.youtube.com/watch?v=KIvl-VY8H0Y&ab_channel=PromptEngineering
  
  * This is a lightweight app using the Web Research Retriever. It uses langchain to search and chat on web data on streamlit.

      ![image](https://github.com/ParthaPRay/LLM-Learning-Sources/assets/1689639/531acb8f-9e56-498e-8c96-0f0af82fd46d)


    https://github.com/langchain-ai/web-explorer/tree/main
    
  * Your LLM Powered Financial Analyst, https://www.youtube.com/watch?v=JeruKKuMxCg&ab_channel=PromptEngineering
 

  * AutoGen + Panel Ep.3 - Web UI for Multi-agent with Document Retrieval

    https://www.youtube.com/watch?v=98Ri4VVBP_8&t=431s

     https://github.com/yeyu2/Youtube_demos


  * How to Create a Web UI for AutoGen by Using Panel

    https://www.youtube.com/watch?v=mFmPDyLlj1E

    https://github.com/yeyu2/Youtube_demos


  * Create Full Function UI for AutoGen Powered by Panel (Human Input Enabled)

    https://www.youtube.com/watch?v=9lSaRP9GLCY

  * AutoGen + Function Calling + Open Source LLMs, Here is How

    https://www.youtube.com/watch?v=UIBerUGqHjc&ab_channel=YeyuLab

  * Use Open Source LLMs in AutoGen powered by Fireworks AI, without GPU/CPU

    https://www.youtube.com/watch?v=HN96PTdiseo&ab_channel=YeyuLab

   
  * Development with Large Language Models Tutorial – OpenAI, Langchain, Agents, Chroma

    https://www.youtube.com/watch?v=xZDB1naRUlk

  * Make an offline GPT voice assistant in Python
  
    https://youtu.be/w5unVTO7mLQ?si=LLictvhoG4hm2JJy

  * Build and Run a Medical Chatbot using Llama 2 on CPU Machine: All Open Source, https://www.youtube.com/watch?v=kXuHxI5ZcG0&ab_channel=AIAnytime
    
  * Chat With Websites Using ChainLit / Streamlit, LangChain, Ollama & Mistral 🧠, https://www.youtube.com/watch?v=FZrkm0vaYYQ&ab_channel=DataScienceBasics

    https://github.com/sudarshan-koirala/chat-with-website

  * LocalGPT API: Serve Multiple Users At the Same time, https://www.youtube.com/watch?v=z9wDKwgQojM&ab_channel=PromptEngineering
    
  * Deploy and Use any Open Source LLMs using RunPod, https://www.youtube.com/watch?v=nHuHGoLSXb0&ab_channel=AIAnytime

  * CPU-based SLMs for AI Agents and Function Calling by LLMWare, https://www.youtube.com/watch?v=0MOMBJjytkQ&ab_channel=AIAnytime

  * Function Calling using Open Source LLM (Mistral 7B), https://www.youtube.com/watch?v=MQmfSBdIfno&t=337s&ab_channel=AIAnytime

  * 4 LLM frameworks to build AI apps with voice data

     ![image](https://github.com/ParthaPRay/LLM-Learning-Sources/assets/1689639/6ba1f1d8-2f1b-4ed4-b54f-2596d92c53a7)

     - LeMUR: https://www.assemblyai.com/docs/getting-started/apply-llms-to-audio-files
     - LangChain: https://www.langchain.com/langchain
     - LlamaIndex: https://www.llamaindex.ai/
     - Haystack: https://haystack.deepset.ai/

      https://www.youtube.com/watch?v=wdF-0CGkoeQ&ab_channel=AssemblyAI

      
  
    
  * vector search, RAG, and Azure AI search,

    https://speakerdeck.com/pamelafox/vector-search-and-retrieval-for-generative-ai-app-microsoft-ai-tour-sf

    https://www.youtube.com/live/vuOA13Y_Qzk?si=bT6zY4piPt_yUn_Q

    https://github.com/pamelafox/vector-search-demos

    https://pamelafox.github.io/vectors-comparison

    https://github.com/Azure-Samples/azure-search-openai-demo


  * Manage vector databases and long term memory in flowwise, AI vector tools Review part 1
  
   https://youtu.be/d7nAcshOe4w?si=kArGQ_Ua8pFdvzFy

  * Learn how to use LlamaIndex with LanChainin Flowwise,LlamaIndex vs Langchain part 2,
  
    https://youtu.be/KVOWPeV9s00?si=T9K6edpHIcAr0BBS

  * Create a Web Interface for your LLM in Python

    https://huggingface.co/blog/Alex1337/create-a-web-interface-for-your-llm-in-python

    Turns Data and AI algorithms into production-ready web applications in no time.

    https://github.com/Avaiga/taipy

    https://www.taipy.io/

  * I made AI to auto categorise 10000 comments on Google Sheet with 0$
 
     https://youtu.be/wXiTuNnh2h4?si=P58oj6TLjhqOmtOD


  * Build a medical RAG app using Biomistral, Qdrant and Llama.cpp

    https://youtu.be/A_m3tCqdts4?si=23s00oY8opM8i2PR


  * Steerable AI with Pinecone + Semantic router, https://youtu.be/qjRrMxT20T0?si=hQj7YxUJAj2Y2unV

  * Constitutional AI with Open LLMs

    https://huggingface.co/blog/constitutional_ai

    https://github.com/huggingface/alignment-handbook/tree/main/recipes/constitutional-ai

  * Stop paying for ChatGPT with these two tools | LMStudio x AnythingLLM\

    https://www.youtube.com/watch?v=-Rs8-M-xBFI&ab_channel=TimCarambat

  * Create Chat UI Using ChainLit, LangChain, Ollama & Gemma 🧠
    https://www.youtube.com/watch?v=n9AMtXLveMs&t=11s&ab_channel=DataScienceBasics

  * LangSmith For Beginners | Must know LLM Evaluation Platform 🔥
    https://www.youtube.com/watch?v=FgG-trkAMwU&ab_channel=DataScienceBasics


 * Create-Llama: deploy LlamaIndex RAG App to Vercel
 
   https://youtu.be/D8PM89Xry7Q?si=N52WpnQn-CsUqHex


 * PhiData: How to Seamlessly Integrate AI into Your Application

   https://www.youtube.com/watch?v=fLGj63fiYfM&ab_channel=MervinPraison

      ![image](https://github.com/ParthaPRay/LLM-Learning-Sources/assets/1689639/c03206bd-997c-432c-be61-5452fd0fdd85)

      ![image](https://github.com/ParthaPRay/LLM-Learning-Sources/assets/1689639/80c10a2b-66fd-4968-a2a5-a1374b22a057)

 * Boost Gmail Efficiency with AI: Python Tutorial (CrewAI, LangChain, LangGraph)

    https://www.youtube.com/watch?v=o4-4NvrcOvs&ab_channel=AIFORDEVS 

    https://github.com/joaomdmoura/crewAI
    
  * Create Complex Research Analysis with AI Agents using SLIM models on CPU with LLMWare

    https://www.youtube.com/watch?v=y4WvwHqRR60&ab_channel=llmware
    
      ![image](https://github.com/ParthaPRay/LLM-Learning-Sources/assets/1689639/d6bdb92a-21e1-4ca6-9324-83bf63f352ac)

    https://huggingface.co/llmware

    https://github.com/llmware-ai/llmware

    https://github.com/llmware-ai/llmware/tree/main/examples/SLIM-Agents/


  * Taipy: Creating Production-Grade Apps with Taipy vs Streamlit, https://www.youtube.com/watch?v=MgAIrGxnN-8&ab_channel=WorldofAI

  * Anthropic Claude API: Supercharge Your AI App with Large Context, https://www.youtube.com/watch?v=Wtt9tuO8UPY&ab_channel=MervinPraison

  * Build an AI Applicant Tracking System(ATS) for Your Resumes with LLMs|Get JOBS 100X FASTER, https://www.youtube.com/watch?v=7lP7fune0Gw&ab_channel=DataInsightEdge
    
 * Build & Chat with Invoices using Google’s Gemini Pro VisionIStreamlit + Use Case| Tutorial, https://www.youtube.com/watch?v=7_926xGDbDY&ab_channel=DataInsightEdge

 * Chat with Multiple Documents using Gemini Pro with LangChain| Step-by-Step Tutorial #ai #llm, https://www.youtube.com/watch?v=UXLWLFOB0Xo&ab_channel=DataInsightEdge

 * GEMINI Pro with LangChain | Chat, MultiModal and Chat with your Documents, https://www.youtube.com/watch?v=7h8ZHSkAkas&ab_channel=PromptEngineering
 
 * Gemini Pro + LangChain - Chains, Mini RAG, PAL + Multimodal, https://www.youtube.com/watch?v=G3-YOEVg-xc&ab_channel=SamWitteveen

 * LangGraph + function call + Yahoofinance = Multi-agent application, https://youtu.be/r2PvHdkaXWc?si=alEiCMZwy0xAwNwG
   


   
 * CrewAI agents for stock analysis (works with local Ollama LLMs),
 https://youtu.be/U_Sg3Odf1vk?si=gzDboL0gLYTPn7Q6

 * Build an LLM powered chrome extension, https://youtu.be/9RKXffJsJhs?si=Ly_ocxdSttphdhKk

  * LangGraph and OpenGPTs: building agent forward applications with Langchain
, https://www.youtube.com/live/NdF609kO8FY?si=OLcaLpy3ALBUeOUF


* Claude 3 Function Calling: How to Integrate your own Software?, https://www.youtube.com/watch?v=LuBROahHvfo&ab_channel=MervinPraison

  ![image](https://github.com/ParthaPRay/LLM-Learning-Sources/assets/1689639/a0853d80-410f-4f36-99b4-00762ebbfd1f)

  
* I Created AI Assistants to Automate Recruitment Process: Crew AI

   https://www.youtube.com/watch?v=OQJ4gp70Zg0&ab_channel=MervinPraison

* Building Production-Grade LLM Apps

  https://www.youtube.com/watch?v=fo0F-DAum7E&ab_channel=DeepLearningAI


* Images Interpolation with Stable Diffusion

  This notebook shows how to use Stable Diffusion to interpolate between images. Image interpolation using Stable Diffusion is the process of creating intermediate images that smoothly transition from one given image to another, using a generative model based on diffusion.

   ![image](https://github.com/ParthaPRay/LLM-Learning-Sources/assets/1689639/5121241e-0a38-415c-8fce-63ed55f91db5)

  
  https://huggingface.co/learn/cookbook/stable_diffusion_interpolation

  https://colab.research.google.com/github/huggingface/cookbook/blob/main/notebooks/en/stable_diffusion_interpolation.ipynb



* Building Google's Dramatron with LangGraph JS & Anthropic's Claude 3

  https://www.youtube.com/watch?v=alHnQjyn7hg&ab_channel=LangChain


* VectorShift + Pipelines + System Prompt = Ai Agent Chatbot

  https://youtu.be/0HxHkNT4_EU?si=qeOsaRbRC6gt-rtA


* AutoGen + Knowledge Graph + GPT-4 = Graph Chatbot

  https://youtu.be/dS06WQaxmjk?si=rh6rtw4EDDlph3xE


* AutoGen + LangChian + SQLite + Schema Function = Super SQL Chabot

  https://youtu.be/YB9M5tNAZVs?si=9NzLEB6okREdlpkL

* Microsoft PHI-2 + Huggine Face + Langchain = Super Tiny Chatbot 
        
  
  https://youtu.be/_WmH2WSuT_8?si=Jq-r8eib1G9bVjrj






# LPU

   * How does Groq LPU work? (w/ Head of Silicon Igor Arsovski!), https://www.youtube.com/watch?v=WQDMKTEgQnY&ab_channel=AleksaGordi%C4%87-TheAIEpiphany
   
   * Getting Started with Groq API | Making Near Real Time Chatting with LLMs Possible

    https://www.youtube.com/watch?v=S53BanCP14c&ab_channel=PromptEngineering

  * Groq API: Make your AI Applications Lighting Speed, https://www.youtube.com/watch?v=vKWtFVqr6Wc&t=96s&ab_channel=MervinPraison


  * Builx an Agent with Long-Term personalized memory,
    https://youtu.be/oPCKB9MUP6c?si=FGDDaDm1KuXVazhP

  * Build the fastest AI chatbot with memory using Groq, gradio, Langchain 

    https://youtu.be/a5l7E3tzsIY?si=V4Jzwu3J_ja1HsO2
    
https://github.com/InsightEdge01/GroqchatbotwithMemory/tree/main
  
  * **Fastest talking AI I could build
    deepgram + groq**
  
  
  https://youtu.be/J2sbC8X5Pp8?si=6L4sqm2izVXkDgR7
   
    https://aura-tts-demo.deepgram.com


    Code: https://github.com/gkamradt/QuickAgent





# HuggingFace

* Huggingface docs, https://huggingface.co/docs

* Hugging Face Text Generation Inference available for AWS Inferentia2

  https://huggingface.co/blog/text-generation-inference-on-inferentia2

  This tutorial shows how easy it is to deploy a state-of-the-art LLM, such as **Zephyr 7B**, on **AWS Inferentia2** using **Amazon SageMaker**. Zephyr is a 7B _fine-tuned_ version of **mistralai/Mistral-7B-v0.1** that was trained on a mix of publicly available and synthetic datasets using **Direct Preference Optimization (DPO)**, as described in detail in the technical report. The model is released under the **Apache 2.0 license**, ensuring wide accessibility and use.

      Following steps are performed:
      
      1. Setup development environment
      2. Retrieve the TGI Neuronx Image
      3 .Deploy Zephyr 7B to Amazon SageMaker
      4. Run inference and chat with the model


* Pushing Models and Adapters to HuggingFace | Free Notebook, 

  https://www.youtube.com/watch?v=Kd4JL7GnR8Y&ab_channel=TrelisResearch

  https://github.com/TrelisResearch/install-guides/blob/main/Pushing_to_Hub.ipynb

  https://awsdocs-neuron.readthedocs-hosted.com/en/latest/

  https://huggingface.co/docs/optimum-neuron/index


* Deep Dive: Hugging Face models on AWS AI Accelerators

  https://www.youtube.com/watch?v=66JUlAA8nOU&ab_channel=JulienSimon

* A guide to setting up **your own** Hugging Face **leaderboard**: an end-to-end example with Vectara's hallucination leaderboard

  https://huggingface.co/blog/leaderboards-on-the-hub-vectara


* The Hallucinations Leaderboard, an Open Effort to Measure Hallucinations in Large Language Models
  https://huggingface.co/blog/leaderboards-on-the-hub-hallucinations

* Creating open machine learning datasets? Share them on the Hugging Face Hub!
  https://huggingface.co/blog/researcher-dataset-sharing

* Deploy Embedding Models with Hugging Face Inference Endpoints
  https://huggingface.co/blog/inference-endpoints-embeddings

* Bhilding a self-corrective coding assistant from scratch
  https://youtu.be/MvNdgmM7uyc?si=b78VIhFapFo2U8NV
  




# Pipepline

   * ML pipeline with Pandas and Sklearn, https://www.youtube.com/watch?v=Zpy9npXnW00&ab_channel=RicardoCalix
   * LangChain for LLM Application Development, https://www.deeplearning.ai/short-courses/langchain-for-llm-application-development/
   * How would you build an enterprise solution for AutoML?
  ![image](https://github.com/ParthaPRay/LLM-Learning-Sources/assets/1689639/f5321166-62e1-4551-8d56-f6545d444bcf)

  Here are the different components to consider:

- Frontend client: we need to allow the user to input parameters to set up the model training and start the process. The user should be able to visualize the results of a specific run along with its related metrics. We could also provide a way to compare training runs for a better model selection process.

- A backend server: this is where the logic displayed on the frontend is implemented. It connects to a Run Metadata database that captures the different run parameters and metrics. This database should contain all the information necessary to restart identical training runs. MLFLow is an excellent example of a training runs management system.

- A message queue for training requests: Because we may have multiple users submitting training requests simultaneously, we need to buffer those requests. If we have a cap on the number of training servers we can use simultaneously, it is better to buffer requests until enough machines are available for the next requests.

- An orchestration scheduler: The orchestration system can plan the various stages and restart one in case of failure. Airflow and Kubeflow are examples of such a system. The scheduler will monitor the message queue and trigger a training pipeline once a user request is received.

- A training pipeline: The different steps are captured in a DAG and are handled by the orchestration workers.

- The Data pull module: we need to establish a logic to pull the correct data from the feature store. Once the data is pulled, it must be validated to ensure that it follows the requirements for the particular training run and is consistent with features metadata.

- The Data processing module: once the data is ready, we need, at the very least, to carve out a validation set for model performance evaluation.

- The Model selection module: this is where most of the process will be spent. That module handles the model selection process, including choosing the ML model, the hyperparameters, the model architecture, and performing the feature selection. The result of this module is a trained optimal model.

- The model validation module: after training the model, we need to capture the different validation metrics that will help the user make an educated decision about the resulting model. Beyond ML metrics, we must capture information about hardware utilization, such as memory and CPU usage. We need to send the resulting metadata to the Run Metadata database.

- The model push module: the resulting model needs to be pushed to a model registry and its version number.


* What is CI/CD/CT for machine learning

  ![image](https://github.com/ParthaPRay/LLM-Learning-Sources/assets/1689639/fd45a649-b413-484a-bdd3-eafd57a6e5ea)

   If you are working in a big tech company on ML projects, chances are you are working on some version of Continuous Integration / Continuous Deployment (CI/CD). It represents a high level of maturity in MLOps with Continuous Training (CT) at the top. This level of automation really helps ML engineers to solely focus on experimenting with new ideas while delegating repetitive tasks to engineering pipelines and minimizing human errors.

On a side note, when I was working at Meta, the level of automation was of the highest degree. That was simultaneously fascinating and quite frustrating! I had spent so many years learning how to deal with ML deployment and management that I had learned to like it. I was becoming good at it, and suddenly all that work seemed meaningless as it was abstracted away in some automation. I think this is what many people are feeling when it comes to AutoML: a simple call to a "fit" function seems to replace what took years of work and experience for some people to learn.

There are many ways to implement CI/CD/CT for Machine Learning but here is a typical process:

- The experimental phase - The ML Engineer wants to test a new idea (let's say a new feature transformation). He modifies the code base to implement the new transformation, trains a model, and validates that the new transformation indeed yields higher performance. The resulting outcome at this point is just a piece of code that needs to be included in the master repo.

- Continuous integration - The engineer then creates a Pull Request (PR) that automatically triggers unit testing (like a typical CI process) but also triggers the instantiation of the automated training pipeline to retrain the model, potentially test it through integration tests or test cases and push it to a model registry. There is a manual process for another engineer to validate the PR and performance reading of the new model. 

- Continuous deployment - Activating a deployment triggers a canary deployment to make sure the model fits in a serving pipeline and runs an A/B test experiment to test it against the production model. After satisfactory results, we can propose the new model as a replacement for the production one. 

- Continuous training - as soon as the model enters the model registry, it deteriorates and you might want to activate recurring training right away. For example, each day the model can be further fine-tuned with the new training data of the day, deployed, and the serving pipeline is rerouted to the updated model.

The Google Cloud documentation is a good read on the subject:

https://lnkd.in/g-w3hFz 

https://lnkd.in/giQrUzfq
   

* Machine Learning Engineering for Production (MLOps) 

  https://www.youtube.com/watch?v=NgWujOrCZFo&list=PLkDaE6sCZn6GMoA0wbpJLi3t34Gd8l0aK&ab_channel=DeepLearningAI
  



* Let's Learn LangChain! // Applied AI Workshops March 2024

    https://github.com/justingrammens/LetsLearnLangChain
    
    https://www.youtube.com/watch?v=QT3wALFDZBo&ab_channel=AppliedAI





# LLM Threats

* Navigating LLM Threats: Detecting Prompt Injections and Jailbreaks

 https://www.youtube.com/watch?v=kH4ZoZSvddM&ab_channel=DeepLearningAI






# Pervasive Generative AI

  * Using Ollama to Run Local LLMs on the Raspberry Pi 5, https://www.youtube.com/watch?v=ewXANEIC8pY&ab_channel=IanWootten

  * Private AI Revolution: Setting Up Ollama with WebUI on Raspberry Pi 5!, https://www.youtube.com/watch?v=jJKbYj8mIy8&ab_channel=KevinMcAleer

  * I Ran Advanced LLMs on the Raspberry Pi 5!, https://www.youtube.com/watch?v=Y2ldwg8xsgE&ab_channel=DataSlayer

  * How to Run a ChatGPT-like AI on Your Raspberry Pi, https://www.youtube.com/watch?v=idZctq7WIq4&ab_channel=GaryExplains

  * Local AI Just Got Easy (and Cheap), https://www.youtube.com/watch?v=mdOEaNV8NXw&ab_channel=DataSlayer

      Following boards are needed:
      
      1. Zima Board
      2. Coral USB TPU
      3. Coral PCie TPU
      4. M.2 Adapter
      5. Raspberry Pi 5


  * Power of Generative AI + Common-Sense of Reasoning AI = All-Pervasive Conversational Ux, https://www.youtube.com/watch?v=j1uZ1NpC_4M&ab_channel=Travellingwave

    Paper Link: www.isca-speech.org/archive/pdfs/interspeech_2023/rao23_interspeech.pdf or www.travellingwave.com/TwIS2023.pdf

  * Running SDXL on the Raspberry Pi 5 is now POSSIBLE!, https://www.youtube.com/watch?v=XVS8oiuU6sA&ab_channel=AiFlux

  * World's Easiest GPT-like Voice Assistant
    https://github.com/nickbild/local_llm_assistant?tab=readme-ov-file







# Cloud GPUs

https://fullstackdeeplearning.com/cloud-gpus/

By Sergey Karayev and Charles Frye. Updated October 30, 2023.

Discussion of this page on Hacker News [https://news.ycombinator.com/item?id=36025099] May 21, 2023.

* **GPU Cloud Server Comparison**

     - The table below does not include all possible configurations for all providers, as providers differ in their configuration strategy.
     - Most providers, including AWS, Azure, and Lambda, provide instances with pre-set configurations.
     - On GCP, any suitable machine can be connected to a configuration of GPUs.
     - On other providers, like Oblivus Cloud, Cudo Compute, and RunPod, users have precise control over the resources they request. Note that RunPod's Community Cloud, Oblivus, and Cudo are all "open clouds", meaning compute is provided by third parties.
     - For providers without pre-set instance configurations, we have selected configurations that are roughly equivalent to AWS's options. Generally, these configurations are good for workloads that require heavy inter-GPU communication.
     - Where possible, regions were set to be the west or central parts of the United States. GPU availability depends on the region.
     - Raw data can be found in a csv on GitHub, https://github.com/full-stack-deep-learning/website/blob/main/docs/cloud-gpus/cloud-gpus.csv.
     - Costs can be substantially reduced via preemption recovery and failover across clouds. If you don't want to roll your own, consider a tool like SkyPilot - https://github.com/skypilot-org/skypilot. See discussion of their launch on Hacker News - https://news.ycombinator.com/item?id=33964285, December 13, 2022.
 


* **How do I choose GPU?**

    - This page is intended to track and make explorable the current state of pricing and hardware for cloud GPUs.
   
    - If you want advice on which machines and cards are best for your use case, we recommend Tim Dettmer's blog post on GPUs for deep learning.
   
    - The whole post is a tutorial and FAQ on GPUS for DNNs, but if you just want the resulting heuristics for decision-making, see the "GPU Recommendations" section, which is the source of the chart below.

    ![image](https://github.com/ParthaPRay/LLM-Learning-Sources/assets/1689639/b108b975-5919-41bf-9660-03790ae15945)




* **GPU Raw Performance Numbers and Datasheets**

| Model | Arch   | FP32 | Mixed-precision | FP16 | Source    |
|-------|--------|------|-----------------|------|-----------|
| A100  | Ampere | 19.5 | 156             | 312  | [Datasheet](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet-us-nvidia-1758950-r4-web.pdf)|
| A10G  | Ampere | 35   | 35              | 70   | [Datasheet](https://d1.awsstatic.com/product-marketing/ec2/NVIDIA_AWS_A10G_DataSheet_FINAL_02_17_2022.pdf) |
| A6000 | Ampere | 38   | ?               | ?    | [Datasheet](https://www.nvidia.com/content/dam/en-zz/Solutions/design-visualization/quadro-product-literature/proviz-print-nvidia-rtx-a6000-datasheet-us-nvidia-1454980-r9-web%20(1).pdf) |
| V100  | Volta  | 14   | 112             | 28   | [Datasheet](https://images.nvidia.com/content/technologies/volta/pdf/tesla-volta-v100-datasheet-letter-fnl-web.pdf) |
| T4    | Turing | 8.1  | 65              | ?    | [Datasheet](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/tesla-t4/t4-tensor-core-datasheet-951643.pdf) |
| P4    | Pascal | 5.5  | N/A             | N/A  | [Datasheet](https://images.nvidia.com/content/pdf/tesla/184457-Tesla-P4-Datasheet-NV-Final-Letter-Web.pdf) |
| P100  | Pascal | 9.3  | N/A             | 18.7 | [Datasheet](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/tesla-p100/pdf/nvidia-tesla-p100-PCIe-datasheet.pdf) |
| K80   | Kepler | 8.73 | N/A             | N/A  | [Datasheet](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/tesla-product-literature/Tesla-K80-BoardSpec-07317-001-v05.pdf) |
| A40   | Ampere | 37   | 150             | 150  | [Datasheet](https://images.nvidia.com/content/Solutions/data-center/a40/nvidia-a40-datasheet.pdf) |

  




* **GPU Performance Benchmarks**
  
  Below are some basic benchmarks for GPUs on common deep learning tasks.

   ![image](https://github.com/ParthaPRay/LLM-Learning-Sources/assets/1689639/2e3c4550-f2ee-4119-802e-c1d8a10a5aa3)


  Benchmark of different GPUs on a single ImageNet epoch, by [AIME](https://www.aime.info/en/blog/deep-learning-gpu-benchmarks-2021/)

    ![image](https://github.com/ParthaPRay/LLM-Learning-Sources/assets/1689639/62dd8605-1b0d-44e8-897b-85a614b443b6)

   Benchmark of different GPUs on a mix of tasks, by [Lambda Labs](https://lambdalabs.com/gpu-benchmarks)

    






# AGI

* **OpenAI-backed "AGI ROBOT" SHOCKED The ENTIRE Industry**, https://www.youtube.com/watch?v=yauNW4C-Tfo&ab_channel=MatthewBerman



# Explainable AI

* Explainable machine learning: LIME

   ![image](https://github.com/ParthaPRay/LLM-Learning-Sources/assets/1689639/4c4ee1e4-ba2e-4283-8cda-4950a6c6e69a)

  It is so intuitive that I couldn't believe that nobody really thought about it before. Well, it is easy to be surprised after the facts! It is very reminiscent of Partial Dependence plots or ICE plots, but instead of looking at the global contributions of the different features, it provides local explanations for each prediction.

LIME (Local Interpretable Model-agnostic Explanations) looks at an ML model as a black box, and it tries to estimate the local variations of a prediction by perturbing the feature values of the specific data instance. The process is as follows:

- Choose a data instance x with the prediction y you want to explain
- Sample multiple data points around the initial data point by perturbing the values of the features
- Take those new samples and get the related inferences from our ML model
- We now have data points with features X' and predictions y' 
=> Train a simple linear model on those data points and weigh the samples by how far they are from the original data point x in the feature space (low weights for high distance and high weights for low distance).

Linear models are readily interpretable. For example, if we have

y = w_1 x_1 + w_2 x_2 + w_3 * x_3

 w_1 * x_1 is the contribution to the prediction of the feature X_1 for the specific data instance, and a high value means a high contribution. So with this linear model, we can rank and quantify in an additive manner the contributions of each feature and for each instance to the predictions, and this is what we call "explanations" for the predictions.

LIME works a bit differently for different data types:

- For tabular data, we can perturb the feature by simply adding some small noise to the continuous variables. For categorical variables, it is more delicate as the concept of distance is more subjective. Another way to do it is to choose another value of the feature from the dataset.

- For text data, the features are usually the words or the tokens. The typical way to perturb the features is to remove at random a few words from the original sentence. It is intuitive to think that if we remove an important word, the predictions should change quite a bit.

- For image data, pixels are not really representative of what "matters" in an image. "Super-pixels" are created by segmenting the image (clustering similar close pixels) and then serve as the main features. We can turn on and off those new features by zeroing their values. By turning off a few super-pixels, we effectively perturb the feature set enough to estimate which segments contribute the most to the predictions.

Here is the original paper: [“Why Should I Trust You?”
Explaining the Predictions of Any Classifier](https://arxiv.org/pdf/1602.04938.pdf), and the [Python package](https://github.com/marcotcr/lime).


* Explainable AI: SHAP


   ![image](https://github.com/ParthaPRay/LLM-Learning-Sources/assets/1689639/410bacf3-b39c-467d-b773-ada830abbcac)

   SHAP is certainly one of the most used techniques for explainable AI these days, but I think many people don't know why. Some researchers had a huge impact on the history of ML, and most people will never know about them.

SHAP (SHapley Additive exPlanations) is a framework that provides explanations of predictions as a sum of the contributions of the underlying features used in the model. We have known about the Shapley value since 1951 (https://lnkd.in/e6jBm8YD), and since then, people have tried to use them as a way to measure feature attributions in Machine Learning models, but it was not until 2017 that a team from the University of Washington proposed a unified framework to apply those in any ML models.

- Kernel SHAP is a black box method that builds on top of LIME (https://lnkd.in/gpjdUNxw). Let's say you want to explain a specific prediction p with the related features values x. The idea is to create many news samples around x by replacing some of the values with others pulled at random from the data set and to see the predictions of those new samples by the model. We can then use those samples and predictions to train a linear model and use the fitted weights to understand the local contributions of the different features. The difference between LIME and SHAP is the way the samples are weighted in the MSE loss function. LIME uses a Gaussian, whereas SHAP uses the Shapley weights.

- Tree SHAP is the exact and faster estimate of those numbers by utilizing the structure of tree-based algorithms. In a tree, we can compute the exact predictions with a subset of the features by skipping the removed features and averaging the predictions of the resulting subtrees. We understand the contribution of a feature by measuring the variation of the predictions with and without it. In 2019, the same team proposed an algorithm to explore all the feature contributions of the feature power-set at once: https://lnkd.in/gDhHeQJP. 

- Linear SHAP is the exact analytic simplification of the original formula for linear models. For a model f(x) = w_1 * x_1 + w_2 * x_2 + …, the contribution of the feature x_1 is simply 
w_1 * ( x_1 - E[x_1]).

- Deep SHAP is an application of DeepLIFT (https://lnkd.in/gtRtxhZq) using the Shapley values as a measure of contribution. DeepLIFT is a way to decompose the predictions of Neural Networks as a linear combination of contributions of the underlying features. The idea is that we can backpropagate the contributions as we do the gradient.

You can find the original SHAP papers here: https://lnkd.in/gWfEGkHt, https://lnkd.in/gDhHeQJP. SHAP is obviously, for most people, a [Python package](https://shap.readthedocs.io/en/latest/), and make sure to check it out if you haven't. 


# Responsible AI

  https://youtube.com/playlist?list=PL8P_Z6C4GcuVMxhwT9JO_nKuW0QMSJ-cZ&si=vtxnKLMZwB8SGz6y

  https://github.com/aws-samples/aws-machine-learning-university-responsible-ai/





# General ML, DL


* **How to convert any problem into a machine learning problem**

  https://www.youtube.com/watch?v=-MTW39At8F0&ab_channel=RicardoCalix

* **Intro to Reinforcement Learning through Human Feedbacks (RLHF)**
  
  https://www.youtube.com/watch?v=A8YqZKGRTAM&ab_channel=RicardoCalix

* **A Simple Generative Adversarial Network (GAN) in PyTorch**

  https://www.youtube.com/watch?v=BGtSw0XNthY&ab_channel=RicardoCalix

* **Learn More about ML and AI and Gen AI on** https://www.youtube.com/@ricardocalix188/videos

* Super VIP Cheatsheet: Deep Learning

  ![image](https://github.com/ParthaPRay/LLM-Learning-Sources/assets/1689639/1192964a-eb41-47f8-9a6b-c86f2f4b2a62)


  https://github.com/afshinea/stanford-cs-230-deep-learning/blob/master/en/super-cheatsheet-deep-learning.pdf

* **Full Stack Deep Learning Course for Free**
      
         - [FSDL 2022 (Online)](https://fullstackdeeplearning.com/course/2022/): A fully online course, taught via YouTube, Crowdcast, and Discord.
         - [FSDL 2021 (Online)](https://fullstackdeeplearning.com/spring2021/): Contemporaneous with the Berkeley course, we taught an online cohort course.
         - [FSDL 2021 (Berkeley)](https://bit.ly/berkeleyfsdl): Taught as a UC Berkeley undergrad course CS194-080 in Spring 2021
         - [FSDL 2020 (UW)](https://bit.ly/uwfsdl): Taught as University of Washington Professional Master's Program course CSEP 590C in Spring 2020
         - [FSDL 2019 (Online)](https://fall2019.fullstackdeeplearning.com/): Materials from the November 2019 bootcamp held on Berkeley campus organized in a nice online format.
         - [FSDL 2019 (Bootcamp)](https://fullstackdeeplearning.com/course/): Raw materials from the March 2019 bootcamp, held on Berkeley campus.
         - [FSDL 2018 (Bootcamp)](https://fullstackdeeplearning.com/course/): Our first bootcamp, held on Berkeley campus in August 2018

      *  **Deep Learning Fundamentals (Full Stack Deep Learning - Spring 2021)**
      
         https://www.youtube.com/watch?v=fGxWfEuUu0w&list=PL1T8fO7ArWlcWg04OgNiJy91PywMKT2lv&ab_channel=TheFullStack
      
      * **Full Stack Deep Learning - 2022**
      
        https://www.youtube.com/watch?v=-Iob-FW5jVM&list=PL1T8fO7ArWleMMI8KPJ_5D5XSlovTW_Ur&ab_channel=TheFullStack


* What is the difference between the model parameters and the model hyperparameters? 

  ![image](https://github.com/ParthaPRay/LLM-Learning-Sources/assets/1689639/98216dae-fb39-4bb1-848a-df4ca46188c3)

 What is the difference between the model parameters and the model hyperparameters? The hyperparameters are the parameters we cannot co-train with the other parameters through the statistical learning optimization used to learn from the data. So we need to alternate between learning the parameters through minimizing the loss function and tuning the hyperparameters through different optimization techniques. And that can be computationally very expensive! Neural Architecture Search is about tuning the network architecture as hyperparameters and the search space dimension can be as big as 10^40!

One technique that gave me something to think about is DARTS. Instead of tuning the architecture through typical optimization techniques like Reinforcement Learning or Bayesian optimization, we jointly learn the architecture and the model parameters through the gradient descent process. That's AutoML taken to the next level!

The idea is to first establish a SuperNet of all the possible operations you may want to evaluate within your network. For example, you may want to test different convolution strides or kernel sizes, and you may want to discover new useful ways to connect them. Typically, we fix the skeleton of the network (the number of computational blocks - for example, ResNet-50 contains 16 residual blocks), and we search within each block. You put all the operations you want to test in each of the blocks and you create all the possible connections you may want to exist between those operations. Those connections contain parameters you can learn through gradient descent and they parametrize the connection probabilities. To make sure the model generalizes well, the model parameters are learned by minimizing the loss function measured on training data batches while the architecture parameters are learned by minimizing the loss function measured on the validation dataset (as you would in typical hyperparameter optimization).

Once trained, you just keep the connections with the highest probabilities and remove the unused operations. This allows you to discover the optimal sub-network. You can then retrain from scratch using this time the sub-network.

DARTS is the seminal work on differential architecture search and has seen a lot of improvement since then. You can read more about it here: https://lnkd.in/ggwr9afT. If you are interested in learning more about Neural Architecture Search, I would advise reading this review: https://lnkd.in/geAA-c8f.


* ML model optimization


  ![image](https://github.com/ParthaPRay/LLM-Learning-Sources/assets/1689639/dea23cd1-c759-486e-bb52-a602977b7578)

  Do we need to train a model to understand how good it would be? Can't we "guess" its potential predictive power just based on its architecture or training parameters? That's the idea behind Meta-Learning: learn the patterns that make a model better than another one for some learning task!

The concepts are simple: featurize the learning meta-data, train a model to predict performance metrics with those features, and use that meta-model to search the optimization space when tuning another model.

  Featurizing the learning meta-data means that we create features from the training settings. We can capture the architecture of a network as a one-hot encoded feature vector. We can capture the different hyperparameter values and the training parameters such as the number of epochs or the hardware (CPU / GPT). We can extend the meta-feature space to the dataset used for training. For example, we can include a one-hot encoded representation of the features used and the number of samples that were used (this will allow you to perform feature selection as well). We could capture anything that could influence the learning and the resulting performance metrics. The more meta-features you include, the greater the space you will able to optimize over, but also the more difficult it will be to correctly learn the target variable.

Now that you can featurize training experiments, you can train a meta-learner to learn the relationship between the training parameters and a performance metric. Because you will most likely have very few samples, your meta-learner should be a simple model such as a linear regression or a shallow neural network.

Now that you have a model that understands the relationship between the learning meta-data and the performance metrics, you can search for the learning meta-data that maximizes the performance metric. Because you have a model, you can assess billions of different learning meta-data in seconds and converge to the optimal meta-features quickly. The typical approach is to use Reinforcement Learning or supervised fine-tuning. Fine-tuning means that if you have specific training data or if you want to focus on a subset of the search space, you can train a couple of new models on that data and get the resulting performance metrics. This will allow you to fine-tune the meta-learner to get a more optimal optimization search.

This is a good read to get started on the subject: https://lnkd.in/e9VafpST

* What happens when your Machine Learning model breaks?

  ![image](https://github.com/ParthaPRay/LLM-Learning-Sources/assets/1689639/d8237ece-b458-4595-90d6-1604533495a9)


  What happens when your Machine Learning model breaks? Imagine if the Netflix movie ranking model, the Facebook feed ranking model, or the Google search engine model suddenly stopped working. Nothing would show on those websites! Would that be an acceptable user experience?

In reality, those websites are extremely reliable! To run any of them, thousands of microservices or databases are always running in the background, and some of them are doomed to crash from time to time. In many cases, we can make the systems fault tolerant by adding redundancy.

This doesn't always work for ML pipelines! Suddenly your model can start to output unusable predictions or errors. Those predictions may be widely inaccurate or simply non-numerical values. If a prediction request fails, it may be due to some hardware failure, in which case redundancy could solve the problem. It could also be due to bugs that have been introduced due to the way a specific feature is computed, which would lead to any redundant model to fail as well. It is often important to have fallback strategies in place to handle this kind of situation. A fallback model could be a previous version of the production model, a simpler model, or a simple heuristic rule that outputs sub-optimal predictions, but predictions nonetheless. If a request fails, you can have a retry step with exception handling that reroutes the request to a fallback model.

It is quite easy to detect failures when a model throws errors or non-numerical values, but it is much harder when the model seemingly predicts meaningful values. That is why it is always important to monitor input features and model outputs. If some feature statistics start to drastically change over time, you may want to temporarily disable any model feeding on that feature and re-route requests to simpler models not using the feature, or you could simply replace the feature value with a constant while you investigate. Similarly, your prediction statistics, the model calibration, or the online model performance could start shifting, in which case you need to make sure your monitoring system automatically enables re-routing of the requests to a different model. 

Fallback mechanisms become critical in big tech companies. You may have hundreds of engineers working on separate aspects of the ML pipelines, testing different techniques to improve those pipelines. Multiple engineers may deploy a new model, a new feature, a new feature transformation, or a new optimization technique that may lead to the pipelines suddenly failing. The monitoring system may detect outlier behavior but it may take days to debug the problem, and it is often easier to revert to a previous state of the pipelines until the problem is resolved.

Reliability for ML systems can be tricky and it is important to adopt ML specific strategies to handle it!



* Machine Learning: Data Gone Wrong

  
  ![1692035126608](https://github.com/ParthaPRay/LLM-Learning-Sources/assets/1689639/22cb165a-8aae-4110-86e6-d5539baa2a83)

There definitively is no shortage of ways Data can go wrong when it comes to Machine Learning! There are no magic tricks to avoid those but there are ways to mitigate them to some degree.

- Leaky variables are when you are using information you could not have known at the time of prediction in your training data. In a sense, you are including what you are trying to predict as part of your feature set which leads to seemingly overperforming models.

- Concept drift is when the distribution of the underlying input variables remains the same but their relationships to the target variable change. That is why it is important to have periodic retraining or continuous training strategies in place.

- Feedback loops are when the current model's predictions are used to accumulate future training data. Because of it, it leads to selection bias with future models trained on data that do not represent well production data. That happens a lot in recommender engines! That can actually tend to lead to better models but it also can reinforce mistakes made by previous models.

- Stationarity is a fundamental assumption in statistical learning as we assume that samples are identically distributed. If their probability distribution evolves over time (non-stationary), the identical distribution assumption is violated. That is why it is critical to build features that are as stationary as possible. For example dollar amount is not a good feature (because of inflation), but relative dollar changes (Δ$ / $) may be better.

- Population shift is a typical problem leading to concept shift and non-stationarity. The underlying population used for the model to infer changes over time, and the original training data isn't anymore representative of the current population. Again periodic retraining is a good remedy for this problem.

- Regulatory changes are a difficult one! One day, a new data law is voted or the Apple Store changes its privacy policies making capturing a specific feature impossible. Whole companies went bankrupt because they were relying on specific data that Google Play or Apple Store allowed to capture one day, but prevented the next.

- Overfitting is obviously the most well-known one and it is fortunately the one that every ML engineer is well prepared for! This is when the model does not generalize well to test data because it captured too much of the statistical noise within the training data.

- Training data bias is when the sample distribution during training does not well represent the production data distribution, leading to biased models. It is crucial to understand how the bias will affect the inferences.

- Covariate shift is when the input feature distribution P(X) changes but not their relation to the target P(Y|X). This may lead to biases in the training data selection process that may result in inaccurate models.

  






# Metrics for Evaluation

   * All metrics on the Hugging Face Hub
  
  https://huggingface.co/metrics
  

   * MTEB 





  
# Youtube Channels

* **Mervin Praison** https://www.youtube.com/@MervinPraison
* **James Briggs** https://www.youtube.com/@jamesbriggs     **************** 
* **AI Anytime** https://www.youtube.com/@AIAnytime      ****************
* **All About AI** https://www.youtube.com/@AllAboutAI    ****************
* **Sam Witteveen** https://www.youtube.com/@samwitteveenai     ****************
* **AutoGPT Tutorials** https://www.youtube.com/@AutoGPTTutorial     ****************
* **AI Makerspace** https://www.youtube.com/@AI-Makerspace     ****************
* **AssemblyAI** https://www.youtube.com/@AssemblyAI     ****************
* **Venelin Valkov** https://www.youtube.com/@venelin_valkov  ****************
* **Trelus Research** https://youtube.com/@TrelisResearch?si=We9ORBTjY3teMpq4  ****************
* **Ricardo Calix**, https://www.youtube.com/@ricardocalix188  ****************
* **Connor Shorten** https://youtube.com/@connorshorten6311?si=YA9lHWPqWaAdOtSy ****************
* **Julien Simon** https://www.youtube.com/@juliensimonfr    ****************
* **Matthew Berman** https://www.youtube.com/@matthew_berman  ****************
* **DataScience Basics** https://youtube.com/@datasciencebasics?si=7jtQNnu2ovM0p_ge
* **Aleksa Gordić - The AI Epiphany** https://www.youtube.com/@TheAIEpiphany    ****************
  https://github.com/gordicaleksa

* **Jeff Heaton** https://youtube.com/@HeatonResearch?si=hfcA9vNxWsk05Uws    ****************
   www.heatonresearch.com
  
* **Prompt Engineering** https://www.youtube.com/@engineerprompt

* **WorldofAI** https://www.youtube.com/@intheworldofai

* **AlejzndroAO Software and AI**, https://youtube.com/@alejandro_ao?si=1TRHMqnIpQGUjJG6
* **Arize AI**  https://www.youtube.com/@arizeai/videos
* **Learn Data With Mark** https://youtube.com/@learndatawithmark?si=Sf7QWUJd6Jn2K5CR
* **SkillCurb** https://www.youtube.com/@skillcurb
* **Seth Juarez** https://www.youtube.com/@sethjuarez
* **Nicholas Renotte** https://www.youtube.com/@NicholasRenotte/
* **Mat Williams** https://youtube.com/@technovangelist?si=UiLCumC6anKxbzB-
* **Ian Wootten** https://youtube.com/@IanWootten?si=4xbHzdFIIX7n9SMS
* **AI for Devs** https://youtube.com/@ai-for-devs?si=4TrsM8CP7VBO-2a_
* **code_your_own_AI** https://www.youtube.com/@code4AI
  
* **Jeremy Howard** https://www.youtube.com/@howardjeremyp

* **Leon Explains AI** https://www.youtube.com/@leonsaiagency
* **Skill Leap AI** https://www.youtube.com/@AppOfTheDay
* **AI Flux**  https://www.youtube.com/@aifluxchannel
* **AI Jason** https://www.youtube.com/@AIJasonZ
* **Abhishek Thakur** https://www.youtube.com/@abhishekkrthakur
* **Decoder** https://youtube.com/@decoder-sh?si=OtRKUHqzVgSDT8BC
* **Fahd Mirza** https://www.youtube.com/@fahdmirza
* **Gao Dalie** https://www.youtube.com/@GaoDalie97
* **Yeyu Lab** https://www.youtube.com/@yeyulab

* **Entry Point AI** https://www.youtube.com/@EntryPointAI
* **Steve (Builder.io)** https://www.youtube.com/@Steve8708
* **Andrej Karpathy** https://youtu.be/VMj-3S1tku0?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ
* **AI Engineer** https://www.youtube.com/@aiDotEngineer
* **Kurdiez** https://www.youtube.com/@kurdiez_en
* **Whispering AI** https://www.youtube.com/@WhisperingAI/videos
* **Shaw Talebi** https://www.youtube.com/@ShawhinTalebi
* **Greg Kamradt (Data Indy)** https://www.youtube.com/@DataIndependent
* **Deci Ai** https://youtube.com/@deciai?si=udeFtVlH6uTJYMfo
* **Rob Mulla** https://www.youtube.com/@robmulla
* **Edward Hu** https://edwardjhu.com/
* **Deploying AI** https://youtube.com/@deployingai?si=pXZDOETUDdqiB_9I
* **llmware** https://www.youtube.com/@llmware/videos
* **DataInsightEdge** https://www.youtube.com/@DataInsightEdge01
* **LLmware** https://www.youtube.com/@llmware

* **AI Papers Academy**  https://www.youtube.com/@aipapersacademy
* **Predibase**
  https://youtube.com/@Predibase?si=HbdO89yPruuKJp6I
  
* **JakeEh**, https://youtube.com/@jakeeh?si=m1gSOQIkJbhPxJmt




# Prompt Engineeing

 
   ![image](https://github.com/ParthaPRay/LLM-Learning-Sources/assets/1689639/a1c1869d-231f-4b28-8ada-2b6aa36065c1)


  * Token Cost Reduction through LLMLingua's Prompt Compression, https://www.youtube.com/watch?v=xLNL6hSCPhc&ab_channel=AIAnytime
 
  * Prompting Guide, https://www.promptingguide.ai/research/rag

  * Prompt Engineering, RAG, and Fine-tuning: Benefits and When to Use, https://www.youtube.com/watch?v=YVWxbHJakgg&ab_channel=EntryPointAI
 
  * Text to Speech Tortoise versus Openvoice Comparison | How to Clone Any Voice for FREE !!, https://www.youtube.com/watch?v=E9jWEmUSxyo&ab_channel=SkillCurb
 
  * ChatGPT Vision API End to End Project with Zapier and MindStudio, https://www.youtube.com/watch?v=4UsQxuhxB7c&ab_channel=SkillCurb

    
  * Vibe-Based Prompt Engineering with PromptLayer's Jared Zoneraich, https://www.youtube.com/watch?v=SEgwj6SVWyQ&ab_channel=ArizeAI
    
  * Prompt Templates, Functions and Prompt Window Management, https://www.youtube.com/watch?v=YaYaZu6NbS0&ab_channel=ArizeAI
  
  * ChatGPT Prompt Engineering for Developers, https://www.deeplearning.ai/short-courses/chatgpt-prompt-engineering-for-developers/



# Courses and Tutorials

      * **Free Course on** (https://course.fast.ai/) by Jeremy Howard's Fastai
            
      **Practical Deep Learning:** A free course designed for people with some coding experience, who want to learn how to apply deep learning and machine learning to practical problems.

      Book PDF: https://dl.ebooksworld.ir/books/Deep.Learning.for.Coders.with.fastai.and.PyTorch.Howard.Gugger.OReilly.9781492045526.EBooksWorld.ir.pdf
   


* **Learn from Huggingface**

  https://huggingface.co/learn

    
    * Open-Source AI Cookbook: https://huggingface.co/learn/cookbook
    * NLP Course: https://huggingface.co/learn/nlp-course
    * Deep RL Course: https://huggingface.co/learn/deep-rl-course
    * Audio Course: https://huggingface.co/learn/audio-course


* **LLM University**

 LLM University by Cohere
 
 https://docs.cohere.com/docs/llmu



* **🚀 Full Stack LLM Bootcamp 🚀**

  https://fullstackdeeplearning.com/llm-bootcamp/

  https://fullstackdeeplearning.com/llm-bootcamp/spring-2023/
  

  **The Full Stack** (https://www.youtube.com/@The_Full_Stack/videos)

   https://github.com/the-full-stack/website

  **Lectures** https://www.youtube.com/watch?v=twHxmU9OxDU&list=PL1T8fO7ArWleyIqOy37OVXsP4hFXymdOZ&pp=iAQB
  
     - Learn to Spell: Prompt Engineering  https://youtu.be/JnBHR_yL2w8?list=PL1T8fO7ArWleyIqOy37OVXsP4hFXymdOZ
     - LLMOps  https://youtu.be/Fquj2u7ay40?list=PL1T8fO7ArWleyIqOy37OVXsP4hFXymdOZ
     - UX for Language User Interfaces  https://youtu.be/l5mG4z343qg?list=PL1T8fO7ArWleyIqOy37OVXsP4hFXymdOZ
     - Augmented Language Models   https://youtu.be/YdeuQhlHmCA?list=PL1T8fO7ArWleyIqOy37OVXsP4hFXymdOZ
     - Launch an LLM App in One Hour  https://youtu.be/twHxmU9OxDU?list=PL1T8fO7ArWleyIqOy37OVXsP4hFXymdOZ
     - LLM Foundations  https://youtu.be/MyFrMFab6bo?list=PL1T8fO7ArWleyIqOy37OVXsP4hFXymdOZ
     - Project Walkthrough: askFSDL  https://www.youtube.com/watch?v=pUKs4xM1r5U&list=PL1T8fO7ArWleyIqOy37OVXsP4hFXymdOZ&index=5&pp=iAQB
     - What's Next?  https://youtu.be/ax_R4yz1WwM?list=PL1T8fO7ArWleyIqOy37OVXsP4hFXymdOZ
     - UX for Language user interfaces (LLM Bootcamp)  https://www.youtube.com/watch?v=l5mG4z343qg&list=PL1T8fO7ArWleyIqOy37OVXsP4hFXymdOZ&index=6&pp=iAQB
     - Invited Talks
     - Fireside Chat with Peter Welinder  https://www.youtube.com/watch?v=54UThDl00qI&list=PL1T8fO7ArWleyIqOy37OVXsP4hFXymdOZ&index=9&pp=iAQB
     - Harrison Chase: Agents  https://www.youtube.com/watch?v=DWUdGhRrv2c&list=PL1T8fO7ArWleyIqOy37OVXsP4hFXymdOZ&index=10&pp=iAQB
     - Reza Shabani: How To Train Your Own LLM   https://www.youtube.com/watch?v=roEKOzxilq4&list=PL1T8fO7ArWleyIqOy37OVXsP4hFXymdOZ&index=11&pp=iAQB


  

* **Machind Learning University by AWS**, https://youtube.com/@machinelearninguniversity1942?si=pD5dszE0HTiOclcu

  https://github.com/aws-samples/aws-machine-learning-university-accelerated-nlp

  https://github.com/aws-samples/aws-machine-learning-university-accelerated-cv

  https://github.com/aws-samples/aws-machine-learning-university-accelerated-tab

  https://github.com/aws-samples/aws-machine-learning-university-dte

  https://github.com/aws-samples/aws-machine-learning-university-responsible-ai
  

* **PyTorch**

  Real-World PyTorch: From Zero to Hero in Deep Learning & LLMs | Tensors, Operations, Model Training

  Explore PyTorch from basics to advanced model training. Through hands-on examples, learn tensor manipulation, GPU utilization, and model optimization. Ideal for anyone eager to master deep learning with PyTorch, this video ensures you're equipped for the AI revolution.

  https://www.youtube.com/watch?v=dgs_9quxZXk&ab_channel=VenelinValkov
  
  https://github.com/curiousily/Get-Things-Done-with-Prompt-Engineering-and-LangChain


* **CS50**

This is CS50, Harvard University's introduction to the intellectual enterprises of computer science and the art of programming. Demanding, but definitely doable. Social, but educational. A focused topic, but broadly applicable skills. CS50 is the quintessential Harvard (and Yale!) course.

 https://www.youtube.com/@cs50


* **Ahead of AI magazine by Sebastian Raschka**
    https://magazine.sebastianraschka.com/archive


* **Edx:** cs50.edx.org





* **FreeCodeCamp** https://www.youtube.com/@freecodecamp

* Generative AI Full course - Gemini Pro, openAI, Llama, Langchain, Pinecone, vector databases and more, https://youtu.be/mEsleV16qdo?si=K4ZFHW2ZRG7EtL3Q

* The AiEdge

  https://www.linkedin.com/company/the-aiedge-newsletter/posts/?feedView=all


* **Create a Large Language Model from Scratch with Python – Tutorial** https://www.youtube.com/watch?v=UU1WVnMk4E8&t=24s
* **Prompt Engineering for Web Devs - ChatGPT and Bard Tutorial** https://youtu.be/ScKCy2udln8
* **Deep Learning for Computer Vision with Python and TensorFlow – Complete Course** https://youtu.be/IA3WxTTPXqQ
* **Machine Learning with Python and Scikit-Learn – Full Course** https://youtu.be/hDKCxebp88A
* **MLOps Course – Build Machine Learning Production Grade Projects**  https://youtu.be/-dJPoLm_gtE
* **code_your_own_AI** https://www.youtube.com/@code4AI
* **The Ethics of AI & Machine Learning - Full Course** https://youtu.be/qpp1G0iEL_c





* **Google**

**Google Cloud Skills Boost** https://www.cloudskillsboost.google/paths/118
Google Cloud Generative AI Learning Path

     - Introduction to Generative AI https://www.cloudskillsboost.google/course_templates/536
     - Introduction to Large Language Models https://www.cloudskillsboost.google/course_templates/539
     - Generative AI Fundamentals https://www.cloudskillsboost.google/course_templates/556
     - Encoder-Decoder Architecture  https://www.cloudskillsboost.google/course_templates/543
     - Attention Mechanism  https://www.cloudskillsboost.google/course_templates/537
     - Transformer Models and BERT Model  https://www.cloudskillsboost.google/course_templates/538
     - Generative AI Explorer - Vertex AI  https://www.cloudskillsboost.google/quests/299

* **Microsoft Resesrch Blog**
  https://www.microsoft.com/en-us/research/blog/


