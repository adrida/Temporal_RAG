# Temporal Augmented Retrieval

This repository contains the code for a temporal augmented retrieval approach as well as a gradio app that is available through a hugging face space at [LINK]
![TAR Schema](https://github.com/adrida/deltaxplainer/blob/master/assets/delta.png?raw=true) 
## Why the need for temporal augmented retrieval

The main advantage of having a temporal aspect is the ability to factor in temporal dynamics and changes of topics in the underlying data. We apply it on financial tweets in our example but the main use case would be client sales meeting notes.
The use cases are as follows:
### Detect and anticipate market trends and movements
Detecting emerging trends is one of the main reasons why we emphasize the need to include a dynamic aspect of RAG. Let's take a simple example. Say you have a high volume of temporal textual data (social media posts, client meeting notes,...). Let's consider that this data is particularly relevant because it contains market insights. 

Following this example, say we want to know the feeling around the topic "Metaverse". The query could look something like "Tell me how people are feeling about the metaverse". A traditional RAG approach will try to find documents to address directly the question. Meaning that they will look for documents individually talking about people's thoughts around the metaverse and not evaluate what each document is saying. More sophisticated RAG approaches, such as the ones implemented in langchain will go beyond finding topics matching with the query and first use a LLM to augment the query with metadata and examples of documents that might be of interest, for example, you could have the augmented query looking like this

```
augmented query:
- Original query: Tell me how people are feeling about the metaverse
- Metadata: "date: past 5 days", "only documents tagged 'innovation'"
- Examples: "Metaverse is a great opportunity", "Is Metaverse really about to change everything?"
```

This query will then be used to find relevant documents and combine them into a context fed to an LLM that will generate the final answer.

The main limitation of this approach lies in the fact that it will extract knowledge from the data in a static manner, even by having filtered out the data with the date as metadata. This means that the output will give you information on what people are saying about metaverse rather than insights on the evolution of the topic.

In our case, what is of interest is not the fact that they have been talking about Metaverse over the past 5 days but rather if they are talking more today about it compared to last month. The latter gives direct insights into the evolution of this topic and can help identify emerging trends or controversial topics. This can be used to either anticipate new trends by designing products or marketing operations to make the most of them or simply mitigate incoming bad communication (bad PR).

Temporal Augmented Retrieval is the first proposal to try to address this issue and include a dynamic aspect.

### Identify cross-selling opportunities
By understanding the client's discussions evolutions through time we can identify cross-selling opportunities. This is particularly relevant for use cases involving a vast number of clients and products. In large companies (especially financial services), having a sales rep knowing all products and partnerships offered by the company (even outside of his scope of expertise) is often impossible. Digging through client meeting notes can help uncover clients who might benefit from a product offered by another business unit. 
An example of a query could look like this: "Do we have any clients interested in eco-friendly CRMs?"

The traditional RAG method can perform this important task quite well already, but we want to make sure that Temporal Augmented Retrieval doesn't lose this ability when studying the data's dynamics.


## How does it work?
This is a breakdown of how temporal augmented retrieval works. It follows the same global RAG structure (query augmentation, metadata, combining into context). The main difference resides in the metadata part as well as the different prompts used for LLM's intermediary calls. No RAG libraries (langchain,...) nor vector db has been used, everything is implemented to work using pandas and numpy. It works with openAI API at the moment but LLM calls are distinct and easy to change directly. The same goes for semantic search operations, we use openAI embeddings and the function is also easily adaptable. A good improvement in the future will be to add more flexibility on which LLMs and embedding engines can be plugged. 

### 1) Query Augmentation

As in traditional RAG, we augment the initial query to determine two main things: relevant timestamps for a dynamic study and examples on which to perform a semantic search. We emphasize the temporal aspect by specifying in the prompt that timestamps should be determined for a temporal study. We also provide in the prompt a list of unique timestamps available, this will be later parametrized in the function.

System prompt for query augmentation (models: gpt-3.5-turbo-16k/gpt-4):
```
On [current date: 19 July], you'll receive a finance-related question from a sales manager, without direct interaction. Generate a JSON response with the following structure, considering the temporal aspect:

{
"timestamps": # Relevant timestamps to study corresponding tweets for a temporal dynamic aspect (e.g., topic drift). USE THE MINIMAL NUMBER OF TIMESTAMP POSSIBLE ALWAYS ALWAYS!,
"query": # Repeat the user's query,
"similarity_boilerplate": # Boilerplate of relevant documents for cosine similarity search after embedding (it could look like an example of tweets that might help answer the query),
}

Allowed historical timestamps:
['2018-07-18', '2018-07-19', '2018-07-08', '2018-07-09', '2018-07-10', '2018-07-11', '2018-07-12', '2018-07-13', '2018-07-14', '2018-07-15', '2018-07-16', '2018-07-17']

Ensure the output is always in JSON format and never provide any other response.
```


### 2) Meta-temporal data
Regarding metadata, we use different subfunctions to generate relevant insights on the dynamic aspect of the data. This step outputs a list for each timestamp: the number of directly relevant documents and the list of corresponding tweets (with a maximum of 10 - Also a parameter -).
The idea here is to allow for the final combination of the LLM to understand how the volume of tweets evolved as well as a sample of what those actual tweets look like.

- Number of relevant tweets
The first step is to estimate for each timestamp, how many documents directly address the query. This step has two main advantages, first, it will allow the semantic search retrieval to stop at the right number of similar documents and not include unrelated data. Secondly, it will provide a direct estimation of how many tweets are relevant and hence provide the ability to perform time-wise volume comparisons.
To find the number of relevant tweets we first start by performing a semantic search on the considered timestamp. The obtained list is sorted by similarity. We then perform a dichotomic search to find the last relevant tweet from this list. This is the `condition_check` function in the code.

To do this dichotomic search we use the following prompts (models: gpt-3.5-turbo-16k/gpt-4):

System prompt: `Only answer with True or False no matter what`
User chat prompt: 
```
Consider this tweet: [TWEET TO CHECK RELEVANCE]

Is it relevant to the following query: [AUGMENTED USER QUERY]
```

- List of relevant tweets
To find relevant tweets, we simply perform a traditional semantic like the previous step but we only output the minimum number of tweets between the number of relevant tweets and the parameter `number_of_tweets_per_timestamp`. 

This fed context aims to differentiate from traditional RAG by giving temporal information on the evolution of the sought query and steering retrieval towards a true temporal augmented retrieval. The parameters such as the number of tweets per timestamp are directly dependent on the context length accepted by the underlying LLM.

### 3) Merging into one query

The last step consists of combining all the built context into one response. To do so, we consider the original query and the context by using the following chat prompts (models: gpt-3.5-turbo-16k/gpt-4):

System prompt: 
```

        You will be fed a list of tweets each at a specific timestamp and the number of relevant tweets. You need to take into account (if needed) the number of tweets relevant to the query and how this number evolved. Your task is to use those tweets to answer to the best of your knowledge the following question:

        QUESTION: [USER ORIGINAL QUERY]

        SPECIFIC INSTRUCTIONS AND SYSTEM WARNINGS: You redact a properly structured markdown string containing a professional report.
        You ALWAYS specify your sources by citing them (no urls though). Those tweets are samples from the data and are the closest to the query, you should also take into account the volume of tweets obtained.
        Otherwise, it will be considered highly misleading and harmful content.
        You should however always try your best to answer and you need to study in depth the historical relationship between the timestamps and how they answer the QUESTION.
        You never refer to yourself.
        Make it as if a real human provided a well-constructed and structured report/answer extracting the best of the knowledge contained in the context."
```

User prompt:
```
[USER CONTEXT]
```

Based on my experiments I would advise using an LLM with a large context for this step (ideally gpt-4).

### Parameters

The parameters in our code are:

`number_of_tweets_per_timestamp`: Minimal number of tweets to include when doing retrieval for each timestamp

`MODEL_AUGMENT`: LLM used to augment the query and perform the dichotomic checks (steps 1 and 2). Currently only supports OpenAI models

`MODEL_ANSWER`: LLM used to combine all context elements into one answer (step 3). Currently only supports OpenAI models

## Contact

To reach out to me to discuss please visit adrida.github.io
If you would like to contribute, please feel free to open a Github issue
