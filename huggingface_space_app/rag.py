import os
import openai
import time
import numpy as np
import time
import pandas as pd

GPT_MODEL_AUGMENT = "gpt-3.5-turbo-16k"
GPT_MODEL_ANSWER = "gpt-3.5-turbo-16k"


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def get_embedding(text, model="text-embedding-ada-002"):
    try:
        text = text.replace("\n", " ")
    except:
        None
    try:
        return openai.Embedding.create(input = [text], model=model)['data'][0]['embedding']
    except:
        time.sleep(2)

def augment_query(query):

    SYS_PROMPT = """
        On [current date: 19 July] Generate a JSON response with the following structure:

        {
        "timestamps": # Relevant timestamps in which to get data to answer the query,
        "query": # Repeat the user's query,
        }
        Allowed timestamps:
        ['2018-07-18', '2018-07-19', '2018-07-08', '2018-07-09', '2018-07-10', '2018-07-11', '2018-07-12', '2018-07-13', '2018-07-14', '2018-07-15', '2018-07-16', '2018-07-17']

        Ensure the output is always in JSON format and never provide any other response.
        """
    response = openai.chat.completions.create(
                                            model=GPT_MODEL_AUGMENT,
                                            messages=
                                            [
                                                {
                                                "role": "system",
                                                "content": SYS_PROMPT
                                                },
                                                {
                                                "role": "user",
                                                "content": query
                                                }
                                            ],
                                            temperature=1,
                                            max_tokens=1000,
                                            top_p=1,
                                            frequency_penalty=0,
                                            presence_penalty=0,
                                            ).choices[0].message.content
    return response

def semantic_search(df_loc, query,timestamp, nb_programs_to_display=15):
    timestamp = str(timestamp).strip()
    embedding = get_embedding(query, model='text-embedding-ada-002')
    filtered_df = df_loc[df_loc["timestamp"]==timestamp].drop(columns=["url"])
    def wrap_cos(x,y):
        try:
            res = cosine_similarity(x,y)
        except:
            res = 0
        return res
    filtered_df['similarity']  = filtered_df.embedding.apply(lambda x: wrap_cos(x, embedding))

    results = filtered_df.sort_values('similarity', ascending=False).head(nb_programs_to_display)
    return results

def get_relevant_documents(df, query, nb_programs_to_display=15):

    query = eval(query)
    all_retrieved = []
    for timestamp in query["timestamps"]:
        all_retrieved.append({
            "timestamp" : timestamp,
            "tweets" : semantic_search(df, query["query"],timestamp, nb_programs_to_display=nb_programs_to_display)
        })

    return all_retrieved

def get_final_answer(relevant_documents, query):
    context = ""
    for relevant_timestamp in relevant_documents: 
        list_tweets = relevant_timestamp["tweets"]
        context += "\nTimestamp: " + relevant_timestamp["timestamp"] + "\nList of tweets:\n" + str((list_tweets["text"] + "   --- Tweeted by: @" +list_tweets["source"] +  " \n").to_list()) + "\n---"


    USER_PROMPT = f"""
    "We have provided context information below. 
    ---------------------
    {context}
    "\n---------------------\n"
    Given this information, please answer the question: {query}
    """
    response = openai.chat.completions.create(
                                                model=GPT_MODEL_ANSWER,
                                                messages=[
                                                    {
                                                    "role": "user",
                                                    "content": USER_PROMPT
                                                    }
                                                ],

                                                temperature=1,
                                                max_tokens=1000,
                                                top_p=1,
                                                frequency_penalty=0,
                                                presence_penalty=0,
                                                ).choices[0].message.content
    return response

def get_answer(query, df, api_key):
    """This traditional RAG approach has been implemented without using deidcated libraries and include different steps.
    It starts by augmenting the query and then perform a semantic search on the augmented query. Finally it combines the augmented query and the retrieved documents into an answer. 

    Args:
        query (String): Query given by the user
        df (pd.DataFrame()): corpus with embeddings
        api_key (String): OpenAI API key

    Returns:
        String: Answer to the original query
    """
    openai.api_key = api_key
    augmented_query = augment_query(query)
    relevant_documents = get_relevant_documents(df, augmented_query,nb_programs_to_display=10)
    response = get_final_answer(relevant_documents, augmented_query,)
    return response
