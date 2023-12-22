
import pandas as pd
from temporal_augmented_retrival import get_answer as get_temporal_answer
from rag import get_answer as get_rag_answer
from naive_rag import get_answer as get_naive_answer

path_to_csv = "contenu_embedded_august2023_1.csv"
path_to_raw = "stockerbot-export.csv"
df = pd.read_csv(path_to_csv, on_bad_lines='skip').reset_index(drop=True).drop(columns=['Unnamed: 0'])
df["embedding"] = df.embedding.apply(lambda x: eval(x)).to_list()
df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.strftime('%Y-%m-%d')


def get_benchmark(text_query, api_key):
    global df
    tempo = get_temporal_answer(text_query, df, api_key)
    rag = get_rag_answer(text_query, df, api_key)
    naive = get_naive_answer(text_query, df, api_key)
    return(tempo, rag, naive)
    
    
