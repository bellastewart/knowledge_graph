import uuid
import pandas as pd
import numpy as np
from .prompts import extractConcepts
from .prompts import graphPrompt
from .prompts import docsgraphPrompt
from pydantic import BaseModel
from typing import List


def documents2Dataframe(documents) -> pd.DataFrame:
    rows = []
    for chunk in documents:
        row = {
            "text": chunk.page_content,
            **chunk.metadata,
            "chunk_id": uuid.uuid4().hex,
        }
        rows = rows + [row]

    df = pd.DataFrame(rows)
    return df


def df2ConceptsList(dataframe: pd.DataFrame) -> list:
    # dataframe.reset_index(inplace=True)
    results = dataframe.apply(
        lambda row: extractConcepts(
            row.text, {"chunk_id": row.chunk_id, "type": "concept"}
        ),
        axis=1,
    )
    # invalid json results in NaN
    results = results.dropna()
    results = results.reset_index(drop=True)

    ## Flatten the list of lists to one single list of entities.
    concept_list = np.concatenate(results).ravel().tolist()
    return concept_list


def concepts2Df(concepts_list) -> pd.DataFrame:
    ## Remove all NaN entities
    concepts_dataframe = pd.DataFrame(concepts_list).replace(" ", np.nan)
    concepts_dataframe = concepts_dataframe.dropna(subset=["entity"])
    concepts_dataframe["entity"] = concepts_dataframe["entity"].apply(
        lambda x: x.lower()
    )

    return concepts_dataframe


def df2Graph(dataframe: pd.DataFrame, model=None) -> list:
    # dataframe.reset_index(inplace=True)
    results = dataframe.apply(
        lambda row: graphPrompt(row.text, {"chunk_id": row.chunk_id}, model), axis=1
    )
    # invalid json results in NaN
    results = results.dropna()
    results = results.reset_index(drop=True)

    ## Flatten the list of lists to one single list of entities.
    concept_list = np.concatenate(results).ravel().tolist()
    return concept_list


def graph2Df(nodes_list) -> pd.DataFrame:
    ## Remove all NaN entities
    graph_dataframe = pd.DataFrame(nodes_list).replace(" ", np.nan)
    graph_dataframe = graph_dataframe.dropna(subset=["node_1", "node_2"])
    graph_dataframe["node_1"] = graph_dataframe["node_1"].apply(lambda x: x.lower())
    graph_dataframe["node_2"] = graph_dataframe["node_2"].apply(lambda x: x.lower())

    return graph_dataframe


def docs2Graph(documents: list, model=None) -> list:
    # Combine all document texts into one string
    full_text = " ".join([doc.page_content for doc in documents])

    # Optional: collect metadata if you want to keep track
    metadata = {"sources": [doc.metadata for doc in documents]}

    # Call graphPrompt on the full text
    result = docsgraphPrompt(full_text, metadata, model)

    # Handle result and flatten
    if result is None:
        return []

    concept_list = np.array(result).ravel().tolist()
    return concept_list

def docsgraph2Df(nodes_list) -> pd.DataFrame:
    # Convert to DataFrame and clean
    graph_dataframe = pd.DataFrame(nodes_list).replace(" ", np.nan)
    graph_dataframe = graph_dataframe.dropna(subset=["node_1", "node_2"])

    # Normalize text
    graph_dataframe["node_1"] = graph_dataframe["node_1"].str.lower()
    graph_dataframe["node_2"] = graph_dataframe["node_2"].str.lower()

    return graph_dataframe

#define pydantic scheme 
class Edge(BaseModel):
    source: str
    target: str
    relation: str

class Node(BaseModel):
    id: str

class GraphJSON(BaseModel):
    nodes: List[Node]
    edges: List[Edge]