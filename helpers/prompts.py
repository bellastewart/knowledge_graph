import sys
from yachalk import chalk
sys.path.append("..")

import json
import ollama.client as client
import json
import re
import networkx as nx
import hypernetx as hnx


from pydantic import BaseModel
from typing import List



def extractConcepts(prompt: str, metadata={}, model="mistral-openorca:latest"):
    SYS_PROMPT = (
        "[\n"
        "   {\n"
        '       "entity": The Concept,\n'
        '       "importance": The concontextual importance of the concept on a scale of 1 to 5 (5 being the highest),\n'
        '       "category": The Type of Concept,\n'
        "   }, \n"
        "{ }, \n"
        "]\n"
    )
    response, _ = client.generate(model_name=model, system=SYS_PROMPT, prompt=prompt)
    try:
        result = json.loads(response)
        result = [dict(item, **metadata) for item in result]
    except:
        print("\n\nERROR ### Here is the buggy response: ", response, "\n\n")
        result = None
    return result


def graphPrompt(input: str, metadata={}, model="mistral-openorca:latest"):
    if model == None:
        model = "mistral-openorca:latest"

    # model_info = client.show(model_name=model)
    # print( chalk.blue(model_info))

    SYS_PROMPT = (
        "You are a network graph maker who extracts terms and their relations from a given context. "
        "You are provided with a context chunk (delimited by ```) Your task is to extract the ontology "
        "of terms mentioned in the given context. These terms should represent the key concepts as per the context. \n"
        "Thought 1: While traversing through each sentence, Think about the key terms mentioned in it.\n"
            "\tTerms may include object, entity, location, organization, person, \n"
            "\tcondition, acronym, documents, service, concept, etc.\n"
            "\tTerms should be as atomistic as possible\n\n"
        "Thought 2: Think about how these terms can have one on one relation with other terms.\n"
            "\tTerms that are mentioned in the same sentence or the same paragraph are typically related to each other.\n"
            "\tTerms can be related to many other terms\n\n"
        "Thought 3: Find out the relation between each such related pair of terms. \n\n"
        "Format your output as a list of json. Each element of the list contains a pair of terms"
        "and the relation between them, like the follwing: \n"
        "[\n"
        "   {\n"
        '       "node_1": "A concept from extracted ontology",\n'
        '       "node_2": "A related concept from extracted ontology",\n'
        '       "edge": "relationship between the two concepts, node_1 and node_2 in one or two sentences"\n'
        "   }, {...}\n"
        "]"
    )

    USER_PROMPT = f"context: ```{input}``` \n\n output: "
    response, _ = client.generate(model_name=model, system=SYS_PROMPT, prompt=USER_PROMPT)
    try:
        result = json.loads(response)
        result = [dict(item, **metadata) for item in result]
    except:
        print("\n\nERROR ### Here is the buggy response: ", response, "\n\n")
        result = None
    return result


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

def graphjson_to_nx(graph_json: GraphJSON) -> nx.DiGraph:
    G = nx.DiGraph()
    for node in graph_json.nodes: #instane from class Node
        G.add_node(node.id)
    for edge in graph_json.edges:
        G.add_edge(edge.source, edge.target, relation=edge.relation)
    return G

def docsgraphPrompt(input: str, model="mistral-openorca:latest"):
    if model is None:
        model = "mistral-openorca:latest"

    SYS_PROMPT = (
        "You are a network ontology graph maker who extracts terms and their relations from a given context, using category theory. "
        "You are provided with a context chunk (delimited by triple backticks: ```). "
        "Your task is to extract the ontology of terms mentioned in the given context, representing key concepts with well-defined and widely used names of materials, systems, and methods. "
        "Always preserve technical terms or abbreviations exactly as given. "
        "Each edge must include a <relation> that reveals meaningful scientific insight about the relationship from the <source> to the <target>. "
        "Return a JSON with two fields: <nodes> and <edges>. "
        "Each node must have <id>."
        "Each edge must have <source>, <target>, and <relation>."
    )

    USER_PROMPT = f"context: ```{input}```\n\nExtract the knowledge graph in structured JSON: "
    print('Generating triples...')
    response, _ = client.generate(model_name=model, system=SYS_PROMPT, prompt=USER_PROMPT)

    print("=== RAW LLM RESPONSE ===")
    print(response)

    try:
        cleaned_response = response.strip().strip("```")
        cleaned_response = re.sub(r",\s*([}\]])", r"\1", cleaned_response)

        #print("=== CLEANED ===")
        #print(cleaned_response)

        raw_result = json.loads(cleaned_response)

        # Only convert dicts to lists if needed
        if isinstance(raw_result.get("nodes"), dict):
            raw_result["nodes"] = [{"id": k} for k in raw_result["nodes"].keys()]

        if isinstance(raw_result.get("edges"), dict):
            raw_result["edges"] = list(raw_result["edges"].values())

        # Validate using Pydantic
        validated_result = GraphJSON.model_validate(raw_result)

        # Create NetworkX graph
        G = graphjson_to_nx(validated_result)

        print(f"Generated graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
        return G

    except Exception as e:
        print("\n\nERROR ### Could not parse or validate graph JSON. Here is the buggy response:\n", response)
        print("Exception:", e, "\n\n")
        return None
    


### HYPERGRAPH ### 
###
# Define Pydantic schema for hypergraph
class Event(BaseModel):
    id: str
    entities: List[str]  # All co-dependent entities in this event

class HypergraphJSON(BaseModel):
    events: List[Event]  # Each event defines a hyperedge of entities

def docsHypergraphPrompt(input: str, model="mistral-openorca:latest"):
    if model is None:
        model = "mistral-openorca:latest"

    SYS_PROMPT = (
        "You are a hypergraph extractor identifying co-dependent entities from scientific text. "
        "Your task is to detect when multiple entities participate in a single scientific event, and group them under that event. "
        "Each group of entities that co-occur in an event should be listed under an <id> that names the event. "
        "Return a JSON with a single key: <events>, which is a list. Each item must have an <id> (the event name or description) and a list of <entities> (co-dependent terms that belong to this event). "
        "Preserve scientific terminology exactly as written."
    )

    USER_PROMPT = f"context: ```{input}```\n\nExtract the hypergraph-style JSON with co-occurring entities grouped by event: "
    print('Generating hypergraph events...')
    response, _ = client.generate(model_name=model, system=SYS_PROMPT, prompt=USER_PROMPT)

    print("=== RAW LLM RESPONSE ===")
    print(response)

    try:
        cleaned_response = response.strip().strip("```")
        cleaned_response = re.sub(r",\s*([}\]])", r"\\1", cleaned_response)

        raw_result = json.loads(cleaned_response)

        # Validate with Pydantic
        validated_result = HypergraphJSON.model_validate(raw_result)

        # Convert to HyperNetX format
        edge_dict = {event.id: set(event.entities) for event in validated_result.events}
        H = hnx.Hypergraph(edge_dict)

        print(f"Generated hypergraph with {len(H.nodes)} nodes and {len(H.edges)} hyperedges.")
        return H

    except Exception as e:
        print("\n\nERROR ### Could not parse or validate hypergraph JSON. Here is the buggy response:\n", response)
        print("Exception:", e, "\n\n")
        return None