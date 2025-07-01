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

### ORIGINAL GRAPH ### 
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


### TRIPLES GRAPH ###
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
    "You are a scientific knowledge extractor tasked with generating a hypergraph from a given context, identifying groups of co-dependent entities that participate together in scientific events. "
    "Your output will enable construction of a hypergraph, where each event acts as a hyperedge connecting multiple entities (nodes). "
    "You are provided with a context chunk (delimited by triple backticks: ```). "
    "Your goal is to detect scientific events and extract the set of entities that are functionally or physically linked within each event. "
    "Each event must be described with a short but descriptive phrase in a field called `id`. "
    "Each event must also list all co-occurring entities in a field called `entities`, using precise technical terms or abbreviations exactly as they appear in the text. "
    "Return a valid JSON object with a single top-level field: `events`. "
    "Each item in `events` must be a dictionary with two fields: "
    "`id` (the event description), and `entities` (a list of co-dependent entity strings). "
    "Do not include angle brackets around field names. "
    "Ensure the JSON is syntactically correct and fully compatible with standard JSON parsers."
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
    
"""

from pydantic import BaseModel
from typing import List
import json
import re
import hypernetx as hnx

# Define Pydantic schema for hypergraph
class Event(BaseModel):
    id: str
    entities: List[str]

class HypergraphJSON(BaseModel):
    events: List[Event]

def docsHypergraphPrompt(input: str, model="mistral-openorca:latest"):
    if model is None:
        model = "mistral-openorca:latest"

    SYS_PROMPT = (
        "You are a scientific knowledge extractor who builds hypergraphs by identifying co-dependent or co-occurring scientific entities in a given context. "
        "You are provided with a chunk of scientific text (delimited by triple backticks: ```). Your job is to extract events that represent meaningful scientific interactions or formulations.\n\n"

        "Thought 1: As you read each sentence, identify groups of scientific entities that co-occur or co-function in a meaningful way.\n"
        "\tEntities may include materials, chemicals, biological components, processing steps, or devices.\n\n"

        "Thought 2: For each group, write a full sentence that describes what the entities are doing together. "
        "This will be the 'id' field. Use wording from the source text whenever possible.\n\n"

        "Thought 3: Output your results as a JSON list of objects. Each object should have:\n"
        '\t- "id": a complete sentence describing the interaction\n'
        '\t- "entities": a list of strings representing co-acting terms\n\n'

        "Formatting rules:\n"
        "\t- Output must be a valid JSON list (not wrapped in a dictionary)\n"
        "\t- Do NOT include markdown formatting (no ``` or ```json)\n"
        "\t- Do NOT add a comma after the last object in the list\n"
        "\t- Do NOT use angle brackets around keys\n"
        "\t- Always use double quotes for keys and string values\n"
        "\t- Ensure all brackets [ ] and braces { } are properly closed and matched\n\n"

        "Example format:\n"
        '[\n'
        '  {\n'
        '    "id": "PDMS and carbon black combine to form a piezoresistive matrix.",\n'
        '    "entities": ["PDMS", "carbon black"]\n'
        '  },\n'
        '  {\n'
        '    "id": "Chitosan and genipin crosslink to form a hydrogel.",\n'
        '    "entities": ["Chitosan", "genipin"]\n'
        '  }\n'
        ']'
    )

    USER_PROMPT = f"context: ```{input}```\n\nExtract a list of hypergraph-style events with co-acting entities:"
    print('Generating hypergraph events...')
    response, _ = client.generate(model_name=model, system=SYS_PROMPT, prompt=USER_PROMPT)

    print("=== RAW LLM RESPONSE ===")
    print(response)

    try:
        # Clean LLM response and remove markdown formatting
        cleaned_response = response.strip()
        if cleaned_response.startswith("```json"):
            cleaned_response = cleaned_response[len("```json"):].strip()
        elif cleaned_response.startswith("```"):
            cleaned_response = cleaned_response[len("```"):].strip()
        if cleaned_response.endswith("```"):
            cleaned_response = cleaned_response[:-3].strip()

        # Remove trailing commas before closing braces/brackets
        cleaned_response = re.sub(r",\s*([}\]])", r"\1", cleaned_response)

        # Parse flat list JSON
        raw_events_list = json.loads(cleaned_response)

        # Wrap in a dictionary for Pydantic validation
        validated_result = HypergraphJSON(events=raw_events_list)

        # Build HyperNetX hypergraph
        edge_dict = {event.id: set(event.entities) for event in validated_result.events}
        H = hnx.Hypergraph(edge_dict)

        print(f" Generated hypergraph with {len(H.nodes)} nodes and {len(H.edges)} hyperedges.")
        return H

    except Exception as e:
        print("\n\n ERROR: Could not parse or validate hypergraph JSON. Here is the buggy response:\n")
        print(response)
        print("\n Exception:\n", e)
        return None


"""


from pydantic import BaseModel
from typing import List
import json
import re
import hypernetx as hnx

# Define Pydantic schema for hypergraph
class Event(BaseModel):
    id: str
    entities: List[str]

class HypergraphJSON(BaseModel):
    events: List[Event]

def docsHypergraphPrompt(input: str, model="mistral-openorca:latest"):
    if model is None:
        model = "mistral-openorca:latest"

    SYS_PROMPT = (
        "You are a scientific knowledge extractor who builds hypergraphs by identifying co-dependent or co-occurring scientific entities in a given context. "
        "You are provided with a chunk of scientific text (delimited by triple backticks: ```). Your job is to extract events that represent meaningful scientific interactions or formulations.\n\n"

        "Thought 1: As you read each sentence, identify groups of scientific entities that co-occur or co-function in a meaningful way.\n"
        "\tEntities may include materials, chemicals, biological components, processing steps, or devices.\n\n"

        "Thought 2: For each group, write a full sentence that describes what the entities are doing together. "
        "This will be the 'id' field. Use wording from the source text whenever possible.\n\n"

        "Thought 3: Output your results as a JSON list of objects. Each object should have:\n"
        '\t- "id": a complete sentence describing the interaction\n'
        '\t- "entities": a list of strings representing co-acting terms\n\n'

        "Formatting rules:\n"
        "\t- Output must be a valid JSON list (not wrapped in a dictionary)\n"
        "\t- Do NOT include markdown formatting (no ``` or ```json)\n"
        "\t- Do NOT add a comma after the last object in the list\n"
        "\t- Do NOT use angle brackets around keys\n"
        "\t- Always use double quotes for keys and string values\n"
        "\t- Ensure all brackets [ ] and braces { } are properly closed and matched\n\n"

        "Example format:\n"
        '[\n'
        '  {\n'
        '    "id": "PDMS and carbon black combine to form a piezoresistive matrix.",\n'
        '    "entities": ["PDMS", "carbon black"]\n'
        '  },\n'
        '  {\n'
        '    "id": "Chitosan and genipin crosslink to form a hydrogel.",\n'
        '    "entities": ["Chitosan", "genipin"]\n'
        '  }\n'
        ']'
    )

    USER_PROMPT = f"context: ```{input}```\n\nExtract a list of hypergraph-style events with co-acting entities:"
    print('Generating hypergraph events...')
    response, _ = client.generate(model_name=model, system=SYS_PROMPT, prompt=USER_PROMPT)

    print("=== RAW LLM RESPONSE ===")
    print(response)

    try:
        # Clean LLM response and remove markdown formatting
        cleaned_response = response.strip()
        if cleaned_response.startswith("```json"):
            cleaned_response = cleaned_response[len("```json"):].strip()
        elif cleaned_response.startswith("```"):
            cleaned_response = cleaned_response[len("```"):].strip()
        if cleaned_response.endswith("```"):
            cleaned_response = cleaned_response[:-3].strip()

        # Remove trailing commas before closing braces/brackets
        cleaned_response = re.sub(r",\s*([}\]])", r"\1", cleaned_response)

        # Parse flat list JSON
        raw_events_list = json.loads(cleaned_response)

        # Validate against schema
        validated_result = HypergraphJSON(events=raw_events_list)

        # Create new edge labels: e1, e2, ...
        edge_mapping = {f"e{i+1}": event.id for i, event in enumerate(validated_result.events)}

        # Build edge dictionary using new labels and original entity sets
        edge_dict = {
            new_label: set(event.entities)
            for new_label, event in zip(edge_mapping.keys(), validated_result.events)
        }

        # Build hypergraph
        H_simple = hnx.Hypergraph(edge_dict)

        print(f"Generated hypergraph with {len(H_simple.nodes)} nodes and {len(H_simple.edges)} relabeled hyperedges.")
        return H_simple, edge_mapping

    except Exception as e:
        print("\n\n ERROR: Could not parse or validate hypergraph JSON. Here is the buggy response:\n")
        print(response)
        print("\n Exception:\n", e)
        return None, None
