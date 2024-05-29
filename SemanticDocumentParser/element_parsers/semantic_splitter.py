import asyncio
import functools
from typing import List, TypedDict

from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.core.schema import TextNode, Document
from unstructured.documents.elements import Element, Title, NarrativeText


class ElementGroup(TypedDict):
    """Groups of elements split by consecutive Title objects"""

    title_node: Title
    nodes: List[Element]


def _create_element_groups(elements: List[Element]) -> List[ElementGroup]:
    """
    Create element groups between Title elements.

    Elements between Titles represent semantically different units of information, so we have a guaranteed
    semantic boundary we can exploit in chunking.

    :param elements: The element array
    :return: The grouped elements

    """

    element_groups: List[ElementGroup] = []
    current_group = None

    # Parse groups
    for element in elements:

        if isinstance(element, Title):
            if current_group:
                element_groups.append(current_group)

            current_group = ElementGroup(title_node=element, nodes=[])

        elif current_group is not None:
            current_group['nodes'].append(element)

    # Get rid of the remaining group
    if current_group is not None:
        element_groups.append(current_group)

    return element_groups


async def _semantic_split_node(
        title_node: Title,
        node: NarrativeText,
        node_parser: SemanticSplitterNodeParser
) -> List[NarrativeText]:
    """
    Run semantic splitting on each text node to subdivide bulky paragraphs into semantic units

    :param title_node: The Title the node falls under
    :param node: The node to parse
    :param node_parser: The node parser to use
    :return: The unstructured NarrativeText elements

    """

    # Note: Uses a Llama-Index Document type
    document: Document = Document(
        text=node.text
    )

    # Note: Produces Llama-Index nodes
    llama_nodes: List[TextNode] = await asyncio.to_thread(
        functools.partial(
            node_parser.build_semantic_nodes_from_documents,
            documents=[document]
        )
    )

    elements: List[NarrativeText] = []

    # Regenerative NarrativeText elements
    for llama_node in llama_nodes:
        elements.append(
            NarrativeText(
                # The title node may be important to describe the node contents
                text=title_node.text + "\n" + llama_node.text,
                metadata=node.metadata
            )
        )

    return elements


async def _semantic_split_element_group(
        group: ElementGroup,
        node_parser: SemanticSplitterNodeParser
):
    """
    Process an element group. Semantically split paragraphs into further nodes.

    :param group: The element group to process
    :return: The 1D processed node split

    """

    nodes: List[Element] = []

    for node in group['nodes']:

        # Other node types can be parsed as their own semantic units & just need to be passed on
        if not isinstance(node, NarrativeText):
            nodes.append(node)
            continue

        # Add the splits
        nodes.extend(
            await _semantic_split_node(
                group['title_node'],
                node,
                node_parser
            )
        )

    return nodes


async def semantic_splitter(
        elements: List[Element],
        node_parser: SemanticSplitterNodeParser
) -> List[Element]:
    """

    Re-distribute NarrativeTexts as chunks based on semantic similarity of adjacent texts.

    The process roughly follows:
        1. Group by title elements
        2. Run semantic splitting within each group
            i. By grouping consecutive NarrativeTexts
            ii. By running semantic splitting WITHIN these chunks
            iii. By returning a 1D array for each group that gets combined

    Edge Cases Handled:
        - Adjacent titles

    :param node_parser: The parser used to semantically split NarrativeText elements
    :param elements: All elements in the document
    :return: The new list of elements with relationships respected

    """

    # Split into groups between Title elements
    element_groups: List[ElementGroup] = _create_element_groups(elements)
    nodes: List[Element] = []

    for group in element_groups:
        # Add them to the 1D array
        nodes.extend(
            await _semantic_split_element_group(
                group,
                node_parser
            )
        )

    return elements
