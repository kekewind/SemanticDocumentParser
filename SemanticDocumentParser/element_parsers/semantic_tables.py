import asyncio
import json
import logging
import re
import textwrap
import traceback
from json import JSONDecodeError
from typing import List, Awaitable, Optional, Union

from llama_index.core.base.llms.types import ChatResponse
from llama_index.core.llms import LLM
from llama_index_client import ChatMessage
from unstructured.documents.elements import Element, Table, NarrativeText, Title

SemanticUnitsTemplate: ChatMessage = ChatMessage(
    role="system",
    additional_kwargs={},
    content=textwrap.dedent(
        """
        You will be given a table. Convert the table into a valid Python list of natural language strings.
        Do NOT include HTML.
        
        Here is an example response to a table:
        
        ["Tutorial 1 with TA Donald Ipperciel is scheduled for Thursday from 4:30 PM to 5:15 PM in room VH 1018. You can join the Zoom session at https://yorku.zoom.us/j/98541900339.", "Tutorial 2 with TA Susan Cawley is scheduled for Thursday from 4:30 PM to 5:15 PM in room HNE 105. You can join the Zoom session at https://yorku.zoom.us/j/98541900339.", "Tutorial 3 with TA Susan Cawley is scheduled for Thursday from 5:30 PM to 6:15 PM in room VH 2005. You can join the Zoom session at https://yorku.zoom.us/j/98541900339."]
        
        Now you try:
        """
    )
)

SemanticSummaryTemplate: ChatMessage = ChatMessage(
    role="system",
    additional_kwargs={},
    content=textwrap.dedent(
        """
        You will be given a table.
        The header cells may be in the first row, first column, both, or neither. 
        
        First, describe the table.
        Then, determine the header cells, and use them to list the number of each item in the table.
        
        Here is an example response for a table:
        
        Tutorials are offered at different times in different locations.
        There are 3 total tutorials, 2 different TAs, 3 times to meet, 3 rooms to meet, and 3 Zoom URLs.
                
        Now you try:
        """
    )
)


def _parse_llm_json_response(response: ChatResponse) -> list[str]:
    """
    Parse the LLM's JSON response. We must make sure it replied exactly how it should.

    :param response: The LLM response
    :return: The JSON array of response strings

    """

    try:
        element_texts: list[str] = json.loads(response.message.content)
        if not isinstance(element_texts[0], str):
            raise JSONDecodeError("LLM returned invalid JSON string for Table", response.message.content, 0)
        return element_texts
    except JSONDecodeError:
        logging.error(
            "Failed to parse a table! Got invalid reply: "
            + response.message.content + "\n"
            + traceback.format_exc()
        )
        return []


async def _semantic_summarize_table(
        element: Table,
        previous_element: Optional[Union[NarrativeText, Title]],
        llm: LLM
) -> NarrativeText:
    """
    Given a Python table, semantically summarize the elements in the table using an LLM

    :param element: The table to summarize
    :param previous_element: The previous element in the table
    :param llm: The LLM used for the summary task
    :return: The summary elements

    """

    # Query the LLM using Llama-Index
    response: ChatResponse = await llm.achat(
        messages=(
            [
                SemanticSummaryTemplate,
                ChatMessage(
                    role="user",
                    content=element.metadata.text_as_html,
                    additional_kwargs={}
                )
            ]
        )
    )

    # Read the summary as JSON
    element_header: str = previous_element.text + "\n" if previous_element else ""

    # Generate the summary.
    return NarrativeText(
        text=element_header + response.message.content,
        metadata=element.metadata
    )


async def _semantic_parse_table(
        element: Table,
        previous_element: Optional[Union[NarrativeText, Title]],
        llm: LLM
) -> List[NarrativeText]:
    """
    Split a table into semantic units of information using GPT.

    :param element: The element to parse
    :param llm: The LLM used to parse the table
    :return: The parsed table as elements

    """

    # Query the LLM using Llama-Index
    response: ChatResponse = (
        await llm.achat(
            messages=(
                [
                    SemanticUnitsTemplate,
                    ChatMessage(
                        role="user",
                        content=element.metadata.text_as_html,
                        additional_kwargs={}
                    )
                ]
            )
        )
    )

    # Parse the items in the table
    element_texts: list[str] = _parse_llm_json_response(response)
    elements: List[NarrativeText] = []
    element_header: str = previous_element.text + "\n\n" if previous_element else ""

    for idx, element_text in enumerate(element_texts):
        elements.append(
            NarrativeText(
                text=element_header + f"List Item {idx + 1}: {element_text}",
                metadata=element.metadata
            )
        )

    return elements


async def _semantic_ingest_table(
        element: Table,
        previous_element: Optional[Union[NarrativeText, Title]],
        llm: LLM
) -> List[NarrativeText]:
    """
    Parse the table & create a summary of it. Also include a raw copy of the table.

    :param element: The element to parse
    :param previous_element: The previous element before the table, if it was a Title or NarrativeText
    :param llm: The LLM used to parse the table
    :return: List of NarrativeText elements generated from the table

    """

    elements: List[NarrativeText] = [
        NarrativeText(
            text=textwrap.dedent(
                re.sub('<[^<]+?>', '', element.metadata.text_as_html)
            ),
            metadata=element.metadata
        )
    ]

    tasks: List[Awaitable] = [
        _semantic_parse_table(element, previous_element, llm),
        _semantic_summarize_table(element, previous_element, llm)
    ]

    results = await asyncio.gather(*tasks)
    elements.extend(results[0])
    elements.append(results[1])

    return elements


async def semantic_tables(elements: List[Element], llm) -> List[Element]:
    """
    Semantically separate tables into natural language using an LLM

    :param elements: The elements in the table
    :param llm: The LLM to use for comprehension of the table
    :return: The list of elements parsed from the table

    """

    tasks: List[Awaitable] = []
    nodes: List[Element] = []

    # Create the comprehension tasks
    for idx, element in enumerate(elements):

        if not isinstance(element, Table):
            nodes.append(element)
            continue

        previous_element: Optional[Element] = None

        # Only include if the previous node is a TITLE or TEXT element
        if idx > 0:
            if isinstance(elements[idx - 1], NarrativeText) or isinstance(elements[idx - 1], Title):
                previous_element = elements[idx - 1]

        # Add the task
        tasks.append(
            _semantic_ingest_table(
                element, previous_element, llm
            )
        )

    # Return the list as a 1D array
    for item in await asyncio.gather(*tasks):
        if isinstance(item, list):
            nodes.extend(item)
        else:
            nodes.append(item)

    return nodes
