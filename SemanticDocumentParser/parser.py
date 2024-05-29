import io
import time
from typing import List, Tuple, TypedDict, Optional

from llama_index.core.llms import LLM
from llama_index.core.node_parser import SemanticSplitterNodeParser
from pydantic.v1 import BaseModel
from unstructured.documents.elements import Element
from unstructured.partition.auto import partition

from SemanticDocumentParser.element_parsers.list_parser import list_parser
from SemanticDocumentParser.element_parsers.metadata_parser import metadata_parser
from SemanticDocumentParser.element_parsers.semantic_splitter import semantic_splitter
from SemanticDocumentParser.element_parsers.semantic_tables import semantic_tables


class SemanticDocumentParserStats(TypedDict):
    element_parse_time: Optional[int]
    metadata_parse_time: Optional[int]
    paragraph_parse_time: Optional[int]
    list_parse_time: Optional[int]
    table_parse_time: Optional[int]


class SemanticDocumentParser(BaseModel):
    """
    Split nodes into semantic units

    """

    llm_model: LLM
    node_parser: SemanticSplitterNodeParser

    async def aparse(
            self,
            document: io.BytesIO
    ) -> Tuple[List[Element], SemanticDocumentParserStats]:
        """
        Asynchronously (where possible) parse the document

        :param document: The document to parse of any type unstructured supports
        :return: A list of elements existing as distinct chunks of NarrativeText

        """

        # Generate the document-agnostic array
        _1_start_time: int = int(time.time())
        elements: List[Element] = partition(file=document)
        _1_end_time: int = int(time.time())

        # If there are no elements, don't run the parsers
        if len(elements) < 1:
            stats: SemanticDocumentParserStats = {
                "element_parse_time": _1_end_time - _1_start_time,
                "metadata_parse_time": None,
                "paragraph_parse_time": None,
                "list_parse_time": None,
                "table_parse_time": None
            }

            return [], stats

        # Parse metadata
        _2_start_time: int = int(time.time())
        metadata_parser(elements)
        _2_end_time: int = int(time.time())

        # Group elements by Title separation & semantically deconstruct grouped NarrativeText elements
        _3_start_time: int = int(time.time())
        elements = await semantic_splitter(elements, self.node_parser)
        _3_end_time: int = int(time.time())

        # Group ListItem elements
        _4_start_time: int = int(time.time())
        elements = list_parser(elements)
        _4_end_time: int = int(time.time())

        _5_start_time: int = int(time.time())
        elements = await semantic_tables(elements, self.llm_model)
        _5_end_time: int = int(time.time())

        stats: SemanticDocumentParserStats = {
            "element_parse_time": _1_end_time - _1_start_time,
            "metadata_parse_time": _2_end_time - _2_start_time,
            "paragraph_parse_time": _3_end_time - _3_start_time,
            "list_parse_time": _4_end_time - _4_start_time,
            "table_parse_time": _5_end_time - _5_start_time,
        }

        return elements, stats


__all__ = ['SemanticDocumentParser']
