from typing import List, Optional, Generator

from unstructured.documents.elements import Element, ListItem, NarrativeText, Title, PageBreak


def _list_group_parser(elements: List[ListItem], header_node: Optional[NarrativeText]) -> List[NarrativeText]:
    """
    Deconstruct a list group into semantic units that relate to the previous node

    :param elements: The elements in the group
    :param header_node: The node just prior to the ListItem list
    :return: The deconstructed list

    """

    nodes: List[NarrativeText] = []
    header_text: str = header_node.text + "\n\n" if header_node else ""
    footer_text: str = f" There were {len(elements)} items."

    # Create an overall node with everything in it
    nodes.append(
        NarrativeText(
            text=header_text + "\n".join(["- " + element.text for element in elements]) + footer_text,
        )
    )

    # Create NarrativeText elements from each ListItem
    for idx, element in enumerate(elements):
        nodes.append(
            NarrativeText(
                text=header_text + f"List Item #{idx + 1}): " + element.text,
                metadata=element.metadata,
            )
        )

    return nodes


def _iterate_without_page_breaks(elements: List[Element]) -> Generator[Element, None, None]:
    """
    Remove page breaks using a cheeky generator method. Necessary for list parser across multiple pages.

    :param elements: The elements to iterate through
    :return: None

    """

    for element in elements:
        if isinstance(element, PageBreak):
            continue
        yield element


def list_parser(elements: List[Element]) -> List[Element]:
    """
    Each item in a list is its own semantic unit of information.

    Lists should be represented in their TOTAL form, but also with individual items.

    :param elements: All elements of a document
    :return: Elements with lists nodes enhanced properly and converted to NarrativeText

    """

    nodes: List[Element] = []

    header_node: Optional[Element] = None

    list_group: List[ListItem] = []
    last_node: Optional[Element] = None

    for element in _iterate_without_page_breaks(elements):

        # If it's a list item then add it to the current group
        if isinstance(element, ListItem):
            list_group.append(element)

            # If the last node was text & now it's a list, set the header node
            if isinstance(last_node, NarrativeText) or isinstance(last_node, Title):
                header_node = last_node

        else:
            # If not a list item just add it directly
            nodes.append(element)

            # If the last node was a list node & now it isn't, run the parser
            if isinstance(last_node, ListItem):
                nodes.extend(_list_group_parser(list_group, header_node))
                list_group = []
                header_node = None

        last_node = element

    return nodes
