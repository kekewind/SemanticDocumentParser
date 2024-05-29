from typing import List

from unstructured.documents.elements import Element


def _parse_element_urls(element: Element) -> None:
    """
    Replace the URL in-text into the element. In-place modification of element.

    Known Limitation: Unstructured does not parse the hyperlinks within Table elements.

    :param element: The element to parse
    :return: None

    """

    change_delta: int = 0

    for link in element.metadata.links:
        # Deconstruct dict
        start_index: int = link['start_index'] + change_delta
        link_text: str = link['text']
        end_index: int = start_index + len(link_text)

        # Create the embedded language
        new_text = f" (The link URL is {link['url']})"
        change_delta += len(new_text)
        element.text = element.text[:start_index] + link_text + new_text + element.text[end_index:]

    # No need for those anymore!
    element.metadata.link_texts = None
    element.metadata.links = None
    element.metadata.link_urls = None


def metadata_parser(elements: List[Element]) -> None:
    """
    Extract hyperlinks and substitute them in natural language. In-place modification of array.
    Remove extra metadata fields that are unnecessary and annoying to debug with.

    :param elements: Element list
    :return: None

    """

    for element in elements:

        # Must have a link in the element
        if element.metadata.links:
            _parse_element_urls(element)

        # Category / Parent Metadata is garbage and never works
        element.metadata.parent_id = None
        element.metadata.category_depth = None

        # Other stuff we don't care about
        element.metadata.filetype = None
        element.metadata.languages = None
        element.metadata.page_number = None


__all__ = ["metadata_parser"]
