r"""Example 1.  # noqa D208,D415

Problem:
    Our consultancy is partnering with a CPG brand to improve their sales on Amazon.
    One important component we have identified in working with them is product reviews, so we
    are creating a dashboard where the brand can log in and understand how they are being reviewed.
    Our brands do not have a lot of time, so it would be helpful in our dashboard to summarize the
    broad themes of the reviews, and present them to the user.

Problem Distillation:
    Given a set of amazon reviews, return 5 broad positive themes from those reviews,
    and return 5 "improvement points" for the brand to focus on improving to meet customer needs.


Data:
    A sample of Amazon review data was taken from the following url and saved to `example_1_data.py`:
    https://jmcauley.ucsd.edu/data/amazon_v2/categoryFilesSmall/Grocery_and_Gourmet_Food_5.json.gz

    For more information, see: https://nijianmo.github.io/amazon/index.html

Prerequisites:
    You'll need to have an OPENAI API Key available at the environment variable: `OPENAI_API_KEY`.

"""

import functools
import json
import os
from typing import Any
from typing import TypedDict

import openai


class ReviewSummaries(TypedDict):
    """Dict for review summaries."""

    positive: list[str]
    negative: list[str]


@functools.lru_cache
def set_openai_api_key():
    """Set the OpenAI API key from the env var OPENAI_API_KEY."""
    try:
        openai.api_key = os.environ["OPENAI_API_KEY"]
    except KeyError as err:
        raise KeyError(
            "You must set the env var OPENAI_API_KEY to your openai api key."
        ) from err


def format_reviews_for_prompt(reviews: list[dict]) -> str:
    """Return a string with the reviews information for insertion into the prompt."""
    return """List of reviews:
{}
""".format(
        "\n---\nREVIEW:\n".join([review["reviewText"] for review in reviews])
    )


def openai_messages_prompt(reviews: list[dict]) -> list[dict]:
    """The prompt to send to OpenAI For a Chat Completion.

    Notes:
        See: https://platform.openai.com/docs/api-reference/chat/create#chat/create-messages
    """
    reviews_for_prompt = format_reviews_for_prompt(reviews)
    return [
        {
            "role": "user",
            "content": f"""
You are a helpful AI assistant that is designed to summarize broad themes from Amazon Reviews.

You will return data in a JSON format, with keys "positive" and "negative".
Each Key will be a list of 5 broad themes and take aways from the reviews passed to you.
The themes and take aways should apply to many reviews.
Ideally, the themes and take-aways should be actionable for the person selling the product.

Please give me the JSON results for the following reviews:
REVIEWS:
{reviews_for_prompt}
""",
        },
    ]


def read_amazon_reviews(asin: str = "B0009JQK9C") -> list[dict]:
    """Read the amazon reviews from disk for `asin`."""
    # I artificially subset example_1_data.AMAZON_REVIEWS for this asin
    assert asin == "B0009JQK9C"
    from indypy_talk_20230508.example_1_data import AMAZON_REVIEWS

    return AMAZON_REVIEWS


def get_review_summary_openai_api_response_for_asin(
    asin: str = "B0009JQK9C", model: str = "gpt-3.5-turbo"
):
    """Return the response from the OpenAI API with review summaries for `asin`.

    Args:
        asin: The asin to return a response of summaries for.
        model: The OpenAI API model to use.

    """
    reviews = read_amazon_reviews(asin)
    messages = openai_messages_prompt(reviews)
    set_openai_api_key()
    results = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        # return 3 choices for completions, in case we do not get a parsable choice
        # the tradeoff here is number of tokens used vs. reliability
        n=3,
    )

    return results


def parse_openai_api_result_choice(choice: dict) -> ReviewSummaries:
    """Parse a single "choice" from the OpenAI API repsonse.

    Notes:
        This ensures it is a valid response that can be parsed as expected.

    Raises:
        json.JSONDecodeError or AssertionError if it is not parsable.

    """
    d = json.loads(choice["message"]["content"])
    assert "positive" in d and "negative" in d
    return d


def parse_openai_api_response(response: Any) -> ReviewSummaries:
    """Parse the OpenAI API response and return a ReviewSummaries."""
    choices = response["choices"]
    # if we return many choices, take the first one that parses
    for choice in choices:
        try:
            return parse_openai_api_result_choice(choice)
        except (json.JSONDecodeError, AssertionError):
            pass
    else:
        raise ValueError("There were no parsable choices in `response`.")


def get_review_summaries(asin: str = "B0009JQK9C") -> ReviewSummaries:
    """Return dict of positive and negative summaries for Amazon reviews for `asin`."""
    response = get_review_summary_openai_api_response_for_asin(asin)
    return parse_openai_api_response(response)
