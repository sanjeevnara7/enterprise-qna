from semantic_router import Route
from semantic_router.encoders import HuggingFaceEncoder
from semantic_router.layer import RouteLayer

from . import utterance_config

chitchat = Route(
    name="chitchat",
    utterances=utterance_config.small_talk_utterances
)

sales=Route(
    name="sales",
    utterances=utterance_config.sales_utterances
)

tickets=Route(
    name="tickets",
    utterances=utterance_config.ticket_utterances
)

invoices=Route(
    name="invoices",
    utterances=utterance_config.invoice_utterances
)

resumes=Route(
    name="resumes",
    utterances=utterance_config.resume_utterances
)

routes = [chitchat, sales, tickets, invoices, resumes]

def init_route_layer():
    encoder = HuggingFaceEncoder()
    rl = RouteLayer(encoder=encoder, routes=routes)
    return rl