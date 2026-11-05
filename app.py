import os

import gradio as gr
from openai import AsyncAzureOpenAI
from agents import (
    Agent,
    Runner,
    function_tool,
    OpenAIChatCompletionsModel,
    set_tracing_disabled,
    SQLiteSession,
)
from agents.extensions.handoff_prompt import RECOMMENDED_PROMPT_PREFIX

set_tracing_disabled(True)

AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
DEPLOYMENT = "gpt-5.4-ptu"

azure_client = AsyncAzureOpenAI(
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_API_KEY,
    api_version="2024-10-21",
)
azure_model = OpenAIChatCompletionsModel(model=DEPLOYMENT, openai_client=azure_client)

PRODUCT_CATALOG = {
    "novabook air": {"price": "$1,199", "description": "A 13-inch ultralight laptop, 18-hour battery."},
    "novabook pro": {"price": "$1,899", "description": "A 16-inch laptop built for video editing."},
    "novabuds": {"price": "$149", "description": "Wireless earbuds with active noise cancellation."},
}

ORDERS = {
    "1001": {"item": "NovaBook Air", "status": "Shipped, arriving in 2 days"},
    "1042": {"item": "NovaBuds", "status": "Delayed, new estimate in 5 days"},
    "1077": {"item": "NovaBook Pro", "status": "Delivered"},
}


@function_tool
def lookup_product_tool(product_name: str) -> str:
    """Look up a product's price and description from the catalog.

    Args:
        product_name: The name of the product the customer asked about.
    """
    key = product_name.strip().lower()
    if key not in PRODUCT_CATALOG:
        return f"We could not find a product called '{product_name}' in the catalog."
    info = PRODUCT_CATALOG[key]
    return f"{product_name}: {info['price']}. {info['description']}"


@function_tool
def log_lead_tool(name: str, interest: str) -> str:
    """Record a promising sales lead, so a human sales rep can follow up later.

    Args:
        name: The customer's name, if they gave one.
        interest: What product or need they expressed interest in.
    """
    with open("leads.txt", "a", encoding="utf-8") as f:
        f.write(f"{name} - {interest}\n")
    print(f"[LEAD] {name} - {interest}")
    return "Lead logged"


@function_tool
def check_order_status_tool(order_id: str) -> str:
    """Look up the shipping status of a customer's order.

    Args:
        order_id: The order number the customer gave.
    """
    order = ORDERS.get(order_id.strip())
    if not order:
        return f"We could not find an order with number {order_id}."
    return f"Order {order_id} ({order['item']}): {order['status']}"


@function_tool
def log_support_ticket_tool(issue: str) -> str:
    """Record a support issue that needs human follow up.

    Args:
        issue: A short description of the customer's problem.
    """
    with open("support_tickets.txt", "a", encoding="utf-8") as f:
        f.write(issue + "\n")
    print(f"[TICKET] {issue}")
    return "Support ticket logged"


sales_instructions = f"""{RECOMMENDED_PROMPT_PREFIX}

You are a friendly, knowledgeable sales rep for BearCats Gadgets.

Answer product questions using lookup_product_tool. If a customer seems interested in buying, or
asks to be followed up with, use log_lead_tool to record their name and what they are interested in.

If a customer asks about an existing order, a refund, or a shipping problem, that is not your job,
say so and let the conversation be handled by Support.
"""

sales_agent = Agent(
    name="Sales Rep",
    handoff_description="Handles product questions, pricing, and sales leads.",
    instructions=sales_instructions,
    tools=[lookup_product_tool, log_lead_tool],
    model=azure_model,
)

support_instructions = f"""{RECOMMENDED_PROMPT_PREFIX}

You are a calm, patient support agent for BearCats Gadgets.

Use check_order_status_tool when a customer asks about an existing order. If something is wrong
that you cannot resolve yourself, a damaged item, a refund request, a complaint, use
log_support_ticket_tool to record it for a human to follow up on.

If a customer asks about buying a new product, or wants pricing, that is not your job, say so and
let the conversation be handled by Sales Rep.
"""

support_agent = Agent(
    name="Support",
    handoff_description="Handles order status, shipping issues, and returns.",
    instructions=support_instructions,
    tools=[check_order_status_tool, log_support_ticket_tool],
    model=azure_model,
)

sales_tool = sales_agent.as_tool(
    tool_name="consult_sales_rep",
    tool_description="Consult the sales rep for product questions, pricing, or availability.",
)

support_tool = support_agent.as_tool(
    tool_name="consult_support",
    tool_description="Consult support for order status, shipping issues, or returns.",
)

orchestrator_instructions = """You are the front desk for BearCats Gadgets.

Answer the customer yourself when you can. Consult consult_sales_rep for product or pricing
questions, and consult_support for order or shipping questions. If a single message touches both,
consult both tools and combine what they tell you into one clear reply.
"""

orchestrator = Agent(
    name="Orchestrator",
    instructions=orchestrator_instructions,
    tools=[sales_tool, support_tool],
    model=azure_model,
)

chat_session = SQLiteSession("render_visitor", "sales_team.db")


def chat(message, history):
    result = Runner.run_sync(orchestrator, message, session=chat_session)
    return result.final_output


demo = gr.ChatInterface(chat, title="BearCats Gadgets")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port)
