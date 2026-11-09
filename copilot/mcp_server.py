"""MCP server: the copilot's tools for any MCP client (Claude Desktop, Claude Code, IDEs).

    uv run python -m copilot.mcp_server          # stdio transport

Claude Desktop config:
    {"mcpServers": {"kanz": {"command": "uv", "args": ["--directory", "<repo>", "run",
     "python", "-m", "copilot.mcp_server"]}}}

The same Toolbox backs the HTTP agent and MCP, so behaviour (verification, grounding data,
Arabic search) is identical whichever client calls it.
"""

from __future__ import annotations

from mcp.server.mcpserver import MCPServer

from .agent.tools import Toolbox

mcp = MCPServer(
    "kanz-commerce",
    instructions="Search Kanz deals and help articles; look up orders with verification.",
)
_tb = Toolbox()


@mcp.tool()
def search_offers(query: str, max_price: int | None = None, limit: int = 3) -> dict:
    """Search Kanz deals. Arabic, Egyptian dialect, Arabizi and English queries all work."""
    return _tb.search_offers(query, max_price, limit)


@mcp.tool()
def get_offer(offer_id: str) -> dict:
    """Full details of one offer by id, e.g. KZ-0042."""
    return _tb.get_offer(offer_id)


@mcp.tool()
def search_help(query: str) -> dict:
    """Help-centre answers: refunds, redeeming, payments, bookings, account, wallet."""
    return _tb.search_help(query)


@mcp.tool()
def order_status(order_id: str, phone_last4: str) -> dict:
    """Order lookup; needs the order id AND the last 4 digits of the account phone."""
    return _tb.order_status(order_id, phone_last4)


if __name__ == "__main__":
    mcp.run()
