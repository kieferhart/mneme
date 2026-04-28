"""CLI for Mneme.

Provides the 'mneme' command with subcommands:
- init: Initialize the database and schema
- add: Add a new memory
- link: Link two memories with a relationship
- neighbors: Show outgoing neighbors and weights
- ask: Traverse the graph with a query
- reward: Reward a relationship edge
- sessions: List past sessions
- show: Show a memory with all its edges
"""

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .db import get_connection
from .schema import init_schema
from .memory import add_memory, find_memory_by_title, link_memories, get_neighbors, show_memory, backfill_embeddings
from .traverse import find_candidate_nodes, traverse
from .learn import reward_edge
from .session import create_session, log_session_edges, list_sessions
from .creativity import suggest_new_links
from .learn import apply_citation_rewards, apply_pairwise_rewards

app = typer.Typer(help="Mneme — Local AI memory graph")
console = Console()


def _print_memory(mem: dict) -> None:
    """Pretty-print a memory node."""
    panel = Panel(
        Text.from_markup(
            f"[bold]{mem.get('title', '?')}[/bold]\n"
            f"[dim]ID: {mem.get('id', '?')}[/dim]\n"
            f"[dim]Kind: {mem.get('kind', '?')}[/dim]\n"
            f"\n{mem.get('body', '')}"
        ),
        title="Memory",
        border_style="blue",
    )
    console.print(panel)


def _print_edges(edges: list[dict], title_prefix: str) -> None:
    """Pretty-print a list of edges."""
    if not edges:
        console.print(f"[dim]No {title_prefix} edges.[/dim]")
        return

    table = Table(title=f"{title_prefix} Edges")
    table.add_column("Target", style="cyan")
    table.add_column("Type", style="green")
    table.add_column("Acc", justify="right")
    table.add_column("Creat", justify="right")
    table.add_column("Novel", justify="right")
    table.add_column("Conf", justify="right")
    table.add_column("Uses", justify="right")

    for edge in edges:
        target = edge.get("target_title", edge.get("source_title", "?"))
        rel = str(edge.get("kind", ""))
        table.add_row(
            target,
            rel,
            f"{edge.get('accuracy_weight', 0):.1f}",
            f"{edge.get('creative_weight', 0):.1f}",
            f"{edge.get('novelty_weight', 0):.1f}",
            f"{edge.get('confidence', 0):.2f}",
            str(edge.get("use_count", 0)),
        )
    console.print(table)


@app.command()
def init() -> None:
    """Initialize the Mneme database and schema."""
    conn = get_connection()
    init_schema(conn)
    console.print("[green]✓ Mneme database initialized.[/green]")
    import os
    db_path = os.environ.get("MNEME_DB", "data/mneme.kuzu")
    console.print(f"[dim]  Database: {db_path}[/dim]")


@app.command()
def add(title: str, body: str, kind: str = "note") -> None:
    """Add a new memory to the graph.

    TITLE: Human-readable title.
    BODY: Full text content.
    """
    conn = get_connection()
    result = add_memory(conn, title=title, body=body, kind=kind)
    console.print(f"[green]✓ Memory added: {result['title']}[/green]")
    console.print(f"  ID: {result['id']}")
    console.print(f"  Kind: {result['kind']}")
    console.print(f"  Summary: {result['summary']}")


@app.command()
def embed() -> None:
    """Backfill TF-IDF embeddings for all memories without them."""
    conn = get_connection()
    result = backfill_embeddings(conn)
    console.print(f"[green]✓ Embeddings backfilled:[/green]")
    console.print(f"  Total memories: {result['total']}")
    console.print(f"  Updated: {result['updated']}")


@app.command()
def link(source: str, target: str, kind: str = "relates_to",
         reason: str = "") -> None:
    """Link two memories with a relationship.

    SOURCE: Title of the source memory.
    TARGET: Title of the target memory.
    """
    conn = get_connection()

    src = find_memory_by_title(conn, source)
    tgt = find_memory_by_title(conn, target)
    if not src:
        console.print(f"[red]✗ Source memory not found: {source}[/red]")
        raise typer.Exit(code=1)
    if not tgt:
        console.print(f"[red]✗ Target memory not found: {target}[/red]")
        raise typer.Exit(code=1)

    rel_type = "relates_to" if kind == "relates_to" else "analogous_to"
    result = link_memories(conn, source, target, kind=rel_type, reason=reason)

    console.print(f"[green]✓ Link created:[/green]")
    console.print(f"  {source} --[{result['kind']}]--> {target}")
    console.print()
    _print_edges(get_neighbors(conn, source), "Outgoing")


@app.command()
def neighbors(title: str) -> None:
    """Show outgoing neighbors and their weights for a memory."""
    conn = get_connection()
    mem = find_memory_by_title(conn, title)
    if not mem:
        console.print(f"[red]✗ Memory not found: {title}[/red]")
        raise typer.Exit(code=1)

    _print_memory(mem)
    console.print()
    _print_edges(get_neighbors(conn, title), "Outgoing")


@app.command()
def ask(query: str, mode: str = "balanced", max_hops: int = 3) -> None:
    """Traverse the memory graph with a query.

    QUERY: The question or topic to explore.
    """
    conn = get_connection()

    console.print(f"[cyan]Query:[/cyan] {query}")
    console.print(f"[cyan]Mode:[/cyan] {mode}")
    console.print(f"[cyan]Max hops:[/cyan] {max_hops}")
    console.print()

    candidates = find_candidate_nodes(conn, query, top_n=5)
    if not candidates:
        console.print("[yellow]No memory nodes matched the query.[/yellow]")
        return

    console.print("[bold]Starting candidates:[/bold]")
    for i, c in enumerate(candidates, 1):
        console.print(
            f"  {i}. [cyan]{c['title']}[/cyan] "
            f"(overlap: {c['overlap_score']:.2f})",
        )
    console.print()

    best = candidates[0]
    console.print(f"[bold]Starting from:[/bold] {best['title']}")

    # Create session
    session = create_session(conn, user_query=query, mode=mode)
    session_id = session["id"]

    # Traverse
    path_result = traverse(conn, best["id"], query, mode=mode,
                           max_hops=max_hops)

    # Log USED_IN_SESSION
    log_session_edges(conn, session_id, path_result)

    # Apply citation-grounded rewards (Direction A)
    citation_result = apply_citation_rewards(
        conn, path_result["path"], path_result["edges"], query,
    )
    if citation_result.get("status") == "done":
        n_edges = citation_result.get("edges_rewarded", 0)
        console.print(
            f"[dim]Citations: {n_edges} edges rewarded[/dim]",
        )
    elif citation_result.get("status") == "skipped":
        console.print(f"[dim]{citation_result.get('reason', '')}[/dim]")

    # Trigger async pairwise comparison (Direction B)
    apply_pairwise_rewards(conn, path_result, query, mode=mode)

    # Print path
    console.print()
    console.print("[bold]Traversal Path:[/bold]")
    for i, node in enumerate(path_result["path"]):
        t = node.get("title", "?")
        b = node.get("body", "")[:120]
        console.print(f"  {i}. [cyan]{t}[/cyan]")
        if b:
            console.print(f"     {b}")

    # Print edges
    if path_result["edges"]:
        console.print()
        console.print("[bold]Edge Scores:[/bold]")
        for edge in path_result["edges"]:
            arrow = f"--[{edge['rel_type']}]-->"
            console.print(
                f"  {edge['from']} {arrow} {edge['to']}  "
                f"[green]score: {edge['score']}[/green]",
            )
            if edge.get("reason"):
                console.print(f"    Reason: {edge['reason']}")

    # Suggest new links
    if mode in ("creative", "discovery"):
        existing = [
            {"source_title": e.get("from", ""),
             "target_title": e.get("to", "")}
            for e in path_result["edges"]
        ]
        suggestions = suggest_new_links(
            path_result["path"], existing, threshold=0.03, conn=conn,
        )
        if suggestions:
            console.print()
            console.print("[bold yellow]Suggested new links:[/bold yellow]")
            for s in suggestions:
                console.print(
                    f"  [yellow]Suggested:[/yellow] "
                    f"{s['source_title']} "
                    f"--[analogous_to]--> {s['target_title']}",
                )
                console.print(f"    Reason: {s['reason']}")
                console.print(
                    f"    Initial creative_weight: {s['initial_creative_weight']}, "
                    f"confidence: {s['confidence']}",
                )

    console.print()
    console.print(f"[dim]Session ID: {session_id}[/dim]")


@app.command()
def reward(session: str, source: str, target: str,
           weight_field: str = "accuracy_weight",
           amount: float = 3.0) -> None:
    """Reward a relationship edge.

    Increases the chosen edge's weight and proportionally decreases
    sibling edges from the same source (fixed budget mechanism).
    """
    conn = get_connection()

    src = find_memory_by_title(conn, source)
    tgt = find_memory_by_title(conn, target)
    if not src:
        console.print(f"[red]✗ Source memory not found: {source}[/red]")
        raise typer.Exit(code=1)
    if not tgt:
        console.print(f"[red]✗ Target memory not found: {target}[/red]")
        raise typer.Exit(code=1)

    # Determine relationship type
    neighbors_list = get_neighbors(conn, source)
    tgt_neighbor = None
    rel_table = None
    for n in neighbors_list:
        if n["target_title"] == target:
            tgt_neighbor = n
            rt = str(n.get("kind", "")).lower()
            rel_table = "RELATES_TO" if "relates" in rt else "ANALOGOUS_TO"
            break

    if not tgt_neighbor:
        console.print(
            f"[red]✗ No edge from '{source}' to '{target}'.[/red]"
        )
        raise typer.Exit(code=1)

    if rel_table is None:
        rel_table = "RELATES_TO"

    result = reward_edge(conn, source, target, rel_table,
                         weight_field, amount)

    console.print(f"[green]✓ Edge rewarded:[/green]")
    console.print(f"  {source} --[{result['rel_type']}]--> {target}")
    console.print(f"  Weight field: {result['weight_field']}")
    console.print(f"  Amount: +{result['amount']}")
    console.print(
        f"  Before accuracy: {result['before_accuracy_weight']:.4f} "
        f"→ After: {result['after_accuracy_weight']:.4f}"
    )
    console.print(
        f"  Before creative: {result['before_creative_weight']:.4f} "
        f"→ After: {result['after_creative_weight']:.4f}"
    )
    console.print()
    _print_edges(get_neighbors(conn, source), "Outgoing")


@app.command()
def sessions() -> None:
    """List all past sessions."""
    conn = get_connection()
    rows = list_sessions(conn)

    if not rows:
        console.print("[dim]No sessions yet.[/dim]")
        return

    table = Table(title="Sessions")
    table.add_column("ID", style="cyan")
    table.add_column("Query", style="white")
    table.add_column("Mode", style="green")
    table.add_column("Date", style="dim")

    for r in rows:
        q = str(r.get("query", ""))[:50]
        table.add_row(
            r.get("id", "?"),
            q,
            r.get("mode", "?"),
            str(r.get("created_at", ""))[:19],
        )
    console.print(table)


@app.command()
def show(title: str) -> None:
    """Show a memory with all its details and edges."""
    conn = get_connection()
    result = show_memory(conn, title)
    if not result:
        console.print(f"[red]✗ Memory not found: {title}[/red]")
        raise typer.Exit(code=1)

    _print_memory(result)

    if result.get("outgoing_edges"):
        console.print()
        _print_edges(result["outgoing_edges"], "Outgoing")

    if result.get("incoming_edges"):
        console.print()
        _print_edges(result["incoming_edges"], "Incoming")


if __name__ == "__main__":
    app()
