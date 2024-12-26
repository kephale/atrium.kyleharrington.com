# /// script
# title = "Hello World"
# description = "A simple example script that prints a greeting"
# author = "Kyle Harrington <atrium@kyleharrington.com>"
# license = "MIT"
# version = "0.1.0"
# keywords = ["example", "hello-world"]
# repository = "https://github.com/kephale/Kyle Harrington's atrium"
# documentation = "https://github.com/kephale/Kyle Harrington's atrium#readme"
# homepage = "https://kephale.github.io/Kyle Harrington's atrium"
# requires-python = ">=3.12"
# cover_image = "cover.png"
# dependencies = [
# "typer"
# ]
# ///

import typer
from typing import Optional

app = typer.Typer(help="A simple hello world script")

@app.command()
def greet(
    name: Optional[str] = typer.Option(None, help="Name to greet")
):
    """
    Print a greeting message
    """
    if name:
        typer.echo(f"Hello, {name}!")
    else:
        typer.echo("Hello, World!")

if __name__ == "__main__":
    app()