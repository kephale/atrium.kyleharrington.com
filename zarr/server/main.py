# /// script
# title = "FastAPI Zarr Server"
# description = "A simple FastAPI server for sharing Zarr arrays over HTTP with CORS support"
# author = "Kyle Harrington <atrium@kyleharrington.com>"
# license = "MIT"
# version = "0.1.0"
# keywords = ["zarr", "server", "fastapi", "http", "cors", "data-sharing"]
# classifiers = [
#     "Development Status :: 4 - Beta", 
#     "Intended Audience :: Science/Research",
#     "License :: OSI Approved :: MIT License",
#     "Programming Language :: Python :: 3.10",
#     "Programming Language :: Python :: 3.11",
#     "Programming Language :: Python :: 3.12",
#     "Topic :: Scientific/Engineering :: Bio-Informatics",
#     "Topic :: Internet :: WWW/HTTP :: HTTP Servers",
#     "Topic :: System :: Networking"
# ]
# requires-python = ">=3.10"
# dependencies = [
#     "fastapi>=0.103.0",
#     "uvicorn>=0.23.0",
#     "zarr>=2.15.0,<3",
#     "numpy>=1.24.0",
#     "typer>=0.9.0"
# ]
# ///

import os
import sys
import typing
from pathlib import Path
from typing import Optional, List, Union, Dict, Any

import numpy as np
import typer
import uvicorn
import zarr
from fastapi import FastAPI, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

app = typer.Typer()
api = FastAPI(title="FastAPI Zarr Server")

# Global variables to store configuration
zarr_store = None  # Will hold the zarr Group or Array
zarr_path_prefix: str = ""
allow_write: bool = False


def get_zarr_info(z) -> Dict[str, Any]:
    """Get information about a zarr array or group."""
    # Use attribute checks instead of isinstance for more robustness
    if hasattr(z, 'shape') and hasattr(z, 'dtype'):
        # This is an Array
        return {
            "type": "array",
            "shape": z.shape,
            "chunks": z.chunks,
            "dtype": str(z.dtype),
            "compressor": str(z.compressor),
            "fill_value": str(z.fill_value),
            "order": z.order,
            "path": z.path,
            "nbytes": z.nbytes,
            "nbytes_stored": z.nbytes_stored if hasattr(z, "nbytes_stored") else None,
            "attrs": dict(z.attrs.asdict()),
        }
    elif hasattr(z, 'array_keys') and hasattr(z, 'group_keys'):
        # This is a Group
        return {
            "type": "group",
            "path": z.path,
            "attrs": dict(z.attrs.asdict()),
            "arrays": list(z.array_keys()),
            "groups": list(z.group_keys()),
        }
    else:
        return {"type": "unknown"}


@api.get("/info")
async def get_info() -> Dict[str, Any]:
    """Get information about the zarr store."""
    if zarr_store is None:
        return {"error": "No zarr store is currently loaded"}
    
    result = get_zarr_info(zarr_store)
    result["allow_write"] = allow_write
    result["zarr_path"] = zarr_path_prefix
    
    return result


@api.get("/info/{path:path}")
async def get_path_info(path: str) -> Dict[str, Any]:
    """Get information about a specific path in the zarr store."""
    if zarr_store is None:
        return {"error": "No zarr store is currently loaded"}
    
    try:
        obj = zarr_store[path]
        return get_zarr_info(obj)
    except KeyError:
        return {"error": f"Path '{path}' not found in zarr store"}
    except Exception as e:
        return {"error": str(e)}


@api.get("/{path:path}")
async def get_zarr_data(path: str, request: Request) -> Response:
    """Get zarr data at the specified path."""
    if zarr_store is None:
        return Response(status_code=status.HTTP_503_SERVICE_UNAVAILABLE)
    
    full_path = os.path.join(zarr_path_prefix, path) if zarr_path_prefix else path
    
    try:
        data = zarr_store.store[full_path]
        return Response(content=data, media_type="application/octet-stream")
    except KeyError:
        return Response(status_code=status.HTTP_404_NOT_FOUND)


@api.put("/{path:path}")
async def put_zarr_data(path: str, request: Request) -> Response:
    """Update zarr data at the specified path."""
    if zarr_store is None:
        return Response(status_code=status.HTTP_503_SERVICE_UNAVAILABLE)
    
    if not allow_write:
        return Response(
            status_code=status.HTTP_403_FORBIDDEN,
            content="Writing is not allowed in read-only mode",
        )
    
    full_path = os.path.join(zarr_path_prefix, path) if zarr_path_prefix else path
    
    try:
        blob = await request.body()
        zarr_store.store[full_path] = blob
        return Response(status_code=status.HTTP_200_OK)
    except Exception as e:
        return Response(
            content=str(e),
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )


def configure_app(
    zarr_path: str, 
    write_mode: bool = False, 
    allowed_origins: Optional[List[str]] = None
) -> FastAPI:
    """Configure the FastAPI application with a zarr store."""
    global zarr_store, zarr_path_prefix, allow_write
    
    try:
        # Set global variables
        zarr_store = zarr.open(zarr_path, mode="a" if write_mode else "r")
        zarr_path_prefix = zarr_store.path if zarr_store.path else ""
        allow_write = write_mode
        
        # Configure CORS if origins are provided
        if allowed_origins:
            # Print allowed origins
            for origin in allowed_origins:
                print(f"CORS: Allowing {origin}")
            
            api.add_middleware(
                CORSMiddleware,
                allow_origins=allowed_origins,
                allow_credentials=True,
                allow_methods=["GET", "PUT"] if write_mode else ["GET"],
                allow_headers=["*"],
            )
        
        return api
    except Exception as e:
        print(f"Error configuring app: {e}")
        raise


def run_server(app, host, port, **kwargs):
    """Run the uvicorn server directly without module reference"""
    config = uvicorn.Config(app=app, host=host, port=port, **kwargs)
    server = uvicorn.Server(config)
    server.run()


@app.command()
def serve(
    path: str = typer.Argument(..., help="Path to zarr file/directory"),
    host: str = typer.Option("127.0.0.1", help="Host to bind server to"),
    port: int = typer.Option(8000, help="Port to bind server to"),
    cors: Optional[str] = typer.Option(None, help="Origin to allow CORS from (use '*' for any)"),
    allow_write: bool = typer.Option(False, "--allow-write", "-w", help="Allow writing to zarr store"),
    reload: bool = typer.Option(False, help="Enable auto-reload for development"),
    workers: int = typer.Option(1, help="Number of worker processes")
):
    """Serve a zarr file/directory over HTTP using FastAPI."""
    print(f"Starting FastAPI Zarr Server on {host}:{port}")
    print(f"Serving zarr from: {path}")
    print(f"Write mode: {'enabled' if allow_write else 'disabled'}")
    
    # Configure CORS
    allowed_origins = None
    if cors:
        if cors == "*":
            allowed_origins = ["*"]
            print("CORS: Allowing requests from all origins")
        else:
            allowed_origins = [origin.strip() for origin in cors.split(",")]
            print(f"CORS: Allowing requests from: {', '.join(allowed_origins)}")
    
    # Configure FastAPI app
    api_app = configure_app(path, allow_write, allowed_origins)
    
    # Instead of using module reference, we pass the app directly
    run_server(
        app=api_app,
        host=host,
        port=port,
        reload=reload,
        workers=workers
    )


@app.command()
def list_info(
    path: str = typer.Argument(..., help="Path to zarr file/directory"),
):
    """List information about a zarr store."""
    z = zarr.open(path, mode="r")
    
    print(f"\nZarr Store Information: {path}")
    print("-" * 50)
    
    info = get_zarr_info(z)
    
    if info["type"] == "group":
        print(f"Type: Group")
        print(f"Path: {info['path']}")
        print(f"Arrays: {', '.join(info['arrays']) if info['arrays'] else 'None'}")
        print(f"Groups: {', '.join(info['groups']) if info['groups'] else 'None'}")
        
        # Print attributes
        if info["attrs"]:
            print("\nAttributes:")
            for key, value in info["attrs"].items():
                print(f"  {key}: {value}")
            
        # Print array examples if there are any
        if info["arrays"]:
            print("\nArray Examples:")
            for i, array_key in enumerate(info["arrays"][:3]):  # Show first 3 arrays
                array_info = get_zarr_info(z[array_key])
                print(f"  {array_key}: {array_info['shape']} {array_info['dtype']}")
                
    elif info["type"] == "array":
        print(f"Type: Array")
        print(f"Shape: {info['shape']}")
        print(f"Chunks: {info['chunks']}")
        print(f"Dtype: {info['dtype']}")
        print(f"Compressor: {info['compressor']}")
        print(f"Size: {info['nbytes']/1024/1024:.2f} MB (uncompressed)")
        if info['nbytes_stored']:
            print(f"Stored: {info['nbytes_stored']/1024/1024:.2f} MB (compressed)")
            print(f"Ratio: {info['nbytes']/info['nbytes_stored']:.2f}x")
        
        # Print attributes
        if info["attrs"]:
            print("\nAttributes:")
            for key, value in info["attrs"].items():
                print(f"  {key}: {value}")


def main():
    """CLI entry point."""
    app()


if __name__ == "__main__":
    main()
