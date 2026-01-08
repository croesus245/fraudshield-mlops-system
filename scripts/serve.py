"""
Serve the model.

Simple wrapper to run the FastAPI server.
"""

import argparse
import uvicorn


def main():
    parser = argparse.ArgumentParser(description="Serve fraud detection model")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host")
    parser.add_argument("--port", type=int, default=8000, help="Port")
    parser.add_argument("--reload", action="store_true", help="Auto-reload on changes")
    parser.add_argument("--workers", type=int, default=1, help="Number of workers")
    
    args = parser.parse_args()
    
    print("=" * 50)
    print("FRAUDSHIELD API SERVER")
    print("=" * 50)
    print(f"Host: {args.host}")
    print(f"Port: {args.port}")
    print(f"Workers: {args.workers}")
    print(f"Reload: {args.reload}")
    print("=" * 50)
    print(f"API docs: http://{args.host}:{args.port}/docs")
    print("=" * 50)
    
    uvicorn.run(
        "src.serving.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=args.workers if not args.reload else 1,
    )


if __name__ == "__main__":
    main()
