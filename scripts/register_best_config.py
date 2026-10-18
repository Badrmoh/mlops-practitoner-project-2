from __future__ import annotations

import argparse
from pathlib import Path

import yaml
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient


DEFAULT_TRACKING_URI = "https://mlflow.badribrahim.com"
DEFAULT_MODEL_NAME = "legrag-best-chunking-config"
DEFAULT_ALIAS = "best"
PARAMS_PATH = Path("dvc/params.yaml")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Register a selected MLflow run as the best Legal RAG chunking config.",
    )
    parser.add_argument(
        "--run-id",
        required=True,
        help="MLflow run ID selected as the best run.",
    )
    parser.add_argument(
        "--tracking-uri",
        default=DEFAULT_TRACKING_URI,
        help=f"MLflow tracking URI. Default: {DEFAULT_TRACKING_URI}",
    )
    parser.add_argument(
        "--model-name",
        default=DEFAULT_MODEL_NAME,
        help=f"Registered model name. Default: {DEFAULT_MODEL_NAME}",
    )
    parser.add_argument(
        "--alias",
        default=DEFAULT_ALIAS,
        help=f"Alias to assign to the created model version. Default: {DEFAULT_ALIAS}",
    )
    return parser.parse_args()


def config_tags() -> dict[str, str]:
    if not PARAMS_PATH.exists():
        return {"config_path": str(PARAMS_PATH)}

    params = yaml.safe_load(PARAMS_PATH.read_text()) or {}
    chunk = params.get("chunk", {})
    embed = params.get("embed", {})
    eval_params = params.get("eval", {})

    return {
        "config_path": str(PARAMS_PATH),
        "chunk.max_tokens_per_chunk": str(chunk.get("max_tokens_per_chunk", "")),
        "chunk.chunk_overlap_tokens": str(chunk.get("chunk_overlap_tokens", "")),
        "embed.embedding_model": str(embed.get("embedding_model", "")),
        "embed.collection_name": str(embed.get("collection_name", "")),
        "eval.top_k": str(eval_params.get("top_k", "")),
        "eval.agent_model": str(eval_params.get("agent_model", "")),
        "eval.judge_model": str(eval_params.get("judge_model", "")),
    }


def ensure_registered_model(client: MlflowClient, model_name: str, run_id: str) -> None:
    try:
        client.create_registered_model(
            model_name,
            tags={
                "kind": "chunking-config",
                "best_run_id": run_id,
                "source": str(PARAMS_PATH),
            },
            description="Best Legal RAG chunking configuration selected by MLflow evaluation metrics.",
        )
    except MlflowException as exc:
        if "already exists" not in str(exc).lower():
            raise


def main() -> None:
    args = parse_args()
    client = MlflowClient(tracking_uri=args.tracking_uri)

    run = client.get_run(args.run_id)
    ensure_registered_model(client, args.model_name, args.run_id)

    tags = {
        "kind": "chunking-config",
        "source_run_id": args.run_id,
        **config_tags(),
    }

    version = client.create_model_version(
        name=args.model_name,
        source=run.info.artifact_uri,
        run_id=args.run_id,
        tags=tags,
        description="Best chunking config. See config/params.yaml artifact on the source run.",
    )
    client.set_registered_model_alias(args.model_name, args.alias, version.version)
    client.set_registered_model_tag(args.model_name, "best_run_id", args.run_id)

    print(f"registered_model={args.model_name}")
    print(f"version={version.version}")
    print(f"alias={args.alias}")
    print(f"source_run_id={args.run_id}")
    print(f"source={run.info.artifact_uri}")


if __name__ == "__main__":
    main()
