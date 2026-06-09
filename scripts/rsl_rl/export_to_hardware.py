# Copyright 2026 Zachary Olkin. All rights reserved.

import sys
import os
import argparse
from train_policy import ENVIRONMENTS, EXPERIMENT_NAMES

# Define locally to avoid importing train_policy.py which loads AppLauncher

def main():
    parser = argparse.ArgumentParser(description="Export the policy to be used on hardware using hugging face.")

    # Load in the hugging face repo
    parser.add_argument(
        "--hf_repo_id",
        type=str,
        default=None,
        help="Hugging Face repository ID (e.g., 'username/repo-name'). If provided, exported policy will be uploaded."
    )

    # Folder name to export the policy to
    parser.add_argument(
        "--folder_name",
        type=str,
        required=True,
        help="Name of the folder in the HuggingFace repo to upload policy.pt and policy_parameters.yaml to."
    )

    # The environment to load
    parser.add_argument(
        "--env_type",
        type=str,
        required=True,
        choices=list(ENVIRONMENTS.keys()),
        help="Type of environment to play (vanilla/custom/clf)"
    )

    parser.add_argument("--load_run", type=str, default=None, help="Name of the run folder to load from.")
    parser.add_argument("--checkpoint", type=str, default="model_.*", help="Checkpoint regex pattern to match.")
    args_cli = parser.parse_args()

    # Set the task based on environment type
    args_cli.task = ENVIRONMENTS[args_cli.env_type]

    # Upload to Hugging Face if repo_id provided
    if args_cli.hf_repo_id:
        from huggingface_hub import HfApi, repo_exists, login, whoami

        try:
            whoami()
            print("HF already logged in.")
        except Exception:
            print("Not authenticated, logging in...")
            login()

        print(f"[DEBUG] Uploading policy to Hugging Face repository: {args_cli.hf_repo_id}")
        api = HfApi()

        # Check if repository exists
        if not repo_exists(args_cli.hf_repo_id):
            print(f"[ERROR] Repository {args_cli.hf_repo_id} does not exist. Please create it first.")
            sys.exit(1)

        # Get the location of the exported file
        base_log_path = os.path.join("logs", "g1_policies", EXPERIMENT_NAMES[args_cli.env_type])
        log_root_path = os.path.join(base_log_path, args_cli.env_type)
        log_root_path = os.path.abspath(log_root_path)

        # Import AppLauncher after parsing args to avoid sys.argv consumption
        from isaaclab.app import AppLauncher
        print("[DEBUG] Launching Omniverse app")
        app_launcher = AppLauncher(headless=True)

        from isaaclab_tasks.utils import get_checkpoint_path
        import robot_rl.tasks  # noqa: F401

        checkpoint_path = get_checkpoint_path(log_root_path, args_cli.load_run, args_cli.checkpoint)
        export_model_dir = os.path.join(os.path.dirname(checkpoint_path), "exported")

        print(f"[INFO] Looking for the exported policy in {export_model_dir}")

        # Upload both policy.pt and policy_parameters.yaml
        policy_file = os.path.join(export_model_dir, "policy.pt")
        params_file = os.path.join(export_model_dir, "policy_parameters.yaml")

        # Upload files to the specified folder
        if os.path.exists(policy_file):
            repo_path = f"{args_cli.env_type}/{args_cli.folder_name}/policy.pt"
            api.upload_file(
                path_or_fileobj=policy_file,
                path_in_repo=repo_path,
                repo_id=args_cli.hf_repo_id,
                commit_message=f"Upload policy.pt to {args_cli.folder_name}"
            )
            print(f"[INFO] Uploaded {repo_path} to {args_cli.hf_repo_id}")
        else:
            print(f"[ERROR] Policy file not found: {policy_file}")
            sys.exit(1)

        if os.path.exists(params_file):
            repo_path = f"{args_cli.env_type}/{args_cli.folder_name}/policy_parameters.yaml"
            api.upload_file(
                path_or_fileobj=params_file,
                path_in_repo=repo_path,
                repo_id=args_cli.hf_repo_id,
                commit_message=f"Upload policy_parameters.yaml to {args_cli.folder_name}"
            )
            print(f"[INFO] Uploaded {repo_path} to {args_cli.hf_repo_id}")
        else:
            print(f"[ERROR] Policy parameters file not found: {params_file}")
            sys.exit(1)

        print(f"[INFO] Successfully uploaded policy to https://huggingface.co/{args_cli.hf_repo_id}/tree/main/{args_cli.env_type}/{args_cli.folder_name}")

if __name__ == "__main__":
    main()