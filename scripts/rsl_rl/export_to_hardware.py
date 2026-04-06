import sys
import os
import argparse


def main():
    """Export a trained policy to HuggingFace for hardware deployment."""
    parser = argparse.ArgumentParser(description="Export the policy to be used on hardware using hugging face.")

    parser.add_argument(
        "--hf_repo_id",
        type=str,
        required=True,
        help="Hugging Face repository ID (e.g., 'username/repo-name')."
    )
    parser.add_argument(
        "--folder_name",
        type=str,
        required=True,
        help="Folder path in the HuggingFace repo to upload policy.pt and policy_parameters.yaml to."
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        required=True,
        help="Path to the log root directory containing run folders with checkpoints."
    )
    parser.add_argument("--load_run", type=str, default=None, help="Name of the run folder to load from.")
    parser.add_argument("--checkpoint", type=str, default="model_.*", help="Checkpoint regex pattern to match.")
    args_cli = parser.parse_args()

    from huggingface_hub import HfApi, repo_exists, login, whoami

    try:
        whoami()
        print("HF already logged in.")
    except Exception:
        print("Not authenticated, logging in...")
        login()

    print(f"[INFO] Uploading policy to Hugging Face repository: {args_cli.hf_repo_id}")
    api = HfApi()

    if not repo_exists(args_cli.hf_repo_id):
        print(f"[ERROR] Repository {args_cli.hf_repo_id} does not exist. Please create it first.")
        sys.exit(1)

    log_root_path = os.path.abspath(args_cli.log_dir)

    # Import AppLauncher after parsing args to avoid sys.argv consumption
    from isaaclab.app import AppLauncher
    print("[INFO] Launching Omniverse app")
    app_launcher = AppLauncher(headless=True)

    from isaaclab_tasks.utils import get_checkpoint_path
    import robot_rl.tasks  # noqa: F401

    checkpoint_path = get_checkpoint_path(log_root_path, args_cli.load_run, args_cli.checkpoint)
    export_model_dir = os.path.join(os.path.dirname(checkpoint_path), "exported")

    print(f"[INFO] Looking for the exported policy in {export_model_dir}")

    policy_file = os.path.join(export_model_dir, "policy.pt")
    params_file = os.path.join(export_model_dir, "policy_parameters.yaml")

    for filename, local_path in [("policy.pt", policy_file), ("policy_parameters.yaml", params_file)]:
        if not os.path.exists(local_path):
            print(f"[ERROR] File not found: {local_path}")
            sys.exit(1)

        repo_path = f"{args_cli.folder_name}/{filename}"
        api.upload_file(
            path_or_fileobj=local_path,
            path_in_repo=repo_path,
            repo_id=args_cli.hf_repo_id,
            commit_message=f"Upload {filename} to {args_cli.folder_name}"
        )
        print(f"[INFO] Uploaded {repo_path} to {args_cli.hf_repo_id}")

    print(f"[INFO] Successfully uploaded policy to https://huggingface.co/{args_cli.hf_repo_id}/tree/main/{args_cli.folder_name}")

if __name__ == "__main__":
    main()