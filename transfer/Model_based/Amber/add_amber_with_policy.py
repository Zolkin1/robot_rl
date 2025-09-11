# add_amber_main.py
import argparse
from pathlib import Path
import yaml
import numpy as np
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(
    description="Add Amber to Isaac Lab, but drive it with a trained policy."
)
parser.add_argument(
    "--use_casadi_ik", action="store_true",
    help="Use CasADi IK instead of Isaac Differential IK (default = False)"
)
parser.add_argument(
    "--config_file", type=Path, required=True,
    help="YAML with keys: checkpoint_path, dt, num_obs, num_action, period, "
            "action_scale, default_angles, qvel_scale, ang_vel_scale, command_scale",
)
parser.add_argument(
    "--csv_out", type=Path, default=Path("transfer/Model_based/amber_joint_log3D.csv"),
    help="Where to write joint‐position logs",
)
parser.add_argument(
    "--desired_vel", type=float, nargs=3, default=[-1.0, 0.0, 0.0],
    help="Desired base command [vx, vy, vyaw]"
)
parser.add_argument(
    "--num_envs", type=int, default=1,
    help="Number of environments to spawn (policy will be applied to env 0 only)."
)
parser.add_argument(
    "--policy", type=int, default=0,nargs="?", const=1,
    help="Just load up the policy"
)
parser.add_argument(
    "--lip", type=int, default=0,nargs="?", const=1,
    help="Just load up the policy"
)
parser.add_argument("--video", action="store_true", default=False,
                    help="Save a viewport MP4.")
parser.add_argument("--video_length", type=int, default=2000,
                    help="Number of physics steps to record.")


def str2bool(v):
    """
    Accepts several spellings of a boolean value.
    """
    if isinstance(v, bool):
        return v
    v = v.lower()
    if v in ("yes", "true", "t", "1"):
        return True
    if v in ("no", "false", "f", "0"):
        return False
    raise argparse.ArgumentTypeError("boolean value expected")


parser.add_argument(
    "--debug",
    nargs="?",
    const=True,            # `--debug`   → True
    default=False,         # omitted     → False
    type=str2bool,         # `--debug t` or `--debug false`
    help="Enable verbose debug output (default: False)",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# ─── launch Kit & IsaacLab ───
app_launcher   = AppLauncher(args_cli)
simulation_app = app_launcher.app
import torch
import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene
if args_cli.video:
    args_cli.enable_cameras = True
    from omni.kit.viewport.utility import get_active_viewport, capture_viewport_to_file  # :contentReference[oaicite:0]{index=0}
if args_cli.lip == 1:
    from transfer.Model_based.Amber.amber_rl_wrapper_lip import RLPolicy
    from transfer.Model_based.Amber.amber_utils_lip import run_simulator

else:
    from transfer.Model_based.Amber.amber_rl_wrapper import RLPolicy
    # print("yes")
    from transfer.Model_based.Amber.amber_utils import run_simulator

from transfer.Model_based.Amber.amber_cfg import NewRobotsSceneCfg
# from transfer.Model_based.Amber.amber_utils import run_simulator
# from transfer.Model_based.Amber.amber_utils_policy import run_simulator
# from transfer.Model_based.Amber.amber_utils_lip_at_policy_rate import run_simulator
def _start_video_capture(filename: str, fps: float):
    """
    Begins recording the active viewport to <filename>.mp4  at <fps> frames/s.
    """
    import omni.kit.viewport.utility as vp
    from omni.kit.capture import ImageCapture

    vp.get_active_viewport().set_resolution_policy("RESIZE_FILL")
    ImageCapture.start_capture(output_path=filename, framerate=int(round(fps)))

def _stop_video_capture():
    """Stops an ongoing ImageCapture session (if any)."""
    from omni.kit.capture import ImageCapture

    try:
        ImageCapture.stop_capture()
    except RuntimeError:
        # already stopped / never started
        pass

def main():
    
    # now that Kit is up, import other IsaacLab bits
    from isaaclab.scene import InteractiveScene

    # ─── build simulation & scene ───
    sim_cfg = sim_utils.SimulationCfg(
        dt=0.005, render_interval=5, device=args_cli.device
    )
    sim   = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view([-9.5/3, 3.2/3, 1.3], [0.0, 0.0, 1])

    scene = InteractiveScene(NewRobotsSceneCfg(args_cli.num_envs, env_spacing=4.0))
    sim.reset(); scene.reset()
    print("[INFO] Isaac & scene initialized.")

    # ─── Load your policy ───
    with open(args_cli.config_file) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    # print("num action:",cfg["num_action"])
    policy = RLPolicy(
        dt=cfg["dt"],
        checkpoint_path=str(cfg["checkpoint_path"]),
        num_obs=cfg["num_obs"],
        num_action=cfg["num_action"],
        cmd_scale=cfg["command_scale"],
        period=cfg["period"],
        action_scale=cfg["action_scale"],
        default_angles=np.array(cfg["default_angles"], dtype=np.float32),
        qvel_scale=cfg["qvel_scale"],
        ang_vel_scale=cfg["ang_vel_scale"],
    )
    print(f"[INFO] Loaded policy from {cfg['checkpoint_path']}")
    # 1) Show the actual JIT module
    print("[DEBUG] JIT policy module:", policy.policy)
    # 2) List its parameters / state_dict keys to confirm non-empty weights
    try:
        sd = policy.policy.state_dict()
        print("[DEBUG] Policy state_dict keys (first 10):", list(sd.keys())[:10])
    except Exception as e:
        print("[DEBUG] Can't get state_dict:", e)

    # (optional) quick policy sanity check omitted for brevity…
    # 3) Run a dummy forward with zeros to see if it spits out anything non-zero
    dummy_obs = torch.zeros(1, policy.num_obs)
    if torch.cuda.is_available():
        dummy_obs = dummy_obs.cuda()
    with torch.inference_mode():
        out = policy.policy(dummy_obs)
    print("[DEBUG] policy(dummy_obs) →", out.detach().cpu().numpy().squeeze())
    # ─── run! ───
    if args_cli.video:
        vp = get_active_viewport()
        if vp is None:
            raise RuntimeError("Could not find an active viewport for recording!")
        frame_dir = Path("videos/frames")
        frame_dir.mkdir(parents=True, exist_ok=True)
        max_frames = args_cli.video_length
        print(f"[INFO] Capturing {max_frames} frames to '{frame_dir}'")
    run_simulator(sim, scene, policy, simulation_app, args_cli)

    simulation_app.close()

if __name__ == "__main__":
    main()