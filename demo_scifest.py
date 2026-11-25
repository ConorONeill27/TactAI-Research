"""Side-by-side SciFest MuJoCo insertion demo with live tactile telemetry."""

import argparse

import matplotlib.pyplot as plt
from matplotlib.widgets import Button, RadioButtons, Slider
import mujoco
import numpy as np

from src.controllers.baseline_controllers import UnconstrainedPolicyBaseline
from src.controllers.learned_policy import LearnedChunkPolicy
from src.controllers.insertion_protocol import tactai
from src.envs.unknown_grasp_env import UnknownGraspEnv
from scripts.run_grasp_development import A2_DEV_QP, A2_STEPS


class Episode:
    def __init__(self, mode):
        self.env = UnknownGraspEnv(seed=42)
        self.env.model.geom_rgba[self.env.peg_geom_id] = [0.02, 0.65, 0.83, 1]
        self.select(mode)
        self.policy = LearnedChunkPolicy()
        self.camera = mujoco.MjvCamera()
        self.camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.camera.lookat[:] = [0, 0, 0.025]
        self.camera.distance = 0.18
        self.camera.azimuth = 125
        self.camera.elevation = -14

    def select(self, mode):
        self.mode = mode
        self.controller = (UnconstrainedPolicyBaseline() if mode == "Unconstrained" else
                           tactai(self.env.dt, unknown_grasp_mode=True,
                                  wrist_ft_only=mode == "Wrist F/T-only QP", **A2_DEV_QP))

    def reset(self, tilt, offset, grasp):
        self.obs, self.info = self.env.reset(seed=42, ood_tilt_deg=(tilt, 0),
                                            ood_offset_mm=(offset, 0), grasp_tilt_deg=(grasp, 0))
        self.controller.reset()
        self.policy.reset(seed=42)
        self.time_ms = [0]
        self.forces = [0.0]
        self.stage = 1 if self.mode != "Unconstrained" else None
        self.status = "Running"

    def tick(self):
        if self.status != "Running":
            return
        nominal = self.policy.get_action(self.obs)
        wrench, diagnostics = self.controller.compute_safe_action(nominal, self.obs)
        self.obs, _, terminal, truncated, self.info = self.env.step(wrench)
        self.stage = diagnostics.get("stage")
        self.time_ms.append(len(self.time_ms) * self.env.dt * 1000)
        self.forces.append(self.info["current_force_n"])
        if self.info["is_jammed"]:
            self.status = "Jammed"
        elif self.info["is_success"]:
            self.status = "Inserted"
        elif terminal or truncated or len(self.time_ms) > A2_STEPS:
            self.status = "Trial ended"


class Dashboard:
    def __init__(self):
        self.fig, self.axes = plt.subplots(3, 2, figsize=(12, 8),
                                           gridspec_kw={"height_ratios": [1.7, 1, 1.1]})
        self.fig.subplots_adjust(bottom=0.25, top=0.91, hspace=0.52, wspace=0.26)
        self.fig.suptitle("TactAI · Peg insertion under misalignment", fontsize=17)
        manager = self.fig.canvas.manager
        manager.set_window_title("SciFest | TactAI")
        self.episodes = [Episode("Wrist F/T-only QP"), Episode("TactAI")]
        self.renderers = []
        self.artists = []
        for col, name in enumerate(("Wrist F/T-only QP", "TactAI")):
            episode = self.episodes[col]
            renderer = mujoco.Renderer(episode.env.model, height=200, width=320)
            self.renderers.append(renderer)
            scene, force, tactile = self.axes[:, col]
            scene.set_title(name, fontweight="bold")
            scene.axis("off")
            view = scene.imshow(np.zeros((200, 320, 3), dtype=np.uint8))
            telemetry = scene.text(0.5, -0.05, "", transform=scene.transAxes,
                                   ha="center", va="top", fontsize=9)
            line, = force.plot([], [], color="#2563eb", linewidth=1.8)
            force.axhline(15, linestyle="--", color="#b45309", label="15 N limit")
            force.axhline(40, linestyle=":", color="#b91c1c", label="40 N jam")
            force.set(xlim=(0, A2_STEPS), ylim=(0, 55), xlabel="Simulated time (ms)",
                      ylabel="Contact force (N)")
            force.legend(loc="upper right", ncol=2, fontsize=8)
            force.grid(alpha=0.2)
            pad = episode.env.sensor_left
            extent = np.array([pad.U.min(), pad.U.max(), pad.V.min(), pad.V.max()]) * 1000
            image = tactile.imshow(np.zeros_like(pad.U), extent=extent, origin="lower",
                                   vmin=0, vmax=100, cmap="magma", aspect="auto")
            marker, = tactile.plot([], [], "c+", markersize=10, markeredgewidth=2)
            tactile.set(xlabel="Pad u (mm)", ylabel="Pad v (mm)",
                        title="Left GelSight pressure and CoP", aspect="equal")
            self.fig.colorbar(image, ax=tactile, label="kPa", fraction=0.05, pad=0.04)
            self.artists.append((view, telemetry, line, image, marker))

        self.tilt = Slider(self.fig.add_axes([0.19, 0.17, 0.44, 0.025]),
                           "Initial tilt (°)", -15, 15, valinit=3, valstep=0.5)
        self.offset = Slider(self.fig.add_axes([0.19, 0.12, 0.44, 0.025]),
                             "Lateral offset (mm)", -0.92, 0.92, valinit=0, valstep=0.02)
        self.grasp = Slider(self.fig.add_axes([0.19, 0.068, 0.44, 0.025]),
                            "In-hand grasp tilt (?)", -10, 10, valinit=7, valstep=.5)
        self.mode = RadioButtons(self.fig.add_axes([0.67, 0.055, 0.17, 0.14]),
                                 ["Unconstrained", "Wrist F/T-only QP", "TactAI"], active=1)
        self.replay = Button(self.fig.add_axes([0.85, 0.137, 0.13, 0.048]), "Run / restart")
        self.pause = Button(self.fig.add_axes([0.85, 0.078, 0.13, 0.048]), "Pause")
        self.replay.on_clicked(self.reset)
        self.pause.on_clicked(self.toggle_pause)
        self.mode.on_clicked(self.reset)
        self.timer = self.fig.canvas.new_timer(interval=33)
        self.timer.add_callback(self.update)
        self.fig.canvas.mpl_connect("close_event", self.close)
        self.fig.canvas.mpl_connect("key_press_event", self.keypress)
        self.reset()

    def reset(self, _=None):
        self.episodes[0].select(self.mode.value_selected)
        self.axes[0, 0].set_title(self.mode.value_selected, fontweight="bold")
        for episode in self.episodes:
            episode.reset(self.tilt.val, self.offset.val, self.grasp.val)
        self.paused = False
        self.pause.label.set_text("Pause")
        self.draw()

    def toggle_pause(self, _=None):
        self.paused = not self.paused
        self.pause.label.set_text("Resume" if self.paused else "Pause")
        self.fig.canvas.draw_idle()

    def keypress(self, event):
        if event.key == " ":
            self.toggle_pause()
        elif event.key in ("r", "enter"):
            self.reset()

    def update(self):
        if self.paused or all(e.status != "Running" for e in self.episodes):
            return
        for episode in self.episodes:
            for _ in range(4):
                episode.tick()
        self.draw()

    def draw(self):
        pressures = [e.env.left_pad_pressure / 1000 for e in self.episodes]
        vmax = max(1, *(p.max() for p in pressures))
        upper = max(55, 1.05 * max(max(e.forces) for e in self.episodes))
        for col, episode in enumerate(self.episodes):
            renderer = self.renderers[col]
            view, text, line, image, marker = self.artists[col]
            renderer.update_scene(episode.env.data, episode.camera)
            view.set_data(renderer.render())
            line.set_data(episode.time_ms, episode.forces)
            self.axes[1, col].set_ylim(0, upper)
            stage = f"Stage {episode.stage}" if episode.stage is not None else "Automaton off"
            wrist = np.rad2deg(episode.obs["ee_rpy"][0])
            peg = np.rad2deg(episode.env.peg_tilt_rad()[0])
            text.set_text(f"{episode.status}  ·  {stage}  ·  Tilt {episode.info['tilt_deg']:.2f}°\n"
                          f"Force {episode.forces[-1]:.1f} N  ·  Peak {max(episode.forces):.1f} N")
            image.set_data(pressures[col])
            image.set_clim(0, vmax)
            u, v = episode.obs["tactile"]["left_pad"]["center_of_pressure"] * 1000
            marker.set_data([u], [v])
        self.fig.canvas.draw_idle()

    def close(self, _=None):
        self.timer.stop()
        for renderer in self.renderers:
            renderer.close()
        self.renderers.clear()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true",
                        help="Exercise the dashboard without entering the GUI event loop.")
    args = parser.parse_args()
    if args.smoke:
        plt.switch_backend("Agg")
    dashboard = Dashboard()
    if args.smoke:
        try:
            while any(e.status == "Running" for e in dashboard.episodes):
                for episode in dashboard.episodes:
                    episode.tick()
            dashboard.draw()
            for episode in dashboard.episodes:
                print(f"{episode.status}: peak {max(episode.forces):.2f} N")
            assert dashboard.episodes[0].status != "Inserted"
            assert dashboard.episodes[1].status == "Inserted"
            dashboard.mode.set_active(0)
            assert dashboard.episodes[0].mode == "Unconstrained"
            dashboard.mode.set_active(2)
            assert dashboard.episodes[0].mode == "TactAI"
        finally:
            dashboard.close()
        return
    dashboard.timer.start()
    plt.show()


if __name__ == "__main__":
    main()
