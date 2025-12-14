import gradio as gr
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# ---------- Helpers: UI drawing ----------
def make_click_line_image(width=900, height=120, x_min=-1.0, x_max=1.0, marker_x=None):
    """
    Draw a horizontal line. Optionally draw a marker at marker_x in value coordinates [x_min, x_max].
    Returns a PIL image.
    """
    fig, ax = plt.subplots(figsize=(width / 100, height / 100), dpi=100)
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)

    y = height // 2
    ax.plot([50, width - 50], [y, y], linewidth=4)

    # tick labels
    ax.text(50, y + 25, f"{x_min:.1f}", ha="center", va="bottom")
    ax.text(width - 50, y + 25, f"{x_max:.1f}", ha="center", va="bottom")

    if marker_x is not None:
        # map value -> pixel
        px = value_to_pixel(marker_x, width=width, x_min=x_min, x_max=x_max)
        ax.scatter([px], [y], s=200)

    ax.axis("off")
    fig.tight_layout(pad=0)
    fig.canvas.draw()

    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    return Image.fromarray(img)

def pixel_to_value(px, width=900, x_min=-1.0, x_max=1.0):
    # clickable line spans [50, width-50]
    left, right = 50, width - 50
    px_clamped = float(np.clip(px, left, right))
    frac = (px_clamped - left) / (right - left)
    return x_min + frac * (x_max - x_min)

def value_to_pixel(x, width=900, x_min=-1.0, x_max=1.0):
    left, right = 50, width - 50
    frac = (x - x_min) / (x_max - x_min)
    return left + frac * (right - left)

# ---------- Toy environment generator (change-point with hazard) ----------
def generate_sequence(n_trials, hazard_early, hazard_late, switch_trial, obs_sigma, seed=0):
    rng = np.random.default_rng(int(seed))
    x_true = np.zeros(n_trials, dtype=float)
    x_true[0] = rng.uniform(-1, 1)

    for t in range(1, n_trials):
        h = hazard_early if t < switch_trial else hazard_late
        if rng.random() < h:
            x_true[t] = rng.uniform(-1, 1)  # new latent state
        else:
            x_true[t] = x_true[t - 1]

    obs = x_true + rng.normal(0, obs_sigma, size=n_trials)
    obs = np.clip(obs, -1, 1)
    return x_true, obs

# ---------- Session state ----------
def new_state(n_trials=120, switch_trial=60, hazard_early=0.02, hazard_late=0.30,
              obs_sigma=0.15, seed=0, mode="hidden_switch"):
    x_true, obs = generate_sequence(
        n_trials=n_trials,
        hazard_early=hazard_early,
        hazard_late=hazard_late,
        switch_trial=switch_trial,
        obs_sigma=obs_sigma,
        seed=seed
    )
    return {
        "t": 0,
        "n_trials": n_trials,
        "switch_trial": switch_trial,
        "hazard_early": hazard_early,
        "hazard_late": hazard_late,
        "obs_sigma": obs_sigma,
        "seed": int(seed),
        "mode": mode,  # "hidden_switch" or "dial"
        "x_true": x_true,
        "obs": obs,
        "clicks": [],  # list of (t, x_click)
        "last_click": None,
    }

def reset_app(seed, n_trials, switch_trial, hazard_early, hazard_late, obs_sigma, mode):
    st = new_state(
        n_trials=int(n_trials),
        switch_trial=int(switch_trial),
        hazard_early=float(hazard_early),
        hazard_late=float(hazard_late),
        obs_sigma=float(obs_sigma),
        seed=int(seed),
        mode=mode
    )
    line_img = make_click_line_image(marker_x=None)
    status = f"Reset. Trial 1/{st['n_trials']}. Mode={mode}. (Hidden switch at t={st['switch_trial']})"
    return st, status, float(st["obs"][0]), line_img, plot_progress(st)

def plot_progress(st):
    """
    Plot: true latent vs observations vs user clicks.
    Returns a PIL image.
    """
    n = st["n_trials"]
    t = st["t"]

    clicks_t = [tt for tt, _ in st["clicks"]]
    clicks_x = [xx for _, xx in st["clicks"]]

    fig, ax = plt.subplots(figsize=(9, 3), dpi=120)
    ax.plot(st["x_true"], label="true")
    ax.plot(st["obs"], label="obs", alpha=0.7)
    if clicks_t:
        ax.scatter(clicks_t, clicks_x, label="your clicks", s=30)

    ax.axvline(st["switch_trial"], linestyle="--", alpha=0.6, label="hazard switch")
    ax.scatter([t], [st["obs"][t]], s=120, marker="x", label="current obs")

    ax.set_ylim(-1.05, 1.05)
    ax.set_xlabel("trial")
    ax.set_ylabel("value")
    ax.legend(loc="upper right", ncol=4, fontsize=8)
    fig.tight_layout()
    fig.canvas.draw()

    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    return Image.fromarray(img)

# ---------- Click handler ----------
def on_click(evt: gr.SelectData, st):
    """
    evt.index for Image is typically (row, col). We'll use col as x pixel.
    """
    if st is None:
        st = new_state()

    # evt.index often: (y, x)
    y_px, x_px = evt.index
    x_val = pixel_to_value(x_px)

    st["last_click"] = x_val
    # Show marker where they clicked
    line_img = make_click_line_image(marker_x=x_val)

    return st, f"Clicked x={x_val:.3f}. Now press 'Submit / Next trial'.", line_img

def submit_and_advance(st, hazard_dial):
    if st is None:
        st = new_state()

    if st["last_click"] is None:
        return st, "Click on the line first.", float(st["obs"][st["t"]]), make_click_line_image(None), plot_progress(st)

    # store click
    t = st["t"]
    st["clicks"].append((t, float(st["last_click"])))

    # advance
    if t < st["n_trials"] - 1:
        st["t"] += 1
    st["last_click"] = None  # require new click next trial

    # If in "dial" mode, regenerate the sequence using the slider as BOTH hazards (constant hazard).
    # This is the "unlock the hazard dial" exploration mode.
    if st["mode"] == "dial":
        h = float(hazard_dial)
        st["hazard_early"] = h
        st["hazard_late"] = h
        st["x_true"], st["obs"] = generate_sequence(
            n_trials=st["n_trials"],
            hazard_early=h,
            hazard_late=h,
            switch_trial=st["switch_trial"],
            obs_sigma=st["obs_sigma"],
            seed=st["seed"]
        )

    status = f"Trial {st['t']+1}/{st['n_trials']}. (Mode={st['mode']})"
    return st, status, float(st["obs"][st["t"]]), make_click_line_image(marker_x=None), plot_progress(st)

# ---------- UI ----------
with gr.Blocks(title="Glaze Click Scaffold") as demo:
    gr.Markdown("## Click-to-estimate scaffold (hazard switch + hazard dial)\n"
                "This is a toy environment to validate interaction + deployment. Next step: plug Glaze updates in parallel.")

    st = gr.State(None)

    with gr.Row():
        mode = gr.Dropdown(["hidden_switch", "dial"], value="hidden_switch", label="Mode")
        seed = gr.Number(value=0, precision=0, label="Seed")
        n_trials = gr.Slider(30, 300, value=120, step=1, label="Trials")
        switch_trial = gr.Slider(5, 200, value=60, step=1, label="Hidden switch trial")

    with gr.Row():
        hazard_early = gr.Slider(0.0, 0.8, value=0.02, step=0.01, label="Hazard early (hidden_switch mode)")
        hazard_late = gr.Slider(0.0, 0.8, value=0.30, step=0.01, label="Hazard late (hidden_switch mode)")
        obs_sigma = gr.Slider(0.01, 0.6, value=0.15, step=0.01, label="Observation noise σ (fixed)")

    hazard_dial = gr.Slider(0.0, 0.8, value=0.05, step=0.01, label="Hazard dial (dial mode)")

    reset_btn = gr.Button("Reset / Generate sequence")

    status = gr.Textbox(label="Status", interactive=False)
    current_obs = gr.Number(label="Current observation (shown to user)", interactive=False)

    click_img = gr.Image(
        value=make_click_line_image(marker_x=None),
        label="Click on the line to report your estimate",
        interactive=True
    )

    next_btn = gr.Button("Submit / Next trial")

    progress_plot = gr.Image(label="Progress", interactive=False)

    # Wire events
    reset_btn.click(
        reset_app,
        inputs=[seed, n_trials, switch_trial, hazard_early, hazard_late, obs_sigma, mode],
        outputs=[st, status, current_obs, click_img, progress_plot]
    )

    click_img.select(
        on_click,
        inputs=[st],
        outputs=[st, status, click_img]
    )

    next_btn.click(
        submit_and_advance,
        inputs=[st, hazard_dial],
        outputs=[st, status, current_obs, click_img, progress_plot]
    )

demo.launch()
