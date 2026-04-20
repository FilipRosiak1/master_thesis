from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import streamlit as st


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
MODELS = (
    "char_vae",
    "grammar_vae",
    "grammar_vae_masked",
    "tree_vae",
    "transformer_vae",
    "vq_grammar_ae",
)


def run_script(script_name: str, args: list[str]) -> tuple[int, str, str, str]:
    script_path = SCRIPTS / script_name
    cmd = [sys.executable, str(script_path), *args]
    cmd_text = " ".join(cmd)
    completed = subprocess.run(
        cmd,
        cwd=str(ROOT),
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.returncode, completed.stdout, completed.stderr, cmd_text


def render_command_result(title: str, rc: int, stdout: str, stderr: str, cmd: str) -> None:
    st.markdown(f"### {title}")
    st.code(cmd, language="bash")
    if rc == 0:
        st.success("Command finished successfully")
    else:
        st.error(f"Command failed with exit code {rc}")
    st.markdown("**stdout**")
    st.code(stdout or "(empty)", language="text")
    if stderr:
        st.markdown("**stderr**")
        st.code(stderr, language="text")


def train_tab() -> None:
    st.subheader("Train")
    with st.form("train_form"):
        model = st.selectbox("Model", MODELS, index=2)
        data_path = st.text_input("Data path", "datasets/f1/f1_dataset.txt")
        output_root = st.text_input("Output root", "models/f1")
        epochs = st.number_input("Epochs override (0 = use default)", min_value=0, value=0)
        seed = st.number_input("Seed (-1 = none)", value=-1)
        config_model = st.text_input("Config model (optional)", "")
        config_data = st.text_input("Config data (optional)", "")
        config_train = st.text_input("Config train (optional)", "")
        submitted = st.form_submit_button("Run training")

    if submitted:
        args = ["--model", model, "--data-path", data_path, "--output-root", output_root]
        if epochs > 0:
            args.extend(["--epochs", str(int(epochs))])
        if seed >= 0:
            args.extend(["--seed", str(int(seed))])
        if config_model:
            args.extend(["--config-model", config_model])
        if config_data:
            args.extend(["--config-data", config_data])
        if config_train:
            args.extend(["--config-train", config_train])
        rc, out, err, cmd = run_script("train.py", args)
        render_command_result("Train Result", rc, out, err, cmd)


def eval_tab() -> None:
    st.subheader("Evaluate")
    with st.form("eval_form"):
        model = st.selectbox("Model", MODELS, index=2, key="eval_model")
        weights = st.text_input("Weights (.pth)")
        data_path = st.text_input("Data path", "datasets/f1/f1_dataset.txt", key="eval_data")
        mode = st.selectbox("Mode", ("reconstruct", "mutate"))
        max_items = st.number_input("Max items (0 = all)", min_value=0, value=0)
        num_samples = st.number_input("Mutation samples", min_value=1, value=10)
        noise_scale = st.number_input("Noise scale", min_value=0.0, value=1.0)
        submitted = st.form_submit_button("Run evaluation")

    if submitted:
        args = ["--model", model, "--weights", weights, "--data-path", data_path, "--mode", mode]
        if mode == "reconstruct" and max_items > 0:
            args.extend(["--max-items", str(int(max_items))])
        if mode == "mutate":
            args.extend(["--num-samples", str(int(num_samples)), "--noise-scale", str(float(noise_scale))])
        rc, out, err, cmd = run_script("eval.py", args)
        render_command_result("Eval Result", rc, out, err, cmd)


def infer_tab() -> None:
    st.subheader("Infer")
    with st.form("infer_form"):
        model = st.selectbox("Model", MODELS, index=2, key="infer_model")
        weights = st.text_input("Weights (.pth)", key="infer_weights")
        mode = st.selectbox("Mode", ("reconstruct", "mutate"), key="infer_mode")
        input_string = st.text_input("Input genotype", "RX")
        noise_scale = st.number_input("Noise scale", min_value=0.0, value=1.0, key="infer_noise")
        submitted = st.form_submit_button("Run inference")

    if submitted:
        args = [
            "--model",
            model,
            "--weights",
            weights,
            "--mode",
            mode,
            "--input-string",
            input_string,
            "--noise-scale",
            str(float(noise_scale)),
        ]
        rc, out, err, cmd = run_script("infer.py", args)
        render_command_result("Infer Result", rc, out, err, cmd)


def optimize_tab() -> None:
    st.subheader("Optimize latent z")
    with st.form("optimize_form"):
        model = st.selectbox("Model", MODELS, index=2, key="opt_model")
        weights = st.text_input("Weights (.pth)", key="opt_weights")
        fitness_fn = st.text_input("Fitness function (module:function or path.py:function)")
        algorithm = st.selectbox("Algorithm", ("cmaes", "cem"))
        cma_backend = st.selectbox("CMA backend", ("auto", "internal"))
        iterations = st.number_input("Iterations", min_value=1, value=80)
        population_size = st.number_input("Population size (0 = default)", min_value=0, value=0)
        elite_fraction = st.number_input("Elite fraction (CEM)", min_value=0.01, max_value=0.5, value=0.2)
        submitted = st.form_submit_button("Run optimization")

    if submitted:
        args = [
            "--model",
            model,
            "--weights",
            weights,
            "--fitness-fn",
            fitness_fn,
            "--algorithm",
            algorithm,
            "--iterations",
            str(int(iterations)),
        ]
        if algorithm == "cmaes":
            args.extend(["--cma-backend", cma_backend])
        else:
            args.extend(["--elite-fraction", str(float(elite_fraction))])
        if population_size > 0:
            args.extend(["--population-size", str(int(population_size))])
        rc, out, err, cmd = run_script("optimize_latent.py", args)
        render_command_result("Optimization Result", rc, out, err, cmd)


def pipeline_tab() -> None:
    st.subheader("Full pipeline")
    st.caption("Runs: train -> eval reconstruct -> eval mutate -> optional optimize")
    with st.form("pipeline_form"):
        model = st.selectbox("Model", MODELS, index=2, key="pipe_model")
        data_path = st.text_input("Data path", "datasets/f1/f1_dataset.txt", key="pipe_data")
        output_root = st.text_input("Output root", "models/f1", key="pipe_out")
        epochs = st.number_input("Epochs override (0 = default)", min_value=0, value=0, key="pipe_epochs")
        mutation_samples = st.number_input("Mutation samples", min_value=1, value=10)
        noise_scale = st.number_input("Noise scale", min_value=0.0, value=1.0, key="pipe_noise")
        fitness_fn = st.text_input("Fitness function (optional)", "")
        algorithm = st.selectbox("Optimize algorithm", ("cmaes", "cem"), key="pipe_alg")
        optimize_iterations = st.number_input("Optimize iterations", min_value=1, value=80)
        submitted = st.form_submit_button("Run pipeline")

    if submitted:
        args = [
            "pipeline",
            "--model",
            model,
            "--data-path",
            data_path,
            "--output-root",
            output_root,
            "--mutation-samples",
            str(int(mutation_samples)),
            "--noise-scale",
            str(float(noise_scale)),
            "--algorithm",
            algorithm,
            "--optimize-iterations",
            str(int(optimize_iterations)),
        ]
        if epochs > 0:
            args.extend(["--epochs", str(int(epochs))])
        if fitness_fn:
            args.extend(["--fitness-fn", fitness_fn])
        rc, out, err, cmd = run_script("workbench.py", args)
        render_command_result("Pipeline Result", rc, out, err, cmd)


def main() -> None:
    st.set_page_config(page_title="F1 VAE Workbench", page_icon="🧬", layout="wide")
    st.title("F1 VAE Workbench")
    st.caption("Streamlit frontend for train/eval/infer/optimize workflows")

    tabs = st.tabs(["Train", "Evaluate", "Infer", "Optimize", "Pipeline"])
    with tabs[0]:
        train_tab()
    with tabs[1]:
        eval_tab()
    with tabs[2]:
        infer_tab()
    with tabs[3]:
        optimize_tab()
    with tabs[4]:
        pipeline_tab()


if __name__ == "__main__":
    main()
