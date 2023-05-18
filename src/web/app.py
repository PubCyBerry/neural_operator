# Type hint
# Config
from pathlib import Path
from typing import Any, Optional, Tuple

# webapp
import streamlit as st
from streamlit import session_state as sess
from streamlit.components import v1 as components

# from streamlit_extras import switch_page_button

# ML backend
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf


# user-defined libs
from src.utils import plotting
from src.utils.utils import load_model, make_mesh
from src.data.create_data import Data_Generator


############ CONTENTS ###############
def web_figure(target_func: str, *args: Any, **kwds: Any):
    fig = getattr(plotting, target_func)(*args, **kwds)
    st.pyplot(fig)


def web_animation(target_func: str, *args: Any, **kwds: Any):
    anim = getattr(plotting, target_func)(*args, **kwds)
    components.html(anim.to_jshtml(), width=800, height=550, scrolling=False)


#####################################


def main():
    # Set title
    app_title: str = "PDE with Deep Learning"
    st.set_page_config(page_title=app_title, page_icon=":hourglass:")
    st.title(app_title)

    top_container = st.container()
    top1, top2, top3 = top_container.columns([1, 1, 1])

    # PDE Configuration
    with st.expander("PDE Configuration"):
        # Set PDE
        target_pde: str = st.selectbox(
            "choose equation", ("Burgers_spectral", "KdV_spectral"), label_visibility="collapsed"
        )
        st.subheader("Parameter of PDE")

        # kinematic viscosity
        coefficient: float = st.number_input(
            "kinematic viscosity", value=0.01, format="%.4f", label_visibility="collapsed"
        )

        st.subheader("Spatial Domain")
        x_left, x_right, grid_x = st.columns(3)
        xmin: float = x_left.number_input("Left Boundary(m)", value=0.0)
        xmax: float = x_right.number_input("Right Boundary(m)", value=1.0)
        Nx: int = grid_x.number_input("Spatial Grids", value=1024, key="Nx")

        st.subheader("Temporal Domain")
        t_left, t_right, grid_t = st.columns(3)
        tmin: float = t_left.number_input("Initial Time(s)", value=0.0)
        tmax: float = t_right.number_input("Final Time(s)", value=1.0)
        Nt: int = grid_t.number_input("Temporal Grids", value=512, key="Nt")

        sess.xlim: Tuple[float, float] = (xmin, xmax)
        sess.tlim: Tuple[float, float] = (tmin, tmax)

        sess.generator = Data_Generator(
            target_pde=target_pde,
            xlim=sess.xlim,
            tlim=sess.tlim,
            Nx=Nx,
            Nt=Nt,
            coefficient=coefficient,
            data_dir="data",
            backend="torch",
        )

    with st.expander("Initial Condition Configuration"):
        col1, col2 = st.columns([1, 1.5], gap="large")
        num_data: int = 1
        with col1:
            m: float = st.number_input("m : mean", 0.0)
            sigma: float = st.number_input("σ : scaling factor", 49.0)
            tau: float = st.number_input("τ : inverse length scale of random field", 7.0)
            gamma: float = st.number_input("γ : regularity of random field", 2.5)
            is_parallel: bool = False
            current_variance = (
                f"$C={int(sigma)}" + r"\left(-\dfrac{d^2}{dx^2}" + rf"+{int(tau)}I \right)" + "^{" + f"{-gamma}" + "}$"
            )

        with col2:
            st.markdown("#### Gaussian Random Field(GRF)")
            st.markdown(r"Probability measure $\mu\sim\mathcal{N}(m,C)$")
            st.markdown(r"where $C=\sigma^2(-\Delta + \tau^2 I)^{-\gamma}$")
            st.markdown(r"Choose initial condtion $u(x,0)=u_0 \sim \mu$")
            st.markdown(r"##### Current setting")
            st.markdown(current_variance)

    if top1.button("Solve PDE", use_container_width=True):
        with st.spinner("Calculation..."):
            sess.data = sess.generator.create_data(
                num_data=num_data,
                m=m,
                sigma=sigma,
                tau=tau,
                gamma=gamma,
                is_parallel=is_parallel,
            ).squeeze()
            st.success("Calculation Complete", icon="✅")

    if top2.button("Load default", use_container_width=True):
        with st.spinner("Loading..."):
            data = np.load("data/demo.npz")
            sess.data = torch.Tensor(data["ys"])
            st.success("Calculation Complete", icon="✅")

    if top3.button("Plot Solution", use_container_width=True) and "data" in sess:
        with st.expander("Solution", expanded=True):
            with st.spinner("Drawing Full Field..."):
                web_figure("plot_solution", sess.generator.xs, sess.generator.ts, sess.data)
            with st.spinner("Animating Solution..."):
                web_animation(
                    "animate_solution",
                    sess.generator.xs,
                    sess.data,
                    sess.generator.ts,
                    save_img=False,
                )

    with st.expander("DNN"):
        model_name = "DNN"

        # Create batch
        xcol, tcol, btn = st.columns(3)
        Nx: int = xcol.number_input("Spatial Grids", value=sess.Nx, key=f"Nx_{model_name}")
        Nt: int = tcol.number_input("Temporal Grids", value=sess.Nt, key=f"Nt_{model_name}")
        btn.write("\n")
        btn.write("\n")
        if btn.button("Infer", use_container_width=True, key=f"button_{model_name}"):
            model = load_model(model_name)
            data = make_mesh(torch.linspace(*sess.xlim, Nx), torch.linspace(*sess.tlim, Nt)).cuda()

            # Perform inference
            preds = model((data[:, :1], data[:, 1:])).view(Nx, Nt).detach().cpu()

            # Plot result
            with st.spinner("Drawing Full Field..."):
                web_figure("plot_solution", sess.xlim, sess.tlim, sess.data, preds)
            with st.spinner("Animating Solution..."):
                web_animation("animate_solution", sess.xlim, sess.tlim, sess.data, preds, save_img=False)
        del sess[f"Nx_{model_name}"], sess[f"Nt_{model_name}"], sess[f"button_{model_name}"]

    with st.expander("PINN"):
        model_name = "PINN"

        # Create batch
        xcol, tcol, btn = st.columns(3)
        Nx: int = xcol.number_input("Spatial Grids", value=sess.Nx, key=f"Nx_{model_name}")
        Nt: int = tcol.number_input("Temporal Grids", value=sess.Nt, key=f"Nt_{model_name}")
        btn.write("\n")
        btn.write("\n")
        if btn.button("Infer", use_container_width=True, key=f"button_{model_name}"):
            model = load_model(model_name)
            data = make_mesh(torch.linspace(*sess.xlim, Nx), torch.linspace(*sess.tlim, Nt)).float().cuda()

            # Perform inference
            preds = model((data[:, :1], data[:, 1:])).view(Nx, Nt).detach().cpu()

            # Plot result
            with st.spinner("Drawing Full Field..."):
                web_figure("plot_solution", sess.xlim, sess.tlim, sess.data, preds)
            with st.spinner("Animating Solution..."):
                web_animation("animate_solution", sess.xlim, sess.tlim, sess.data, preds, save_img=False)
        del sess[f"Nx_{model_name}"], sess[f"Nt_{model_name}"], sess[f"button_{model_name}"]

    with st.expander("DeepONet"):
        model_name = "DeepONet"

        # Create batch
        xcol, tcol, btn = st.columns(3)
        Nx: int = xcol.number_input("Spatial Grids", value=sess.Nx, key=f"Nx_{model_name}")
        Nt: int = tcol.number_input("Temporal Grids", value=sess.Nt, key=f"Nt_{model_name}")
        btn.write("\n")
        btn.write("\n")
        if btn.button("Infer", use_container_width=True, key=f"button_{model_name}"):
            model = load_model(model_name)
            y = make_mesh(torch.linspace(*sess.xlim, Nx), torch.linspace(*sess.tlim, Nt)).cuda()
            u = sess.data[:: sess.data.shape[0] // model.num_sensor, 0].unsqueeze(0).cuda()
            st.write(Nx, y.size(), u.size())
            preds = model((u, y)).view(Nx, Nt).detach().cpu()
            st.write(preds.shape)
            # Plot result
            with st.spinner("Drawing Full Field..."):
                web_figure("plot_solution", sess.xlim, sess.tlim, sess.data, preds)
            with st.spinner("Animating Solution..."):
                web_animation("animate_solution", sess.xlim, sess.tlim, sess.data, preds, save_img=False)

        del sess[f"Nx_{model_name}"], sess[f"Nt_{model_name}"], sess[f"button_{model_name}"]
    st.sidebar.write(sess)


if __name__ == "__main__":
    main()
