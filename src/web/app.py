# Type hint
from typing import Any, Optional, Tuple

# webapp
import streamlit as st
from streamlit import session_state as sess
from streamlit.components import v1 as components

# from streamlit_extras import switch_page_button

# ML backend
import numpy as np
import torch
import gc


# user-defined libs
from src.utils import plotting
from src.utils.utils import load_model, make_mesh
from src.data.create_data import Data_Generator


############ CONTENTS ###############
@st.cache_data
def web_figure(tag:str, target_func: str, _xs:torch.tensor, _ts:torch.tensor, _ys:torch.tensor, _preds:torch.tensor=None, *args: Any, **kwds: Any):
    fig_tag = tag
    fig = getattr(plotting, target_func)(_xs, _ts, _ys, _preds, *args, **kwds)
    st.pyplot(fig)


@st.cache_data
def web_animation(tag:str, target_func: str, _xs:torch.tensor, _ts:torch.tensor, _ys:torch.tensor, _preds:torch.tensor=None, *args: Any, **kwds: Any):
    anim_tag = tag
    anim = getattr(plotting, target_func)(_xs, _ts, _ys, _preds, *args, **kwds)
    components.html(anim.to_jshtml(), width=800, height=550, scrolling=False)


def model_page(model_name: str):
    with st.expander(model_name, expanded=True):
        # Create batch
        xcol, tcol, btn = st.columns(3)
        Nx: int = xcol.number_input("Spatial Grids", value=sess.Nx, key=f"Nx_{model_name}")
        Nt: int = tcol.number_input("Temporal Grids", value=sess.Nt, key=f"Nt_{model_name}")
        btn.write("\n")
        btn.write("\n")
        if btn.button("Infer", use_container_width=True, key=f"button_{model_name}"):
            model = load_model(model_name)
            mesh = make_mesh(torch.linspace(*sess.xlim, Nx), torch.linspace(*sess.tlim, Nt))

            # Perform inference
            if model_name == "DeepONet":
                u = sess.data[:: sess.data.shape[0] // model.num_sensor, 0].unsqueeze(0)
                preds = model((u.cuda(), mesh.cuda())).view(Nx, Nt).detach().cpu()

            elif model_name == "FNO":
                u = sess.data[:, :1].unsqueeze(0)
                preds = (
                    model.repeat((u.cuda(), torch.linspace(*sess.xlim, u.shape[1]).view(1, -1, 1).cuda()), Nt)
                    .view(Nx, Nt)
                    .detach()
                    .cpu()
                )
            else:
                preds = model((mesh[:, :1].cuda(), mesh[:, 1:].cuda())).view(Nx, Nt).detach().cpu()

            # Plot result
            with st.spinner("Drawing Full Field..."):
                web_figure(f'{model_name}_Fig',"plot_solution", sess.xlim, sess.tlim, sess.data, preds)
            with st.spinner("Animating Solution..."):
                web_animation(f'{model_name}_Anim',"animate_solution", sess.xlim, sess.tlim, sess.data, preds, save_img=False)
        # del sess[f"Nx_{model_name}"], sess[f"Nt_{model_name}"], sess[f"button_{model_name}"]


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
            device=torch.device('cpu')
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
            st.success("Done!", icon="✅")

    if top2.button("Load default", use_container_width=True):
        with st.spinner("Loading..."):
            data = np.load("data/demo.npz")
            sess.data = torch.Tensor(data["ys"])
            st.success("Done!", icon="✅")

    if top3.button("Plot Solution", use_container_width=True) and "data" in sess:
        with st.expander("Solution", expanded=True):
            with st.spinner("Drawing Full Field..."):
                web_figure('Solution_Fig',"plot_solution", sess.xlim, sess.tlim, sess.data)
            with st.spinner("Animating Solution..."):
                web_animation('Solution_Anim',"animate_solution", sess.xlim, sess.tlim, sess.data, save_img=False)

    model_page("DNN")
    model_page("PINN")
    model_page("DeepONet")
    model_page("FNO")



if __name__ == "__main__":
    main()
