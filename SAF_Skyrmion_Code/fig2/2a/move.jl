using MicroMagnetic
using Printf
using NPZ

cd(@__DIR__)

# ============================
# 1) Geometry factors
# ============================
"""
    geom_factors_from_spin(spin; z_layer, nx, ny, nz, dx, dy)

Compute (for a given layer):
  - Topological charge Q
  - Shape factor η (so that D = η G)
  - Ratio |G|/D = |Q|/η

Supported input:
  - spin :: Vector{Float64}, length = 3*nx*ny*nz
  - spin :: Array{Float64,4}, size = (3, nx, ny, nz)
"""
function geom_factors_from_spin(
        spin;
        z_layer::Int,
        nx::Int,
        ny::Int,
        nz::Int,
        dx::Float64,
        dy::Float64,
    )

    # Normalize spin to (3, nx, ny, nz)
    if ndims(spin) == 1
        @assert length(spin) == 3*nx*ny*nz "spin length mismatch: length(spin)=$(length(spin)), expected $(3*nx*ny*nz)"
        m4 = reshape(spin, 3, nx, ny, nz)
    elseif ndims(spin) == 4
        m4 = spin
        nx = size(m4, 2)
        ny = size(m4, 3)
        nz = size(m4, 4)
    else
        error("spin must be a Vector or 4D array, got ndims = $(ndims(spin))")
    end

    @assert 1 ≤ z_layer ≤ nz "z_layer = $z_layer out of range 1:$nz"

    mx = @view m4[1, :, :, z_layer]
    my = @view m4[2, :, :, z_layer]
    mz = @view m4[3, :, :, z_layer]

    total_D = 0.0
    total_Q = 0.0
    dA = dx * dy

    @inbounds for i in 1:nx, j in 1:ny
        # Periodic boundary conditions (PBC)
        ip = (i == nx) ? 1  : i + 1
        im = (i == 1)  ? nx : i - 1
        jp = (j == ny) ? 1  : j + 1
        jm = (j == 1)  ? ny : j - 1

        # Central differences
        dmx_dx = (mx[ip, j] - mx[im, j]) / (2dx)
        dmy_dx = (my[ip, j] - my[im, j]) / (2dx)
        dmz_dx = (mz[ip, j] - mz[im, j]) / (2dx)

        dmx_dy = (mx[i, jp] - mx[i, jm]) / (2dy)
        dmy_dy = (my[i, jp] - my[i, jm]) / (2dy)
        dmz_dy = (mz[i, jp] - mz[i, jm]) / (2dy)

        # ∫ (|∂x m|^2 + |∂y m|^2) d^2r
        grad_sq_x = dmx_dx^2 + dmy_dx^2 + dmz_dx^2
        grad_sq_y = dmx_dy^2 + dmy_dy^2 + dmz_dy^2
        total_D += (grad_sq_x + grad_sq_y) * dA

        # m · (∂x m × ∂y m)
        cx = dmy_dx*dmz_dy - dmz_dx*dmy_dy
        cy = dmz_dx*dmx_dy - dmx_dx*dmz_dy
        cz = dmx_dx*dmy_dy - dmy_dx*dmx_dy

        m_dot_cross = mx[i,j]*cx + my[i,j]*cy + mz[i,j]*cz
        total_Q += m_dot_cross * dA
    end

    Q = total_Q / (4π)

    # D_xx = (1/2)∬ (|∂x m|^2 + |∂y m|^2) d^2r
    D_integral_xx = total_D / 2.0
    η = D_integral_xx / (4π)              # so that D = η G

    ratio_G_D = (η != 0) ? abs(Q) / η : 0.0

    return Q, η, ratio_G_D
end

# ============================
# 2) Simulation parameters
# ============================

@using_gpu()

const NX = 300
const NY = 85
const NZ = 3
const SIM_DX = 1e-9
const SIM_DY = 1e-9
const SIM_DZ = 2e-9

const Rd = 7
const lambda_pin = 0.05
const current_J = -1.6e12
const alpha_val = 0.08

# Ms profile: layer 2 is non-magnetic (Ms=0)
spatial_Ms = (i,j,k,dx,dy,dz) -> k == 2 ? 0.0 : 5.8e5

# Ku profile: Gaussian pinning well in top layer (k==3)
function spatial_Ku_pin(i, j, k, dx, dy, dz)
    r = sqrt((i - 30)^2 + (j - 30)^2)
    if k == 3
        return 8e5 * (1 - lambda_pin * exp(-r^2 / Rd^2))
    end
    return 8e5
end

# Initial skyrmion-like texture with opposite polarity between layers 1 and 3
function init_single_skx(i, j, k, dx, dy, dz)
    r  = 20
    cx = 30
    cy = 30
    inside = (i - cx)^2 + (j - cy)^2 < r^2

    if k == 3
        return inside ? (0.0, 0.01, -1.0) : (0.0, 0.01,  1.0)
    elseif k == 1
        return inside ? (0.0, 0.01,  1.0) : (0.0, 0.01, -1.0)
    else
        return (0.0, 0.0, 0.0)
    end
end

# Current distribution: applied to layers 1 and 3
function spatial_J(i, j, k, dx, dy, dz)
    if k == 3 || k == 1
        return (current_J, 0.0, 0.0)
    else
        return (0.0, 0.0, 0.0)
    end
end

# Non-adiabaticity xi: different values for layers 1 and 3
function spatial_xi(i, j, k, dx, dy, dz)
    if k == 3
        return 0.05
    elseif k == 1
        return 0.03
    else
        return 0.0
    end
end

# ============================
# 3) Main routine
# ============================

function run_system()
    # Output directories
    isdir("npy")  || mkpath("npy")
    isdir("data") || mkpath("data")

    # Setup + relax
    mesh = FDMesh(nx=NX, ny=NY, nz=NZ, dx=SIM_DX, dy=SIM_DY, dz=SIM_DZ, pbc="xy")
    sim  = Sim(mesh, driver="SD", name="saf_skx")

    set_Ms(sim, spatial_Ms)
    add_exch(sim, 15e-12)
    add_dmi(sim, 3.5e-3, type="interfacial")
    add_exch_int(sim, -8e-4)
    add_anis(sim, spatial_Ku_pin, axis=(0, 0, 1))

    init_m0(sim, init_single_skx)

    println("Relaxing system...")
    relax(sim, maxsteps=20_000, stopping_dmdt=0.01)

    # Dynamics (LLG + STT)
    set_driver(sim, driver="LLG", alpha=alpha_val)
    add_stt(sim, model=:zhang_li, Ms=5.8e5, xi=spatial_xi, J=spatial_J, P=0.5)

    dt    = 2e-10
    steps = 30

    txt_filename = "data/dynamics_tracking.txt"
    println("Starting dynamics with J=$current_J. Saving to $txt_filename ...")

    open(txt_filename, "w") do f
        write(f, "# Time(s)  Rx_top(m)  Ry_top(m)  Q_top  Eta_top  Rx_bot(m)  Ry_bot(m)  Q_bot  Eta_bot  Eta_SAF_avg\n")

        for i in 0:steps
            time = dt * i
            run_until(sim, time)

            # Copy spin from GPU to CPU
            spin_vec = Array(sim.spin)

            # Top layer (z=3)
            Rx3, Ry3 = compute_guiding_center(sim, z=3)
            Q3, Eta3, _ = geom_factors_from_spin(
                spin_vec;
                z_layer=3,
                nx=NX, ny=NY, nz=NZ,
                dx=SIM_DX, dy=SIM_DY,
            )

            # Bottom layer (z=1)
            Rx1, Ry1 = compute_guiding_center(sim, z=1)
            Q1, Eta1, _ = geom_factors_from_spin(
                spin_vec;
                z_layer=1,
                nx=NX, ny=NY, nz=NZ,
                dx=SIM_DX, dy=SIM_DY,
            )

            Eta_avg = 0.5 * (Eta1 + Eta3)

            write(f, @sprintf(
                "%g  %g  %g  %.6f  %.6f  %g  %g  %.6f  %.6f  %.6f\n",
                time, Rx3, Ry3, Q3, Eta3, Rx1, Ry1, Q1, Eta1, Eta_avg,
            ))
            flush(f)

            # Save full spin state
            npzwrite(@sprintf("npy/m_step_%d.npy", i), spin_vec)

            @printf(
                "Step %d: T=%.1f ns, Rx_top=%.1f nm, Q_top=%.2f, Eta3=%.4f\n",
                i, time*1e9, Rx3*1e9, Q3, Eta1,
            )
        end
    end

    println("Simulation finished.")
end

run_system()
