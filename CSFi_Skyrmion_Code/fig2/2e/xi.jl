using MicroMagnetic
using Printf
using NPZ
using LinearAlgebra

cd(@__DIR__)

# ===========================
# 1) Global constants
# ===========================
const Q_thiele   = 1.0
const ETA_shape  = 1.23
const G_thiele   = 4π * Q_thiele
const D_thiele   = 4π * ETA_shape
const G_over_D   = G_thiele / D_thiele

@printf("Thiele parameters: G = %.6f, D = %.6f, G/D = %.6f\n",
        G_thiele, D_thiele, G_over_D)

const NX = 250
const NY = 800
const NZ = 3
const DX = 1e-9
const DY = 1e-9
const DZ = 2e-9
const Lx_box = NX * DX
const Ly_box = NY * DY

# Relaxed initial state cache
const RELAX_FILE = "m_222relaxed.npy"

@using_gpu()

# ===========================
# 2) Helper functions
# ===========================

function compute_eta(spin::AbstractArray; z_layer::Int = 1)
    # Ensure spin is on CPU and in (3, NX, NY, NZ) layout
    m4 = Array(spin)
    m4 = ndims(m4) == 4 ? m4 : reshape(m4, 3, NX, NY, NZ)

    @assert size(m4,1) == 3 && size(m4,2) == NX && size(m4,3) == NY && size(m4,4) == NZ

    mx = @view m4[1, :, :, z_layer]
    my = @view m4[2, :, :, z_layer]
    mz = @view m4[3, :, :, z_layer]

    Sxx = 0.0
    Syy = 0.0

    @inbounds for i in 1:NX, j in 1:NY
        # Periodic boundary conditions (PBC)
        ip = (i == NX) ? 1  : i + 1
        im = (i == 1)  ? NX : i - 1
        jp = (j == NY) ? 1  : j + 1
        jm = (j == 1)  ? NY : j - 1

        # Central differences
        dmx_dx = (mx[ip, j] - mx[im, j]) / (2DX)
        dmy_dx = (my[ip, j] - my[im, j]) / (2DX)
        dmz_dx = (mz[ip, j] - mz[im, j]) / (2DX)

        dmx_dy = (mx[i, jp] - mx[i, jm]) / (2DY)
        dmy_dy = (my[i, jp] - my[i, jm]) / (2DY)
        dmz_dy = (mz[i, jp] - mz[i, jm]) / (2DY)

        dot_xx = dmx_dx^2 + dmy_dx^2 + dmz_dx^2
        dot_yy = dmx_dy^2 + dmy_dy^2 + dmz_dy^2

        Sxx += dot_xx * DX * DY
        Syy += dot_yy * DX * DY
    end

    pref = 1.0 / (4π)
    η_xx = pref * Sxx
    η_yy = pref * Syy
    return η_xx, η_yy, 0.5 * (η_xx + η_yy)
end

function compute_eta(filename::AbstractString; z_layer::Int = 1)
    spin = npzread(filename)
    return compute_eta(spin; z_layer=z_layer)
end

# Ms profile: layers 1 and 3 magnetic, layer 2 non-magnetic
spatial_Ms = (i,j,k,dx,dy,dz) -> k == 1 ? 5.8e5 : (k == 3 ? 5.8e5 : 0)

# Initial skyrmion-like texture (opposite polarity between layers 1 and 3)
function init_single_skx(i, j, k, dx, dy, dz)
    r  = 20; cx = 50; cy = 50
    if k == 3
        return ( (i - cx)^2 + (j - cy)^2 < r^2 ? (0.0, 0.01,  1.0) : (0.0, 0.01, -1.0) )
    elseif k == 1
        return ( (i - cx)^2 + (j - cy)^2 < r^2 ? (0.0, 0.01, -1.0) : (0.0, 0.01,  1.0) )
    else
        return (0.0, 0.0, 0.0)
    end
end

function basic_setup(; driver="SD", m0=(0, 0, 1), name="skx")
    mesh = FDMesh(nx=NX, ny=NY, nz=3, dx=DX, dy=DY, dz=DZ, pbc="xy")
    sim = Sim(mesh, driver=driver, name=name)
    set_Ms(sim, spatial_Ms)
    add_exch(sim, 15e-12)
    add_dmi(sim, 3.5e-3, type="interfacial")
    add_exch_int(sim, -8e-4)
    add_anis(sim, 8e5, axis=(0, 0, 1))
    init_m0(sim, m0)
    return sim
end

# ===========================
# 3) Core simulation:
#    compute theta and tan(theta) from (last step - mid step)
# ===========================
function run_system(; lambda_val=1.0, xi_layer3=0.06, current_J=-3e11, save_dir)

    short_name = @sprintf("sim_lam_%.2f", lambda_val)

    # (1) Initialize (use cached relaxed state if available)
    if isfile(RELAX_FILE)
        m0_relaxed = npzread(RELAX_FILE)
        sim = basic_setup(driver="LLG", m0=m0_relaxed, name=short_name)
    else
        println("Relaxing initial state...")
        sim = basic_setup(driver="SD", m0=init_single_skx, name="relax_init")
        relax(sim, maxsteps=20000, stopping_dmdt=0.01)
        npzwrite(RELAX_FILE, Array(sim.spin))
        sim = basic_setup(driver="LLG", m0=npzread(RELAX_FILE), name=short_name)
    end
    set_driver(sim; driver="LLG", alpha=0.08)

    # (2) STT distributions
    function spatial_J(i, j, k, dx, dy, dz)
        J1 = current_J
        if k == 3
            return (J1, 0, 0)
        elseif k == 1
            return (lambda_val * J1, 0, 0)
        else
            return (0, 0, 0)
        end
    end

    function spatial_xi(i, j, k, dx, dy, dz)
        if k == 3
            return xi_layer3
        elseif k == 1
            return 0.03
        else
            return 0.0
        end
    end

    add_stt(sim, model=:zhang_li, Ms=5.8e5, xi=spatial_xi, J=spatial_J, P=0.5)

    # (3) Time stepping + PBC unwrapping (layer 1)
    txt_filename = joinpath(save_dir, @sprintf("traj_lam_%.2f.txt", lambda_val))

    steps = 50
    dt    = 1e-10
    ts, xs_un, ys_un = Float64[], Float64[], Float64[]

    open(txt_filename, "w") do f
        write(f, "# time  X1_unwrap  Y1_unwrap\n")

        time = 0.0
        Rx1, Ry1 = compute_guiding_center(sim, z=1)
        x_prev, y_prev = Rx1, Ry1
        x_un_prev, y_un_prev = Rx1, Ry1

        push!(ts, time); push!(xs_un, x_un_prev); push!(ys_un, y_un_prev)
        @printf(f, "%.4e  %.6e  %.6e\n", time, x_un_prev, y_un_prev)

        for i in 1:steps
            time = dt * i
            run_until(sim, time, save_data=false)

            Rx1, Ry1 = compute_guiding_center(sim, z=1)

            # Unwrap X (PBC)
            dx = Rx1 - x_prev
            if dx > 0.5Lx_box
                dx -= Lx_box
            elseif dx < -0.5Lx_box
                dx += Lx_box
            end
            x_un = x_un_prev + dx

            # Unwrap Y (PBC)
            dy = Ry1 - y_prev
            if dy > 0.5Ly_box
                dy -= Ly_box
            elseif dy < -0.5Ly_box
                dy += Ly_box
            end
            y_un = y_un_prev + dy

            push!(ts, time); push!(xs_un, x_un); push!(ys_un, y_un)
            @printf(f, "%.4e  %.6e  %.6e\n", time, x_un, y_un)

            x_prev, y_prev = Rx1, Ry1
            x_un_prev, y_un_prev = x_un, y_un
        end
    end

    # (4) theta from (last - mid)
    mid_step = steps ÷ 2
    mid_idx  = mid_step + 1
    end_idx  = length(xs_un)

    dx_tot = xs_un[end_idx] - xs_un[mid_idx]
    dy_tot = ys_un[end_idx] - ys_un[mid_idx]

    tan_val   = dy_tot / dx_tot
    theta_deg = atand(dy_tot, dx_tot)

    @printf("    [Res] λ=%.2f: Δx=%.3e, Δy=%.3e, θ=%.2f°\n",
            lambda_val, dx_tot, dy_tot, theta_deg)

    sim = nothing
    return theta_deg, tan_val
end

# ===========================
# 4) Scan lambda and fit beta1/beta2 from Y(lambda)
# ===========================
function scan_lambda(lambda_list, xi_layer3, current_J)

    J_str  = @sprintf("%.0e", current_J)
    xi_str = @sprintf("%.3f", xi_layer3)

    base_folder = "Data_Scan"
    group_dir   = joinpath(base_folder, "J_$(J_str)", "xi_$(xi_str)")
    isdir(group_dir) || mkpath(group_dir)

    println("\n========================================")
    @printf(">> Group: J=%s, Xi=%s\n", J_str, xi_str)
    @printf(">> Dir  : %s\n", group_dir)
    println("========================================")

    lambdas    = collect(lambda_list)
    thetas_deg = Float64[]
    tan_vals   = Float64[]
    Ys         = Float64[]

    # Sweep lambdas, compute theta and Y
    for λ in lambdas
        θ_deg, tan_v = run_system(lambda_val=λ,
                                  xi_layer3=xi_layer3,
                                  current_J=current_J,
                                  save_dir=group_dir)
        push!(thetas_deg, θ_deg)
        push!(tan_vals,   tan_v)
        Y = (1 - λ) / tan_v
        push!(Ys, Y)
    end

    # Linear fit: Y(λ) = A + B λ  (least squares via backslash)
    X = hcat(ones(length(lambdas)), lambdas)
    p = X \ Ys
    A, B = p

    # Convert to beta: Y = (D/G)(beta1 + beta2 λ) => beta = (G/D) * coeff
    beta1 = G_over_D * A
    beta2 = G_over_D * B

    # Save results
    out_name = joinpath(group_dir, "fit_result.txt")
    open(out_name, "w") do f
        @printf(f, "# Current_J = %.2e\n", current_J)
        @printf(f, "# Xi_layer3 = %.3f\n", xi_layer3)
        @printf(f, "# Thiele parameters: G = %.12e, D = %.12e, G/D = %.12e\n",
                G_thiele, D_thiele, G_over_D)
        @printf(f, "# theta from Δy/Δx (last-mid); tan_val = Δy/Δx\n")
        @printf(f, "# lambda    theta_deg      tan_theta        Y=(1-lambda)/tan(theta)\n")
        for i in eachindex(lambdas)
            @printf(f, "%10.5f  %12.6f  %14.6e  %20.12e\n",
                    lambdas[i], thetas_deg[i], tan_vals[i], Ys[i])
        end
        @printf(f, "\n# linear fit: Y(λ) = A + B λ\n")
        @printf(f, "# A = %.12e\n# B = %.12e\n", A, B)
        @printf(f, "# beta1 = %.12e\n# beta2 = %.12e\n", beta1, beta2)
    end

    println(">> Group Done. Fit saved to $(basename(out_name))")
    return A, B, beta1, beta2
end

# ===========================
# 5) Main sweep
# ===========================
J_list   = [-5e11]
xi3_list = [0.09, 0.10, 0.11, 0.12, 0.13]

lambda_list = 0.1:0.1:0.8

println("Starting Full Matrix Scan...")

for J_val in J_list
    for xi_val in xi3_list
        scan_lambda(lambda_list, xi_val, J_val)
        GC.gc()
    end
end

println("\nAll simulations completed!")
