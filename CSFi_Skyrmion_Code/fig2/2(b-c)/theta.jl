using MicroMagnetic
using Printf
using NPZ

cd(@__DIR__)
@using_gpu()

# ===========================
# 1) Global settings
# ===========================
const NX = 150
const NY = 150
const NZ = 3
const DX = 1e-9
const DY = 1e-9
const DZ = 2e-9
const RELAX_FILE = "mrelaxed.npy"

const TOTAL_STEPS = 40
const MID_STEP    = 20       # drift window start (skip transient)
const DT          = 1e-10
const REPEATS     = 1        # repeats per lambda

const BETA_LAYER1 = 0.03     # k = 1 (bottom)
const BETA_LAYER3 = 0.05     # k = 3 (top)

# ===========================
# 2) Field profiles and init
# ===========================
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

# Voting-based center estimation:
# circshift the spin field along y and pick the most consistent guiding-center result.
function compute_center_voting(sim; z=1)
    mesh = sim.mesh
    nx, ny, nz = mesh.nx, mesh.ny, mesh.nz
    dy = mesh.dy
    Ly = ny * dy

    # Backup original spin (CPU copy)
    m_flat_backup = Array(sim.spin)
    m_3d_backup = reshape(m_flat_backup, (3, nx, ny, nz))

    shifts_idx = [0, div(ny, 3), -div(ny, 3)]
    candidates_y = Float64[]

    for s in shifts_idx
        if s == 0
            sim.spin[:] = m_flat_backup
        else
            m_shifted = circshift(m_3d_backup, (0, 0, s, 0))
            sim.spin[:] = reshape(m_shifted, length(m_flat_backup))
        end

        _, raw_cy = compute_guiding_center(sim, z=z)

        # Undo the artificial shift
        real_cy = raw_cy - (s * dy)
        real_cy = mod(real_cy, Ly)

        push!(candidates_y, real_cy)
    end

    # Restore original spin (critical)
    sim.spin[:] = m_flat_backup

    y1, y2, y3 = candidates_y[1], candidates_y[2], candidates_y[3]

    # Periodic distance on [0, Ly)
    function pbc_dist(a, b, L)
        d = abs(a - b)
        return d > L/2 ? L - d : d
    end

    d12 = pbc_dist(y1, y2, Ly)
    d23 = pbc_dist(y2, y3, Ly)
    d13 = pbc_dist(y1, y3, Ly)

    final_cy = 0.0

    if (d12 <= d23) && (d12 <= d13)
        diff = y2 - y1
        if diff > Ly/2; y2 -= Ly; elseif diff < -Ly/2; y2 += Ly; end
        final_cy = (y1 + y2) / 2
    elseif (d23 <= d12) && (d23 <= d13)
        diff = y3 - y2
        if diff > Ly/2; y3 -= Ly; elseif diff < -Ly/2; y3 += Ly; end
        final_cy = (y2 + y3) / 2
    else
        diff = y3 - y1
        if diff > Ly/2; y3 -= Ly; elseif diff < -Ly/2; y3 += Ly; end
        final_cy = (y1 + y3) / 2
    end

    final_cy = mod(final_cy, Ly)

    # X is typically stable under PBC here; compute once on the unshifted field
    cx, _ = compute_guiding_center(sim; z=z)

    return cx, final_cy
end

# Base simulation setup
function basic_setup(; driver="SD", m0=(0, 0, 1), name="skx")
    mesh = FDMesh(nx=NX, ny=NY, nz=NZ, dx=DX, dy=DY, dz=DZ, pbc="xy")
    sim = Sim(mesh, driver=driver, name=name)
    set_Ms(sim, spatial_Ms)
    add_exch(sim, 15e-12)
    add_dmi(sim, 3.5e-3, type="interfacial")
    add_exch_int(sim, -8e-4)
    add_anis(sim, 8e5, axis=(0, 0, 1))
    init_m0(sim, m0)
    return sim
end

# Pre-relax once and cache the relaxed spin to RELAX_FILE
function pre_relax()
    if !isfile(RELAX_FILE)
        println("  > Pre-relaxing initial state (run once)...")
        sim = basic_setup(driver="SD", m0=init_single_skx, name="relax_init")
        relax(sim, maxsteps=20000, stopping_dmdt=0.01)
        npzwrite(RELAX_FILE, Array(sim.spin))
        save_vtk(sim, "vtktest")
        println("  > Relaxed state saved to $RELAX_FILE")
    else
        println("  > Found relaxed state: $RELAX_FILE, skipping relaxation.")
    end
end

# ===========================
# 3) One run: simulate + save full trajectory + return (dx, dy)
# ===========================
function run_single_simulation(lambda_val, current_J, repeat_idx, save_dir)

    run_id   = @sprintf("L%.3f_R%d", lambda_val, repeat_idx)
    sim_name = "sim_" * run_id
    traj_filename = joinpath(save_dir, "traj_" * run_id * ".txt")

    sim = basic_setup(driver="LLG", m0=npzread(RELAX_FILE), name=sim_name)
    set_driver(sim; driver="LLG", alpha=0.08)

    # STT distributions
    spatial_J = (i, j, k, dx, dy, dz) -> begin
        if k == 3
            return (current_J, 0.0, 0.0)
        elseif k == 1
            return (lambda_val * current_J, 0.0, 0.0)
        else
            return (0.0, 0.0, 0.0)
        end
    end

    spatial_xi = (i, j, k, dx, dy, dz) -> begin
        if k == 3
            return BETA_LAYER3
        elseif k == 1
            return BETA_LAYER1
        else
            return 0.0
        end
    end

    add_stt(sim, model=:zhang_li, Ms=5.8e5, xi=spatial_xi, J=spatial_J, P=0.5)

    # Unwrapping box size
    Lx = sim.mesh.nx * sim.mesh.dx
    Ly = sim.mesh.ny * sim.mesh.dy

    # Layer 1 state
    cx1_prev = NaN
    cy1_prev = NaN
    cx1_u = 0.0
    cy1_u = 0.0
    x1_mid, y1_mid = 0.0, 0.0
    x1_end, y1_end = 0.0, 0.0

    # Layer 3 state (tracked for logging)
    cx3_prev = NaN
    cy3_prev = NaN
    cx3_u = 0.0
    cy3_u = 0.0
    x3_mid, y3_mid = 0.0, 0.0
    x3_end, y3_end = 0.0, 0.0

    open(traj_filename, "w") do f_traj
        write(f_traj, "# Time(ns)   L1_X_unwrap(m)   L1_Y_unwrap(m)   L3_X_unwrap(m)   L3_Y_unwrap(m)\n")

        for i in 1:TOTAL_STEPS
            current_time = i * DT
            run_until(sim, current_time, save_data=false)

            # Raw centers (modded into [0, L))
            cx1, cy1 = compute_center_voting(sim; z=1)
            cx1_mod = mod(cx1, Lx)
            cy1_mod = mod(cy1, Ly)

            cx3, cy3 = compute_center_voting(sim; z=3)
            cx3_mod = mod(cx3, Lx)
            cy3_mod = mod(cy3, Ly)

            # PBC unwrapping
            if i == 1
                cx1_u, cy1_u = cx1_mod, cy1_mod
                cx3_u, cy3_u = cx3_mod, cy3_mod
            else
                dxp1 = cx1_mod - cx1_prev
                dyp1 = cy1_mod - cy1_prev
                if dxp1 >  Lx/2; dxp1 -= Lx; elseif dxp1 < -Lx/2; dxp1 += Lx; end
                if dyp1 >  Ly/2; dyp1 -= Ly; elseif dyp1 < -Ly/2; dyp1 += Ly; end
                cx1_u += dxp1
                cy1_u += dyp1

                dxp3 = cx3_mod - cx3_prev
                dyp3 = cy3_mod - cy3_prev
                if dxp3 >  Lx/2; dxp3 -= Lx; elseif dxp3 < -Lx/2; dxp3 += Lx; end
                if dyp3 >  Ly/2; dyp3 -= Ly; elseif dyp3 < -Ly/2; dyp3 += Ly; end
                cx3_u += dxp3
                cy3_u += dyp3
            end

            cx1_prev, cy1_prev = cx1_mod, cy1_mod
            cx3_prev, cy3_prev = cx3_mod, cy3_mod

            @printf("Step %d (t=%.2fns): L1(%.2e, %.2e) | L3(%.2e, %.2e)\n",
                    i, current_time*1e9, cx1_u, cy1_u, cx3_u, cy3_u)

            @printf(f_traj, "%.6e  %.6e  %.6e  %.6e  %.6e\n",
                    current_time*1e9, cx1_u, cy1_u, cx3_u, cy3_u)
            flush(f_traj)

            if i == MID_STEP
                x1_mid, y1_mid = cx1_u, cy1_u
            elseif i == TOTAL_STEPS
                x1_end, y1_end = cx1_u, cy1_u
            end
        end
    end

    sim = nothing
    GC.gc()

    # Return layer-1 displacement over the second half
    dx = x1_end - x1_mid
    dy = y1_end - y1_mid

    return dx, dy, traj_filename
end

# ===========================
# 4) Main sweep: average over repeats
# ===========================
const J_VAL = -1e12
const LAMBDA_LIST = 0.1:0.1:1

save_dir = "Data_Avg_Traj"
isdir(save_dir) || mkpath(save_dir)

pre_relax()

summary_file = joinpath(save_dir, "Summary_Averaged_Results23.txt")
println("Start Simulation... All data will be saved to: $save_dir")
println("Repeats per lambda: $REPEATS")

open(summary_file, "w") do f_sum
    write(f_sum, "# Averaged over $REPEATS runs per lambda. Temperature = 50K\n")
    @printf(f_sum, "%-10s %-12s %-12s %-12s %-12s\n", "lambda", "avg_theta", "avg_tan", "avg_dx", "avg_dy")

    for lambda in LAMBDA_LIST
        println("\n>> Processing lambda = $lambda ...")

        sum_dx = 0.0
        sum_dy = 0.0

        for r in 1:REPEATS
            dx, dy, saved_traj = run_single_simulation(lambda, J_VAL, r, save_dir)

            sum_dx += dx
            sum_dy += dy

            @printf("    [Run %d/%d] dx=% .3e, dy=% .3e -> Traj saved: %s\n",
                    r, REPEATS, dx, dy, basename(saved_traj))
        end

        avg_dx = sum_dx / REPEATS
        avg_dy = sum_dy / REPEATS

        avg_tan   = avg_dy / avg_dx
        avg_theta = atand(avg_dy, avg_dx)

        @printf("  >> AVG Result: Angle = %.2f deg, Tan = %.4f\n", avg_theta, avg_tan)

        @printf(f_sum, "%-10.4f %-12.6f %-12.6f %-14.6e %-14.6e\n",
                lambda, avg_theta, avg_tan, avg_dx, avg_dy)
        flush(f_sum)
    end
end

println("\nAll Done! Please check the folder '$save_dir' for trajectory files.")
