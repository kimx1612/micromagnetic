using MicroMagnetic
using Printf
using NPZ

cd(@__DIR__)
@using_gpu()

# ===========================
# 1) Global settings
# ===========================
const NX = 300
const NY = 1000
const NZ = 3
const DX = 1e-9
const DY = 1e-9
const DZ = 2e-9
const RELAX_FILE = "mrelaxed.npy"   # temporary spin storage

const TOTAL_STEPS = 50
const MID_STEP    = 25              # drift window start (skip transient)
const DT          = 1e-10

# ===========================
# 2) Field profiles and init
# ===========================
spatial_Ms = (i,j,k,dx,dy,dz) -> k == 1 ? 5.8e5 : (k == 3 ? 5.8e5 : 0)

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

# ===========================
# 3) Run one beta and return Hall angle
# ===========================
function run_and_save(beta_val, current_J, lambda_val, traj_folder)

    sim_name = @sprintf("sim_beta_%.3f", beta_val)

    # (1) Relax initial state (SD)
    println("  > Force Relaxing initial state...")
    sim = basic_setup(driver="SD", m0=init_single_skx, name="relax_init")
    relax(sim, maxsteps=20000, stopping_dmdt=0.01)

    # Save relaxed spin (CPU copy) and reload for dynamics
    npzwrite(RELAX_FILE, Array(sim.spin))

    sim = basic_setup(driver="LLG", m0=npzread(RELAX_FILE), name=sim_name)
    set_driver(sim; driver="LLG", alpha=0.08)

    # (2) STT parameters
    xi_layer3 = beta_val

    spatial_J = (i, j, k, dx, dy, dz) -> begin
        if k == 3
            return (current_J, 0, 0)
        elseif k == 1
            return (lambda_val * current_J, 0, 0)
        else
            return (0, 0, 0)
        end
    end

    spatial_xi = (i, j, k, dx, dy, dz) -> begin
        if k == 3
            return xi_layer3
        elseif k == 1
            return 0.03
        else
            return 0.0
        end
    end

    add_stt(sim, model=:zhang_li, Ms=5.8e5, xi=spatial_xi, J=spatial_J, P=0.5)

    # (3) Run and record trajectory (layer 1 guiding center)
    traj_filename = joinpath(traj_folder, @sprintf("traj_beta_%.3f.txt", beta_val))

    x_mid, y_mid = 0.0, 0.0
    x_end, y_end = 0.0, 0.0

    open(traj_filename, "w") do f
        write(f, "# step   time(s)        x(m)           y(m)\n")

        rx, ry = compute_guiding_center(sim, z=1)
        @printf(f, "%-6d %-14.4e %-14.6e %-14.6e\n", 0, 0.0, rx, ry)

        @printf("    [Step  0] time=0.00e+00, rx=%.6e, ry=%.6e\n", rx, ry)

        for i in 1:TOTAL_STEPS
            run_until(sim, i * DT, save_data=false)

            rx, ry = compute_guiding_center(sim, z=1)

            @printf("    [Step %2d] time=%.2e, rx=%.6e, ry=%.6e\n", i, i*DT, rx, ry)
            @printf(f, "%-6d %-14.4e %-14.6e %-14.6e\n", i, i*DT, rx, ry)

            if i == MID_STEP
                x_mid, y_mid = rx, ry
            elseif i == TOTAL_STEPS
                x_end, y_end = rx, ry
            end
        end
    end

    sim = nothing; GC.gc()

    dx = x_end - x_mid
    dy = y_end - y_mid

    tan_val = dy / dx
    theta_deg = atand(dy, dx)

    return theta_deg, tan_val
end

# ===========================
# 4) Main sweep
# ===========================
const J_VAL = -5e11
const LAMBDA_VAL = 0.0
const BETA_LIST = [0.06, 0.07, 0.08, 0.09, 0.10, 0.12, 0.15, 0.2, 0.25, 0.3]

save_dir = "Data_Simple_Traj"
if !isdir(save_dir) mkpath(save_dir) end

summary_file = joinpath(save_dir, "Summary_Results2.txt")
println("Start... Data saved to: $save_dir")

open(summary_file, "w") do f
    write(f, "# Method: theta = atan(dy/dx), tan_val = dy/dx (from step 15 to 30)\n")
    @printf(f, "%-10s %-12s %-12s\n", "beta", "theta_deg", "tan_val")

    for beta in BETA_LIST
        println("\n>> Beta = $beta ... ")

        theta, tan_v = run_and_save(beta, J_VAL, LAMBDA_VAL, save_dir)

        @printf(">> Result: Angle = %.2f deg, Tan = %.4f\n", theta, tan_v)
        @printf(f, "%-10.4f %-12.6f %-12.6f\n", beta, theta, tan_v)
        flush(f)
    end
end

println("All Done!")
