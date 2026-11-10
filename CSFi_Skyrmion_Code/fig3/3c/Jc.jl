using MicroMagnetic
using Printf
cd(@__DIR__)

@using_gpu()

# --- constants (SI) ---
const Ms_val = 5.8e5
const P_pol  = 0.5
const q_e    = 1.602176634e-19
const muB    = 9.2740100783e-24

# Initial magnetization: skyrmion-like texture in layer 1 and 3
function init_single_skx(i, j, k, dx, dy, dz)
    if k == 3
        return (i - 100)^2 + (j - 80)^2 < 20^2 ? (0, 0.01, -1) : (0, 0.01, 1)
    elseif k == 1
        return (i - 100)^2 + (j - 80)^2 < 20^2 ? (0, 0.01, 1) : (0, 0.01, -1)
    else
        return (0, 0, 0)
    end
end

# Build simulation: 3-layer mesh, Ms=0 in layer 2, spatial Ku pinning in layer 1
function basic_setup(; driver="SD", m0=(0, 0, 1), name="skx", kappa=0.01, Rd=3.0)
    mesh = FDMesh(nx=300, ny=200, nz=3, dx=1e-9, dy=1e-9, dz=2e-9, pbc="xy")
    sim = Sim(mesh, driver=driver, name=name)

    # Ms profile: layer 2 is non-magnetic (Ms=0)
    spatial_Ms = (i, j, k, dx, dy, dz) -> k == 2 ? 0 : 5.8e5
    set_Ms(sim, spatial_Ms)

    add_exch(sim, 15e-12)
    add_dmi(sim, 3.5e-3, type="interfacial")
    add_exch_int(sim, -8e-4)

    # Ku profile: Gaussian pinning well in layer 1 centered at (100,80)
    spatial_Ku = (i, j, k, dx, dy, dz) -> k == 1 ? 8e5*(1-kappa*exp(-(hypot(i - 100, j - 80)/Rd)^2)) : 8e5
    add_anis(sim, spatial_Ku, axis=(0, 0, 1))

    init_m0(sim, m0)
    return sim
end

# Run LLG + Zhang-Li STT; return true if depinned, false if pinned/stuck
function run_system(;Rd=8.0, kappa=0.05, xi1=0.03, xi3=0.03, jx=0.0, lambda=0)
    # Initial state from VTK (expects m0.vts in WORKDIR)
    sim = basic_setup(m0=read_vtk("m0.vts"), Rd=Rd, kappa=kappa)

    set_driver(sim, driver="LLG", alpha=0.08)

    # velocity scale used in pinning check
    u = abs( (P_pol * muB / (q_e * Ms_val)) * jx )

    # Layer-dependent current density J (x-direction)
    function spatial_J(i, j, k, dx, dy, dz)
        if k == 1
            return (-jx, 0, 0)
        elseif k == 3
            return (-jx*lambda, 0, 0)
        else
            return (0, 0, 0)
        end
    end

    # Layer-dependent non-adiabaticity xi
    function spatial_xi(i, j, k, dx, dy, dz)
        if k == 3; return xi3; elseif k == 1; return xi1; else; return 0.0; end
    end

    add_stt(sim, model=:zhang_li, Ms=5.8e5, xi=spatial_xi, J=spatial_J, P=0.5)

    dt = 1e-10

    # Initial guiding centers (layer 1 and 3)
    Rx1_0, Ry1_0 = compute_guiding_center(sim, z=1)
    Rx3_0, Ry3_0 = compute_guiding_center(sim, z=3)

    # For instantaneous speed (layer 1)
    Rx_pre, Ry_pre = Rx1_0, Ry1_0

    println("Start Running: Jx=$jx, Lambda=$lambda")
    println("Step | Time(ns) | L1_X(nm)  L1_Y(nm) | L3_X(nm)  L3_Y(nm) | Speed(m/s)")

    for i in 1:1000
        run_until(sim, dt * i, save_data = false)

        Rx1, Ry1 = compute_guiding_center(sim, z=1)
        Rx3, Ry3 = compute_guiding_center(sim, z=3)

        # Instantaneous speed from layer 1 displacement
        speed = hypot(Rx1 - Rx_pre, Ry1 - Ry_pre) / dt

        @printf("%4d | %6.2f   | %8.2f  %8.2f | %8.2f  %8.2f | %8.5f\n",
                i, (dt*i)*1e9, Rx1*1e9, Ry1*1e9, Rx3*1e9, Ry3*1e9, speed)

        Rx_pre, Ry_pre = Rx1, Ry1

        # Pinned/stuck criterion (avoid early-time false trigger by i>10)
        if speed < 1e-5 * u  && i > 10
             println("  -> Stucked/Pinned at step $i")
             return false
        end

        # Depinning criterion: layer 3 moves > 60 nm from initial position
        if hypot(Rx3 - Rx3_0, Ry3 - Ry3_0) > 60e-9
            println("  -> Depinned! Moved > 60nm")
            return true
        end
    end
    return false
end

# Bisection search for critical jx (depin threshold)
function find_jx(;Rd=3, kappa=0.01, xi1=0.03, xi3=0.03, lambda=1, steps=10, j2_max=2e12)
    println("find_jx xi1  xi3", xi1,xi3)
    j1, j2 = 0, j2_max
    jx = (j1 + j2) / 2

    for i in 1:steps
        jx = (j1 + j2) / 2
        println("i= ", i, "   jx=", jx, "   lambda=", lambda)

        moved = run_system(Rd=Rd, kappa=kappa, xi1=xi1, xi3=xi3,  jx=jx, lambda=lambda)

        if moved
            j2 = jx
        else
            j1 = jx
        end
    end
    return (j1 + j2) / 2
end

# Relaxation to generate and save initial state (m0.vts)
function relax_system()
    sim = basic_setup(m0=init_single_skx)
    relax(sim, maxsteps=20000, stopping_dmdt=0.01)
    save_vtk(sim, "m0")
end

# Generate initial state
relax_system()

# Parameter sweep: lambda -> critical current jc, write to text file
xi1=0.03
xi3=0.05
txt_name = "jx3_lambda_$(xi1)_$xi3.txt"
open(txt_name, "w") do f
    write(f, "#lambda jx1  Rd=3, kappa=0.05 , $xi1 , $xi3 \n")
    for lambda in 0:0.1:2
        jc = find_jx(lambda=lambda, Rd=3.0, kappa=0.05, xi1=xi1, xi3=xi3, steps=8, j2_max=1e12)
        write(f, "$lambda   $jc\n")
        flush(f)
        println("Saved lambda=$lambda")
    end
end

