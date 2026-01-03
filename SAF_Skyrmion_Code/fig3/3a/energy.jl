using MicroMagnetic
using Printf
cd(@__DIR__)

# -------------------------
# Global parameters
# -------------------------
Rd = 7
lambda = 0.05  # global pinning strength (used in Ku profile)

@using_gpu()

# Ms profile: layer 2 is non-magnetic (Ms = 0)
spatial_Ms = (i,j,k,dx,dy,dz) -> k == 2 ? 0 : 5.8e5

# Initial magnetization: skyrmion-like texture in layer 1 and 3
function init_single_skx(i, j, k, dx, dy, dz)
    r=20
    cx=100
    cy=80
    if k == 3
        # Layer 3: core vs background polarity (kept as in original)
        return (i - cx)^2 + (j - cy)^2 < r^2 ? (0, 0.01, -1) : (0, 0.01, 1)
    elseif k == 1
        # Layer 1: opposite polarity (SAF-like coupling)
        return (i - cx)^2 + (j - cy)^2 < r^2 ? (0, 0.01, 1) : (0, 0.01, -1)
    else
        return (0, 0, 0)
    end
end

# Spatial anisotropy Ku with a Gaussian pinning well (layer 3 only)
function spatial_Ku_pin(i, j, k, dx, dy, dz)
    r = sqrt((i - 100)^2 + (j - 80)^2)
    if k == 3
        return 8e5*(1-lambda*exp(-r^2/Rd^2))
    end
    return 8e5
end

# Base simulation setup: mesh + energy terms + initial state
function basic_setup(; driver="SD", m0=(0, 0, 1), name="skx")
    mesh = FDMesh(nx=200, ny=160, nz=3, dx=1e-9, dy=1e-9, dz=2e-9, pbc="xy")
    sim = Sim(mesh, driver=driver, name=name)
    set_Ms(sim, spatial_Ms)
    add_exch(sim, 15e-12)
    add_dmi(sim, 3.5e-3, type="interfacial")
    # Interlayer exchange (negative = antiferromagnetic coupling)
    add_exch_int(sim, -6e-3)
    add_anis(sim, spatial_Ku_pin, axis=(0, 0, 1))
    init_m0(sim, m0)
    return sim
end

function run_system()
    # Output directory for VTK snapshots
    isdir("vtk") || mkpath("vtk")

    # Initialize and relax
    sim = basic_setup(m0=init_single_skx)
    relax(sim, maxsteps=20000, stopping_dmdt=0.01)

    # Dynamics: LLG
    set_driver(sim, driver="LLG", alpha=0.08)

    # Current density (A/m^2)
    current_J = -2e12

    # Layer-dependent current density J (x-direction)
    function spatial_J(i, j, k, dx, dy, dz)
        if k == 3
            return (current_J, 0, 0)
        elseif k == 1
            return (current_J, 0, 0)
        else
            return (0, 0, 0)
        end
    end

    # Layer-dependent non-adiabaticity xi
    function spatial_xi(i, j, k, dx, dy, dz)
        if k == 3; return 0.05; elseif k == 1; return 0.03; else; return 0.0; end
    end

    # Zhang-Li STT term
    add_stt(sim, model=:zhang_li, Ms=5.8e5, xi=spatial_xi, J=spatial_J, P=0.5)

    # Time stepping and output file
    steps = 200
    dt = 1e-11
    name = @sprintf("XY_Rd_%g_lambda_%g.txt", Rd, lambda)

    # Write (time, guiding center, total energy) to file
    open(name, "w") do f
        write(f, "# Time(s)  Rx1(m)  Ry1(m)  TotalEnergy(J)\n")

        for i in 1:steps
            time = dt * i
            run_until(sim, time, save_data=true)

            total_energy = sum(sim.energy)
            Rx1, Ry1 = compute_guiding_center(sim, z=1) # layer 1
            Rx3, Ry3 = compute_guiding_center(sim, z=3) # layer 3

            # Save field snapshot
            save_vtk(sim, "vtk/step_$i")

            # Save scalar time series
            write(f, @sprintf("%g  %g  %g   %0.10g\n", time, Rx1, Ry1, total_energy))

            # Console log (positions in nm for readability)
            println("λ=$(lambda)  step=$i ; ",
                    "Rx1=$(Rx1*1e9)nm ; Ry1=$(Ry1*1e9)nm ; ",
                    "Rx3=$(Rx3*1e9)nm ; Ry3=$(Ry3*1e9)nm")
        end
    end
end

run_system()
