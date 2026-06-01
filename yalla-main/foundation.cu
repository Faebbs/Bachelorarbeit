#include "./include/dtypes.cuh"
#include "./include/inits.cuh"
#include "./include/property.cuh"
#include "./include/solvers.cuh"
#include "./include/vtk.cuh"

//const auto r_max = 8.f; // Maximum distance for interaction; Cutoff distance
const auto r_max = 2.f; // Maximum distance for interaction; Cutoff distance
//const auto n_cells = 20u*25u; // Number of cells in the simulation
const auto n_time_steps = 1000u;
const auto dt = 0.05; // Time step size

// Parameters for Morse potential
//const auto r_e = 1.0f; // Equilibrium distance
const auto r_e = 0.3f; // Equilibrium distance
const auto D_e = 2.0f; // Depth of the potential well
const auto alpha = 1.0f; // Width of the potential

const auto r_start = r_e / 2;         // Resting radius
const auto r_activated = r_start / 2; // Target radius when activated
const auto r_decay = 0.9f;            // Exponential decay factor (0 < r_decay < 1): smaller = slower

const auto activation_delay = 5;  // Timesteps in delay state, before going to activated state
const auto activation_duration = 10;  // Timesteps in activated state, before refreactory state
const auto refractory_duration = 40;  // Timesteps in refractory state, before going back to deactivated state

//const auto sensitivity = 1.0f;  // Number of necessary neigbbors to trigger activation
//const auto n_lifetime = log(10000)/(r_activated + (r_start - r_activated) * exp(-r_decay)); 
//const auto force_threshold = r_activated + (r_start - r_activated) * exp(-r_decay * n_lifetime);  // Cumulative force needed to trigger activation
const auto force_threshold = 0.12f;  // Half-activation force (K in Hill function)
const auto use_hill_function = false;  // Whether to use Hill function for activation probability instead of hard threshold
const auto n_hill = 2.0f;            // Hill coefficient: steepness (higher = closer to hard threshold)

// Timesteps at which activation is triggered
int activation_steps[] = {100, 200, 300, 400};

__device__ int *d_neig;
__device__ int *d_activated;
__device__ int *d_prev_activated;
__device__ float *d_radius;
__device__ float3 *d_force_accum;


// Stuff that happens once every time step before time step is taken
int alter_cells_before(Solution<float3, Gabriel_solver>& cells, Property<float>& h_radius,
                Property<int>& h_activated, Property<float>& h_force_mag,
                Property<int>& h_state_timer, int n_cells)
{
    // Cell state update
    for (int i = 0; i < n_cells; i++) {
        switch (h_activated.h_prop[i]){
            case 0: {  // Deactivated: Hill-function activation probability
                if (use_hill_function){
                    //Hill function with stochastic
                    float F = h_force_mag.h_prop[i];
                    float Fn = powf(F, n_hill);
                    float Kn = powf(force_threshold, n_hill);
                    float P = Fn / (Kn + Fn);
                    if ((float)rand() / RAND_MAX < P) {
                        h_activated.h_prop[i] = 7;
                    }
                } else {
                    // Hard threshold
                    if (h_force_mag.h_prop[i] > force_threshold) {
                        h_activated.h_prop[i] = 7;
                    }
                }

                break;
            }
            case 1: // Activated state: radius shrinks
                h_radius.h_prop[i] = r_activated + (h_radius.h_prop[i] - r_activated) * expf(-r_decay);
                // Timer for how long cell is in state 1, goes to 2 after defined duration
                h_state_timer.h_prop[i]++;
                if (h_state_timer.h_prop[i] >= activation_duration) {
                    h_activated.h_prop[i] = 2;
                    h_state_timer.h_prop[i] = 0;
                }
                break;
            case 2: // Refractory state: radius grows back to r_start
                h_radius.h_prop[i] = r_start + (h_radius.h_prop[i] - r_start) * expf(-r_decay);
                // Timer for how long cell is in state 2, goes to 0 after defined duration
                h_state_timer.h_prop[i]++;
                if (h_state_timer.h_prop[i] >= refractory_duration) {
                    h_activated.h_prop[i] = 0;
                    h_state_timer.h_prop[i] = 0;
                }
                break;
            case 7: // Delay activation
                h_state_timer.h_prop[i]++;
                // Timer for how long cell is in state 7, goes to 1 after defined duration
                if (h_state_timer.h_prop[i] >= activation_delay) {
                    h_activated.h_prop[i] = 1;
                    h_state_timer.h_prop[i] = 0;
                } 
                // Stage of cell is reset to 0 if force drops under threshold during delay phase
                if (h_force_mag.h_prop[i] < force_threshold) {
                    h_activated.h_prop[i] = 0;
                    h_state_timer.h_prop[i] = 0;
                }
                break;
        }
    }

    h_radius.copy_to_device();
    h_activated.copy_to_device();

    return 0;
}

// Simulates Cell interaction, goes through all pairs of cells and calculates the force on cell i exerted by cell j
__device__ float3 simulation_step (float3 Xi, float3 r, float dist, int i, int j)
{
    float3 dF{0};
    if (i == j) return dF;

    if (dist > r_max) return dF;

    d_neig[i] += 1;

    // Calculation of the force which is exuded by cell j on cell i. 
    // Depends on the distance between the cells and the potential we choose.

    // Radius of cells
    auto r_i = d_radius[i];
    auto r_j = d_radius[j];
    auto surface_dist = dist - (r_i + r_j);
    // Morse Potential: auto F = D_e * (expf(-2*alpha*(dist - r_e)) - 2*expf(-alpha*(dist - r_e)));
    // Kraft abgeleitet daraus:
    auto phi = expf(-alpha * (surface_dist - r_e));
    auto F = -2.0f * D_e * alpha * (1.0f - phi) * phi;
    // auto phi = expf(-alpha * (dist - r_e));
    // auto F = -2.0f * D_e * alpha * (1.0f - phi) * phi;

    // Vector of force on cell i exerted by cell j
    dF = r * F / dist;
    // Accumulate forces from all neighbors (atomic to avoid race conditions between threads)
    atomicAdd(&d_force_accum[i].x, dF.x);
    atomicAdd(&d_force_accum[i].y, dF.y);
    atomicAdd(&d_force_accum[i].z, dF.z);
    return dF;
}

int main(int argc, const char* argv[])
{
    // Load initial conditions from VTK
    Vtk_input input{"/home/fabian/Bachelorarbeit/initial_conditions_mesh_1.vtk"};
    const auto n_cells = input.n_points;

    printf("Loaded %d cells from VTK\n", n_cells);
    printf("Cell resting radius: %f\n", r_start);
    printf("Cell activated radius: %f\n", r_activated);
    printf("Force threshold for activation: %f\n", force_threshold);

    // Prepare initial state
    Solution<float3, Gabriel_solver> cells{n_cells, 50, r_max};
    input.read_positions(cells); // Reads coordinates of vtk file
    //random_sphere(r_min, cells);
    //regular_rectangle(2.0f*r_e, 20, cells);

    // n_neighbor property
    cells.copy_to_device();
    Property<int> h_neig{n_cells, "neighbours"};
    cudaMemcpyToSymbol(d_neig, &h_neig.d_prop, sizeof(d_neig));
    h_neig.copy_to_device();

    // activated property: 0=deactivated, 1=activated, 2=refractory
    Property<int> h_activated{n_cells, "activated"};
    thrust::fill(h_activated.h_prop, h_activated.h_prop + n_cells, 2);
    //thrust::fill(h_activated.h_prop, h_activated.h_prop + n_cells, 2);
    h_activated.copy_to_device();
    cudaMemcpyToSymbol(d_activated, &h_activated.d_prop, sizeof(d_activated));

    // radius property, starts at r_start for all cells
    Property<float> h_radius{n_cells, "radius"};
    thrust::fill(h_radius.h_prop, h_radius.h_prop + n_cells, r_start);  // Standard-Radius, equally for all cells at half of equilibrium distance
    h_radius.copy_to_device();
    cudaMemcpyToSymbol(d_radius, &h_radius.d_prop, sizeof(d_radius));

    // accumulated force vector property (device-side accumulation)
    Property<float3> h_force_accum{n_cells, "force_accum"};
    thrust::fill(h_force_accum.h_prop, h_force_accum.h_prop + n_cells, float3{0});
    h_force_accum.copy_to_device();
    cudaMemcpyToSymbol(d_force_accum, &h_force_accum.d_prop, sizeof(d_force_accum));

    // force magnitude property (host-side, computed from h_force_accum after each step)
    Property<float> h_force_mag{n_cells, "force_mag"};
    thrust::fill(h_force_mag.h_prop, h_force_mag.h_prop + n_cells, 0.f);

    // state timer property, counts how long cell has been in current state, starts at 0 for all cells
    Property<int> h_state_timer{n_cells, "state_timer"};
    thrust::fill(h_state_timer.h_prop, h_state_timer.h_prop + n_cells, 0);
    
    // Resets neighbors and accumulated forces to 0, before each time step
    auto fun = [&](const int n, const float3* __restrict__ d_X, float3* d_dX) {
        thrust::fill(thrust::device, h_neig.d_prop, h_neig.d_prop + n, 0);
        thrust::fill(thrust::device, h_force_accum.d_prop, h_force_accum.d_prop + n, float3{0});
    };
    
    // Integrate cell positions
    Vtk_output output{"file"};

    // Randomly select cells to be activated
    bool random_cells = false;
    int number_of_random_cells = 200;
    int randomnumbers[number_of_random_cells] = {0};
    if (random_cells) {
        for (auto i = 0; i < number_of_random_cells; i++) randomnumbers[i] = rand() % n_cells;
    };

    // Time step loop
    for (auto time_step = 0; time_step <= n_time_steps; time_step++) {
        cells.copy_to_host();

        // Check if current timestep is in the activation list
        bool trigger_activation = false;
        for (int as : activation_steps) if (time_step == as) { trigger_activation = true; break; }

        if (trigger_activation){
            if (random_cells) {
                for (int i = 0; i < number_of_random_cells; i++) {
                    h_activated.h_prop[randomnumbers[i]] = 1;
                }
            } else {
                // Activate cells in a radius
                float3 spot_center = {10.f, -12.f, 0.f};  // Sphere where activation starts
                float spot_radius = 4.0f;
                for (int i = 0; i < n_cells; i++) {
                    float3 p = cells.h_X[i];
                    float dx = p.x - spot_center.x;
                    float dy = p.y - spot_center.y;
                    float dz = p.z - spot_center.z;
                    if (sqrtf(dx*dx + dy*dy + dz*dz) < spot_radius) {
                        h_activated.h_prop[i] = 1;
                    }
                }
                //h_activated.h_prop[86] = 1;
                //h_activated.h_prop[85] = 1;
                //h_activated.h_prop[107] = 1;
                //h_activated.h_prop[106] = 1;
                //h_activated.h_prop[105] = 1;
            }
            h_activated.copy_to_device();
        }

        // Update single cell state
        alter_cells_before(cells, h_radius, h_activated, h_force_mag, h_state_timer, n_cells);
        // Exude neighbor forces
        cells.take_step<simulation_step, friction_on_background>(dt, fun);

        h_neig.copy_to_host();
        h_activated.copy_to_host();
        h_radius.copy_to_host();
        h_force_accum.copy_to_host();

        for (int i = 0; i < n_cells; i++) {
            auto f = h_force_accum.h_prop[i];
            h_force_mag.h_prop[i] = sqrtf(f.x*f.x + f.y*f.y + f.z*f.z);
        }

        output.write_positions(cells);
        output.write_property(h_neig);
        output.write_property(h_activated);
        output.write_property(h_radius);
        output.write_property(h_force_mag);
    }

    return 0;
}
