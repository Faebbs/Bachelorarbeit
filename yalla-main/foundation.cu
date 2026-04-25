#include "./include/dtypes.cuh"
#include "./include/inits.cuh"
#include "./include/property.cuh"
#include "./include/solvers.cuh"
#include "./include/vtk.cuh"

const auto r_max = 3.f; // Maximum distance for interaction; Cutoff distance
const auto n_cells = 20u; 
const auto n_time_steps = 1000u;
const auto dt = 0.05; // Time step size

// Parameters for Morse potential
const auto r_e = 0.5f; // Equilibrium distance
const auto D_e = 2.0f; // Depth of the potential well
const auto alpha = 1.0f; // Width of the potential

__device__ int *d_neig;
__device__ int *d_activated;
__device__ int *d_prev_activated;

// Simulates Cell interaction
__device__ float3 simulation_step(
    float3 Xi, float3 r, float dist, int i, int j)
{
    float3 dF{0};
    if (i == j) return dF;

    if (dist > r_max) return dF;

    d_neig[i] += 1;

    // Activation, check if neighbor was activated last timestep
    // if (d_prev_activated[j] == 1 && d_activated[i] == 0) {
    //     d_activated[i] = 1;  // Activate cell i if neighbor j was activated previously
    // }
    switch(d_activated[i]){
        case 0:  // Activate cell i if neighbor j was activated previously
            if (d_prev_activated[j] == 1) {
                d_activated[i] = 1;  
            }
            break;
        case 1: // Refractory state
            d_activated[i] = 2;
            break;
        case 2: // Deactivated state
            d_activated[i] = 0;
    }
    // Calculation of the force which is exuded by cell j on cell i. 
    // Depends on the distance between the cells and the potential we choose.

    // Morse Potential: auto F = D_e * (expf(-2*alpha*(dist - r_e)) - 2*expf(-alpha*(dist - r_e)));
    // Kraft abgeleitet daraus:
    auto phi = expf(-alpha * (dist - r_e));
    auto F = -2.0f * D_e * alpha * (1.0f - phi) * phi;

    // Vector of force on cell i exerted by cell j
    dF = r * F / dist;
    return dF;
}

int main(int argc, const char* argv[])
{
    // Prepare initial state
    Solution<float3, Gabriel_solver> cells{n_cells, 50, r_max};
    //random_sphere(r_min, cells);
    regular_rectangle(r_e+0.4, 20, cells);

    // n_neighbor property
    cells.copy_to_device();
    Property<int> h_neig{n_cells, "neighbours"};
    cudaMemcpyToSymbol(d_neig, &h_neig.d_prop, sizeof(d_neig));
    h_neig.copy_to_device();

    // activated property: 0=deactivated, 1=activated, 2=refractory
    Property<int> h_activated{n_cells, "activated"};
    thrust::fill(h_activated.h_prop, h_activated.h_prop + n_cells, 0);
    h_activated.h_prop[0] = 1; 
    h_activated.copy_to_device();
    cudaMemcpyToSymbol(d_activated, &h_activated.d_prop, sizeof(d_activated));

    // previous activated property (starts all 0)
    Property<int> h_prev_activated{n_cells, "prev_activated"};
    thrust::fill(h_prev_activated.h_prop, h_prev_activated.h_prop + n_cells, 0);
    h_prev_activated.copy_to_device();
    cudaMemcpyToSymbol(d_prev_activated, &h_prev_activated.d_prop, sizeof(d_prev_activated));
    
    auto fun = [&](const int n, const float3* __restrict__ d_X, float3* d_dX) {
        thrust::fill(thrust::device, h_neig.d_prop, h_neig.d_prop + n, 0);
    };
    
    // Integrate cell positions
    Vtk_output output{"file"};
    for (auto time_step = 0; time_step <= n_time_steps; time_step++) {
        cells.copy_to_host();

        // Copy current activated state to previous before taking step
        cudaMemcpy(h_prev_activated.d_prop, h_activated.d_prop, n_cells * sizeof(int), cudaMemcpyDeviceToDevice);

        // Initialize first cell as activated
        // if (time_step == 1){
        //     h_activated.h_prop[0] = 1;
        //     std::cout << "Test" << std::endl;
        // };

        cells.take_step<simulation_step, friction_on_background>(dt, fun);

        h_neig.copy_to_host();
        h_activated.copy_to_host();


        output.write_positions(cells);
        output.write_property(h_neig);
        output.write_property(h_activated);
    }

    return 0;
}
