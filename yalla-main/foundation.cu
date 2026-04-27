#include "./include/dtypes.cuh"
#include "./include/inits.cuh"
#include "./include/property.cuh"
#include "./include/solvers.cuh"
#include "./include/vtk.cuh"

const auto r_max = 3.f; // Maximum distance for interaction; Cutoff distance
const auto n_cells = 20u; 
const auto n_time_steps = 500u;
const auto dt = 0.05; // Time step size

// Parameters for Morse potential
const auto r_e = 1.0f; // Equilibrium distance
const auto D_e = 2.0f; // Depth of the potential well
const auto alpha = 1.0f; // Width of the potential

const auto r_start = r_e / 2; // Resting radius
const auto r_activated = r_start / 2;   // Target radius when activated
const auto r_decay = 0.75f;        // Exponential decay factor (0 < r_decay < 1): smaller = faster

__device__ int *d_neig;
__device__ int *d_activated;
__device__ int *d_prev_activated;
__device__ float *d_radius;


// Stuff that happens once every time step
int alter_cells(Solution<float3, Gabriel_solver>& cells, Property<float>& h_radius, Property<int>& h_activated)
{
    for (int i = 0; i < n_cells; i++) {
        switch (h_activated.h_prop[i]){
            case 0:  // Deactivated state
                break;
            case 1: // Activated state
                h_radius.h_prop[i] = r_activated + (h_radius.h_prop[i] - r_activated) * expf(-r_decay) ;
                break;
            case 2: // Refractory state
                h_radius.h_prop[i] = r_start + (h_radius.h_prop[i] - r_start) * r_decay;
                break;
        }

    }

    h_radius.copy_to_device();

    return 0;
}

// Simulates Cell interaction, goes through all pairs of cells and calculates the force on cell i exerted by cell j
__device__ float3 simulation_step (float3 Xi, float3 r, float dist, int i, int j)
{
    float3 dF{0};
    if (i == j) return dF;

    if (dist > r_max) return dF;

    d_neig[i] += 1;

    // Activation, check if neighbor was activated last timestep
    // if (d_prev_activated[j] == 1 && d_activated[i] == 0) {
    //     d_activated[i] = 1;  // Activate cell i if neighbor j was activated previously
    // }
    // switch(d_activated[i]){
    //     case 0:  // Activate cell i if neighbor j was activated previously
    //         if (d_prev_activated[j] == 1) {
    //             d_activated[i] = 1;  
    //         }
    //         break;
    //     case 1: // Refractory state
    //         d_activated[i] = 2;
    //         break;
    //     case 2: // Deactivated state
    //         d_activated[i] = 0;
    // }
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
    return dF;
}

int main(int argc, const char* argv[])
{
    printf("Cell resting radius: %f\n", r_start);
    printf("Cell activated radius: %f\n", r_activated);
    // Prepare initial state
    Solution<float3, Gabriel_solver> cells{n_cells, 50, r_max};
    //random_sphere(r_min, cells);
    regular_rectangle(2.0f*r_e, 20, cells);

    // n_neighbor property
    cells.copy_to_device();
    Property<int> h_neig{n_cells, "neighbours"};
    cudaMemcpyToSymbol(d_neig, &h_neig.d_prop, sizeof(d_neig));
    h_neig.copy_to_device();

    // activated property: 0=deactivated, 1=activated, 2=refractory
    Property<int> h_activated{n_cells, "activated"};
    thrust::fill(h_activated.h_prop, h_activated.h_prop + n_cells, 0);
    h_activated.copy_to_device();
    cudaMemcpyToSymbol(d_activated, &h_activated.d_prop, sizeof(d_activated));

    Property<float> h_radius{n_cells, "radius"};
    thrust::fill(h_radius.h_prop, h_radius.h_prop + n_cells, r_start);  // Standard-Radius, equally for all cells at half of equilibrium distance
    h_radius.copy_to_device();
    cudaMemcpyToSymbol(d_radius, &h_radius.d_prop, sizeof(d_radius));

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
        // Copy current activated state to previous before taking step
        cudaMemcpy(h_prev_activated.d_prop, h_activated.d_prop, n_cells * sizeof(int), cudaMemcpyDeviceToDevice);

        // Initialize first cell as activated
        if (time_step == 10){
            if (random_cells) {
                for (int i = 0; i < number_of_random_cells; i++) {
                    h_activated.h_prop[randomnumbers[i]] = 1;
                }
            } else {
                h_activated.h_prop[1] = 1;
                h_activated.h_prop[7] = 1;
            }

            h_activated.copy_to_device();
        } else if (time_step == 50)
        {
            h_activated.h_prop[1] = 2;
            h_activated.h_prop[7] = 2;
            h_activated.copy_to_device();
        }
        
        // Update single cell state
        alter_cells(cells, h_radius, h_activated);
        // Exude neighbor forces
        cells.take_step<simulation_step, friction_on_background>(dt, fun);

        h_neig.copy_to_host();
        h_activated.copy_to_host();
        h_radius.copy_to_host();

        

        output.write_positions(cells);
        output.write_property(h_neig);
        output.write_property(h_activated);
        output.write_property(h_radius);
    }

    return 0;
}
