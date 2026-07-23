#include "./include/dtypes.cuh"
#include "./include/inits.cuh"
#include "./include/property.cuh"
#include "./include/solvers.cuh"
#include "./include/vtk.cuh"
#include <algorithm>
#include <cstring>
#include <fstream>
#include <utility>
#include <vector>

// ============================ Model selection ============================
// Choose the geometry by defining exactly ONE of the macros below.
//   MODEL_EGG   : 3D egg surface loaded from VTK, cells pinned to the surface
//   MODEL_SHEET : flat 2D sheet projected onto a sphere-cap height field
//                 (curved center, flat outside the equator; weak-curvature case)
//   MODEL_CAP   : sheet mapped onto a sphere cap via the exponential map
//                 (uniform curvature everywhere, fixed geodesic patch size)
#define MODEL_EGG
//#define MODEL_SHEET
//#define MODEL_CAP

#if (defined(MODEL_EGG) + defined(MODEL_SHEET) + defined(MODEL_CAP)) != 1
#error "Define exactly one of MODEL_EGG, MODEL_SHEET or MODEL_CAP"
#endif

// Cell sheet dimensions (only used for MODEL_SHEET)
const auto sheet_nx = 20u;  // cells per row
const auto sheet_ny = 25u;  // number of rows
// ========================================================================

//const auto r_max = 8.f; // Maximum distance for interaction; Cutoff distance
const auto r_max = 2.f; // Maximum distance for interaction; Cutoff distance
const auto n_time_steps = 1000u;
const auto dt = 0.05; // Time step size

// Which force?
#ifdef MODEL_EGG
const char force_type = 'l'; // egg: capped linear force (repulsion/adhesion, no long-range cohesion); 'm' Morse
#else
const char force_type = 'm'; // (l)inear or (m)orse
#endif

// Parameters for Morse potential
#ifdef MODEL_EGG
const auto r_e = 1.111f; // Egg: cell radius r_start = r_e/2 = 0.556; equilibrium centre-distance 2*r_start = 1.111 = Joern's mean neighbour spacing
#else
const auto r_e = 1.0f; // Equilibrium distance, Sheet
#endif
const auto D_e = 2.0f; // Depth of the potential well
const auto alpha = 1.0f; // Width of the potential

// Coefficients for the linear (Joern-style) pairwise force that currently
// replaces the Morse force in simulation_step (repulsion / adhesion slopes).
const auto lin_repulsion = 1.0f;
const auto lin_adhesion  = 1.0f;

float r_start = r_e / 2;         // Resting radius (egg: set from the global spacing in main)
float r_activated = r_start / 2; // Target radius when activated (follows r_start)

// Per-cell radii (egg): each cell's rest radius is derived from its local neighbour
// distances (~half the mean nearest-neighbour spacing) so cells reach up to their
// neighbours' boundaries and fill the surface, even where packing is sparse. The
// state machine shrinks/grows each cell relative to its OWN radius. Sheet/cap fill
// these uniformly from the globals above.
std::vector<float> pc_r_start;        // per-cell resting radius
std::vector<float> pc_r_activated;    // per-cell contracted radius
const float egg_shrink_factor = 0.5f; // activated = shrink_factor * rest (0.1 = to a tenth)
// Render-only overlap: the extra VTK field "render_radius" is written as
// egg_render_overlap * the physics radius so drawn spheres OVERLAP and fill the
// interstitial gaps that just-touching circles always leave (~9% by geometry).
// Colour/size by "render_radius" in Vedo. Does NOT affect the simulation.
const float egg_render_overlap = 1.4f;

const auto r_decay_shrink = 5.0f;            // Exponential decay factor (0 < r_decay < 1): smaller = slower
const auto r_decay_grow = 0.1f;

const auto activation_delay = 2;  // Timesteps in delay state, before going to activated state
const auto activation_duration = 10;  // Timesteps in activated state, before refreactory state
const auto refractory_duration = 200;  // Timesteps in refractory state, before going back to deactivated state

float force_threshold = 0.15f;  // Half-activation force (K in Hill function); overridable via argv[4]
const auto use_hill_function = false;  // Whether to use Hill function for activation probability instead of hard threshold
const auto n_hill = 2.0f;            // Hill coefficient: steepness (higher = closer to hard threshold)

// Timesteps at which activation is triggered
int activation_steps[] = {220};

// Initial activation seed: activate the nearest cells to the seed point. Given
// as a FRACTION of the total cell count so the stimulus scales with the
// (R-dependent) patch size and stays comparable across the curvature scan.
const auto activation_fraction = 0.01f;  // 5 % of all cells
const auto activation_min = 3;           // floor for very small (high-curvature) caps

// Stiffness of the force that pins cells onto the egg surface
#ifdef MODEL_EGG
const auto surface_stiffness = 3.0f;   // soft pinning (Joern's value) so cells glide freely on the surface
#else
const auto surface_stiffness = 12.0f;
#endif

// EGG cell size (single uniform type). Radius r_start = r_e/2 = 0.556 so the
// equilibrium centre-distance (2*r_start, with d_r_e = 0) equals Joern's relaxed
// mean neighbour spacing 1.111. The cells start at the dense input spacing (~0.54)
// and spread out under the linear force to cover the whole egg.
const int   egg_relax_steps = 500;            // settling steps performed by the 'relax' mode (writes egg_cells_relaxed.vtk and exits). Not part of the wave run, so no frame offset.
// Interaction cutoff: keeps ~6 neighbours interacting and no long-range cohesion.
// Grid cube_size stays r_max, this only gates forces.
const float egg_r_cut = 1.5f;

// CAP model: fixed angular half-extent of the spherical cap. The cell count is
// derived from this and R (N proportional to R^2), so the curvature scan stays
// 1-D in R (no extra free parameter) and the patch never wraps past the equator
// (keep <= 90 deg).
const auto cap_half_angle_deg = 75.0f;

__device__ int *d_neig;
__device__ int *d_activated;
__device__ int *d_prev_activated;
__device__ float *d_radius;
__device__ float3 *d_force_accum;
__device__ int *d_active_neighbor;  // per cell: 1 if a neighbour is currently in state 1
__device__ float d_r_e;  // force equilibrium (rest surface_dist); egg uses 0 (surface contact), sheet/cap use r_e
__device__ float d_r_cut; // interaction cutoff (egg uses a short range to avoid long-range Morse clumping)

// Pins every cell onto a static surface (frozen copy of the egg mesh).
// For each cell the nearest surface grid point is found and a restoring force
// along that point's normal is applied. This removes the displacement component
// perpendicular to the surface while leaving tangential (sliding) motion free.
__global__ void pin_to_surface_kernel(
    const int n,
    const float3* __restrict__ d_X,
    float3* d_dX,
    const int n_grid,
    const float3* __restrict__ d_gridX,
    const float3* __restrict__ d_normX)
{
    auto i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float3 Xi = d_X[i];

    // Find nearest surface grid point
    int idx = 0;
    float min_d2 = 3.4e38f;
    for (int e = 0; e < n_grid; e++) {
        float3 D = Xi - d_gridX[e];
        float d2 = D.x * D.x + D.y * D.y + D.z * D.z;
        if (d2 < min_d2) {
            min_d2 = d2;
            idx = e;
        }
    }

    // Restoring force: remove the component of the offset along the normal
    float3 r = Xi - d_gridX[idx];
    float3 nrm = d_normX[idx];
    float proj = r.x * nrm.x + r.y * nrm.y + r.z * nrm.z;
    float F = -surface_stiffness * proj;
    d_dX[i] += F * nrm;
}

void pin_to_surface(
    const int n,
    const float3* __restrict__ d_X,
    float3* d_dX,
    const int n_grid,
    const float3* __restrict__ d_gridX,
    const float3* __restrict__ d_normX)
{
    pin_to_surface_kernel<<<(n + 32 - 1) / 32, 32>>>(
        n, d_X, d_dX, n_grid, d_gridX, d_normX);
}

#if defined(MODEL_SHEET) || defined(MODEL_CAP)
// Patch center (pole of the sphere cap), shared by the sheet and cap models.
__device__ __host__ float sheet_center_x() { return sheet_nx * r_e; }
__device__ __host__ float sheet_center_y() { return sheet_ny * r_e * 0.87f; }
#endif

#ifdef MODEL_SHEET
// --- Curved surface via height field: spherical cap ---
// The sheet (a rectangle in x/y) is wrapped onto a sphere of radius R whose
// pole sits at the sheet center, giving constant Gaussian curvature 1/R^2.
// Height field z = R - sqrt(R^2 - rho^2), rho = distance from the sheet center.
// Note: beyond the equator (rho > R) the surface is clamped flat.
__device__ __host__ float sheet_height(float x, float y, float R)
{
    float dx = x - sheet_center_x();
    float dy = y - sheet_center_y();
    float arg = R * R - (dx * dx + dy * dy);
    if (arg < 0.f) arg = 0.f;  // guard: outside the sphere -> clamp at equator
    return R - sqrtf(arg);
}

// Unit surface normal via finite differences of the height field (general:
// works for any sheet_height without deriving gradients by hand).
__device__ __host__ float3 sheet_normal(float x, float y, float R)
{
    const float h = 1e-3f;
    float fx = (sheet_height(x + h, y, R) - sheet_height(x - h, y, R)) / (2.f * h);
    float fy = (sheet_height(x, y + h, R) - sheet_height(x, y - h, R)) / (2.f * h);
    float inv = 1.f / sqrtf(fx * fx + fy * fy + 1.f);
    return {-fx * inv, -fy * inv, inv};
}

// Pins each cell onto the analytic curved surface: restoring force along the
// local normal, removing the perpendicular offset while leaving tangential
// (sliding) motion on the surface free.
__global__ void pin_to_sheet_kernel(
    const int n, const float3* __restrict__ d_X, float3* d_dX, const float R)
{
    auto i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float3 P = d_X[i];
    float z_s = sheet_height(P.x, P.y, R);
    float3 nrm = sheet_normal(P.x, P.y, R);
    // Vertical offset from the surface, projected onto the normal
    float proj = (P.z - z_s) * nrm.z;
    float F = -surface_stiffness * proj;
    d_dX[i] += F * nrm;
}

void pin_to_sheet(const int n, const float3* __restrict__ d_X, float3* d_dX,
    const float R)
{
    pin_to_sheet_kernel<<<(n + 32 - 1) / 32, 32>>>(n, d_X, d_dX, R);
}
#endif

#ifdef MODEL_CAP
// --- Uniformly curved spherical cap via the exponential map ---
// Cells are placed on a sphere of radius R: the flat layout distance from the
// patch center is taken as the GEODESIC distance from the pole. So the whole
// patch is curved with constant Gaussian curvature 1/R^2 (no flat region), and
// the geodesic size of the patch is identical for every R - only the curvature
// changes. The surface is a true sphere centered at C = (cx, cy, R), so pinning
// is an exact radial projection.

// Map a flat layout coordinate (tangent-plane / geodesic polar coordinates
// around the patch center) onto the sphere cap via the exponential map.
__host__ float3 place_on_cap(float x_flat, float y_flat, float R)
{
    float dx = x_flat - sheet_center_x();
    float dy = y_flat - sheet_center_y();
    float s = sqrtf(dx * dx + dy * dy);   // geodesic distance from the pole
    float theta = s / R;                  // polar angle on the sphere
    float phi = (s > 1e-6f) ? atan2f(dy, dx) : 0.f;
    float3 P;
    P.x = sheet_center_x() + R * sinf(theta) * cosf(phi);
    P.y = sheet_center_y() + R * sinf(theta) * sinf(phi);
    P.z = R * (1.f - cosf(theta));        // pole at z = 0, rising outward
    return P;
}

// Pins each cell onto the sphere of radius R centered at (cx, cy, R): exact
// radial restoring force toward the surface (= along the sphere normal),
// leaving tangential sliding on the surface free.
__global__ void pin_to_cap_kernel(
    const int n, const float3* __restrict__ d_X, float3* d_dX, const float R)
{
    auto i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float3 C{sheet_center_x(), sheet_center_y(), R};
    float3 d = d_X[i] - C;
    float len = norm3df(d.x, d.y, d.z);
    if (len < 1e-6f) return;
    float3 nrm = d / len;                  // outward sphere normal
    float proj = len - R;                  // signed distance to the sphere
    d_dX[i] += (-surface_stiffness * proj) * nrm;
}

void pin_to_cap(const int n, const float3* __restrict__ d_X, float3* d_dX,
    const float R)
{
    pin_to_cap_kernel<<<(n + 32 - 1) / 32, 32>>>(n, d_X, d_dX, R);
}
#endif

// Per-cell rest radius from local neighbour distances: half the mean distance to
// the kmax nearest neighbours within cutoff. Anchors each cell's size at its
// neighbours' boundaries (Joern-style variable radius) so the sheet has no gaps.
std::vector<float> compute_cell_radii(const float3* X, int n, float cutoff, int kmax)
{
    std::vector<float> rad(n);
    for (int i = 0; i < n; i++) {
        std::vector<float> d;
        for (int j = 0; j < n; j++) {
            if (j == i) continue;
            float dx = X[i].x - X[j].x, dy = X[i].y - X[j].y, dz = X[i].z - X[j].z;
            float dd = sqrtf(dx * dx + dy * dy + dz * dz);
            if (dd < cutoff) d.push_back(dd);
        }
        std::sort(d.begin(), d.end());
        int k = (int)d.size() < kmax ? (int)d.size() : kmax;
        float m = 0.f;
        for (int t = 0; t < k; t++) m += d[t];
        rad[i] = (k > 0) ? 0.5f * m / k : 0.5f * cutoff;
    }
    return rad;
}

// Stuff that happens once every time step before time step is taken
int alter_cells_before(Solution<float3, Gabriel_solver>& cells, Property<float>& h_radius,
                Property<int>& h_activated, Property<float>& h_force_mag,
                Property<int>& h_state_timer, Property<int>& h_active_neighbor,
                bool require_active_neighbor, int n_cells)
{
    // Cell state update
    for (int i = 0; i < n_cells; i++) {
        switch (h_activated.h_prop[i]){
            case 0: {  // Deactivated: Hill-function activation probability
                // In scan mode require a currently contracting neighbour, so only
                // wave-propagated (not spontaneous) activation is allowed.
                bool neighbour_ok =
                    !require_active_neighbor || h_active_neighbor.h_prop[i] == 1;
                if (use_hill_function){
                    //Hill function with stochastic
                    float F = h_force_mag.h_prop[i];
                    float Fn = powf(F, n_hill);
                    float Kn = powf(force_threshold, n_hill);
                    float P = Fn / (Kn + Fn);
                    if (neighbour_ok && (float)rand() / RAND_MAX < P) {
                        h_activated.h_prop[i] = 7;
                    }
                } else {
                    // Hard threshold
                    if (neighbour_ok && h_force_mag.h_prop[i] > force_threshold) {
                        h_activated.h_prop[i] = 7;
                    }
                }

                break;
            }
            case 1: // Activated state: radius shrinks toward this cell's own target
                h_radius.h_prop[i] = pc_r_activated[i] + (h_radius.h_prop[i] - pc_r_activated[i]) * expf(-r_decay_shrink);
                // Timer for how long cell is in state 1, goes to 2 after defined duration
                h_state_timer.h_prop[i]++;
                if (h_state_timer.h_prop[i] >= activation_duration) {
                    h_activated.h_prop[i] = 2;
                    h_state_timer.h_prop[i] = 0;
                }
                break;
            case 2: // Refractory state: radius grows back to this cell's own rest radius
                h_radius.h_prop[i] = pc_r_start[i] + (h_radius.h_prop[i] - pc_r_start[i]) * expf(-r_decay_grow);
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

    if (dist > d_r_cut) return dF;

    d_neig[i] += 1;

    // Mark cell i if a neighbour j is currently contracting (state 1). Used in
    // scan mode to allow only wave-propagated activation (no spontaneous firing).
    if (d_activated[j] == 1) d_active_neighbor[i] = 1;

    // Calculation of the force which is exuded by cell j on cell i. 
    // Depends on the distance between the cells and the potential we choose.

    // Radius of cells
    auto r_i = d_radius[i];
    auto r_j = d_radius[j];
    auto surface_dist = dist - (r_i + r_j);

    double F;
    switch (force_type) {
        case 'm':  // Morse potential
            // --- Morse force ---
            // Morse Potential: auto F = D_e * (expf(-2*alpha*(dist - r_e)) - 2*expf(-alpha*(dist - r_e)));
            // Kraft abgeleitet daraus:
            auto phi = expf(-alpha * (surface_dist - d_r_e));
            F = -2.0f * D_e * alpha * (1.0f - phi) * phi;
            break;
        case 'l':  // Linear force
            // --- Linear force ---
            auto repulsion = lin_repulsion * fmaxf(0.0f, d_r_e - surface_dist);
            auto adhesion  = lin_adhesion  * fmaxf(0.0f, surface_dist - d_r_e);
            F = repulsion - adhesion;
            break;
        default:
            // Unknown force type
            break;
    }

    // Vector of force on cell i exerted by cell j
    dF = r * F / dist;
    // Accumulate forces from all neighbors (atomic to avoid race conditions between threads)
    atomicAdd(&d_force_accum[i].x, dF.x);
    atomicAdd(&d_force_accum[i].y, dF.y);
    atomicAdd(&d_force_accum[i].z, dF.z);
    return dF;
}

// Writes cell positions as a minimal legacy-VTK POLYDATA point cloud (POINTS +
// VERTICES), readable by Vtk_input::read_positions. Used to persist the relaxed
// egg cell sheet so a later wave run can start from the settled configuration.
void write_positions_vtk(const std::string& fn, const float3* X, int n)
{
    std::ofstream f(fn);
    f << "# vtk DataFile Version 3.0\nrelaxed egg cells\nASCII\nDATASET POLYDATA\n";
    f << "POINTS " << n << " float\n";
    for (int i = 0; i < n; i++) f << X[i].x << ' ' << X[i].y << ' ' << X[i].z << '\n';
    f << "VERTICES " << n << ' ' << 2 * n << '\n';
    for (int i = 0; i < n; i++) f << "1 " << i << '\n';
}

int main(int argc, const char* argv[])
{
    // Command line:  ./a.out <R> <seed> <mode> <threshold>
    //   <R>         sphere-cap radius (small R = strong curvature)   [default 15]
    //   <seed>      random seed for the cell layout                  [default 42]
    //   <mode>      "vtk"      -> VTK files, spontaneous activation (full model) [default]
    //               "scan"     -> no VTK, SCAN_RESULT metrics, propagation-only
    //               "vtk_prop" -> VTK files, propagation-only (visualise the wave)
    //   <threshold> excitability (Hill half-activation force)        [default 0.15]
    float surface_R = 15.0f;
    if (argc > 1) surface_R = atof(argv[1]);

    unsigned int random_seed = 42;
    if (argc > 2) random_seed = (unsigned int)atoi(argv[2]);
    srand(random_seed);

    const char* mode = (argc > 3) ? argv[3] : "vtk";
    bool scan_mode = (strcmp(mode, "scan") == 0);  // suppress VTK, print metrics
    // Relax mode (egg only): load the raw cells, settle them, write the relaxed
    // positions to egg_cells_relaxed.vtk and exit. The normal run then loads that
    // pre-relaxed file, so the wave starts on a settled sheet (no frame offset,
    // no spontaneous pre-firing). Driven as a first step by main.py.
    bool relax_mode = (strcmp(mode, "relax") == 0);
    // Propagation-only: a cell may only activate if a neighbour is contracting
    // (no spontaneous firing). On for the scan and for the "vtk_prop" viz mode.
    bool require_active_neighbor = scan_mode || (strcmp(mode, "vtk_prop") == 0);

    if (argc > 4) force_threshold = atof(argv[4]);

    // Optional: activation seed size as a FRACTION of all cells (argv[5]) and
    // the seed / activation point (argv[6..8]). Used by the egg activation scan
    // to sweep the stimulus size and move the seed to different curvature spots.
    float act_fraction = activation_fraction;
    if (argc > 5) act_fraction = atof(argv[5]);

    bool seed_override = false;
    float3 seed_cli{0.f, 0.f, 0.f};
    if (argc > 8) {
        seed_cli = {(float)atof(argv[6]), (float)atof(argv[7]), (float)atof(argv[8])};
        seed_override = true;
    }

    printf("Config: R=%g  seed=%u  mode=%s  threshold=%g  "
           "(VTK %s, %s activation)\n",
        surface_R, random_seed, mode, force_threshold,
        scan_mode ? "off" : "on",
        require_active_neighbor ? "propagation-only" : "spontaneous");

#ifdef MODEL_EGG
    // ----- Egg: cells and pinning surface loaded from SEPARATE files -----
    // Cells: raw biological set (horizontal_egg, ~3000 pts) in RELAX mode, or the
    // pre-relaxed egg_cells_relaxed.vtk (produced by a relax run) for the wave
    // run. Pinning surface = the finer target surface (~33164 pts). All in the
    // same coordinate frame, so cells start on the surface (pin force ~ 0).
    const char* raw_cells = "Joern_Projekt/initial_data/dr1p75_0.vtk";
    const char* relaxed_cells = "egg_cells_relaxed.vtk";
    const char* cell_file = raw_cells;
    if (!relax_mode) {
        std::ifstream test(relaxed_cells);
        if (test.good()) cell_file = relaxed_cells;
        else printf("WARNING: %s not found - using raw cells. Run mode 'relax' first.\n", relaxed_cells);
    }
    printf("Cells loaded from: %s\n", cell_file);
    Vtk_input input{cell_file};
    const int n_cells = input.n_points;
    Solution<float3, Gabriel_solver> cells{n_cells, 50, r_max};
    input.read_positions(cells);

    // --- Static surface for pinning: the fine egg mesh (never integrated) ---
    Vtk_input surf_input{"Joern_Projekt/initial_data/surfaces/target_surface_tribolium_1.vtk"};
    const int n_grid = surf_input.n_points;
    Solution<float3, Tile_solver> surface{n_grid};
    surf_input.read_positions(surface);
    surface.copy_to_device();
    Solution<float3, Tile_solver> surface_norm{n_grid};
    surf_input.read_normals(surface_norm);
    surface_norm.copy_to_device();

    printf("Egg: %d cells (horizontal_egg) + %d surface pts (target_surface)\n",
        n_cells, n_grid);
    // Radius comes straight from r_start = r_e/2 = 1.111 (no override).
#elif defined(MODEL_SHEET)
    // ----- Cell sheet: flat rectangle projected onto a sphere-cap height field -----
    const int n_cells = sheet_nx * sheet_ny;
    printf("Generated %d cells (sheet model), sphere-cap R = %f (curvature %f)\n",
        n_cells, surface_R, 1.0f / (surface_R * surface_R));

    Solution<float3, Gabriel_solver> cells{n_cells, 50, r_max};
    regular_rectangle(2.0f * r_e, sheet_nx, cells); // spacing = equilibrium distance
    // Lift the flat sheet onto the curved surface so cells start on it
    for (int i = 0; i < n_cells; i++) {
        cells.h_X[i].z = sheet_height(cells.h_X[i].x, cells.h_X[i].y, surface_R);
    }
#else
    // ----- Spherical cap: uniform curvature via the exponential map -----
    // Couple the cell count to R via the fixed angular extent (cap_half_angle_deg):
    // N = 0.9069 * (R * theta_max / r_e)^2, so every cap covers the same cone and
    // the scan stays 1-D in R. random_disk reproduces this angular extent because
    // it derives its disk radius from N with the same packing density.
    const float theta_max = cap_half_angle_deg * 0.0174532925f;  // deg -> rad
    int n_tmp = (int)lroundf(0.9069f * powf(surface_R * theta_max / r_e, 2.f));
    const int n_cells = n_tmp < 20 ? 20 : n_tmp;  // floor for very small caps

    // The neighbour grid must span the whole cap. The pole sits at
    // (cx, cy, 0) and cells reach out to ~R from it, so size grid_size (the
    // grid extent is +/- grid_size*cube_size/2 = +/- grid_size) accordingly.
    int cap_extent = (int)(sheet_ny * r_e * 0.87f + surface_R) + 12;
    int grid_size = cap_extent < 50 ? 50 : cap_extent;
    Solution<float3, Gabriel_solver> cells{n_cells, grid_size, r_max};
    random_disk(2.0f * r_e, cells); // random circular patch (in the y-z plane)
    // Map the disk onto the sphere cap; track the geodesic patch radius.
    // random_disk places the disk in the y-z plane (x=0), so its (y,z) serve
    // as the tangent-plane coordinates around the pole.
    float s_max = 0.f;
    for (int i = 0; i < n_cells; i++) {
        float u = cells.h_X[i].y;  // tangent-plane coordinates from random_disk
        float v = cells.h_X[i].z;
        float s = sqrtf(u * u + v * v);
        if (s > s_max) s_max = s;
        cells.h_X[i] = place_on_cap(
            sheet_center_x() + u, sheet_center_y() + v, surface_R);
    }
    float theta_max_deg = s_max / surface_R * 57.29578f;
    printf("Generated %d cells (cap model), R = %f (curvature %f)\n",
        n_cells, surface_R, 1.0f / (surface_R * surface_R));
    printf("Patch geodesic radius = %.2f, angular extent = %.1f deg "
           "(90 = hemisphere)\n", s_max, theta_max_deg);
    if (theta_max_deg > 180.f)
        printf("WARNING: patch wraps past the south pole - R too small!\n");
#endif

    printf("Cell resting radius: %f\n", r_start);
    printf("Cell activated radius: %f\n", r_activated);
    printf("Force threshold for activation: %f\n", force_threshold);

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

    // Per-cell radii: uniform by default; for the egg WAVE run (not relax) derive
    // each cell's rest radius from its local neighbour distances so cells fill up to
    // their neighbours' boundaries (variable radius, Joern-style). Relax keeps a
    // uniform radius so cells first spread out to the target spacing.
    pc_r_start.assign(n_cells, r_start);
    pc_r_activated.assign(n_cells, r_activated);
#ifdef MODEL_EGG
    if (!relax_mode) {
        pc_r_start = compute_cell_radii(cells.h_X, n_cells, egg_r_cut, 6);
        for (int i = 0; i < n_cells; i++)
            pc_r_activated[i] = pc_r_start[i] * egg_shrink_factor;
        float rmin = pc_r_start[0], rmax = pc_r_start[0], rmean = 0.f;
        for (int i = 0; i < n_cells; i++) {
            rmin = fminf(rmin, pc_r_start[i]);
            rmax = fmaxf(rmax, pc_r_start[i]);
            rmean += pc_r_start[i];
        }
        printf("Per-cell rest radius: min=%.3f mean=%.3f max=%.3f (shrink factor %.2f)\n",
            rmin, rmean / n_cells, rmax, egg_shrink_factor);
    }
#endif

    // radius property, initialised to each cell's own rest radius
    Property<float> h_radius{n_cells, "radius"};
    for (int i = 0; i < n_cells; i++) h_radius.h_prop[i] = pc_r_start[i];
    h_radius.copy_to_device();
    cudaMemcpyToSymbol(d_radius, &h_radius.d_prop, sizeof(d_radius));

    // Render-only radius (VTK "render_radius"): overlap-scaled copy of the physics
    // radius so drawn spheres fill the surface. Physics uses h_radius unchanged.
    float render_overlap = 1.0f;
#ifdef MODEL_EGG
    render_overlap = egg_render_overlap;
#endif
    Property<float> h_render_radius{n_cells, "render_radius"};

    // accumulated force vector property (device-side accumulation)
    Property<float3> h_force_accum{n_cells, "force_accum"};
    thrust::fill(h_force_accum.h_prop, h_force_accum.h_prop + n_cells, float3{0});
    h_force_accum.copy_to_device();
    cudaMemcpyToSymbol(d_force_accum, &h_force_accum.d_prop, sizeof(d_force_accum));

    // force magnitude property (host-side, computed from h_force_accum after each step)
    Property<float> h_force_mag{n_cells, "force_mag"};
    thrust::fill(h_force_mag.h_prop, h_force_mag.h_prop + n_cells, 0.f);

    // active-neighbour flag: 1 if a neighbour is currently in state 1 (device-side)
    Property<int> h_active_neighbor{n_cells, "active_neighbor"};
    thrust::fill(h_active_neighbor.h_prop, h_active_neighbor.h_prop + n_cells, 0);
    h_active_neighbor.copy_to_device();
    cudaMemcpyToSymbol(d_active_neighbor, &h_active_neighbor.d_prop, sizeof(d_active_neighbor));

    // state timer property, counts how long cell has been in current state, starts at 0 for all cells
    Property<int> h_state_timer{n_cells, "state_timer"};
    thrust::fill(h_state_timer.h_prop, h_state_timer.h_prop + n_cells, 0);

    // Resets neighbors and accumulated forces to 0, before each time step
    auto fun = [&](const int n, const float3* __restrict__ d_X, float3* d_dX) {
        thrust::fill(thrust::device, h_neig.d_prop, h_neig.d_prop + n, 0);
        thrust::fill(thrust::device, h_force_accum.d_prop, h_force_accum.d_prop + n, float3{0});
        thrust::fill(thrust::device, h_active_neighbor.d_prop, h_active_neighbor.d_prop + n, 0);
#if defined(MODEL_EGG)
        // Pin cells onto the egg surface (frozen mesh)
        pin_to_surface(n, d_X, d_dX, n_grid, surface.d_X, surface_norm.d_X);
#elif defined(MODEL_SHEET)
        // Pin cells onto the sphere-cap height field
        pin_to_sheet(n, d_X, d_dX, surface_R);
#else
        // Pin cells onto the uniformly curved spherical cap
        pin_to_cap(n, d_X, d_dX, surface_R);
#endif
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

    // Seed location for activation (= pole of the cap / center of the sheet,
    // or the fixed spot on the egg). Also used as the patch center for metrics.
#ifdef MODEL_EGG
    float3 seed_center = {-9.f, -12.f, 0.f};
#else
    float3 seed_center = {sheet_nx * r_e, sheet_ny * r_e * 0.87f, 0.f};
#endif
    if (seed_override) seed_center = seed_cli;

    // --- Scan metrics ---
    std::vector<char> ever_activated(n_cells, 0);  // cells that fired at least once
    float avg_neighbors_relaxed = -1.f;            // interior avg neighbours pre-wave
    int n_interior = 0;
    int n_seed = 0;                  // actual number of cells seeded at the trigger
    int n_early = -1;                // cells fired within early_window after trigger
    const int early_window = 50;     // timesteps after trigger for the early count
    int peak_active = 0;             // max #cells SIMULTANEOUSLY in state 1 (front size)
    int t_peak = -1;                 // timestep at which that peak occurs

    // Interaction cutoff + force equilibrium (surface_dist at rest).
#ifdef MODEL_EGG
    const float nb_spacing = 1.111f;   // ~neighbour spacing (neighbour-count metric only)
    // Equilibrium at surface contact (d_r_e = 0): rest centre-distance = 2*r_start
    // = 1.111. The capped linear force + cutoff give ~6 neighbours and no clumping.
    { float rc = egg_r_cut; cudaMemcpyToSymbol(d_r_cut, &rc, sizeof(float)); }
    { float d = 0.f;        cudaMemcpyToSymbol(d_r_e,   &d,  sizeof(float)); }

    // RELAX MODE: settle the raw cells (forces + pinning only, no activation),
    // write the relaxed positions to egg_cells_relaxed.vtk, then exit. The wave
    // run (any other mode) skips this - it already loaded the relaxed cells.
    if (relax_mode) {
        printf("Relaxing %d cells for %d steps ...\n", n_cells, egg_relax_steps);
        for (int rs = 0; rs < egg_relax_steps; rs++)
            cells.take_step<simulation_step, friction_on_background>(dt, fun);
        cells.copy_to_host();
        write_positions_vtk("egg_cells_relaxed.vtk", cells.h_X, n_cells);
        printf("Wrote egg_cells_relaxed.vtk (%d cells). Done.\n", n_cells);
        return 0;
    }
#else
    const float nb_spacing = 2.f * r_e;   // regular layouts
    { float rc = r_max; cudaMemcpyToSymbol(d_r_cut, &rc, sizeof(float)); }
    { float d = r_e;    cudaMemcpyToSymbol(d_r_e,   &d,  sizeof(float)); }
#endif

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
                // Activate a fixed FRACTION of cells around the seed point, so
                // the stimulus scales with the (R-dependent) patch size and is
                // comparable across the curvature scan.
                n_seed = (int)lroundf(act_fraction * n_cells);
                if (n_seed < activation_min) n_seed = activation_min;
                if (n_seed > n_cells) n_seed = n_cells;

                // Activate the n_seed cells nearest to the seed point
                std::vector<std::pair<float, int>> dist(n_cells);
                for (int i = 0; i < n_cells; i++) {
                    float dx = cells.h_X[i].x - seed_center.x;
                    float dy = cells.h_X[i].y - seed_center.y;
                    float dz = cells.h_X[i].z - seed_center.z;
                    dist[i] = {dx * dx + dy * dy + dz * dz, i};
                }
                std::nth_element(dist.begin(), dist.begin() + n_seed, dist.end());
                for (int k = 0; k < n_seed; k++)
                    h_activated.h_prop[dist[k].second] = 1;
                printf("t=%d: activated %d / %d cells (%.1f%%)\n", time_step,
                    n_seed, n_cells, 100.f * n_seed / n_cells);
            }
            h_activated.copy_to_device();
        }

        // Update single cell state (propagation-only when require_active_neighbor)
        alter_cells_before(cells, h_radius, h_activated, h_force_mag, h_state_timer,
            h_active_neighbor, require_active_neighbor, n_cells);
        // Exude neighbor forces
        cells.take_step<simulation_step, friction_on_background>(dt, fun);

        h_neig.copy_to_host();
        h_activated.copy_to_host();
        h_radius.copy_to_host();
        h_force_accum.copy_to_host();
        h_active_neighbor.copy_to_host();

        for (int i = 0; i < n_cells; i++) {
            auto f = h_force_accum.h_prop[i];
            h_force_mag.h_prop[i] = sqrtf(f.x*f.x + f.y*f.y + f.z*f.z);
        }

        // Track (a) cells that have fired at least once (cumulative reach) and
        // (b) cells SIMULTANEOUSLY contracting right now (instantaneous wave-
        // front size). On a closed surface the cumulative count saturates at ~1,
        // so the peak simultaneous count is the curvature-sensitive observable.
        int now_active = 0;
        for (int i = 0; i < n_cells; i++)
            if (h_activated.h_prop[i] == 1) { ever_activated[i] = 1; now_active++; }
        if (now_active > peak_active) { peak_active = now_active; t_peak = time_step; }

        // Local early-spread metric: how many cells have fired a fixed window
        // after the trigger. Distinguishes a vigorous launch from a sputtering
        // one even when the eventual global reach ends up similar.
        if (time_step == activation_steps[0] + early_window)
            n_early = (int)std::count(ever_activated.begin(), ever_activated.end(), (char)1);

        // Record the interior neighbourhood in the relaxed state, right before
        // the wave is triggered. Neighbours are counted with a dedicated radius
        // (1.3x equilibrium spacing) instead of the simulation cutoff, and a
        // fixed-width boundary ring is excluded so edge cells don't bias it.
        if (time_step == activation_steps[0] - 1) {
            const float nb_cut = 1.3f * nb_spacing;  // neighbour-counting radius
            const float edge = 1.5f * nb_spacing;    // boundary ring to exclude
            float r_patch = 0.f;
            std::vector<float> rad(n_cells);
            for (int i = 0; i < n_cells; i++) {
                float dx = cells.h_X[i].x - seed_center.x;
                float dy = cells.h_X[i].y - seed_center.y;
                float dz = cells.h_X[i].z - seed_center.z;
                rad[i] = sqrtf(dx * dx + dy * dy + dz * dz);
                if (rad[i] > r_patch) r_patch = rad[i];
            }
            double sum = 0; int cnt = 0;
            for (int i = 0; i < n_cells; i++) {
                if (rad[i] >= r_patch - edge) continue;  // skip edge cells
                int nb = 0;
                for (int j = 0; j < n_cells; j++) {
                    if (j == i) continue;
                    float dx = cells.h_X[i].x - cells.h_X[j].x;
                    float dy = cells.h_X[i].y - cells.h_X[j].y;
                    float dz = cells.h_X[i].z - cells.h_X[j].z;
                    if (dx * dx + dy * dy + dz * dz < nb_cut * nb_cut) nb++;
                }
                sum += nb; cnt++;
            }
            n_interior = cnt;
            avg_neighbors_relaxed = cnt > 0 ? (float)(sum / cnt) : -1.f;
        }

        if (!scan_mode) {
            output.write_positions(cells);
            output.write_property(h_neig);
            output.write_property(h_activated);
            output.write_property(h_radius);
            for (int i = 0; i < n_cells; i++)
                h_render_radius.h_prop[i] = render_overlap * h_radius.h_prop[i];
            output.write_property(h_render_radius);
            output.write_property(h_force_mag);
        }
    }

    // Final scan metrics
    int n_fired = (int)std::count(ever_activated.begin(), ever_activated.end(), (char)1);
    printf("SCAN_RESULT R %f curvature %f threshold %f frac_activated %f "
           "avg_neighbors %f n_interior %d N %d act_fraction %f n_seed %d "
           "n_early %d peak_active %d frac_peak_active %f t_peak %d "
           "seed_x %f seed_y %f seed_z %f\n",
        surface_R, 1.f / (surface_R * surface_R), force_threshold,
        (float)n_fired / n_cells, avg_neighbors_relaxed, n_interior, n_cells,
        act_fraction, n_seed, n_early, peak_active,
        (float)peak_active / n_cells, t_peak,
        seed_center.x, seed_center.y, seed_center.z);

    return 0;
}
