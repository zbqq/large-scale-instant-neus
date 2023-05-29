#include "utils.h"



PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
    

    m.def("morton3D",&morton3D);
    m.def("morton3D_invert",&morton3D_invert);

    m.def("packbits_u32",&packbits_u32);
    m.def("un_packbits_u32",&un_packbits_u32);
    m.def("distance_mask",&distance_mask);
    m.def("mega_nerf_mask",&mega_nerf_mask);

    m.def("near_far_from_aabb",&near_far_from_aabb);
    m.def("sph_from_ray",&sph_from_ray);
    m.def("march_rays_train",&march_rays_train);
    
}

