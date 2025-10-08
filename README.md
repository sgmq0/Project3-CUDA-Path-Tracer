CUDA Path Tracer
================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 3**

* Raymond Feng
  * [LinkedIn](https://www.linkedin.com/in/raymond-ma-feng/), [personal website](https://www.rfeng.dev/)
* Tested on: Windows 11, i9-9900KF @ 3.60GHz 32GB, NVIDIA GeForce RTX 2070 SUPER (Personal Computer)

![](img/cover_image.png)

![](img/minecraft1.png)

![](img/minecraft2.png)

![](img/dragons_rainbow.png)

![](img/env_map_clouds.png)

# Project Summary
This is a GPU-based renderer built with C++ and CUDA. Originally made for a computer graphics course, I plan to add additional features in the future. Currently, the renderer supports diffuse, perfectly specular, and transmissive/glass materials. It also supports imported models in glTF 2.0 format, albedo and normal maps, and HDR environment maps. The renderer uses a bounding volume heirarchy to render large models efficiently.

**Core Features**
- Stream Compaction
- BSDF evaluation for diffuse and perfectly specular surfaces
- Stochastic sampled anti aliasing
- Material sorting

## Stream Compaction
In order for our pathtracer to run properly, we need to store each ray's remaining bounces after each iteration, and compact away any finished rays. This happens when a ray hits a light source, doesn't hit anything, or reaches the hit limit-- That is, it has 0 remaining bounces. 

To compact away the finished rays, I used the thrust library's `stable_partition` function to separate the successful and unsuccessful rays. Once a ray is discarded in this way, it is no longer casted into the scene. I also keep track of the number of remaining rays with a `num_paths` variable. When `num_paths` reaches 0, we know that the raytracing iteration has finished.

## Diffuse Surfaces
As a basic feature, the renderer supports perfectly diffuse surfaces through pathtracing. When a ray hits a diffuse surface, a cosined-sampled direction is calculated based on the surface normal, and the ray is bounced along that normal. This results in basic lambertian shading. 

| Diffuse (sphere)      | Diffuse (dragon)      |
| ------------- | ------------- |
| ![](img/diffuse_sphere.png) | ![](img/diffuse_dragon.png) |

## Stochastic Sampled Anti-aliasing
The renderer implements stochastic sampled anti-aliasing to enable smoothed renders. For this feature, I referenced ![Paul Bourke](https://paulbourke.net/miscellaneous/raytracing/)'s stochastic sampling notes. 

When a ray is casted from the camera into the screen, its direction lands at the center of the target pixel. With stochastic sampled anti-aliasing, the ray is jittered within the pixel, varying the direction slightly. This gives us a nice effect that smooths out any sharp edges in our scene.

| Without anti-aliasing      | With anti-aliasing     |
| ------------- | ------------- |
| ![](img/AA_off.png) | ![](img/AA_on.png) |

## Refractive Materials
With refractive materials, light passes through the object, sampling the scene behind and around it. When a ray enters an object, it is *refracted* in a direction calculated by Snell's law using the incoming ray direction and the material's index of refraction. When the ray exits the object, it is refracted again. 

In real life, materials are often both reflective and transmissive. We represent this by randomly choosing to sample a reflected direction or a refracted direction, with the ![Fresnel factor](https://pbr-book.org/4ed/Reflection_Models/Specular_Reflection_and_Transmission) as the threshold. Combined, this results in a convincing transmissive surface.

| Glass (IOR: 1.500)      | Diamond (IOR: 2.418)    |
| ------------- | ------------- |
| ![](img/refraction_sphere_1.png) | ![](img/refraction_sphere_2.png) |

| Spheres and Cubes     | Various Objects |
| ------------- | ------------- |
| ![](img/refraction_gems_1.png) | ![](img/refraction_gems_2.png) |

## Environment Maps
Without an environment map, any ray that does not hit any object in the scene returns the color black. Using an environment map, we can sample an image instead. The result is a convincing-looking sky.

I modified the `Scene` class to load in a `HDRI` object, which contains an image. This image is then sampled during the shading step of pathtracing to determine what color a pixel is when it doesn't hit any object in the scene.

| No Environment Map      | Environment Map |
| ------------- | ------------- |
| ![](img/cornell_env_off.png) | ![](img/cornell_env_on.png) |

| Studio     | Cloudy Sky |
| ------------- | ------------- |
| ![](img/env_map_studio.png) | ![](img/env_map_clouds.png) |

## GLTF Loading
Thanks to the ![tinygltf](https://github.com/syoyo/tinygltf) library, the pathtracer is able to read in `.glb` files. Unlike `.obj`s, textures are stored directly within the `.glb` file, making texture mapping possible with a single file!

| Minecraft Forest (30k tris)    | Multiple dragons (400k tris) |
| ------------- | ------------- |
| ![](img/cover_image.png) | ![](img/dragons_rainbow.png) |

## Texture Mapping (Albedo + Normals)
In a `.glb` file, each material stores the indices of its associated texture maps. For example, its albedo texture might be located at index 0, and its normal texture might be located at index 1. Inside the `MyMaterial` struct, `colorTextureIdx` and `normalTextureIdx` keeps track of these indices.

After iterating through each texture in order and loading it into our scene, we can access an array of `MyTexture` objects using the indices stored in `MyMaterial`. Each `Triangle` is also assigned an ID corresponding to the `MyMaterial` it should have.

In the shading step, the interpolated UVs are used to sample the correct texture map.

| Texture Mapping      | Texture + Normal Mapping |
| ------------- | ------------- |
| ![](img/normal_maps_off.png) | ![](img/normal_maps_on.png) |

### Normals
To calculate how a normal map affects the computed normal of an intersection point, I store the normal, tangent, and bitangent vectors of every vertex, which are pre-computed during BVH loading. 

When the intersection is calculated, I first find the interpolated normal, tangent, and bitangent, then compute a TBN transformation matrix. Using the interpolated UV coordinates, I sample the normal texture then apply the TBN transformation to it. The transformed normal is returned during the intersection calculation step.

| No Normal Mapping      | Normal Mapping | Normal Colors    |
| ------------- | ------------- | ------------- |
| ![](img/carbon_fibre_normals_off.png) | ![](img/carbon_fibre_normals_on.png) | ![](img/carbon_fibre_normal_map.png)

## BVH Acceleration 
Because I wanted to add custom mesh support, I needed a faster way to render each and every triangle. Naively, we can iterate through every triangle in the scene, testing a ray-triangle intersection with all of them... Obviously, with big meshes, this gets very slow.

A bounding volume hierarchy (BVH) is a data structure that greatly speeds up mesh intersection testing. Essentially, every triangle in the scene is partitioned into two groups. Then the groups are partitioned into two more groups. So on and so forth, until a binary tree structure of polygons is built. 

When computing an intersection, instead of checking the ray against every polygon in the scene, you check it against the bounding box of every node in the hierarchy. Then, only once you reach a leaf (a node containing only triangles) do you compute your ray-triangle intersections. 

For this feature, I referenced ![Jacco Bikker](https://jacco.ompf2.com/2022/04/13/how-to-build-a-bvh-part-1-basics/)'s How To Build a BVH guide.

| Dragon (BVH off)    | Dragon (BVH on) |
| ------------- | ------------- |
| ![](img/dragon_no_bvh.png) | ![](img/dragon_bvh.png) |

### Performance
This feature sped up my implementation by a *lot*!

As you can see, BVH speeds up mesh loading by hundreds of times. This is incredibly noticable on large models, such as the model with 130k tris, going from 10 seconds(!) per frame to just 67 ms. This is a 173x speedup.

The speed up is less noticable but still very much there with a smaller mesh like the duck with 4k tris, resulting in a 4x speedup.

Generally, it seems that BVH acceleration has more striking results with larger meshes. Since the BVH is constructed on the CPU, the amount of data to parse through doesn't affect the renderer's performance except in the number of triangles to render.

![](img/bvh_comparison.png)

## Russian Roulette Path Termination
Russian Roulette path termination ![(PBRTv3 13.7)](https://pbr-book.org/3ed-2018/Monte_Carlo_Integration/Russian_Roulette_and_Splitting) is an optimization that essentially discards rays without enough light. That is, they don't contribute enough to the final output. The threshold `p` I picked was `0.05`.

When we terminate rays early, we expect the pathtracer to speed up, and to get a brighter final image. 

| Russian Roulette OFF    | Russian Roulette ON |
| ------------- | ------------- |
| ![](img/cornell_rr_off.png) | ![](img/cornell_rr_on.png) |

### Performance
Unfortunately, it seemed like Russian roulette path termination caused a noticable increase in render time in the closed scene, but caused a small (3 ms/frame) decrease in render time in the open scene. 

I'm not sure why this could be happening. My hypothesis was that in a closed scene with few lights (like a Cornell box), rays are less likely to hit a light source, thus decreasing their brightness and increasing their odds of being terminated by Russian roulette.

![](img/rr_comparison.png)

## Material Sorting
In scenes with a large amount of materials, it can be more efficient to group the rays by the intersected material type, reducing divergence and speeding up the render. I used the `sort_by_key` function in the `thrust` library, and created a custom `sort_materials` predicate.

### Performance
![](img/material_sort_comparison.png)

Regardless of how many materials are in the scene, material sorting seems to require too much overhead to be worth it. When testing on open vs closed scenes, it seemed like there was a slight improvement in speed when using many materials in an open scene, but only by about 4 ms/frame. Perhaps this would change with 100 or even 1000 materials...

## Future Ideas
Though I'm done with this for now, in the near future I want to add improvements to the materials, such as a GGX microfacet model or subsurface scattering. Eventually, I want to implement something along the lines of the Disney BSDF model to support more artist-friendly materials.

I also want to add some post-process effects, like distance fog, to improve the look of the Minecraft scenes.

## Sources
- Made liberal use of the ![PBR Textbook](https://pbr-book.org/3ed-2018/contents)
- All models are from Sketchfab, all HDRIs are from Polyhaven.
- https://www.scratchapixel.com/lessons/3d-basic-rendering/ray-tracing-rendering-a-triangle/moller-trumbore-ray-triangle-intersection.html
- https://tavianator.com/2022/ray_box_boundary.html
- https://jacco.ompf2.com/2022/04/13/how-to-build-a-bvh-part-1-basics/
- https://www.khronos.org/files/gltf20-reference-guide.pdf
