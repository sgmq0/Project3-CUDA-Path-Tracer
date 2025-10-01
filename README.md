CUDA Path Tracer
================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 3**

* (TODO) YOUR NAME HERE
* Tested on: (TODO) Windows 22, i7-2222 @ 2.22GHz 22GB, GTX 222 222MB (Moore 2222 Lab)

### (TODO: Your README)

*DO NOT* leave the README to the last minute! It is a crucial part of the
project, and we will not be able to grade you without a good README.

IMPLEMENTATION NOTES:
- gltf loading loads every triangle into a vector stored in the scene object
- whichever geom object is loaded, if it's a mesh, stores the start and end index of the mesh's corresponding triangles. this is so each geom can be transformed independently of each other based on the json

SOURCES:
- pbr textbook
- https://www.scratchapixel.com/lessons/3d-basic-rendering/ray-tracing-rendering-a-triangle/moller-trumbore-ray-triangle-intersection.html
- https://tavianator.com/2022/ray_box_boundary.html