#pragma once

#ifndef GLTF_LOADER_H_
#define GLTF_LOADER_H_

#include <tiny_gltf.h>
#include "sceneStructs.h"
#include "glm/glm.hpp"

bool LoadGLTF(const std::string& filename, std::vector<Triangle>& triangles, int& numTriangles, int& start, int& end,
    glm::vec3& min, glm::vec3& max);

#endif