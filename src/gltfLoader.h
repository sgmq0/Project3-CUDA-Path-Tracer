#pragma once

#ifndef GLTF_LOADER_H_
#define GLTF_LOADER_H_

#include <tiny_gltf.h>
#include "sceneStructs.h"
#include "glm/glm.hpp"

bool LoadGLTF(const std::string& filename, 
    std::vector<Triangle>& triangles, 
    std::vector<MyTexture>& textures,
    std::vector<MyMaterial>& materials,
    std::unordered_map<std::string, uint32_t>& materialMap,
    glm::mat4 transform,
    int materialID,
    bool useMaterial);

#endif