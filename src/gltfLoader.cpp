// Define these only in *one* .cc file.
#define TINYGLTF_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
// #define TINYGLTF_NOEXCEPTION // optional. disable exception handling.

#include <iostream>
#include "gltfLoader.h"

using namespace tinygltf;

bool LoadGLTF(const std::string& filename, std::vector<Triangle>& triangles) {
    tinygltf::Model model;
    tinygltf::TinyGLTF loader;
    std::string err;
    std::string warn;
    const std::string ext = GetFilePathExtension(filename);

    bool ret = false;
    if (ext.compare("glb") == 0) {
        // assume binary glTF.
        ret = loader.LoadBinaryFromFile(&model, &err, &warn, filename.c_str());
    }
    else {
        // assume ascii glTF.
        ret = loader.LoadASCIIFromFile(&model, &err, &warn, filename.c_str());
    }

    if (!warn.empty()) {
    std::cout << "glTF parse warning: " << warn << std::endl;
    }

    if (!err.empty()) {
    std::cerr << "glTF parse error: " << err << std::endl;
    }
    if (!ret) {
    std::cerr << "Failed to load glTF: " << filename << std::endl;
    return false;
    }

    std::cout << "loaded glTF file has:\n"
    << model.accessors.size() << " accessors\n"
    << model.animations.size() << " animations\n"
    << model.buffers.size() << " buffers\n"
    << model.bufferViews.size() << " bufferViews\n"
    << model.materials.size() << " materials\n"
    << model.meshes.size() << " meshes\n"
    << model.nodes.size() << " nodes\n"
    << model.textures.size() << " textures\n"
    << model.images.size() << " images\n"
    << model.skins.size() << " skins\n"
    << model.samplers.size() << " samplers\n"
    << model.cameras.size() << " cameras\n"
    << model.scenes.size() << " scenes\n"
    << model.lights.size() << " lights\n";

    for (const auto& gltfMesh : model.meshes) {
        std::cout << "Current mesh has " << gltfMesh.primitives.size() << " primitives:\n";

        // Create a mesh object
        Mesh loadedMesh;

        loadedMesh.name = gltfMesh.name;

        std::cout << "Current mesh name: " << gltfMesh.name << "\n";

        for (const auto& meshPrimitive : gltfMesh.primitives) {
            // load positions
            const auto& positionAccessor = model.accessors[meshPrimitive.attributes.at("POSITION")];
            const auto& posBufferView = model.bufferViews[positionAccessor.bufferView];
            const auto& posBuffer = model.buffers[posBufferView.buffer];

            // calculate byte offset
            const unsigned char* posDataPtr = posBuffer.data.data() + posBufferView.byteOffset + positionAccessor.byteOffset;
            size_t vertexCount = positionAccessor.count;

            std::vector<glm::vec3> positions;

            // parse positions
            for (size_t i = 0; i < vertexCount; ++i) {
                const float* pos = reinterpret_cast<const float*>(posDataPtr + i * 12); // 3 floats * 4 bytes
                positions.emplace_back(pos[0], pos[1], pos[2]);
            }

            // load indices
            const auto& indexAccessor = model.accessors[meshPrimitive.indices];
            const auto& idxBufferView = model.bufferViews[indexAccessor.bufferView];
            const auto& idxBuffer = model.buffers[idxBufferView.buffer];

            const unsigned char* idxDataPtr = idxBuffer.data.data() + idxBufferView.byteOffset + indexAccessor.byteOffset;
            size_t indexCount = indexAccessor.count;

            std::vector<uint32_t> indices;

            std::cout << "Loaded " << (positions.size() / 3) << " vertices.\n";
            switch (indexAccessor.componentType) {
                case TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE: {
                    const uint8_t* buf = reinterpret_cast<const uint8_t*>(idxDataPtr);
                    for (size_t i = 0; i < indexCount; ++i) indices.push_back(buf[i]);
                    break;
                }
                case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT: {
                    const uint16_t* buf = reinterpret_cast<const uint16_t*>(idxDataPtr);
                    for (size_t i = 0; i < indexCount; ++i) indices.push_back(buf[i]);
                    break;
                }
                case TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT: {
                    const uint32_t* buf = reinterpret_cast<const uint32_t*>(idxDataPtr);
                    for (size_t i = 0; i < indexCount; ++i) indices.push_back(buf[i]);
                    break;
                }
                default:
                    std::cerr << "Unsupported index type: " << indexAccessor.componentType << "\n";
                    continue;
            }

            // group positions into triangles
            if (indices.size() % 3 != 0) {
                std::cerr << "Index count is not a multiple of 3.\n";
                continue;
            }

            for (size_t i = 0; i < indices.size(); i += 3) {
                uint32_t i0 = indices[i];
                uint32_t i1 = indices[i + 1];
                uint32_t i2 = indices[i + 2];

                if (i0 >= positions.size() || i1 >= positions.size() || i2 >= positions.size()) {
                std::cerr << "Index out of range!\n";
                continue;
                }

                triangles.push_back({
                    positions[i0],
                    positions[i1],
                    positions[i2]
                });
            }

            std::cout << "Loaded " << (indices.size() / 3) << " triangles.\n";
        }
    }

}
