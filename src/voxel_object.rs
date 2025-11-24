use std::sync::Arc;

use ash::vk;
use nalgebra_glm::Vec3;

use crate::render_context::RenderContext;

struct Vertex {
    pos: [u8; 3]
}

impl Vertex {
    pub const fn get_binding_description() -> vk::VertexInputBindingDescription {
        vk::VertexInputBindingDescription {
            binding: 0,
            stride: 3,
            input_rate: vk::VertexInputRate::VERTEX
        }
    }

    pub const fn get_attribute_descriptions() -> &'static [vk::VertexInputAttributeDescription] {
        &[
            vk::VertexInputAttributeDescription {
                binding: 0,
                location: 0,
                format: vk::Format::R8G8B8_UINT,
                offset: 3,
            }
        ]
    }
}

const CUBE_VERTICIES: &[Vertex] = &[
    // Bottom
    Vertex {
        pos: [0, 0, 0],
    },
    Vertex {
        pos: [1, 0, 0],
    },
    Vertex {
        pos: [0, 1, 0],
    },
    Vertex {
        pos: [1, 1, 0],
    },
    // Top
    Vertex {
        pos: [0, 0, 1],
    },
    Vertex {
        pos: [1, 0, 1],
    },
    Vertex {
        pos: [0, 1, 1],
    },
    Vertex {
        pos: [1, 1, 1],
    },
    // Left
    Vertex {
        pos: [0, 0, 0],
    },
    Vertex {
        pos: [0, 0, 1],
    },
    Vertex {
        pos: [1, 0, 0],
    },
    Vertex {
        pos: [1, 0, 1],
    },
    // Right
    Vertex {
        pos: [0, 1, 0],
    },
    Vertex {
        pos: [0, 1, 1],
    },
    Vertex {
        pos: [1, 1, 0],
    },
    Vertex {
        pos: [1, 1, 1],
    },
    // Back
    Vertex {
        pos: [0, 0, 0],
    },
    Vertex {
        pos: [0, 0, 1],
    },
    Vertex {
        pos: [0, 1, 0],
    },
    Vertex {
        pos: [0, 1, 1],
    },
    // Front
    Vertex {
        pos: [1, 0, 0],
    },
    Vertex {
        pos: [1, 0, 1],
    },
    Vertex {
        pos: [1, 1, 0],
    },
    Vertex {
        pos: [1, 1, 1],
    },
]

struct VoxelData {
    data: Vec<u8>,
}

impl VoxelData {
    pub fn new(width: usize, height: usize, depth: usize) -> Self {
        let size = width * height * depth;

        let data = vec![0x00; size];

        Self {
            data
        }
    }
}

struct VoxelObject {
    position: Vec3,
    voxel_data: VoxelData,
    material_instance: Arc<VoxelMaterialInstance>,
    volume_texture: vk::Image,
    volume_texture_view: vk::ImageView,
    volume_texture_sampler: vk::Sampler,
}

impl VoxelObject {
    pub fn draw(&self, render_context: &RenderContext, command_buffer: &mut vk::CommandBuffer) {
        todo!()
    }

    fn get_projection() -> ()
}

struct VoxelMaterialInstance {
    pipeline: VoxelMaterialPipeline,
    set: vk::DescriptorSet,
}

struct VoxelMaterialPipeline {
    pipeline: Arc<vk::Pipeline>,
    layout: vk::PipelineLayout,
}