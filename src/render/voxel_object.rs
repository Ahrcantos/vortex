use std::ffi::c_void;

use ash::vk;
use nalgebra_glm::{Mat4, Vec3};

use crate::render_context::RenderContext;

use super::voxel_material::{VoxelMaterialPipeline, VoxelVertex};

#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
struct TransformUBO {
    model: Mat4,
    view: Mat4,
    proj: Mat4,
}

struct VoxelObject {
    position: Vec3,
    descriptor_set: vk::DescriptorSet,
    volume_texture: vk::Image,
    volume_texture_view: vk::ImageView,
    volume_texture_sampler: vk::Sampler,
    uniform_buffer: vk::Buffer,
    uniform_buffer_memory: vk::DeviceMemory,
    uniform_buffer_mapped: *mut c_void,
    vertex_buffer: vk::Buffer,
    vertex_buffer_memory: vk::DeviceMemory,
    index_buffer: vk::Buffer,
    index_buffer_memory: vk::DeviceMemory,
}

impl VoxelObject {
    pub fn new(render_context: &RenderContext, material: &VoxelMaterialPipeline) -> Self {
        let device = render_context.device();
        let descriptor_pool = render_context.descriptor_pool();

        let (vertex_buffer, vertex_buffer_memory) = Self::create_vertex_buffer(render_context);
        let (index_buffer, index_buffer_memory) = Self::create_index_buffer(render_context);
        let (uniform_buffer, uniform_buffer_memory, uniform_buffer_mapped) =
            Self::create_uniform_buffer(render_context);

        let (image, image_memory) = Self::create_voxel_texture(render_context);
        let image_view = Self::create_voxel_texture_image_view(render_context, image);
        let image_sampler = Self::create_texture_sampler(render_context);

        let set_layouts = &[material.descriptor_set_layout];

        let alloc_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(descriptor_pool)
            .set_layouts(set_layouts);

        let descriptor_set = unsafe {
            device
                .allocate_descriptor_sets(&alloc_info)
                .expect("Failed to allocated descriptor set")
                .into_iter()
                .next()
                .expect("No descriptor set allocated")
        };

        // Configure descriptor set
        {
            let range = std::mem::size_of::<TransformUBO>() as u64;
            let buffer_info = vk::DescriptorBufferInfo::default()
                .buffer(uniform_buffer)
                .offset(0)
                .range(range);

            let buffer_info = &[buffer_info];

            let transform_ubo_descriptor_write = vk::WriteDescriptorSet::default()
                .dst_set(descriptor_set)
                .dst_binding(0)
                .dst_array_element(0)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .descriptor_count(1)
                .buffer_info(buffer_info);

            let image_info = vk::DescriptorImageInfo::default()
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .image_view(image_view)
                .sampler(image_sampler);

            let image_info = &[image_info];

            let voxel_texture_descriptor_write = vk::WriteDescriptorSet::default()
                .dst_set(descriptor_set)
                .dst_binding(1)
                .dst_array_element(0)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(1)
                .image_info(image_info);

            let descriptor_writes = &[
                transform_ubo_descriptor_write,
                voxel_texture_descriptor_write,
            ];

            unsafe {
                device.update_descriptor_sets(descriptor_writes, &[]);
            }
        }

        Self {
            position: Vec3::new(0.0, 0.0, 0.0),
            vertex_buffer,
            vertex_buffer_memory,
            index_buffer,
            index_buffer_memory,
            uniform_buffer,
            uniform_buffer_memory,
            uniform_buffer_mapped,
            descriptor_set,
            volume_texture: image,
            volume_texture_view: image_view,
            volume_texture_sampler: image_sampler,
        }
    }

    pub fn get_transform_ubo() -> TransformUBO {
        todo!()
    }

    pub fn create_texture_sampler(render_context: &RenderContext) -> vk::Sampler {
        let device = render_context.device();

        let sampler_info = vk::SamplerCreateInfo::default()
            .mag_filter(vk::Filter::LINEAR)
            .min_filter(vk::Filter::LINEAR)
            .address_mode_u(vk::SamplerAddressMode::MIRRORED_REPEAT)
            .address_mode_v(vk::SamplerAddressMode::MIRRORED_REPEAT)
            .address_mode_w(vk::SamplerAddressMode::MIRRORED_REPEAT)
            .anisotropy_enable(false)
            .border_color(vk::BorderColor::INT_OPAQUE_BLACK)
            .unnormalized_coordinates(false)
            .compare_enable(false)
            .compare_op(vk::CompareOp::ALWAYS)
            .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
            .mip_lod_bias(0.0)
            .min_lod(0.0)
            .max_lod(0.0);

        unsafe {
            device
                .create_sampler(&sampler_info, None)
                .expect("Failed to create sampler")
        }
    }

    fn create_voxel_texture_image_view(
        render_context: &RenderContext,
        image: vk::Image,
    ) -> vk::ImageView {
        let device = render_context.device();

        let subresource_range = vk::ImageSubresourceRange::default()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .base_mip_level(0)
            .level_count(1)
            .base_array_layer(0)
            .layer_count(1);

        let view_info = vk::ImageViewCreateInfo::default()
            .image(image)
            .view_type(vk::ImageViewType::TYPE_3D)
            .format(vk::Format::R8_SRGB)
            .subresource_range(subresource_range);

        unsafe {
            device
                .create_image_view(&view_info, None)
                .expect("Failed to create image view")
        }
    }

    pub fn create_voxel_texture(render_context: &RenderContext) -> (vk::Image, vk::DeviceMemory) {
        let device = render_context.device();

        const DATA_SIZE: usize = 4 * 4 * 4;
        let mut voxel_data: [u8; DATA_SIZE] = [0x00; DATA_SIZE];
        voxel_data[0] = 0xFF;

        let (staging_buffer, staging_buffer_memory) = create_buffer(
            render_context,
            DATA_SIZE as u64,
            vk::BufferUsageFlags::TRANSFER_SRC,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        );

        unsafe {
            let data = device
                .map_memory(
                    staging_buffer_memory,
                    0,
                    DATA_SIZE as u64,
                    vk::MemoryMapFlags::empty(),
                )
                .expect("Failed to map memory");

            std::ptr::copy_nonoverlapping(voxel_data.as_ptr(), data as *mut u8, DATA_SIZE);

            device.unmap_memory(staging_buffer_memory);
        }

        let image_info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_3D)
            .extent(vk::Extent3D::default().width(4).height(4).depth(4))
            .mip_levels(1)
            .array_layers(1)
            .format(vk::Format::R8_SRGB)
            .tiling(vk::ImageTiling::OPTIMAL)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .usage(vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .samples(vk::SampleCountFlags::TYPE_1)
            .flags(vk::ImageCreateFlags::empty());

        let image = unsafe {
            device
                .create_image(&image_info, None)
                .expect("Failed to create image")
        };

        let memory_requirements = unsafe { device.get_image_memory_requirements(image) };
        let memory_type_index = find_memory_type(
            render_context,
            memory_requirements.memory_type_bits,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        );

        let alloc_info = vk::MemoryAllocateInfo::default()
            .allocation_size(memory_requirements.size)
            .memory_type_index(memory_type_index);

        let image_memory = unsafe {
            device
                .allocate_memory(&alloc_info, None)
                .expect("Failed to allocate memory")
        };

        unsafe {
            device
                .bind_image_memory(image, image_memory, 0)
                .expect("Failed to bind memory");
        }

        transition_image_layout(
            render_context,
            image,
            vk::Format::R8_SRGB,
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        );

        copy_buffer_to_image(render_context, staging_buffer, image);

        transition_image_layout(
            render_context,
            image,
            vk::Format::R8_SRGB,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        );

        unsafe {
            device.destroy_buffer(staging_buffer, None);
            device.free_memory(staging_buffer_memory, None);
        }

        (image, image_memory)
    }

    pub fn create_uniform_buffer(
        render_context: &RenderContext,
    ) -> (vk::Buffer, vk::DeviceMemory, *mut c_void) {
        let device = render_context.device();
        let buffer_size = std::mem::size_of::<TransformUBO>() as u64;

        let (buffer, buffer_memory) = create_buffer(
            render_context,
            buffer_size,
            vk::BufferUsageFlags::UNIFORM_BUFFER,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        );

        let mapped = unsafe {
            device
                .map_memory(buffer_memory, 0, buffer_size, vk::MemoryMapFlags::empty())
                .expect("Failed to map memory")
        };

        (buffer, buffer_memory, mapped)
    }

    pub fn create_index_buffer(render_context: &RenderContext) -> (vk::Buffer, vk::DeviceMemory) {
        let device = render_context.device();
        let command_pool = render_context.command_pool();
        let graphics_queue = render_context.graphics_queue();

        let buffer_size = (std::mem::size_of::<u16>() * CUBE_INDICES.len()) as u64;
        let (staging_buffer, staging_buffer_memory) = create_buffer(
            render_context,
            buffer_size,
            vk::BufferUsageFlags::TRANSFER_SRC,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        );

        unsafe {
            let data = device
                .map_memory(
                    staging_buffer_memory,
                    0,
                    (std::mem::size_of::<u16>() * CUBE_INDICES.len()) as u64,
                    vk::MemoryMapFlags::empty(),
                )
                .expect("Failed to map");

            let bytes: &[u8] = bytemuck::cast_slice(CUBE_INDICES);

            std::ptr::copy_nonoverlapping(bytes.as_ptr(), data as *mut u8, buffer_size as usize);

            device.unmap_memory(staging_buffer_memory);
        };

        let (index_buffer, index_buffer_memory) = create_buffer(
            render_context,
            buffer_size,
            vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::INDEX_BUFFER,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        );

        // Copy over memory
        let command_buffer = {
            let alloc_info = vk::CommandBufferAllocateInfo::default()
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_pool(command_pool)
                .command_buffer_count(1);

            unsafe {
                device
                    .allocate_command_buffers(&alloc_info)
                    .expect("Failed to allocate command buffer")
                    .into_iter()
                    .next()
                    .expect("At least one command buffer should be created")
            }
        };

        let begin_info = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

        unsafe {
            device
                .begin_command_buffer(command_buffer, &begin_info)
                .expect("Failed to begin command buffer")
        }

        unsafe {
            let regions = &[vk::BufferCopy::default()
                .src_offset(0)
                .dst_offset(0)
                .size(buffer_size)];

            device.cmd_copy_buffer(command_buffer, staging_buffer, index_buffer, regions);
        }

        unsafe {
            device
                .end_command_buffer(command_buffer)
                .expect("Failed to end command buffer")
        }

        let command_buffers = &[command_buffer];
        let submit_info = vk::SubmitInfo::default().command_buffers(command_buffers);
        unsafe {
            device
                .queue_submit(graphics_queue, &[submit_info], vk::Fence::null())
                .expect("Failed to submit to queue");
        }

        unsafe {
            device
                .queue_wait_idle(graphics_queue)
                .expect("Failed to wait for queue")
        }

        unsafe {
            device.free_command_buffers(command_pool, &[command_buffer]);
            device.destroy_buffer(staging_buffer, None);
            device.free_memory(staging_buffer_memory, None);
        }

        (index_buffer, index_buffer_memory)
    }

    fn create_vertex_buffer(render_context: &RenderContext) -> (vk::Buffer, vk::DeviceMemory) {
        let device = render_context.device();
        let command_pool = render_context.command_pool();
        let graphics_queue = render_context.graphics_queue();

        let buffer_size = (std::mem::size_of::<VoxelVertex>() * CUBE_VERTICIES.len()) as u64;
        let (staging_buffer, staging_buffer_memory) = create_buffer(
            render_context,
            buffer_size,
            vk::BufferUsageFlags::TRANSFER_SRC,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        );

        unsafe {
            let data = device
                .map_memory(
                    staging_buffer_memory,
                    0,
                    (std::mem::size_of::<VoxelVertex>() * CUBE_VERTICIES.len()) as u64,
                    vk::MemoryMapFlags::empty(),
                )
                .expect("Failed to map");

            let bytes: &[u8] = bytemuck::cast_slice(CUBE_VERTICIES);

            std::ptr::copy_nonoverlapping(bytes.as_ptr(), data as *mut u8, buffer_size as usize);

            device.unmap_memory(staging_buffer_memory);
        };

        let (vertex_buffer, vertex_buffer_memory) = create_buffer(
            render_context,
            buffer_size,
            vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::VERTEX_BUFFER,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        );

        // Copy over memory
        let command_buffer = {
            let alloc_info = vk::CommandBufferAllocateInfo::default()
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_pool(command_pool)
                .command_buffer_count(1);

            unsafe {
                device
                    .allocate_command_buffers(&alloc_info)
                    .expect("Failed to allocate command buffer")
                    .into_iter()
                    .next()
                    .expect("At least one command buffer should be created")
            }
        };

        let begin_info = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

        unsafe {
            device
                .begin_command_buffer(command_buffer, &begin_info)
                .expect("Failed to begin command buffer")
        }

        unsafe {
            let regions = &[vk::BufferCopy::default()
                .src_offset(0)
                .dst_offset(0)
                .size(buffer_size)];

            device.cmd_copy_buffer(command_buffer, staging_buffer, vertex_buffer, regions);
        }

        unsafe {
            device
                .end_command_buffer(command_buffer)
                .expect("Failed to end command buffer")
        }

        let command_buffers = &[command_buffer];
        let submit_info = vk::SubmitInfo::default().command_buffers(command_buffers);
        unsafe {
            device
                .queue_submit(graphics_queue, &[submit_info], vk::Fence::null())
                .expect("Failed to submit to queue");
        }

        unsafe {
            device
                .queue_wait_idle(graphics_queue)
                .expect("Failed to wait for queue")
        }

        unsafe {
            device.free_command_buffers(command_pool, &[command_buffer]);
            device.destroy_buffer(staging_buffer, None);
            device.free_memory(staging_buffer_memory, None);
        }

        (vertex_buffer, vertex_buffer_memory)
    }
}

fn copy_buffer_to_image(render_context: &RenderContext, buffer: vk::Buffer, image: vk::Image) {
    const DATA_SIZE: usize = 4 * 4 * 4;

    let device = render_context.device();
    let command_buffer = render_context.begin_single_time_commands();

    let image_subresource = vk::ImageSubresourceLayers::default()
        .aspect_mask(vk::ImageAspectFlags::COLOR)
        .mip_level(0)
        .base_array_layer(0)
        .layer_count(1);

    let region = vk::BufferImageCopy::default()
        .buffer_offset(0)
        .buffer_row_length(0)
        .buffer_image_height(0)
        .image_subresource(image_subresource)
        .image_offset(vk::Offset3D::default().x(0).y(0).z(0))
        .image_extent(vk::Extent3D::default().width(4).height(4).depth(4));

    unsafe {
        device.cmd_copy_buffer_to_image(
            command_buffer,
            buffer,
            image,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            &[region],
        );
    }

    render_context.end_single_time_commands(command_buffer);
}

fn transition_image_layout(
    render_context: &RenderContext,
    image: vk::Image,
    format: vk::Format,
    old_layout: vk::ImageLayout,
    new_layout: vk::ImageLayout,
) {
    let device = render_context.device();
    let command_buffer = render_context.begin_single_time_commands();

    let (src_access_mask, dst_access_mask, source_stage, destination_stage) =
        match (old_layout, new_layout) {
            (vk::ImageLayout::UNDEFINED, vk::ImageLayout::TRANSFER_DST_OPTIMAL) => (
                vk::AccessFlags::empty(),
                vk::AccessFlags::TRANSFER_WRITE,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::TRANSFER,
            ),
            (vk::ImageLayout::TRANSFER_DST_OPTIMAL, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL) => (
                vk::AccessFlags::TRANSFER_WRITE,
                vk::AccessFlags::SHADER_READ,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::FRAGMENT_SHADER,
            ),
            _ => panic!("Unsupported layout transition!"),
        };

    let subresource_range = vk::ImageSubresourceRange::default()
        .aspect_mask(vk::ImageAspectFlags::COLOR)
        .base_mip_level(0)
        .level_count(1)
        .base_array_layer(0)
        .layer_count(1);

    let barrier = vk::ImageMemoryBarrier::default()
        .old_layout(old_layout)
        .new_layout(new_layout)
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .image(image)
        .subresource_range(subresource_range)
        .src_access_mask(src_access_mask)
        .dst_access_mask(dst_access_mask);

    unsafe {
        device.cmd_pipeline_barrier(
            command_buffer,
            source_stage,
            destination_stage,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &[barrier],
        );
    }
    render_context.end_single_time_commands(command_buffer);
}

fn create_buffer(
    render_context: &RenderContext,
    size: vk::DeviceSize,
    usage: vk::BufferUsageFlags,
    properties: vk::MemoryPropertyFlags,
) -> (vk::Buffer, vk::DeviceMemory) {
    let device = render_context.device();

    let create_info = vk::BufferCreateInfo::default()
        .size(size)
        .usage(usage)
        .sharing_mode(vk::SharingMode::EXCLUSIVE);

    let buffer = unsafe {
        device
            .create_buffer(&create_info, None)
            .expect("Failed to create buffer")
    };

    let memory_requirements = { unsafe { device.get_buffer_memory_requirements(buffer) } };

    let alloc_info = vk::MemoryAllocateInfo::default()
        .allocation_size(size)
        .memory_type_index(find_memory_type(
            render_context,
            memory_requirements.memory_type_bits,
            properties,
        ));

    let buffer_memory = unsafe {
        device
            .allocate_memory(&alloc_info, None)
            .expect("Failed to allocated buffer memory")
    };

    unsafe {
        device
            .bind_buffer_memory(buffer, buffer_memory, 0)
            .expect("Failed to bind memory");
    }

    (buffer, buffer_memory)
}

fn find_memory_type(
    render_context: &RenderContext,
    type_filter: u32,
    properties: vk::MemoryPropertyFlags,
) -> u32 {
    let instance = render_context.instance();
    let physical_device = render_context.physical_device();

    let mem_properties = unsafe { instance.get_physical_device_memory_properties(physical_device) };

    *mem_properties
        .memory_types
        .iter()
        .enumerate()
        .find(|(index, mem)| {
            (type_filter & (1 << index) != 0x00) && (mem.property_flags & properties) == properties
        })
        .map(|(index, _)| index)
        .iter()
        .next()
        .expect("Failed to find suitable memory type") as u32
}

const CUBE_VERTICIES: &[VoxelVertex] = &[
    // Bottom
    VoxelVertex { pos: [0, 0, 0] },
    VoxelVertex { pos: [1, 0, 0] },
    VoxelVertex { pos: [0, 1, 0] },
    VoxelVertex { pos: [1, 1, 0] },
    // Top
    VoxelVertex { pos: [0, 0, 1] },
    VoxelVertex { pos: [1, 0, 1] },
    VoxelVertex { pos: [0, 1, 1] },
    VoxelVertex { pos: [1, 1, 1] },
    // Left
    VoxelVertex { pos: [0, 0, 0] },
    VoxelVertex { pos: [0, 0, 1] },
    VoxelVertex { pos: [1, 0, 0] },
    VoxelVertex { pos: [1, 0, 1] },
    // Right
    VoxelVertex { pos: [0, 1, 0] },
    VoxelVertex { pos: [0, 1, 1] },
    VoxelVertex { pos: [1, 1, 0] },
    VoxelVertex { pos: [1, 1, 1] },
    // Back
    VoxelVertex { pos: [0, 0, 0] },
    VoxelVertex { pos: [0, 0, 1] },
    VoxelVertex { pos: [0, 1, 0] },
    VoxelVertex { pos: [0, 1, 1] },
    // Front
    VoxelVertex { pos: [1, 0, 0] },
    VoxelVertex { pos: [1, 0, 1] },
    VoxelVertex { pos: [1, 1, 0] },
    VoxelVertex { pos: [1, 1, 1] },
];

const CUBE_INDICES: &[u16] = &[
    0, 2, 1, 1, 2, 3, 4, 5, 6, 5, 7, 6, 8, 10, 9, 9, 10, 11, 12, 13, 14, 13, 15, 14, 16, 17, 18,
    17, 19, 18, 20, 22, 21, 21, 22, 23,
];
