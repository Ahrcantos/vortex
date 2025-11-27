use std::ffi::{CString, c_char, c_void}; use ash::{Device, Entry, Instance, vk}; use winit::{event_loop::EventLoop, raw_window_handle::HasDisplayHandle};
use crate::{UniformBufferObject, vertex::Vertex};

pub struct RenderContext {
    entry: Entry,
    instance: Instance,
    physical_device: vk::PhysicalDevice,
    device: Device,

    graphics_queue: vk::Queue,
    present_queue: vk::Queue,

    command_pool: vk::CommandPool, // A command pool is tied to a specific queue family? New command pool per render pass?

    // I think for these there can be multiple? Better to tie them to a different structure
    pipeline: vk::Pipeline,
    render_pass: vk::RenderPass,
    pipeline_layout: vk::PipelineLayout,
    descriptor_set_layout: vk::DescriptorSetLayout,
    descriptor_pool: vk::DescriptorPool,
}

impl RenderContext {
    pub fn new<T>(event_loop: &EventLoop<T>) -> Self {
        let entry = Entry::linked();
        let instance = {
            let app_name = CString::new("vortex").unwrap();
            let engine_name = CString::new("Vulkan Engine").unwrap();

            let app_info = vk::ApplicationInfo::default()
                .application_name(&app_name)
                .application_version(vk::make_api_version(0, 0, 0, 1))
                .engine_name(&engine_name)
                .engine_version(vk::make_api_version(0, 0, 0, 1))
                .api_version(vk::API_VERSION_1_0);

            let layer_names = &[{ c"VK_LAYER_KHRONOS_validation".as_ptr() }];

            let mut extension_names: Vec<*const c_char> =
                ash_window::enumerate_required_extensions(
                    event_loop
                        .display_handle()
                        .expect("Could not retrieve display handle")
                        .as_raw(),
                )
                .expect("Failed to enumerate required extensions")
                .into_iter()
                .map(|extension| *extension)
                .collect();

            extension_names.push(c"VK_EXT_debug_utils".as_ptr());

            let create_info = vk::InstanceCreateInfo::default()
                .flags(vk::InstanceCreateFlags::empty())
                .enabled_layer_names(layer_names)
                .enabled_extension_names(&extension_names[..])
                .application_info(&app_info);

            unsafe {
                entry
                    .create_instance(&create_info, None)
                    .expect("Failed to create instance")
            }
        };

        let physical_devices = unsafe {
            instance
                .enumerate_physical_devices()
                .expect("Failed to enumerate physical devices")
        };

        dbg!(&physical_devices);

        let physical_device = physical_devices
            .into_iter()
            .find(|device| is_device_suitable(&instance, device.clone()))
            .expect("No suiteable device found");

        let device = {
            let queue_create_info = vk::DeviceQueueCreateInfo::default()
                .queue_family_index(0) // TODO: look up index
                .queue_priorities(&[1.0]);

            let device_features = vk::PhysicalDeviceFeatures::default();

            let queue_create_infos = &[queue_create_info];

            let extension_names = &[c"VK_KHR_swapchain".as_ptr()];

            let device_create_info = vk::DeviceCreateInfo::default()
                .queue_create_infos(queue_create_infos)
                .enabled_features(&device_features)
                .enabled_extension_names(extension_names);

            unsafe {
                instance
                    .create_device(physical_device, &device_create_info, None)
                    .expect("Failed to create logical device")
            }
        };

        let command_pool = {
            let create_info = vk::CommandPoolCreateInfo::default()
                .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
                .queue_family_index(0); // TODO: Figure out queue index

            unsafe {
                device
                    .create_command_pool(&create_info, None)
                    .expect("Failed to create command pool")
            }
        };

        let graphics_queue = unsafe { device.get_device_queue(0, 0) };
        let present_queue = unsafe { device.get_device_queue(0, 0) };

        let descriptor_set_layout = {
            let ubo_layout_binding = vk::DescriptorSetLayoutBinding::default()
                .binding(0)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::VERTEX);

            let sampler_layout_binding = vk::DescriptorSetLayoutBinding::default()
                .binding(1)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::FRAGMENT);

            let bindings = &[ubo_layout_binding, sampler_layout_binding];

            let create_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(bindings);

            unsafe {
                device
                    .create_descriptor_set_layout(&create_info, None)
                    .expect("Failed to create descriptor set layout")
            }
        };

        let descriptor_pool = {
            let ubo_pool_size = vk::DescriptorPoolSize::default()
                .descriptor_count(1024) // was 2
                .ty(vk::DescriptorType::UNIFORM_BUFFER);

            let sampler_pool_size = vk::DescriptorPoolSize::default()
                .descriptor_count(1024)
                .ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER);

            let pool_sizes = &[ubo_pool_size, sampler_pool_size];
            let pool_info = vk::DescriptorPoolCreateInfo::default()
                .pool_sizes(pool_sizes)
                .max_sets(2);

            unsafe {
                device
                    .create_descriptor_pool(&pool_info, None)
                    .expect("Failed to create descriptor pool")
            }
        };

        let (graphics_pipeline, render_pass, pipeline_layout) = {
            let vertex_shader =
                create_shader_module(&device, include_bytes!("../shaders/vert.spv"));

            let fragment_shader =
                create_shader_module(&device, include_bytes!("../shaders/frag.spv"));

            let vertex_shader_stage = vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::VERTEX)
                .name(c"main")
                .module(vertex_shader);

            let fragment_shader_stage = vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::FRAGMENT)
                .name(c"main")
                .module(fragment_shader);

            let dynamic_state = vk::PipelineDynamicStateCreateInfo::default()
                .dynamic_states(&[vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR]);

            let binding_descriptions = &[Vertex::get_binding_description()];
            let attribute_descriptions = Vertex::get_attribute_descriptions();

            let vertex_input_state = vk::PipelineVertexInputStateCreateInfo::default()
                .vertex_binding_descriptions(binding_descriptions)
                .vertex_attribute_descriptions(attribute_descriptions);

            let input_assembly_state = vk::PipelineInputAssemblyStateCreateInfo::default()
                .topology(vk::PrimitiveTopology::TRIANGLE_LIST)
                .primitive_restart_enable(false);

            let viewport_state = vk::PipelineViewportStateCreateInfo::default()
                .viewport_count(1)
                .scissor_count(1);

            let rasterization_state = vk::PipelineRasterizationStateCreateInfo::default()
                .depth_clamp_enable(false)
                .rasterizer_discard_enable(false)
                .polygon_mode(vk::PolygonMode::FILL)
                .line_width(1.0)
                .cull_mode(vk::CullModeFlags::FRONT)
                .front_face(vk::FrontFace::CLOCKWISE)
                .depth_bias_enable(false);

            let multisample_state = vk::PipelineMultisampleStateCreateInfo::default()
                .sample_shading_enable(false)
                .rasterization_samples(vk::SampleCountFlags::TYPE_1);

            let color_blend_attachment = vk::PipelineColorBlendAttachmentState::default()
                .color_write_mask(
                    vk::ColorComponentFlags::R
                        | vk::ColorComponentFlags::G
                        | vk::ColorComponentFlags::B
                        | vk::ColorComponentFlags::A,
                )
                .blend_enable(false);

            let color_blend_attachments = &[color_blend_attachment];

            let color_blend_state = vk::PipelineColorBlendStateCreateInfo::default()
                .logic_op_enable(false)
                .attachments(color_blend_attachments);

            let set_layouts = &[descriptor_set_layout];

            let pipeline_layout = {
                let create_info = vk::PipelineLayoutCreateInfo::default().set_layouts(set_layouts);

                unsafe {
                    device
                        .create_pipeline_layout(&create_info, None)
                        .expect("Failed to create pipeline layout")
                }
            };

            let render_pass = {
                let color_attachment = vk::AttachmentDescription::default()
                    .format(vk::Format::B8G8R8A8_SRGB)
                    .samples(vk::SampleCountFlags::TYPE_1)
                    .load_op(vk::AttachmentLoadOp::CLEAR)
                    .store_op(vk::AttachmentStoreOp::STORE)
                    .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
                    .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
                    .initial_layout(vk::ImageLayout::UNDEFINED)
                    .final_layout(vk::ImageLayout::PRESENT_SRC_KHR);

                let color_attachment_ref = vk::AttachmentReference::default()
                    .attachment(0)
                    .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);

                let color_attachment_refs = &[color_attachment_ref];

                let subpass = vk::SubpassDescription::default()
                    .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
                    .color_attachments(color_attachment_refs);

                let color_attachments = &[color_attachment];
                let subpasses = &[subpass];

                let dependency = vk::SubpassDependency::default()
                    .src_subpass(vk::SUBPASS_EXTERNAL)
                    .dst_subpass(0)
                    .src_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
                    .src_access_mask(vk::AccessFlags::empty())
                    .dst_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
                    .dst_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE);

                let dependencies = &[dependency];

                let create_info = vk::RenderPassCreateInfo::default()
                    .attachments(color_attachments)
                    .subpasses(subpasses)
                    .dependencies(dependencies);

                unsafe {
                    device
                        .create_render_pass(&create_info, None)
                        .expect("Failed to create render pass")
                }
            };

            let stages = &[vertex_shader_stage, fragment_shader_stage];

            let pipeline_create_info = vk::GraphicsPipelineCreateInfo::default()
                .stages(stages)
                .vertex_input_state(&vertex_input_state)
                .input_assembly_state(&input_assembly_state)
                .viewport_state(&viewport_state)
                .rasterization_state(&rasterization_state)
                .multisample_state(&multisample_state)
                .color_blend_state(&color_blend_state)
                .dynamic_state(&dynamic_state)
                .layout(pipeline_layout)
                .render_pass(render_pass)
                .subpass(0);

            let pipeline = unsafe {
                device
                    .create_graphics_pipelines(
                        vk::PipelineCache::null(),
                        &[pipeline_create_info],
                        None,
                    )
                    .expect("Failed to create graphics pipeline")
                    .into_iter()
                    .next()
                    .expect("No pipeline was created")
            };

            unsafe {
                device.destroy_shader_module(vertex_shader, None);
                device.destroy_shader_module(fragment_shader, None);
            }

            (pipeline, render_pass, pipeline_layout)
        };

        Self {
            entry,
            instance,
            physical_device,
            device,

            graphics_queue,
            present_queue,

            command_pool,

            pipeline: graphics_pipeline,
            render_pass,
            pipeline_layout,
            descriptor_set_layout,
            descriptor_pool,
        }
    }

    pub fn create_texture_sampler(&self) -> vk::Sampler {
        let device = self.device();

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

    pub fn create_voxel_texture_image_view(&self, image: vk::Image) -> vk::ImageView {
        let device = self.device();

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

    pub fn create_voxel_texture(&self) -> (vk::Image, vk::DeviceMemory) {
        let device = self.device();

        const DATA_SIZE: usize = 4 * 4 * 4;
        let mut voxel_data: [u8; DATA_SIZE] = [0x00; DATA_SIZE];
        voxel_data[0] = 0xFF;

        let (staging_buffer, staging_buffer_memory) = self.create_buffer(
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
        let memory_type_index = self.find_memory_type(
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

        self.transition_image_layout(
            image,
            vk::Format::R8_SRGB,
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        );

        self.copy_buffer_to_image(staging_buffer, image);

        self.transition_image_layout(
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

    pub fn create_vertex_buffer(&self, verticies: &[Vertex]) -> (vk::Buffer, vk::DeviceMemory) {
        let buffer_size = (std::mem::size_of::<Vertex>() * verticies.len()) as u64;
        let (staging_buffer, staging_buffer_memory) = self.create_buffer(
            buffer_size,
            vk::BufferUsageFlags::TRANSFER_SRC,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        );

        unsafe {
            let data = self
                .device
                .map_memory(
                    staging_buffer_memory,
                    0,
                    (std::mem::size_of::<Vertex>() * verticies.len()) as u64,
                    vk::MemoryMapFlags::empty(),
                )
                .expect("Failed to map");

            let bytes: &[u8] = bytemuck::cast_slice(verticies);

            std::ptr::copy_nonoverlapping(bytes.as_ptr(), data as *mut u8, buffer_size as usize);

            self.device.unmap_memory(staging_buffer_memory);
        };

        let (vertex_buffer, vertex_buffer_memory) = self.create_buffer(
            buffer_size,
            vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::VERTEX_BUFFER,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        );

        // Copy over memory
        let command_buffer = {
            let alloc_info = vk::CommandBufferAllocateInfo::default()
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_pool(self.command_pool)
                .command_buffer_count(1);

            unsafe {
                self.device
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
            self.device
                .begin_command_buffer(command_buffer, &begin_info)
                .expect("Failed to begin command buffer")
        }

        unsafe {
            let regions = &[vk::BufferCopy::default()
                .src_offset(0)
                .dst_offset(0)
                .size(buffer_size)];

            self.device
                .cmd_copy_buffer(command_buffer, staging_buffer, vertex_buffer, regions);
        }

        unsafe {
            self.device
                .end_command_buffer(command_buffer)
                .expect("Failed to end command buffer")
        }

        let command_buffers = &[command_buffer];
        let submit_info = vk::SubmitInfo::default().command_buffers(command_buffers);
        unsafe {
            self.device
                .queue_submit(self.graphics_queue, &[submit_info], vk::Fence::null())
                .expect("Failed to submit to queue");
        }

        unsafe {
            self.device
                .queue_wait_idle(self.graphics_queue)
                .expect("Failed to wait for queue")
        }

        unsafe {
            self.device
                .free_command_buffers(self.command_pool, &[command_buffer]);
            self.device.destroy_buffer(staging_buffer, None);
            self.device.free_memory(staging_buffer_memory, None);
        }

        (vertex_buffer, vertex_buffer_memory)
    }

    pub fn create_uniform_buffer(&self) -> (vk::Buffer, vk::DeviceMemory, *mut c_void) {
        let device = self.device();
        let buffer_size = std::mem::size_of::<UniformBufferObject>() as u64;

        let (buffer, buffer_memory) = self.create_buffer(
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

    pub fn create_index_buffer(&self, indices: &[u16]) -> (vk::Buffer, vk::DeviceMemory) {
        let buffer_size = (std::mem::size_of::<u16>() * indices.len()) as u64;
        let (staging_buffer, staging_buffer_memory) = self.create_buffer(
            buffer_size,
            vk::BufferUsageFlags::TRANSFER_SRC,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        );

        unsafe {
            let data = self
                .device
                .map_memory(
                    staging_buffer_memory,
                    0,
                    (std::mem::size_of::<u16>() * indices.len()) as u64,
                    vk::MemoryMapFlags::empty(),
                )
                .expect("Failed to map");

            let bytes: &[u8] = bytemuck::cast_slice(indices);

            std::ptr::copy_nonoverlapping(bytes.as_ptr(), data as *mut u8, buffer_size as usize);

            self.device.unmap_memory(staging_buffer_memory);
        };

        let (index_buffer, index_buffer_memory) = self.create_buffer(
            buffer_size,
            vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::INDEX_BUFFER,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        );

        // Copy over memory
        let command_buffer = {
            let alloc_info = vk::CommandBufferAllocateInfo::default()
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_pool(self.command_pool)
                .command_buffer_count(1);

            unsafe {
                self.device
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
            self.device
                .begin_command_buffer(command_buffer, &begin_info)
                .expect("Failed to begin command buffer")
        }

        unsafe {
            let regions = &[vk::BufferCopy::default()
                .src_offset(0)
                .dst_offset(0)
                .size(buffer_size)];

            self.device
                .cmd_copy_buffer(command_buffer, staging_buffer, index_buffer, regions);
        }

        unsafe {
            self.device
                .end_command_buffer(command_buffer)
                .expect("Failed to end command buffer")
        }

        let command_buffers = &[command_buffer];
        let submit_info = vk::SubmitInfo::default().command_buffers(command_buffers);
        unsafe {
            self.device
                .queue_submit(self.graphics_queue, &[submit_info], vk::Fence::null())
                .expect("Failed to submit to queue");
        }

        unsafe {
            self.device
                .queue_wait_idle(self.graphics_queue)
                .expect("Failed to wait for queue")
        }

        unsafe {
            self.device
                .free_command_buffers(self.command_pool, &[command_buffer]);
            self.device.destroy_buffer(staging_buffer, None);
            self.device.free_memory(staging_buffer_memory, None);
        }

        (index_buffer, index_buffer_memory)
    }

    pub fn create_buffer(
        &self,
        size: vk::DeviceSize,
        usage: vk::BufferUsageFlags,
        properties: vk::MemoryPropertyFlags,
    ) -> (vk::Buffer, vk::DeviceMemory) {
        let create_info = vk::BufferCreateInfo::default()
            .size(size)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        let buffer = unsafe {
            self.device
                .create_buffer(&create_info, None)
                .expect("Failed to create buffer")
        };

        let memory_requirements = { unsafe { self.device.get_buffer_memory_requirements(buffer) } };

        let alloc_info = vk::MemoryAllocateInfo::default()
            .allocation_size(size)
            .memory_type_index(
                self.find_memory_type(memory_requirements.memory_type_bits, properties),
            );

        let buffer_memory = unsafe {
            self.device
                .allocate_memory(&alloc_info, None)
                .expect("Failed to allocated buffer memory")
        };

        unsafe {
            self.device
                .bind_buffer_memory(buffer, buffer_memory, 0)
                .expect("Failed to bind memory");
        }

        (buffer, buffer_memory)
    }

    fn copy_buffer_to_image(&self, buffer: vk::Buffer, image: vk::Image) {
        const DATA_SIZE: usize = 4 * 4 * 4;

        let device = self.device();
        let command_buffer = self.begin_single_time_commands();

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

        self.end_single_time_commands(command_buffer);
    }

    fn transition_image_layout(
        &self,
        image: vk::Image,
        format: vk::Format,
        old_layout: vk::ImageLayout,
        new_layout: vk::ImageLayout,
    ) {
        let device = self.device();
        let command_buffer = self.begin_single_time_commands();

        let (src_access_mask, dst_access_mask, source_stage, destination_stage) =
            match (old_layout, new_layout) {
                (vk::ImageLayout::UNDEFINED, vk::ImageLayout::TRANSFER_DST_OPTIMAL) => (
                    vk::AccessFlags::empty(),
                    vk::AccessFlags::TRANSFER_WRITE,
                    vk::PipelineStageFlags::TOP_OF_PIPE,
                    vk::PipelineStageFlags::TRANSFER,
                ),
                (
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                ) => (
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
        self.end_single_time_commands(command_buffer);
    }

    pub fn begin_single_time_commands(&self) -> vk::CommandBuffer {
        let device = self.device();

        let command_buffer = {
            let alloc_info = vk::CommandBufferAllocateInfo::default()
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_pool(self.command_pool)
                .command_buffer_count(1);

            unsafe {
                self.device
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
                .expect("Failed to begin command buffer");
        }

        command_buffer
    }

    pub fn end_single_time_commands(&self, command_buffer: vk::CommandBuffer) {
        let device = self.device();

        unsafe {
            device
                .end_command_buffer(command_buffer)
                .expect("Failed to end command buffer")
        }

        let command_buffers = &[command_buffer];
        let submit_info = vk::SubmitInfo::default().command_buffers(command_buffers);
        unsafe {
            device
                .queue_submit(self.graphics_queue, &[submit_info], vk::Fence::null())
                .expect("Failed to submit to queue");
        }

        unsafe {
            device
                .queue_wait_idle(self.graphics_queue)
                .expect("Failed to wait for queue")
        }

        unsafe {
            device.free_command_buffers(self.command_pool, &[command_buffer]);
        }
    }

    fn find_memory_type(&self, type_filter: u32, properties: vk::MemoryPropertyFlags) -> u32 {
        let mem_properties = unsafe {
            self.instance
                .get_physical_device_memory_properties(self.physical_device)
        };

        *mem_properties
            .memory_types
            .iter()
            .enumerate()
            .find(|(index, mem)| {
                (type_filter & (1 << index) != 0x00)
                    && (mem.property_flags & properties) == properties
            })
            .map(|(index, _)| index)
            .iter()
            .next()
            .expect("Failed to find suitable memory type") as u32
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn entry(&self) -> &Entry {
        &self.entry
    }

    pub fn instance(&self) -> &Instance {
        &self.instance
    }

    pub fn physical_device(&self) -> vk::PhysicalDevice {
        self.physical_device
    }

    pub fn command_pool(&self) -> vk::CommandPool {
        self.command_pool
    }

    pub fn descriptor_pool(&self) -> vk::DescriptorPool {
        self.descriptor_pool
    }

    pub fn descriptor_set_layout(&self) -> vk::DescriptorSetLayout {
        self.descriptor_set_layout
    }

    pub fn render_pass(&self) -> vk::RenderPass {
        self.render_pass
    }

    pub fn pipeline(&self) -> vk::Pipeline {
        self.pipeline
    }

    pub fn pipeline_layout(&self) -> vk::PipelineLayout {
        self.pipeline_layout
    }

    pub fn graphics_queue(&self) -> vk::Queue {
        self.graphics_queue
    }

    pub fn present_queue(&self) -> vk::Queue {
        self.present_queue
    }
}

impl Drop for RenderContext {
    fn drop(&mut self) {
        unsafe {
            self.device
                .destroy_descriptor_pool(self.descriptor_pool, None);
            self.device
                .destroy_descriptor_set_layout(self.descriptor_set_layout, None);
            self.device.destroy_command_pool(self.command_pool, None);
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}

fn is_device_suitable(instance: &Instance, device: vk::PhysicalDevice) -> bool {
    let device_properties = unsafe { instance.get_physical_device_properties(device) };
    let device_features = unsafe { instance.get_physical_device_features(device) };

    device_properties.device_type == vk::PhysicalDeviceType::DISCRETE_GPU
        && device_features.geometry_shader == 1
}

fn create_shader_module(device: &Device, code: &[u8]) -> vk::ShaderModule {
    let create_info = vk::ShaderModuleCreateInfo {
        p_code: code.as_ptr() as *const u32,
        code_size: code.len(),
        ..Default::default()
    };

    let shader_module = unsafe {
        device
            .create_shader_module(&create_info, None)
            .expect("Failed to create shader module")
    };

    shader_module
}
