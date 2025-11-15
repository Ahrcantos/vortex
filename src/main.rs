mod render_context;

use core::f32;
use std::{
    ffi::{CStr, c_void},
    u64, usize,
};

use ash::vk;
use bevy_ecs::{change_detection::Res, resource::Resource, schedule::Schedule, world::World};
use nalgebra_glm::{Mat4, Vec2, Vec3};
use winit::{
    application::ApplicationHandler,
    dpi::PhysicalSize,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    raw_window_handle::{HasDisplayHandle, HasWindowHandle},
    window::{Window, WindowId},
};

use crate::render_context::RenderContext;

const WIDTH: u32 = 800;
const HEIGHT: u32 = 600;

unsafe extern "system" fn vulkan_debug_utils_callback(
    message_severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    message_type: vk::DebugUtilsMessageTypeFlagsEXT,
    p_callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT,
    _p_user_data: *mut c_void,
) -> vk::Bool32 {
    let severity = match message_severity {
        vk::DebugUtilsMessageSeverityFlagsEXT::VERBOSE => "[Verbose]",
        vk::DebugUtilsMessageSeverityFlagsEXT::WARNING => "[Warning]",
        vk::DebugUtilsMessageSeverityFlagsEXT::ERROR => "[Error]",
        vk::DebugUtilsMessageSeverityFlagsEXT::INFO => "[Info]",
        _ => "[Unknown]",
    };
    let types = match message_type {
        vk::DebugUtilsMessageTypeFlagsEXT::GENERAL => "[General]",
        vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE => "[Performance]",
        vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION => "[Validation]",
        _ => "[Unknown]",
    };

    let message = unsafe { CStr::from_ptr((*p_callback_data).p_message) };

    println!("{} {} {:?}", severity, types, message);

    vk::FALSE
}

#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
struct Vertex {
    pos: Vec2,
    color: Vec3,
}

impl Vertex {
    pub const fn get_binding_description() -> vk::VertexInputBindingDescription {
        vk::VertexInputBindingDescription {
            binding: 0,
            stride: 20,
            input_rate: vk::VertexInputRate::VERTEX,
        }
    }

    pub const fn get_attribute_descriptions() -> &'static [vk::VertexInputAttributeDescription] {
        &[
            vk::VertexInputAttributeDescription {
                binding: 0,
                location: 0,
                format: vk::Format::R32G32_SFLOAT,
                offset: 0,
            },
            vk::VertexInputAttributeDescription {
                binding: 0,
                location: 1,
                format: vk::Format::R32G32B32_SFLOAT,
                offset: 8,
            },
        ]
    }
}

const VERTICES: &[Vertex] = &[
    Vertex {
        pos: Vec2::new(-0.5, -0.5),
        color: Vec3::new(1.0, 0.0, 0.0),
    },
    Vertex {
        pos: Vec2::new(0.5, -0.5),
        color: Vec3::new(0.0, 1.0, 0.0),
    },
    Vertex {
        pos: Vec2::new(0.5, 0.5),
        color: Vec3::new(0.0, 0.0, 1.0),
    },
    Vertex {
        pos: Vec2::new(-0.5, 0.5),
        color: Vec3::new(0.0, 0.0, 1.0),
    },
];

const INDICES: &[u16] = &[0, 1, 2, 2, 3, 0];

#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
struct UniformBufferObject {
    model: Mat4,
    view: Mat4,
    proj: Mat4,
}

impl UniformBufferObject {
    fn from_time(time: f32, aspect: f32) -> Self {
        // TODO: Fix perspective flipping image
        let model = {
            let mut m = Mat4::identity();
            let angle = time * (f32::consts::TAU / 4.0);
            let axis = Vec3::new(0.0, 0.0, 1.0);
            nalgebra_glm::rotate(&mut m, angle, &axis)
        };

        let view = {
            let eye = Vec3::new(2.0, 2.0, 2.0);
            let center = Vec3::new(0.0, 0.0, 0.0);
            let up = Vec3::new(0.0, 0.0, 1.0);
            nalgebra_glm::look_at(&eye, &center, &up)
        };

        let proj = nalgebra_glm::perspective(aspect, f32::consts::TAU / 8.0, 0.1, 10.0);

        Self { model, view, proj }
    }
}

struct App {
    window: Option<Window>,
    render_context: RenderContext,
    graphics_queue: vk::Queue,
    present_queue: vk::Queue,
    descriptor_set_layout: vk::DescriptorSetLayout,
    pipeline: vk::Pipeline,
    render_pass: vk::RenderPass,
    _command_pool: vk::CommandPool,
    command_buffer: vk::CommandBuffer,
    image_available_semaphore: vk::Semaphore,
    render_finished_semaphore: vk::Semaphore,
    in_flight_fence: vk::Fence,
    window_size: PhysicalSize<u32>,
    schedule: Schedule,
    world: World,
    swapchain_data: Option<SwapchainData>,
    vertex_buffer: vk::Buffer,
    vertex_buffer_memory: vk::DeviceMemory,
    index_buffer: vk::Buffer,
    index_buffer_memory: vk::DeviceMemory,
    // Only one uniform buffer because only one frame is in flight
    uniform_buffer: vk::Buffer,
    uniform_buffer_memory: vk::DeviceMemory,
    uniform_buffer_mapped: *mut c_void,
    descriptor_pool: vk::DescriptorPool,
    descriptor_set: vk::DescriptorSet,
    pipeline_layout: vk::PipelineLayout,
    delta: f32,
}

#[derive(Resource, Default)]
struct FrameCounter(usize);

impl App {
    pub fn new<T>(event_loop: &EventLoop<T>) -> Self {
        let render_context = RenderContext::new(event_loop);

        let device = render_context.device();
        let instance = render_context.instance();
        let entry = render_context.entry();
        let physical_device = render_context.physical_device();

        let dbg_instance = ash::ext::debug_utils::Instance::new(&entry, &instance);

        {
            let create_info = vk::DebugUtilsMessengerCreateInfoEXT::default()
                .message_severity(
                    vk::DebugUtilsMessageSeverityFlagsEXT::VERBOSE
                        | vk::DebugUtilsMessageSeverityFlagsEXT::WARNING
                        | vk::DebugUtilsMessageSeverityFlagsEXT::ERROR,
                )
                .message_type(
                    vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                        | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION
                        | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE,
                )
                .pfn_user_callback(Some(vulkan_debug_utils_callback));

            unsafe {
                dbg_instance
                    .create_debug_utils_messenger(&create_info, None)
                    .expect("Failed to create debug messenger");
            }
        }

        let graphics_queue = unsafe { device.get_device_queue(0, 0) };
        let present_queue = unsafe { device.get_device_queue(0, 0) };

        let descriptor_set_layout = {
            let ubo_layout_binding = vk::DescriptorSetLayoutBinding::default()
                .binding(0)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::VERTEX);

            let bindings = &[ubo_layout_binding];

            let create_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(bindings);

            unsafe {
                device
                    .create_descriptor_set_layout(&create_info, None)
                    .expect("Failed to create descriptor set layout")
            }
        };

        let descriptor_pool = {
            let pool_size = vk::DescriptorPoolSize::default()
                .descriptor_count(1)
                .ty(vk::DescriptorType::UNIFORM_BUFFER);
            let pool_sizes = &[pool_size];
            let pool_info = vk::DescriptorPoolCreateInfo::default()
                .pool_sizes(pool_sizes)
                .max_sets(1);

            unsafe {
                device
                    .create_descriptor_pool(&pool_info, None)
                    .expect("Failed to create descriptor pool")
            }
        };

        let descriptor_set = {
            let set_layouts = &[descriptor_set_layout];

            let alloc_info = vk::DescriptorSetAllocateInfo::default()
                .descriptor_pool(descriptor_pool)
                .set_layouts(set_layouts);

            unsafe {
                device
                    .allocate_descriptor_sets(&alloc_info)
                    .expect("Failed to allocate descriptor sets")
                    .into_iter()
                    .next()
                    .expect("No descriptor sets allocated")
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
                .cull_mode(vk::CullModeFlags::BACK)
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

        let command_pool = {
            let create_info = vk::CommandPoolCreateInfo::default()
                .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
                .queue_family_index(0);

            unsafe {
                device
                    .create_command_pool(&create_info, None)
                    .expect("Failed to create command pool")
            }
        };

        let (vertex_buffer, vertex_buffer_memory) = Self::create_vertex_buffer(
            &instance,
            physical_device,
            &device,
            command_pool,
            graphics_queue,
        );

        let (index_buffer, index_buffer_memory) = Self::create_index_buffer(
            &instance,
            physical_device,
            &device,
            command_pool,
            graphics_queue,
        );

        let (uniform_buffer, uniform_buffer_memory, uniform_buffer_mapped) =
            Self::create_uniform_buffer(&instance, physical_device, &device);

        // Configure descriptor set
        {
            let range = std::mem::size_of::<UniformBufferObject>() as u64;
            let buffer_info = vk::DescriptorBufferInfo::default()
                .buffer(uniform_buffer)
                .offset(0)
                .range(range);

            let buffer_info = &[buffer_info];

            let descriptor_write = vk::WriteDescriptorSet::default()
                .dst_set(descriptor_set)
                .dst_binding(0)
                .dst_array_element(0)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .descriptor_count(1)
                .buffer_info(buffer_info);

            let descriptor_writes = &[descriptor_write];

            unsafe {
                device.update_descriptor_sets(descriptor_writes, &[]);
            }
        }

        let command_buffer = {
            let alloc_info = vk::CommandBufferAllocateInfo::default()
                .command_pool(command_pool)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(1);

            unsafe {
                device
                    .allocate_command_buffers(&alloc_info)
                    .expect("Failed to create command buffer")
                    .pop()
                    .expect("No command buffers created")
            }
        };

        let image_available_semaphore = unsafe {
            device
                .create_semaphore(&vk::SemaphoreCreateInfo::default(), None)
                .expect("Failed to create semaphore")
        };

        let render_finished_semaphore = unsafe {
            device
                .create_semaphore(&vk::SemaphoreCreateInfo::default(), None)
                .expect("Failed to create semaphore")
        };

        let in_flight_fence = unsafe {
            device
                .create_fence(
                    &vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED),
                    None,
                )
                .expect("Failed to create fence")
        };

        let mut world = World::new();
        world.insert_resource(FrameCounter::default());
        let mut schedule = Schedule::default();

        schedule.add_systems(print_frame_count);

        Self {
            window: None,
            render_context,
            graphics_queue,
            present_queue,
            descriptor_set_layout,
            pipeline: graphics_pipeline,
            render_pass,
            _command_pool: command_pool.clone(),
            command_buffer,
            image_available_semaphore,
            render_finished_semaphore,
            in_flight_fence,
            window_size: PhysicalSize {
                width: WIDTH,
                height: HEIGHT,
            },
            world,
            schedule,
            swapchain_data: None,
            vertex_buffer,
            vertex_buffer_memory,
            index_buffer,
            index_buffer_memory,

            uniform_buffer,
            uniform_buffer_memory,
            uniform_buffer_mapped,

            descriptor_pool,
            descriptor_set,

            pipeline_layout,
            delta: 0.0,
        }
    }

    fn record_command_buffer(&self, command_buffer: vk::CommandBuffer, image_index: u32) {
        let begin_info = vk::CommandBufferBeginInfo::default();

        let device = self.render_context.device();

        unsafe {
            device
                .begin_command_buffer(command_buffer, &begin_info)
                .expect("Failed to begin command buffer")
        }

        let frame_buffer =
            self.swapchain_data.as_ref().unwrap().framebuffers.clone()[image_index as usize];

        let render_pass_begin_info = vk::RenderPassBeginInfo::default()
            .render_pass(self.render_pass.clone())
            .framebuffer(frame_buffer)
            .render_area(
                vk::Rect2D::default()
                    .extent(vk::Extent2D {
                        width: self.window_size.width,
                        height: self.window_size.height,
                    })
                    .offset(vk::Offset2D { x: 0, y: 0 }),
            )
            .clear_values(&[vk::ClearValue {
                color: vk::ClearColorValue {
                    float32: [0.0, 0.0, 0.0, 1.0],
                },
            }]);

        unsafe {
            device.cmd_begin_render_pass(
                command_buffer,
                &render_pass_begin_info,
                vk::SubpassContents::INLINE,
            );
        };

        unsafe {
            device.cmd_bind_pipeline(
                command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipeline,
            );
        };

        unsafe {
            device.cmd_bind_vertex_buffers(command_buffer, 0, &[self.vertex_buffer], &[0]);
        }

        unsafe {
            device.cmd_bind_index_buffer(
                command_buffer,
                self.index_buffer,
                0,
                vk::IndexType::UINT16,
            );
        }

        let viewport = vk::Viewport::default()
            .x(0.0)
            .y(0.0)
            .width(self.window_size.width as f32) // TODO: Take the widht and height of the swapchain
            .height(self.window_size.height as f32)
            .min_depth(0.0)
            .max_depth(1.0);

        unsafe {
            device.cmd_set_viewport(command_buffer, 0, &[viewport]);
        };

        let scissor = vk::Rect2D {
            offset: vk::Offset2D { x: 0, y: 0 },
            extent: vk::Extent2D {
                width: self.window_size.width,
                height: self.window_size.height,
            },
        };

        unsafe {
            device.cmd_set_scissor(command_buffer, 0, &[scissor]);
        }

        unsafe {
            let descriptor_sets = &[self.descriptor_set];
            device.cmd_bind_descriptor_sets(
                command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipeline_layout,
                0,
                descriptor_sets,
                &[],
            );
        }

        unsafe {
            device.cmd_draw_indexed(command_buffer, INDICES.len() as u32, 1, 0, 0, 0);
        }

        unsafe {
            device.cmd_end_render_pass(command_buffer);
        }

        unsafe {
            device
                .end_command_buffer(command_buffer)
                .expect("Failed to record command buffer");
        }
    }

    fn find_memory_type(
        instance: &ash::Instance,
        device: vk::PhysicalDevice,
        type_filter: u32,
        properties: vk::MemoryPropertyFlags,
    ) -> u32 {
        let mem_properties = unsafe { instance.get_physical_device_memory_properties(device) };

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

    fn create_uniform_buffer(
        instance: &ash::Instance,
        physical_device: vk::PhysicalDevice,
        device: &ash::Device,
    ) -> (vk::Buffer, vk::DeviceMemory, *mut c_void) {
        let buffer_size = std::mem::size_of::<UniformBufferObject>() as u64;

        let (buffer, buffer_memory) = Self::create_buffer(
            instance,
            physical_device,
            device,
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

    fn create_index_buffer(
        instance: &ash::Instance,
        physical_device: vk::PhysicalDevice,
        device: &ash::Device,
        command_pool: vk::CommandPool,
        graphics_queue: vk::Queue,
    ) -> (vk::Buffer, vk::DeviceMemory) {
        let buffer_size = (std::mem::size_of::<u16>() * INDICES.len()) as u64;
        let (staging_buffer, staging_buffer_memory) = Self::create_buffer(
            instance,
            physical_device,
            device,
            buffer_size,
            vk::BufferUsageFlags::TRANSFER_SRC,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        );

        unsafe {
            let data = device
                .map_memory(
                    staging_buffer_memory,
                    0,
                    (std::mem::size_of::<Vertex>() * VERTICES.len()) as u64,
                    vk::MemoryMapFlags::empty(),
                )
                .expect("Failed to map");

            let bytes: &[u8] = bytemuck::cast_slice(INDICES);

            std::ptr::copy_nonoverlapping(bytes.as_ptr(), data as *mut u8, buffer_size as usize);

            device.unmap_memory(staging_buffer_memory);
        };

        let (index_buffer, index_buffer_memory) = Self::create_buffer(
            instance,
            physical_device,
            device,
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

    fn create_vertex_buffer(
        instance: &ash::Instance,
        physical_device: vk::PhysicalDevice,
        device: &ash::Device,
        command_pool: vk::CommandPool,
        graphics_queue: vk::Queue,
    ) -> (vk::Buffer, vk::DeviceMemory) {
        let buffer_size = (std::mem::size_of::<Vertex>() * VERTICES.len()) as u64;
        let (staging_buffer, staging_buffer_memory) = Self::create_buffer(
            instance,
            physical_device,
            device,
            buffer_size,
            vk::BufferUsageFlags::TRANSFER_SRC,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        );

        unsafe {
            let data = device
                .map_memory(
                    staging_buffer_memory,
                    0,
                    (std::mem::size_of::<Vertex>() * VERTICES.len()) as u64,
                    vk::MemoryMapFlags::empty(),
                )
                .expect("Failed to map");

            let bytes: &[u8] = bytemuck::cast_slice(VERTICES);

            std::ptr::copy_nonoverlapping(bytes.as_ptr(), data as *mut u8, buffer_size as usize);

            device.unmap_memory(staging_buffer_memory);
        };

        let (vertex_buffer, vertex_buffer_memory) = Self::create_buffer(
            instance,
            physical_device,
            device,
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

    fn create_buffer(
        instance: &ash::Instance,
        physical_device: vk::PhysicalDevice,
        device: &ash::Device,
        size: vk::DeviceSize,
        usage: vk::BufferUsageFlags,
        properties: vk::MemoryPropertyFlags,
    ) -> (vk::Buffer, vk::DeviceMemory) {
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
            .memory_type_index(Self::find_memory_type(
                &instance,
                physical_device,
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
}

fn is_device_suitable(instance: &ash::Instance, device: vk::PhysicalDevice) -> bool {
    let device_properties = unsafe { instance.get_physical_device_properties(device) };
    let device_features = unsafe { instance.get_physical_device_features(device) };

    device_properties.device_type == vk::PhysicalDeviceType::INTEGRATED_GPU
        && device_features.geometry_shader == 1
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let window = event_loop
            .create_window(
                Window::default_attributes().with_inner_size(PhysicalSize::new(WIDTH, HEIGHT)),
            )
            .unwrap();

        let swapchain_data =
            SwapchainData::setup(&self.render_context, self.render_pass.clone(), &window);

        self.window = Some(window);
        self.swapchain_data = Some(swapchain_data);
    }

    fn suspended(&mut self, _event_loop: &ActiveEventLoop) {
        // self.cleanup_swapchain();
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        let device = self.render_context.device();
        let instance = self.render_context.instance();

        match event {
            WindowEvent::CloseRequested => {
                // Wait for rendering to finish and only then clean up
                unsafe { device.device_wait_idle().expect("Failed to wait") };

                event_loop.exit();
            }

            WindowEvent::Resized(size) => {
                unsafe {
                    device.device_wait_idle().expect("Failed to wait");
                }

                self.window_size = size;

                if let Some(swapchain) = &mut self.swapchain_data {
                    swapchain.recreate(
                        &self.render_context,
                        self.render_pass,
                        self.window.as_ref().expect("Window not present"),
                    );
                }
            }

            WindowEvent::RedrawRequested => {
                // Wait for the last frame to be fully drawn
                unsafe {
                    device
                        .wait_for_fences(&[self.in_flight_fence], true, u64::MAX)
                        .expect("Failed to wait for fence");
                };

                unsafe {
                    device
                        .reset_fences(&[self.in_flight_fence])
                        .expect("Failed to reset fence");
                };

                // Aquire next image from the swap chain
                // The returned image index is an index into swap chain images
                // for which we need to pick the associated frame buffer

                let image_index = {
                    let device = ash::khr::swapchain::Device::new(instance, device);
                    let (image_index, _) = unsafe {
                        device
                            .acquire_next_image(
                                self.swapchain_data
                                    .as_ref()
                                    .expect("Swap chain data not present")
                                    .swapchain,
                                u64::MAX,
                                self.image_available_semaphore,
                                vk::Fence::null(),
                            )
                            .expect("Failed to aquire next image")
                    };

                    image_index
                };

                let mut frame_counter = self
                    .world
                    .get_resource_mut::<FrameCounter>()
                    .expect("Frame Count not found");

                frame_counter.0 += 1;

                self.schedule.run(&mut self.world);

                // Recording the command buffer
                unsafe {
                    device
                        .reset_command_buffer(
                            self.command_buffer,
                            vk::CommandBufferResetFlags::empty(),
                        )
                        .expect("Failed to reset command buffer");
                };
                self.record_command_buffer(self.command_buffer, image_index);

                // Submitting command buffer
                {
                    let wait_semaphores = &[self.image_available_semaphore];
                    let signal_semaphores = &[self.render_finished_semaphore];
                    let wait_stages: &[vk::PipelineStageFlags] =
                        &[vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
                    let command_buffers = &[self.command_buffer];

                    let submit_info = vk::SubmitInfo::default()
                        .wait_semaphores(wait_semaphores)
                        .wait_dst_stage_mask(wait_stages)
                        .command_buffers(command_buffers)
                        .signal_semaphores(signal_semaphores);

                    unsafe {
                        device
                            .queue_submit(self.graphics_queue, &[submit_info], self.in_flight_fence)
                            .expect("Failed to submit to queue")
                    }
                };

                let aspect = self.swapchain_data.as_ref().unwrap().aspect();
                let ubo = UniformBufferObject::from_time(self.delta, aspect);

                if self.delta <= 6.2 {
                    self.delta += 0.001;
                } else {
                    self.delta = 0.0;
                }

                unsafe {
                    let data = &[ubo];
                    let data: &[u8] = bytemuck::cast_slice(data);
                    let size = std::mem::size_of::<UniformBufferObject>();
                    std::ptr::copy_nonoverlapping(
                        data.as_ptr(),
                        self.uniform_buffer_mapped as *mut u8,
                        size,
                    );
                }

                // Submit final image back to swap chain (present)
                {
                    let signal_semaphores = &[self.render_finished_semaphore];
                    let swapchains = &[self
                        .swapchain_data
                        .as_ref()
                        .expect("No swap chain defined")
                        .swapchain
                        .clone()];
                    let image_indicies = &[image_index];
                    let present_info = vk::PresentInfoKHR::default()
                        .wait_semaphores(signal_semaphores)
                        .swapchains(swapchains)
                        .image_indices(image_indicies);

                    let device = ash::khr::swapchain::Device::new(instance, device);

                    let suboptimal = unsafe {
                        device
                            .queue_present(self.present_queue, &present_info)
                            .expect("Failed to present")
                    };
                };

                self.window.as_ref().unwrap().request_redraw();
            }
            _ => (),
        }
    }
}

fn create_shader_module(device: &ash::Device, code: &[u8]) -> vk::ShaderModule {
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

fn print_frame_count(frame_counter: Res<FrameCounter>) {
    // println!("Frame {}", frame_counter.0);
}

fn main() {
    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(ControlFlow::Poll);

    let mut app = App::new(&event_loop);
    event_loop.run_app(&mut app).expect("Failed to run app");
}

struct SwapchainData {
    pub surface: vk::SurfaceKHR,
    pub swapchain: vk::SwapchainKHR,
    pub images: Vec<vk::Image>,
    pub image_views: Vec<vk::ImageView>,
    pub framebuffers: Vec<vk::Framebuffer>,
    pub surface_format: vk::SurfaceFormatKHR,
    pub presentation_mode: vk::PresentModeKHR,
    pub extent: vk::Extent2D,
}

impl SwapchainData {
    pub fn aspect(&self) -> f32 {
        self.extent.width as f32 / self.extent.height as f32
    }

    pub fn setup(
        render_context: &RenderContext,
        render_pass: vk::RenderPass,
        window: &Window,
    ) -> Self {
        let entry = render_context.entry();
        let instance = render_context.instance();
        let physical_device = render_context.physical_device();
        let device = render_context.device();

        let surface = unsafe {
            ash_window::create_surface(
                entry,
                instance,
                window.display_handle().unwrap().as_raw(),
                window.window_handle().unwrap().as_raw(),
                None,
            )
            .expect("Failed to create surface")
        };

        let khr_instance = ash::khr::surface::Instance::new(entry, instance);
        let surface_capabilities = unsafe {
            khr_instance
                .get_physical_device_surface_capabilities(physical_device, surface)
                .expect("Failed to get device surface capabilities")
        };

        let surface_formats = unsafe {
            khr_instance
                .get_physical_device_surface_formats(physical_device, surface)
                .expect("Failed to get surface formats")
        };

        let presentation_mode = {
            let surface_presentation_modes = unsafe {
                khr_instance
                    .get_physical_device_surface_present_modes(physical_device, surface)
                    .expect("Failed to get surface present modes")
            };

            surface_presentation_modes
                .into_iter()
                .find(|mode| *mode == vk::PresentModeKHR::MAILBOX)
                .unwrap_or(vk::PresentModeKHR::FIFO)
        };

        let surface_format = surface_formats
            .iter()
            .find(|format| {
                format.format == vk::Format::B8G8R8A8_SRGB
                    && format.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR
            })
            .and_then(|format| Some(format.clone()))
            .unwrap_or(surface_formats[0].clone());

        let extent = Self::choose_swap_extent(&surface_capabilities);

        let khr_device = ash::khr::swapchain::Device::new(instance, device);

        let (swapchain, swapchain_images) = {
            let create_info = vk::SwapchainCreateInfoKHR::default()
                .present_mode(presentation_mode)
                .min_image_count(2)
                .image_format(surface_format.format)
                .image_color_space(surface_format.color_space)
                .image_extent(extent)
                .image_array_layers(1)
                .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
                .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
                .clipped(true)
                .flags(vk::SwapchainCreateFlagsKHR::empty())
                .surface(surface)
                .pre_transform(vk::SurfaceTransformFlagsKHR::IDENTITY)
                .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT);

            let swapchain = unsafe {
                khr_device
                    .create_swapchain(&create_info, None)
                    .expect("Failed to create swapchain")
            };

            let swapchain_images = unsafe {
                khr_device
                    .get_swapchain_images(swapchain)
                    .expect("Failed to get swap chain images")
            };

            (swapchain, swapchain_images)
        };

        let image_views: Vec<_> = swapchain_images
            .iter()
            .map(|image| {
                let create_info = vk::ImageViewCreateInfo::default()
                    .image(image.clone())
                    .view_type(vk::ImageViewType::TYPE_2D)
                    .format(vk::Format::B8G8R8A8_SRGB)
                    .components(
                        vk::ComponentMapping::default()
                            .r(vk::ComponentSwizzle::IDENTITY)
                            .g(vk::ComponentSwizzle::IDENTITY)
                            .b(vk::ComponentSwizzle::IDENTITY)
                            .a(vk::ComponentSwizzle::IDENTITY),
                    )
                    .subresource_range(
                        vk::ImageSubresourceRange::default()
                            .aspect_mask(vk::ImageAspectFlags::COLOR)
                            .base_mip_level(0)
                            .level_count(1)
                            .base_array_layer(0)
                            .layer_count(1),
                    );

                unsafe {
                    device
                        .create_image_view(&create_info, None)
                        .expect("Failed to create image view")
                }
            })
            .collect();

        // Look up if cloning for vulkan objects is an expensive operation or if they
        // are just references
        let framebuffers: Vec<_> = image_views
            .clone()
            .into_iter()
            .map(|image| {
                let attachments = &[image];
                let create_info = vk::FramebufferCreateInfo::default()
                    .render_pass(render_pass)
                    .attachments(attachments)
                    .width(extent.width)
                    .height(extent.height)
                    .layers(1);

                unsafe {
                    device
                        .create_framebuffer(&create_info, None)
                        .expect("Failed to create frame buffer")
                }
            })
            .collect();

        Self {
            surface,
            swapchain,
            images: swapchain_images,
            image_views,
            framebuffers,
            presentation_mode,
            surface_format,
            extent,
        }
    }

    pub fn recreate(
        &mut self,
        render_context: &RenderContext,
        render_pass: vk::RenderPass,
        window: &Window,
    ) {
        let entry = render_context.entry();
        let instance = render_context.instance();
        let physical_device = render_context.physical_device();
        let device = render_context.device();

        let khr_instance = ash::khr::surface::Instance::new(entry, instance);
        let khr_device = ash::khr::swapchain::Device::new(instance, device);

        unsafe {
            khr_device.destroy_swapchain(self.swapchain, None);
        };

        // Recreate

        let surface = unsafe {
            ash_window::create_surface(
                entry,
                instance,
                window.display_handle().unwrap().as_raw(),
                window.window_handle().unwrap().as_raw(),
                None,
            )
            .expect("Failed to create surface")
        };

        let surface_capabilities = unsafe {
            khr_instance
                .get_physical_device_surface_capabilities(physical_device, surface)
                .expect("Failed to get device surface capabilities")
        };

        let extent = Self::choose_swap_extent(&surface_capabilities);

        let (swapchain, swapchain_images) = {
            let create_info = vk::SwapchainCreateInfoKHR::default()
                .present_mode(self.presentation_mode)
                .min_image_count(2)
                .image_format(self.surface_format.format)
                .image_color_space(self.surface_format.color_space)
                .image_extent(extent)
                .image_array_layers(1)
                .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
                .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
                .clipped(true)
                .flags(vk::SwapchainCreateFlagsKHR::empty())
                .surface(self.surface)
                .pre_transform(vk::SurfaceTransformFlagsKHR::IDENTITY)
                .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT);

            let swapchain = unsafe {
                khr_device
                    .create_swapchain(&create_info, None)
                    .expect("Failed to create swapchain")
            };

            let swapchain_images = unsafe {
                khr_device
                    .get_swapchain_images(swapchain)
                    .expect("Failed to get swap chain images")
            };

            (swapchain, swapchain_images)
        };

        let image_views: Vec<_> = swapchain_images
            .iter()
            .map(|image| {
                let create_info = vk::ImageViewCreateInfo::default()
                    .image(image.clone())
                    .view_type(vk::ImageViewType::TYPE_2D)
                    .format(vk::Format::B8G8R8A8_SRGB)
                    .components(
                        vk::ComponentMapping::default()
                            .r(vk::ComponentSwizzle::IDENTITY)
                            .g(vk::ComponentSwizzle::IDENTITY)
                            .b(vk::ComponentSwizzle::IDENTITY)
                            .a(vk::ComponentSwizzle::IDENTITY),
                    )
                    .subresource_range(
                        vk::ImageSubresourceRange::default()
                            .aspect_mask(vk::ImageAspectFlags::COLOR)
                            .base_mip_level(0)
                            .level_count(1)
                            .base_array_layer(0)
                            .layer_count(1),
                    );

                unsafe {
                    device
                        .create_image_view(&create_info, None)
                        .expect("Failed to create image view")
                }
            })
            .collect();

        // Look up if cloning for vulkan objects is an expensive operation or if they
        // are just references
        let framebuffers: Vec<_> = image_views
            .clone()
            .into_iter()
            .map(|image| {
                let attachments = &[image];
                let create_info = vk::FramebufferCreateInfo::default()
                    .render_pass(render_pass)
                    .attachments(attachments)
                    .width(extent.width)
                    .height(extent.height)
                    .layers(1);

                unsafe {
                    device
                        .create_framebuffer(&create_info, None)
                        .expect("Failed to create frame buffer")
                }
            })
            .collect();

        // Cleanup of old data
        for framebuffer in std::mem::replace(&mut self.framebuffers, framebuffers) {
            unsafe { device.destroy_framebuffer(framebuffer, None) };
        }

        for image_view in std::mem::replace(&mut self.image_views, image_views) {
            unsafe {
                device.destroy_image_view(image_view, None);
            }
        }

        self.swapchain = swapchain;
        self.images = swapchain_images;
        self.extent = extent;
    }

    fn choose_swap_extent(capabilities: &vk::SurfaceCapabilitiesKHR) -> vk::Extent2D {
        if capabilities.current_extent.width != u32::MAX {
            capabilities.current_extent
        } else {
            todo!()
        }
    }
}
