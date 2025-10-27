use std::{ffi::CString, u64, usize};

use ash::{Entry, vk};
use bevy_ecs::{change_detection::Res, resource::Resource, schedule::Schedule, world::World};
use winit::{
    application::ApplicationHandler,
    dpi::PhysicalSize,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    raw_window_handle::{HasDisplayHandle, HasWindowHandle},
    window::{Window, WindowId},
};

const WIDTH: u32 = 800;
const HEIGHT: u32 = 600;

struct App {
    window: Option<Window>,
    entry: ash::Entry,
    instance: ash::Instance,
    physical_device: vk::PhysicalDevice,
    device: ash::Device,
    graphics_queue: vk::Queue,
    present_queue: vk::Queue,
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
}

#[derive(Resource, Default)]
struct FrameCounter(usize);

impl App {
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

            let extension_names = ash_window::enumerate_required_extensions(
                event_loop
                    .display_handle()
                    .expect("Could not retrieve display handle")
                    .as_raw(),
            )
            .expect("Failed to enumerate required extensions");

            let create_info = vk::InstanceCreateInfo::default()
                .flags(vk::InstanceCreateFlags::empty())
                .enabled_layer_names(layer_names)
                .enabled_extension_names(extension_names)
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

        let graphics_queue = unsafe { device.get_device_queue(0, 0) };
        let present_queue = unsafe { device.get_device_queue(0, 0) };

        let (graphics_pipeline, render_pass) = {
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

            let vertex_input_state = vk::PipelineVertexInputStateCreateInfo::default()
                .vertex_binding_descriptions(&[])
                .vertex_attribute_descriptions(&[]);

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

            let pipeline_layout = {
                let create_info = vk::PipelineLayoutCreateInfo::default();

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

            (pipeline, render_pass)
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
            entry,
            instance,
            physical_device,
            device,
            graphics_queue,
            present_queue,
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
        }
    }

    fn record_command_buffer(&self, command_buffer: vk::CommandBuffer, image_index: u32) {
        let begin_info = vk::CommandBufferBeginInfo::default();

        unsafe {
            self.device
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
            self.device.cmd_begin_render_pass(
                command_buffer,
                &render_pass_begin_info,
                vk::SubpassContents::INLINE,
            );
        };

        unsafe {
            self.device.cmd_bind_pipeline(
                command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipeline,
            );
        };

        let viewport = vk::Viewport::default()
            .x(0.0)
            .y(0.0)
            .width(self.window_size.width as f32) // TODO: Take the widht and height of the swapchain
            .height(self.window_size.height as f32)
            .min_depth(0.0)
            .max_depth(1.0);

        unsafe {
            self.device.cmd_set_viewport(command_buffer, 0, &[viewport]);
        };

        let scissor = vk::Rect2D {
            offset: vk::Offset2D { x: 0, y: 0 },
            extent: vk::Extent2D {
                width: self.window_size.width,
                height: self.window_size.height,
            },
        };

        unsafe {
            self.device.cmd_set_scissor(command_buffer, 0, &[scissor]);
        }

        unsafe {
            self.device.cmd_draw(command_buffer, 3, 1, 0, 0);
        }

        unsafe {
            self.device.cmd_end_render_pass(command_buffer);
        }

        unsafe {
            self.device
                .end_command_buffer(command_buffer)
                .expect("Failed to record command buffer");
        }
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

        let swapchain_data = SwapchainData::setup(
            &self.entry,
            &self.instance,
            self.physical_device.clone(),
            &self.device,
            self.render_pass.clone(),
            &window,
        );

        self.window = Some(window);
        self.swapchain_data = Some(swapchain_data);
    }

    fn suspended(&mut self, _event_loop: &ActiveEventLoop) {
        dbg!("Suspended");
        // self.cleanup_swapchain();
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        match event {
            WindowEvent::CloseRequested => {
                // Wait for rendering to finish and only then clean up
                unsafe { self.device.device_wait_idle().expect("Failed to wait") };

                event_loop.exit();
            }

            WindowEvent::Resized(size) => {
                unsafe {
                    self.device.device_wait_idle().expect("Failed to wait");
                }

                self.window_size = size;

                if let Some(swapchain) = &mut self.swapchain_data {
                    swapchain.recreate(
                        &self.entry,
                        &self.instance,
                        self.physical_device,
                        &self.device,
                        self.render_pass,
                        self.window.as_ref().expect("Window not present"),
                    );
                }
            }

            WindowEvent::RedrawRequested => {
                // Wait for the last frame to be fully drawn
                unsafe {
                    self.device
                        .wait_for_fences(&[self.in_flight_fence], true, u64::MAX)
                        .expect("Failed to wait for fence");
                };

                unsafe {
                    self.device
                        .reset_fences(&[self.in_flight_fence])
                        .expect("Failed to reset fence");
                };

                // Aquire next image from the swap chain
                // The returned image index is an index into swap chain images
                // for which we need to pick the associated frame buffer

                let image_index = {
                    let device = ash::khr::swapchain::Device::new(&self.instance, &self.device);
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
                    self.device
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
                        self.device
                            .queue_submit(self.graphics_queue, &[submit_info], self.in_flight_fence)
                            .expect("Failed to submit to queue")
                    }
                };

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

                    let device = ash::khr::swapchain::Device::new(&self.instance, &self.device);

                    unsafe {
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
}

impl SwapchainData {
    pub fn setup(
        entry: &ash::Entry,
        instance: &ash::Instance,
        physical_device: vk::PhysicalDevice,
        device: &ash::Device,
        render_pass: vk::RenderPass,
        window: &Window,
    ) -> Self {
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

        let extend = Self::choose_swap_extend(&surface_capabilities);

        let khr_device = ash::khr::swapchain::Device::new(instance, device);

        let (swapchain, swapchain_images) = {
            let create_info = vk::SwapchainCreateInfoKHR::default()
                .present_mode(presentation_mode)
                .min_image_count(2)
                .image_format(surface_format.format)
                .image_color_space(surface_format.color_space)
                .image_extent(extend)
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
                    .width(extend.width)
                    .height(extend.height)
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
        }
    }

    pub fn recreate(
        &mut self,
        entry: &ash::Entry,
        instance: &ash::Instance,
        physical_device: vk::PhysicalDevice,
        device: &ash::Device,
        render_pass: vk::RenderPass,
        window: &Window,
    ) {
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

        let extend = Self::choose_swap_extend(&surface_capabilities);

        let (swapchain, swapchain_images) = {
            let create_info = vk::SwapchainCreateInfoKHR::default()
                .present_mode(self.presentation_mode)
                .min_image_count(2)
                .image_format(self.surface_format.format)
                .image_color_space(self.surface_format.color_space)
                .image_extent(extend)
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
                    .width(extend.width)
                    .height(extend.height)
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
    }

    fn choose_swap_extend(capabilities: &vk::SurfaceCapabilitiesKHR) -> vk::Extent2D {
        if capabilities.current_extent.width != u32::MAX {
            capabilities.current_extent
        } else {
            todo!()
        }
    }
}
