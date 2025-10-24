use std::ffi::CString;

use ash::{
    Entry,
    vk::{self, KHR_DISPLAY_NAME},
};
use winit::{
    application::ApplicationHandler,
    dpi::PhysicalSize,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    raw_window_handle::{HasDisplayHandle, HasWindowHandle},
    window::{Window, WindowId},
};

struct App {
    window: Option<Window>,
    entry: ash::Entry,
    instance: ash::Instance,
    physical_device: vk::PhysicalDevice,
    device: ash::Device,
    _graphics_queue: vk::Queue,
    _present_queue: vk::Queue,
    surface: Option<vk::SurfaceKHR>,
    swap_chain: Option<vk::SwapchainKHR>,
    swap_chain_images: Option<Vec<vk::Image>>,
    image_views: Option<Vec<vk::ImageView>>,
    pipeline: vk::Pipeline,
    render_pass: Option<vk::RenderPass>,
}

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

            // let layer_properties = unsafe { entry.enumerate_instance_layer_properties().unwrap() };
            // dbg!(layer_properties);

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

            let viewport = vk::Viewport::default()
                .x(0.0)
                .y(0.0)
                .width(600.0) // TODO: Take the widht and height of the swapchain
                .height(800.0)
                .min_depth(0.0)
                .max_depth(1.0);

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

                let create_info = vk::RenderPassCreateInfo::default()
                    .attachments(color_attachments)
                    .subpasses(subpasses);

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

        // instance.get_physical_device_surface_

        Self {
            window: None,
            surface: None,
            entry,
            instance,
            physical_device,
            device,
            _graphics_queue: graphics_queue,
            _present_queue: present_queue,
            swap_chain: None,
            swap_chain_images: None,
            image_views: None,
            pipeline: graphics_pipeline,
            render_pass: Some(render_pass),
        }
    }
}

fn is_device_suitable(instance: &ash::Instance, device: vk::PhysicalDevice) -> bool {
    let device_properties = unsafe { instance.get_physical_device_properties(device) };
    let device_features = unsafe { instance.get_physical_device_features(device) };

    // let exts = unsafe { instance.enumerate_device_extension_properties(device) };
    // dbg!(exts);

    device_properties.device_type == vk::PhysicalDeviceType::INTEGRATED_GPU
        && device_features.geometry_shader == 1
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let window = event_loop
            .create_window(
                Window::default_attributes().with_inner_size(PhysicalSize::new(800, 600)),
            )
            .unwrap();

        let surface = unsafe {
            ash_window::create_surface(
                &self.entry,
                &self.instance,
                window.display_handle().unwrap().as_raw(),
                window.window_handle().unwrap().as_raw(),
                None,
            )
            .expect("Failed to create surface")
        };

        {
            let instance = ash::khr::surface::Instance::new(&self.entry, &self.instance);
            let surface_capabilities = unsafe {
                instance
                    .get_physical_device_surface_capabilities(self.physical_device, surface)
                    .expect("Failed to get device surface capabilities")
            };

            let surface_formats = unsafe {
                instance
                    .get_physical_device_surface_formats(self.physical_device, surface)
                    .expect("Failed to get surface formats")
            };

            let surface_presentation_modes = unsafe {
                instance
                    .get_physical_device_surface_present_modes(self.physical_device, surface)
                    .expect("Failed to get surface present modes")
            };

            dbg!(surface_capabilities);
            dbg!(surface_formats);
            dbg!(surface_presentation_modes);
        }

        let (swap_chain, swap_chain_images) = {
            let create_info = vk::SwapchainCreateInfoKHR::default()
                .min_image_count(2)
                .image_format(vk::Format::B8G8R8A8_SRGB)
                .image_color_space(vk::ColorSpaceKHR::SRGB_NONLINEAR)
                .image_extent(vk::Extent2D {
                    width: 800,
                    height: 600,
                })
                .image_array_layers(1)
                .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
                .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
                .clipped(true)
                .flags(vk::SwapchainCreateFlagsKHR::empty())
                .surface(surface)
                .pre_transform(vk::SurfaceTransformFlagsKHR::IDENTITY)
                .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT);

            let instance = ash::khr::swapchain::Device::new(&self.instance, &self.device);
            let swap_chain = unsafe {
                instance
                    .create_swapchain(&create_info, None)
                    .expect("Failed to create swapchain")
            };

            let swap_chain_images = unsafe {
                instance
                    .get_swapchain_images(swap_chain)
                    .expect("Failed to get swap chain images")
            };

            (swap_chain, swap_chain_images)
        };

        let image_views: Vec<_> = swap_chain_images
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
                    self.device
                        .create_image_view(&create_info, None)
                        .expect("Failed to create image view")
                }
            })
            .collect();

        // Look up if cloning for vulkan objects is an expensive operation or if they
        // are just references
        let frame_buffers: Vec<_> = image_views
            .clone()
            .into_iter()
            .map(|image| {
                let attachments = &[image];
                let create_info = vk::FramebufferCreateInfo::default()
                    .render_pass(self.render_pass.expect("Render pass not yet created"))
                    .attachments(attachments)
                    .width(800)
                    .height(600)
                    .layers(1);

                unsafe {
                    self.device
                        .create_framebuffer(&create_info, None)
                        .expect("Failed to create frame buffer")
                }
            })
            .collect();

        self.window = Some(window);
        self.surface = Some(surface);
        self.swap_chain = Some(swap_chain);
        self.swap_chain_images = Some(swap_chain_images);
        self.image_views = Some(image_views)
    }

    fn suspended(&mut self, event_loop: &ActiveEventLoop) {
        self.swap_chain = None;
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        window_id: WindowId,
        event: WindowEvent,
    ) {
        match event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }

            WindowEvent::RedrawRequested => {
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

impl App {
    pub fn main_loop(event_loop: &EventLoop<()>) {
        todo!()
    }
}

fn main() {
    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(ControlFlow::Poll);

    let mut app = App::new(&event_loop);
    event_loop.run_app(&mut app);
}
