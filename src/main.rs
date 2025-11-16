mod frame;
mod render_context;
mod vertex;

use core::f32;
use std::{
    ffi::{CStr, c_void},
    sync::Arc,
    usize,
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

use crate::frame::FrameState;
use crate::render_context::RenderContext;
use crate::vertex::Vertex;

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

const VERTICES: &[Vertex] = &[
    // Bottom
    Vertex {
        pos: Vec3::new(0.0, 0.0, 0.0),
        color: Vec3::new(1.0, 0.0, 0.0),
    },
    Vertex {
        pos: Vec3::new(1.0, 0.0, 0.0),
        color: Vec3::new(0.0, 1.0, 0.0),
    },
    Vertex {
        pos: Vec3::new(0.0, 1.0, 0.0),
        color: Vec3::new(0.0, 0.0, 1.0),
    },
    Vertex {
        pos: Vec3::new(1.0, 1.0, 0.0),
        color: Vec3::new(0.0, 0.0, 1.0),
    },
    // Top
    Vertex {
        pos: Vec3::new(0.0, 0.0, 1.0),
        color: Vec3::new(1.0, 0.0, 0.0),
    },
    Vertex {
        pos: Vec3::new(1.0, 0.0, 1.0),
        color: Vec3::new(0.0, 1.0, 0.0),
    },
    Vertex {
        pos: Vec3::new(0.0, 1.0, 1.0),
        color: Vec3::new(0.0, 0.0, 1.0),
    },
    Vertex {
        pos: Vec3::new(1.0, 1.0, 1.0),
        color: Vec3::new(0.0, 0.0, 1.0),
    },
    // Left
    Vertex {
        pos: Vec3::new(0.0, 0.0, 0.0),
        color: Vec3::new(1.0, 0.0, 0.0),
    },
    Vertex {
        pos: Vec3::new(0.0, 0.0, 1.0),
        color: Vec3::new(0.0, 1.0, 0.0),
    },
    Vertex {
        pos: Vec3::new(1.0, 0.0, 0.0),
        color: Vec3::new(0.0, 0.0, 1.0),
    },
    Vertex {
        pos: Vec3::new(1.0, 0.0, 1.0),
        color: Vec3::new(0.0, 0.0, 1.0),
    },
    // Right
    Vertex {
        pos: Vec3::new(0.0, 1.0, 0.0),
        color: Vec3::new(1.0, 0.0, 0.0),
    },
    Vertex {
        pos: Vec3::new(0.0, 1.0, 1.0),
        color: Vec3::new(0.0, 1.0, 0.0),
    },
    Vertex {
        pos: Vec3::new(1.0, 1.0, 0.0),
        color: Vec3::new(0.0, 0.0, 1.0),
    },
    Vertex {
        pos: Vec3::new(1.0, 1.0, 1.0),
        color: Vec3::new(0.0, 0.0, 1.0),
    },
    // Back
    Vertex {
        pos: Vec3::new(0.0, 0.0, 0.0),
        color: Vec3::new(1.0, 0.0, 0.0),
    },
    Vertex {
        pos: Vec3::new(0.0, 0.0, 1.0),
        color: Vec3::new(0.0, 1.0, 0.0),
    },
    Vertex {
        pos: Vec3::new(0.0, 1.0, 0.0),
        color: Vec3::new(0.0, 0.0, 1.0),
    },
    Vertex {
        pos: Vec3::new(0.0, 1.0, 1.0),
        color: Vec3::new(0.0, 0.0, 1.0),
    },
    // Front
    Vertex {
        pos: Vec3::new(1.0, 0.0, 0.0),
        color: Vec3::new(1.0, 0.0, 0.0),
    },
    Vertex {
        pos: Vec3::new(1.0, 0.0, 1.0),
        color: Vec3::new(0.0, 1.0, 0.0),
    },
    Vertex {
        pos: Vec3::new(1.0, 1.0, 0.0),
        color: Vec3::new(0.0, 0.0, 1.0),
    },
    Vertex {
        pos: Vec3::new(1.0, 1.0, 1.0),
        color: Vec3::new(0.0, 0.0, 1.0),
    },
];

const INDICES: &[u16] = &[
    0, 2, 1, 1, 2, 3, 4, 5, 6, 5, 7, 6, 8, 10, 9, 9, 10, 11, 12, 13, 14, 13, 15, 14, 16, 17, 18,
    17, 19, 18, 20, 22, 21, 21, 22, 23,
];

#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct UniformBufferObject {
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
            nalgebra_glm::look_at_rh(&eye, &center, &up)
        };

        let proj = nalgebra_glm::perspective_rh_zo(aspect, f32::consts::TAU / 8.0, 0.1, 10.0);

        Self { model, view, proj }
    }
}

struct App {
    window: Option<Window>,
    render_context: Arc<RenderContext>,
    window_size: PhysicalSize<u32>,
    schedule: Schedule,
    world: World,
    swapchain_data: Option<SwapchainData>,
    vertex_buffer: vk::Buffer,
    vertex_buffer_memory: vk::DeviceMemory,
    index_buffer: vk::Buffer,
    index_buffer_memory: vk::DeviceMemory,
    delta: f32,
    frame_state: FrameState,
}

#[derive(Resource, Default)]
struct FrameCounter(usize);

impl App {
    pub fn new<T>(event_loop: &EventLoop<T>) -> Self {
        let render_context = RenderContext::new(event_loop);
        let render_context = Arc::new(render_context);

        let instance = render_context.instance();
        let entry = render_context.entry();

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

        let (vertex_buffer, vertex_buffer_memory) = render_context.create_vertex_buffer(VERTICES);
        let (index_buffer, index_buffer_memory) = render_context.create_index_buffer(INDICES);

        let frame_state = FrameState::new(render_context.clone());

        let mut world = World::new();
        world.insert_resource(FrameCounter::default());
        let mut schedule = Schedule::default();

        schedule.add_systems(print_frame_count);

        Self {
            window: None,
            render_context,
            frame_state,
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

            delta: 0.0,
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let window = event_loop
            .create_window(
                Window::default_attributes().with_inner_size(PhysicalSize::new(WIDTH, HEIGHT)),
            )
            .unwrap();

        let render_pass = self.render_context.render_pass();

        let swapchain_data = SwapchainData::setup(&self.render_context, render_pass, &window);

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
        let render_pass = self.render_context.render_pass();

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
                        render_pass,
                        self.window.as_ref().expect("Window not present"),
                    );
                }
            }

            WindowEvent::RedrawRequested => {
                let swapchain = self
                    .swapchain_data
                    .as_ref()
                    .expect("Swapchain data not present");
                let frame = self.frame_state.current_frame();
                frame.wait();

                let aquired_image = frame.aquire_image(swapchain);

                let mut frame_counter = self
                    .world
                    .get_resource_mut::<FrameCounter>()
                    .expect("Frame Count not found");

                frame_counter.0 += 1;

                self.schedule.run(&mut self.world);

                frame.record_command_buffer(
                    swapchain,
                    &aquired_image,
                    self.vertex_buffer,
                    self.index_buffer,
                    INDICES,
                );

                frame.submit();

                let ubo_mapping = frame.get_ubo_mapping();
                let aspect = self.swapchain_data.as_ref().unwrap().aspect();
                let ubo = UniformBufferObject::from_time(self.delta, aspect);

                self.delta += 0.00001;

                unsafe {
                    let data = &[ubo];
                    let data: &[u8] = bytemuck::cast_slice(data);
                    let size = std::mem::size_of::<UniformBufferObject>();
                    std::ptr::copy_nonoverlapping(data.as_ptr(), ubo_mapping as *mut u8, size);
                }

                self.window.as_ref().unwrap().pre_present_notify();

                frame.present(swapchain, &aquired_image);

                self.frame_state.switch_frame();

                self.window.as_ref().unwrap().request_redraw();
            }

            _ => (),
        }
    }
}

fn print_frame_count(_frame_counter: Res<FrameCounter>) {
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
