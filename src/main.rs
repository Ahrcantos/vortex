use std::ffi::CString;
use std::ptr;

use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    window::{Window, WindowId},
};
use ash::vk;

#[derive(Default)]
struct App {
    window: Option<Window>,
}

impl App {
    pub fn new() -> Self {
        let entry = ash::Entry::new().unwrap();

        let instance = {
            let app_name = CString::new("vortex").unwrap();
            let engine_name = CString::new("Vulkan Engine").unwrap();
            let app_info = vk::ApplicationInfo {
                s_type: vk::StructureType::APPLICATION_INFO,
                p_next: ptr::null(),
                p_application_name: app_name.as_ptr(),
                application_version: vk::make_api_version(0, 0, 0, 1),
                p_engine_name: engine_name.as_ptr(),
                engine_version: vk::make_api_version(0, 0, 0, 1),
                api_version: vk::API_VERSION_1_0,
                ..Default::default()
            };
        };

        Self {
            window: None,
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        self.window = Some(
            event_loop
                .create_window(Window::default_attributes())
                .unwrap(),
        );
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

impl App {
    pub fn main_loop(event_loop: &EventLoop<()>) {
        todo!()
    }
}

fn main() {
    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(ControlFlow::Poll);

    let mut app = App::default();
    event_loop.run_app(&mut app);
}
