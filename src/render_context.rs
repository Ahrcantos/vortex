use std::ffi::{CString, c_char};

use ash::{Device, Entry, Instance, vk};
use winit::{event_loop::EventLoop, raw_window_handle::HasDisplayHandle};

pub struct RenderContext {
    entry: Entry,
    instance: Instance,
    physical_device: vk::PhysicalDevice,
    device: Device,
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

        Self {
            entry,
            instance,
            physical_device,
            device,
        }
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
}

impl Drop for RenderContext {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}

fn is_device_suitable(instance: &Instance, device: vk::PhysicalDevice) -> bool {
    let device_properties = unsafe { instance.get_physical_device_properties(device) };
    let device_features = unsafe { instance.get_physical_device_features(device) };

    device_properties.device_type == vk::PhysicalDeviceType::INTEGRATED_GPU
        && device_features.geometry_shader == 1
}
