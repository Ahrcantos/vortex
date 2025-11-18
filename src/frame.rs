use std::{ffi::c_void, sync::Arc};

use ash::vk;

use crate::{SwapchainData, UniformBufferObject, render_context::RenderContext};

pub struct FrameState {
    left_frame: Frame,
    right_frame: Frame,
    current_frame: CurrentFrame,
}

impl FrameState {
    pub fn new(
        render_context: Arc<RenderContext>,
        image_view: vk::ImageView,
        sampler: vk::Sampler,
    ) -> Self {
        let left_frame = Frame::new(render_context.clone(), image_view, sampler);
        let right_frame = Frame::new(render_context, image_view, sampler);

        Self {
            left_frame,
            right_frame,
            current_frame: CurrentFrame::Left,
        }
    }

    pub fn current_frame(&self) -> &Frame {
        match self.current_frame {
            CurrentFrame::Left => &self.left_frame,
            CurrentFrame::Right => &self.right_frame,
        }
    }

    pub fn switch_frame(&mut self) {
        self.current_frame = match self.current_frame {
            CurrentFrame::Right => CurrentFrame::Left,
            CurrentFrame::Left => CurrentFrame::Right,
        }
    }
}

pub enum CurrentFrame {
    Left,
    Right,
}

pub struct Frame {
    render_context: Arc<RenderContext>,
    image_available_semaphore: vk::Semaphore,
    render_finished_semaphore: vk::Semaphore,
    in_flight_fence: vk::Fence,
    uniform_buffer: vk::Buffer,
    uniform_buffer_memory: vk::DeviceMemory,
    uniform_buffer_mapped: *mut c_void,
    descriptor_set: vk::DescriptorSet,
    command_buffer: vk::CommandBuffer,
}

pub struct AquiredImage {
    pub index: u32,
    pub frame_buffer: vk::Framebuffer,
}

impl Frame {
    pub fn new(
        render_context: Arc<RenderContext>,
        image_view: vk::ImageView,
        sampler: vk::Sampler,
    ) -> Self {
        let device = render_context.device();
        let command_pool = render_context.command_pool();
        let descriptor_pool = render_context.descriptor_pool();
        let descriptor_set_layout = render_context.descriptor_set_layout();

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

        let (uniform_buffer, uniform_buffer_memory, uniform_buffer_mapped) =
            render_context.create_uniform_buffer();

        // Configure descriptor set
        {
            let range = std::mem::size_of::<UniformBufferObject>() as u64;
            let buffer_info = vk::DescriptorBufferInfo::default()
                .buffer(uniform_buffer)
                .offset(0)
                .range(range);

            let buffer_info = &[buffer_info];

            let buffer_descriptor_write = vk::WriteDescriptorSet::default()
                .dst_set(descriptor_set)
                .dst_binding(0)
                .dst_array_element(0)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .descriptor_count(1)
                .buffer_info(buffer_info);

            let image_info = vk::DescriptorImageInfo::default()
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .image_view(image_view)
                .sampler(sampler);

            let image_info = &[image_info];

            let image_descriptor_write = vk::WriteDescriptorSet::default()
                .dst_set(descriptor_set)
                .dst_binding(1)
                .dst_array_element(0)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(1)
                .image_info(image_info);

            let descriptor_writes = &[buffer_descriptor_write, image_descriptor_write];

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

        Self {
            render_context,
            image_available_semaphore,
            render_finished_semaphore,
            in_flight_fence,
            command_buffer,
            descriptor_set,
            uniform_buffer,
            uniform_buffer_mapped,
            uniform_buffer_memory,
        }
    }

    pub fn wait(&self) {
        let device = self.render_context.device();

        // Wait for the last frame to be fully drawn
        unsafe {
            device
                .wait_for_fences(&[self.in_flight_fence], true, u64::MAX)
                .expect("Failed to wait for fence");
        };
    }

    pub fn aquire_image(&self, swapchain: &SwapchainData) -> AquiredImage {
        let device = self.render_context.device();
        let instance = self.render_context.instance();

        let image_index = {
            let device = ash::khr::swapchain::Device::new(instance, device);
            let (image_index, _) = unsafe {
                device
                    .acquire_next_image(
                        swapchain.swapchain,
                        u64::MAX,
                        self.image_available_semaphore,
                        vk::Fence::null(),
                    )
                    .expect("Failed to aquire next image")
            };

            image_index
        };

        let frame_buffer = swapchain.framebuffers.clone()[image_index as usize];

        AquiredImage {
            index: image_index,
            frame_buffer,
        }
    }

    pub fn record_command_buffer(
        &self,
        swapchain: &SwapchainData,
        aquired_image: &AquiredImage,
        vertex_buffer: vk::Buffer,
        index_buffer: vk::Buffer,
        indices: &[u16],
    ) {
        let device = self.render_context.device();
        let render_pass = self.render_context.render_pass();

        let frame_buffer = aquired_image.frame_buffer.clone();

        // Recording the command buffer
        unsafe {
            device
                .reset_command_buffer(self.command_buffer, vk::CommandBufferResetFlags::empty())
                .expect("Failed to reset command buffer");
        };

        let begin_info = vk::CommandBufferBeginInfo::default();

        unsafe {
            device
                .begin_command_buffer(self.command_buffer, &begin_info)
                .expect("Failed to begin command buffer")
        }

        let render_pass_begin_info = vk::RenderPassBeginInfo::default()
            .render_pass(render_pass)
            .framebuffer(frame_buffer)
            .render_area(
                vk::Rect2D::default()
                    .extent(vk::Extent2D {
                        // Changed to swapchain extent
                        width: swapchain.extent.width,
                        height: swapchain.extent.height,
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
                self.command_buffer,
                &render_pass_begin_info,
                vk::SubpassContents::INLINE,
            );
        };

        unsafe {
            device.cmd_bind_pipeline(
                self.command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                self.render_context.pipeline(),
            );
        };

        unsafe {
            device.cmd_bind_vertex_buffers(self.command_buffer, 0, &[vertex_buffer], &[0]);
        }

        unsafe {
            device.cmd_bind_index_buffer(
                self.command_buffer,
                index_buffer,
                0,
                vk::IndexType::UINT16,
            );
        }

        let viewport = vk::Viewport::default()
            .x(0.0)
            .y(0.0)
            .width(swapchain.extent.width as f32)
            .height(swapchain.extent.height as f32)
            .min_depth(0.0)
            .max_depth(1.0);

        unsafe {
            device.cmd_set_viewport(self.command_buffer, 0, &[viewport]);
        };

        let scissor = vk::Rect2D {
            offset: vk::Offset2D { x: 0, y: 0 },
            extent: vk::Extent2D {
                width: swapchain.extent.width,
                height: swapchain.extent.height,
            },
        };

        unsafe {
            device.cmd_set_scissor(self.command_buffer, 0, &[scissor]);
        }

        unsafe {
            let descriptor_sets = &[self.descriptor_set];
            device.cmd_bind_descriptor_sets(
                self.command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                self.render_context.pipeline_layout(),
                0,
                descriptor_sets,
                &[],
            );
        }

        unsafe {
            device.cmd_draw_indexed(self.command_buffer, indices.len() as u32, 1, 0, 0, 0);
        }

        unsafe {
            device.cmd_end_render_pass(self.command_buffer);
        }

        unsafe {
            device
                .end_command_buffer(self.command_buffer)
                .expect("Failed to record command buffer");
        }
    }

    pub fn submit(&self) {
        let device = self.render_context.device();

        unsafe {
            device
                .reset_fences(&[self.in_flight_fence])
                .expect("Failed to reset fence");
        };

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

            let graphics_queue = self.render_context.graphics_queue();

            unsafe {
                device
                    .queue_submit(graphics_queue, &[submit_info], self.in_flight_fence)
                    .expect("Failed to submit to queue")
            }
        };
    }

    pub fn present(&self, swapchain: &SwapchainData, aquired_image: &AquiredImage) {
        let device = self.render_context.device();
        let instance = self.render_context.instance();

        // Submit final image back to swap chain (present)
        {
            let signal_semaphores = &[self.render_finished_semaphore];
            let swapchains = &[swapchain.swapchain.clone()];
            let image_indicies = &[aquired_image.index];
            let present_info = vk::PresentInfoKHR::default()
                .wait_semaphores(signal_semaphores)
                .swapchains(swapchains)
                .image_indices(image_indicies);

            let device = ash::khr::swapchain::Device::new(instance, device);

            let present_queue = self.render_context.present_queue();

            unsafe {
                device
                    .queue_present(present_queue, &present_info)
                    .expect("Failed to present")
            };
        };
    }

    pub fn get_ubo_mapping(&self) -> *mut c_void {
        self.uniform_buffer_mapped
    }
}

impl Drop for Frame {
    fn drop(&mut self) {
        let device = self.render_context.device();

        unsafe {
            device.destroy_semaphore(self.image_available_semaphore, None);
            device.destroy_semaphore(self.render_finished_semaphore, None);
            device.destroy_fence(self.in_flight_fence, None);
            device.destroy_buffer(self.uniform_buffer, None);
            // When freeing this memory a
            // pointer to the mapped memory
            // can escape out into the program
            device.free_memory(self.uniform_buffer_memory, None);
        }
    }
}
