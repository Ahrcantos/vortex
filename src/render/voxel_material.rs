use ash::{vk, Device};

use crate::render_context::RenderContext;

struct VoxelMaterialPipeline {
    pipeline: vk::Pipeline,
    layout: vk::PipelineLayout,
    pass: vk::RenderPass,
    descriptor: vk::DescriptorSet,
    descriptor_set_layout: vk::DescriptorSetLayout,
}

impl VoxelMaterialPipeline {
    pub fn new(render_context: &RenderContext) -> Self {
        let device = render_context.device();

        let vertex_shader = create_shader_module(
            &device,
            include_bytes!("../../shaders/voxel/voxel.vert.spv"),
        );

        let fragment_shader = create_shader_module(
            &device,
            include_bytes!("../../shaders/voxel/voxel.frag.spv"),
        );

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

        // TODO: Use dynamic rendering?
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
                .create_graphics_pipelines(vk::PipelineCache::null(), &[pipeline_create_info], None)
                .expect("Failed to create graphics pipeline")
                .into_iter()
                .next()
                .expect("No pipeline was created")
        };

        unsafe {
            device.destroy_shader_module(vertex_shader, None);
            device.destroy_shader_module(fragment_shader, None);
        }

        Self {
            pipeline,
            layout: pipeline_layout,
            pass: render_pass,
            descriptor,
            descriptor_set_layout,
        }
    }

    pub fn cleanup(&mut self, render_context: &RenderContext) {
        let device = render_context.device();

        unsafe {
            device.destroy_pipeline(self.pipeline, None);
            device.destroy_pipeline_layout(self.layout, None);
            device.destroy_render_pass(self.pass, None);
        }
    }
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
