#version 450

layout(location = 0) in vec3 fragColor;
layout(location = 0) out vec4 outColor;

layout(binding = 1) uniform sampler3D voxelSampler;

void main() {
  // outColor = vec4(fragColor, 1.0);

  outColor = texture(voxelSampler, vec3(0.0, 0.0, 0.0));
  // outColor = vec4(1.0, 0.0, 0.0, 1.0);
}
