#version 450

layout(binding = 0) uniform ModelData {
  mat4 model;
  mat4 view;
  mat4 proj;
  vec3 cameraPosition;
  vec3 volumeMin; // In model space (if min is always 0,0,0 do we even need to specify it?)
  vec3 volumeMax;
} data;

layout(location = 0) in vec3 inPosition;

layout(location = 0) out vec3 fragColor;

void main() {
  gl_Position = data.proj * data.view * data.model * vec4(inPosition, 1.0);
  fragColor = vec3(1.0, 1.0, 1.0);
}
