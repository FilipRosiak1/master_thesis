#version 330

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 color;

out vec3 v_color;

uniform mat4 projectionMatrix;
uniform mat4 viewMatrix;
uniform int mode;
uniform float worldSize;
uniform float waterLevel;

void main(void) {
	gl_Position = vec4(position, 1.0);
	v_color = color;

	/*if(mode == 0)
	{
		gl_Position = projectionMatrix * viewMatrix * vec4(position, 1.0);
		height = position.y;
	}
	else
	{
		mat4 transformation = mat4(1);
		transformation[0][0] = worldSize;
		transformation[2][2] = worldSize;
		transformation[3][1] = waterLevel;
		gl_Position = projectionMatrix * viewMatrix  * transformation * vec4(position, 1.0);
		height = 0;
	}
	v_color = color;*/
}