#version 330

layout(triangles) in;
in vec3 v_color[];

layout (triangle_strip, max_vertices = 3) out;
out vec3 g_color;
out float height;
out vec3 normal;
out vec3 position;

uniform mat4 projectionMatrix;
uniform mat4 viewMatrix;
uniform int mode;
uniform float worldSize;
uniform float waterLevel;

vec3 GetNormal()
{
    vec3 a = vec3(gl_in[0].gl_Position) - vec3(gl_in[1].gl_Position);
    vec3 b = vec3(gl_in[2].gl_Position) - vec3(gl_in[1].gl_Position);
    return normalize(cross(a, b));
}

void main(void) {
    for(int i = 0; i < 3; i++)
    {
        if(mode == 0)
        {
            gl_Position = projectionMatrix * viewMatrix * gl_in[i].gl_Position;
            height = gl_in[i].gl_Position.z;
            position = gl_in[i].gl_Position.xyz;
        }
        else
        {
            mat4 transformation = mat4(1);
            transformation[0][0] = worldSize;
            transformation[1][1] = worldSize;
            transformation[3][2] = waterLevel;
            gl_Position = projectionMatrix * viewMatrix  * transformation * gl_in[i].gl_Position;
            position = vec3(transformation * gl_in[i].gl_Position);
            height = 0;
        }
        g_color = v_color[0];
        normal = -GetNormal().xyz;
        EmitVertex();
    }
    EndPrimitive();
}