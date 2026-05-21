#version 330

in vec3 g_color;
in float height;
in vec3 normal;
in vec3 position;

out vec4 out_Color;

uniform int mode;
uniform float worldSize;

const vec3 color_low = vec3(1,0,0);
const vec3 color_high = vec3(0.5,0.5,0);
const float height_min = -5.f;
const float height_max = 5.f;
vec3 light_pos = vec3(worldSize/2, worldSize/2, worldSize);

void main(void){
    if(mode == 0)
    {
        float h = clamp(height, height_min, height_max);
        h -= height_min;
        h /= (height_max - height_min);

        vec3 lightDir = normalize(light_pos - position);
        float diff = max(dot(normal, lightDir), 0.0);
        vec3 diffuse = vec3(diff);

        if(abs(normal.z) < 0.0001)
            out_Color = vec4(diffuse * vec3(0.5, 0.5, 0.5), 1);
        else
            out_Color = vec4(diffuse * mix(color_low, color_high, h), 1);
    }
    else
        out_Color = vec4(g_color, 0.5);
}